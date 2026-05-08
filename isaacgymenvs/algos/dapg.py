"""DAPG-augmented PPO agent for rl_games.

Implements *Demo Augmented Policy Gradient* (Rajeswaran et al. 2017) on top of
rl_games' :class:`A2CAgent`.  At every PPO mini-batch update we add an
auxiliary behaviour-cloning loss on a fixed expert dataset:

.. math::

    \\mathcal{L}_\\text{DAPG} =
        \\mathcal{L}_\\text{PPO}
        + \\lambda(t) \\cdot \\mathbb{E}_{(s,a) \\sim \\mathcal{D}_E}[
            -\\log \\pi_\\theta(a \\mid s) ].

``λ(t)`` decays multiplicatively each PPO epoch (default 0.999) so that the BC
term fades as the policy improves.

Demo file format (``demo_path`` in the train cfg):

* a ``.pt`` saved via ``torch.save({"observations": Tensor, "actions": Tensor}, path)``.
* ``observations`` shape ``(M, obs_dim)`` and ``actions`` shape ``(M, act_dim)``.
* Both tensors are float32, on CPU.  They will be moved to the agent device on
  load.

Activate by setting ``params.algo.name: dapg`` in the rl_games train YAML and
populating ``params.config.dapg.demo_path``.

This file registers the algo via ``register()``; ``isaacgymenvs/__init__.py``
calls it on import so ``rl_games.torch_runner.Runner`` can find the builder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

try:
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from rl_games.algos_torch import torch_ext
    from rl_games.common import common_losses
    from rl_games.torch_runner import Runner
    _RL_GAMES_AVAILABLE = True
except Exception:  # pragma: no cover — optional at import time
    _RL_GAMES_AVAILABLE = False


class _DemoBuffer:
    """In-memory expert dataset; samples uniform mini-batches without replacement."""

    def __init__(self, observations: Tensor, actions: Tensor, device: str | torch.device):
        assert observations.shape[0] == actions.shape[0], (
            f"demo length mismatch: obs={observations.shape[0]} acts={actions.shape[0]}"
        )
        self._obs = observations.to(device).float()
        self._acts = actions.to(device).float()
        self._n = self._obs.shape[0]
        self._device = device

    def __len__(self) -> int:
        return self._n

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if batch_size >= self._n:
            return self._obs, self._acts
        idx = torch.randint(0, self._n, (batch_size,), device=self._device)
        return self._obs[idx], self._acts[idx]


class DAPGAgent(A2CAgent):
    """A2C/PPO agent with an additional BC loss on expert demonstrations."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        cfg = params.get("config", {}).get("dapg", {})
        demo_path = cfg.get("demo_path", None)
        if demo_path is None:
            raise ValueError(
                "DAPGAgent requires params.config.dapg.demo_path to be set"
            )
        # Resolve relative paths against the project root.
        if not os.path.isabs(demo_path):
            project_root = Path(__file__).resolve().parents[2]
            demo_path = str(project_root / demo_path)
        if not os.path.isfile(demo_path):
            raise FileNotFoundError(
                f"DAPG demo file not found: {demo_path}.  "
                "Run tools/collect_demo.py first."
            )

        data = torch.load(demo_path, map_location="cpu")
        if not (isinstance(data, dict) and "observations" in data and "actions" in data):
            raise ValueError(
                f"DAPG demo file {demo_path} must be a dict with keys "
                "'observations' and 'actions'."
            )

        self._demo_buffer = _DemoBuffer(
            observations=data["observations"],
            actions=data["actions"],
            device=self.ppo_device,
        )

        self._lambda = float(cfg.get("lambda_init", 0.1))
        self._lambda_decay = float(cfg.get("lambda_decay", 0.999))
        self._demo_batch_size = int(cfg.get("demo_minibatch_size", 0)) or self.minibatch_size

        # Diagnostics
        self._bc_loss_running = 0.0
        self._bc_loss_running_count = 0

    # ------------------------------------------------------------------
    def _bc_loss(self) -> Tensor:
        """Behaviour-cloning loss on a fresh demo mini-batch."""
        obs_e, act_e = self._demo_buffer.sample(self._demo_batch_size)
        obs_e = self._preproc_obs(obs_e)
        batch_dict = {
            "is_train": True,
            "prev_actions": act_e,
            "obs": obs_e,
        }
        if self.is_rnn:
            # Demos are treated as i.i.d. transitions: zero-init the RNN.
            seq_len = max(1, self.seq_length)
            n = obs_e.shape[0]
            # Pad the demo batch to a multiple of seq_len so the LSTM unroll
            # works.  We chop off any remainder — fine for a stochastic loss.
            nb = (n // seq_len) * seq_len
            if nb == 0:
                return torch.zeros((), device=self.ppo_device)
            obs_e = obs_e[:nb]
            act_e = act_e[:nb]
            batch_dict["obs"] = obs_e
            batch_dict["prev_actions"] = act_e
            batch_dict["seq_length"] = seq_len
            batch_dict["rnn_states"] = [
                torch.zeros(
                    rs.shape[0], nb // seq_len, rs.shape[2], device=self.ppo_device
                )
                for rs in self.rnn_states
            ] if hasattr(self, "rnn_states") and self.rnn_states is not None else None
            # If the policy stores per-step RNN states differently, the user
            # can supply a smaller seq_length via mini_epochs config to stay
            # safe.  We intentionally keep this minimal.
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res = self.model(batch_dict)
            # ``prev_neglogp`` is the negative log prob of the *teacher* action
            # under the current policy — exactly what BC wants to minimise.
            neglogp = res["prev_neglogp"]
            return neglogp.mean()

    # ------------------------------------------------------------------
    def calc_gradients(self, input_dict):
        # First run the standard PPO gradient computation; this populates
        # self.train_result with (a_loss, c_loss, ...).  We then take an extra
        # backward+step on the BC loss multiplied by lambda.
        super().calc_gradients(input_dict)

        if self._lambda <= 0.0:
            return

        bc_loss = self._bc_loss()
        if not torch.isfinite(bc_loss):
            return  # skip pathological mini-batches silently

        scaled = self._lambda * bc_loss
        for p in self.model.parameters():
            p.grad = None
        self.scaler.scale(scaled).backward()
        # Reuse rl_games' gradient clipping/optimiser helper.
        self.trancate_gradients_and_step()

        self._bc_loss_running += float(bc_loss.detach().cpu())
        self._bc_loss_running_count += 1

    # ------------------------------------------------------------------
    def update_epoch(self):
        ep = super().update_epoch()
        # Decay lambda once per PPO epoch.
        self._lambda *= self._lambda_decay
        # Log running BC loss to extras (TensorBoard).
        if self._bc_loss_running_count > 0:
            avg = self._bc_loss_running / self._bc_loss_running_count
            self.diagnostics.epoch(self, "dapg/bc_loss", avg)
            self.diagnostics.epoch(self, "dapg/lambda", self._lambda)
            self._bc_loss_running = 0.0
            self._bc_loss_running_count = 0
        return ep


def register() -> None:
    """Register the DAPG agent with rl_games' algo factory."""
    if not _RL_GAMES_AVAILABLE:
        return
    # Hook into the singleton Runner so that
    # `Runner().algo_factory.create('dapg', ...)` works.  We patch the class
    # method in __init__ so it's idempotent.
    _orig_init = Runner.__init__

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            self.algo_factory.register_builder(
                "dapg", lambda **kw: DAPGAgent(**kw)
            )
        except Exception:
            pass

    Runner.__init__ = patched_init
