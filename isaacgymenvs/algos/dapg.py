"""DAPG-augmented PPO agent for rl_games.

Implements *Demo Augmented Policy Gradient* (Rajeswaran et al. 2017) on top of
rl_games' :class:`A2CAgent`.  The standard PPO loss is augmented with a
behaviour-cloning term computed on a fixed expert dataset, and both terms are
backpropagated in a *single* gradient step:

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

Notes on RNN handling.  The default LSTM-Asymmetric config trains a recurrent
policy.  Because the BC samples are i.i.d. transitions (no temporal structure
implied), we always evaluate the BC loss with ``seq_length = 1`` and a fresh
zeroed RNN state.  This keeps the auxiliary loss well-defined and avoids
hidden-state leakage across unrelated demos.
"""

from __future__ import annotations

import os
from pathlib import Path

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
    """In-memory expert dataset; samples uniform mini-batches with replacement."""

    def __init__(self, observations: Tensor, actions: Tensor, device: str | torch.device):
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
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
        """Behaviour-cloning loss on a fresh demo mini-batch.

        Demos are treated as i.i.d. transitions; under an LSTM policy we still
        evaluate ``seq_length=1`` with a zero-initialised hidden state so that
        each demo example is scored independently — this avoids hidden-state
        leakage between unrelated transitions.
        """
        obs_e, act_e = self._demo_buffer.sample(self._demo_batch_size)
        obs_e = self._preproc_obs(obs_e)
        batch_dict = {
            "is_train": True,
            "prev_actions": act_e,
            "obs": obs_e,
        }
        if self.is_rnn:
            n = obs_e.shape[0]
            batch_dict["seq_length"] = 1
            # rnn_states layout in rl_games: list of tensors with shape
            # (num_layers, n_seq, hidden); each step contains seq_length frames
            # so n_seq = n // seq_length = n.
            try:
                template_states = self.rnn_states  # type: ignore[attr-defined]
            except AttributeError:
                template_states = None
            if template_states is not None:
                batch_dict["rnn_states"] = [
                    torch.zeros(
                        rs.shape[0], n, rs.shape[2],
                        device=self.ppo_device, dtype=rs.dtype,
                    )
                    for rs in template_states
                ]
            # ``zero_rnn_on_done`` requires a dones tensor; provide a dummy.
            if getattr(self, "zero_rnn_on_done", False):
                batch_dict["dones"] = torch.zeros(
                    n, device=self.ppo_device, dtype=torch.bool
                )

        res = self.model(batch_dict)
        # ``prev_neglogp`` is the negative log prob of the *teacher* action
        # under the current policy — exactly what BC wants to minimise.
        return res["prev_neglogp"].mean()

    # ------------------------------------------------------------------
    def calc_gradients(self, input_dict):
        # Straight-line port of the parent's PPO loss assembly with one extra
        # additive term: ``+ λ * BC``.  We intentionally re-implement the
        # body so that the combined loss is back-propagated in a single
        # ``self.scaler.scale(loss).backward()`` call rather than two.
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if getattr(self, "zero_rnn_on_done", False):
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage,
                self.ppo, curr_e_clip,
            )
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip,
                    return_batch, self.clip_value,
                )
            else:
                c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)

            entropy_coef = self.entropy_coef

            losses, sum_mask = torch_ext.apply_masks(
                [
                    a_loss.unsqueeze(1),
                    c_loss,
                    (entropy_coef * entropy).unsqueeze(1),
                    b_loss.unsqueeze(1),
                ],
                rnn_masks,
            )
            a_loss, c_loss, entropy_loss, b_loss = (
                losses[0], losses[1], losses[2], losses[3],
            )

            ppo_loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy_loss
                + b_loss * self.bounds_loss_coef
            )

            # Auxiliary BC term — always evaluated even if lambda is 0 so the
            # loss graph stays stable; the scalar lambda multiplier zeros out
            # the contribution if requested.
            if self._lambda > 0.0:
                bc_loss = self._bc_loss()
                if not torch.isfinite(bc_loss):
                    bc_loss = torch.zeros((), device=self.ppo_device)
            else:
                bc_loss = torch.zeros((), device=self.ppo_device)

            loss = ppo_loss + self._lambda * bc_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        all_grads = self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(),
                old_mu_batch, old_sigma_batch, reduce_kl,
            )
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        # Track BC loss for end-of-epoch reporting.
        self._bc_loss_running += float(bc_loss.detach().cpu())
        self._bc_loss_running_count += 1

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(
            ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip
        ).float()
        extras = {
            "on_policy_contrib": contrib.mean().item(),
            "off_policy_contrib": 0,
            "on_policy_grads": all_grads.detach().cpu(),
            "off_policy_grads": torch.zeros_like(all_grads).cpu(),
        }

        self.train_result = (
            a_loss, c_loss,
            torch_ext.apply_masks([entropy.unsqueeze(1)], rnn_masks)[0][0],
            kl_dist, self.last_lr, lr_mul,
            mu.detach(), sigma.detach(), b_loss, extras,
        )

    # ------------------------------------------------------------------
    def train_epoch(self):
        """Run one PPO epoch then decay lambda.

        rl_games' control flow is::

            while True:
                self.update_epoch()    # increments epoch counter
                self.train_epoch()     # consumes self._lambda inside calc_gradients

        Doing the decay inside ``update_epoch()`` would mean the very first
        training epoch consumes ``lambda_init * lambda_decay`` instead of
        ``lambda_init``.  We therefore tie the decay to the END of
        ``train_epoch`` so that:

          * epoch 0 sees ``lambda_init`` (correct);
          * the value logged for epoch N is the value actually used in epoch N.
        """
        # Snapshot the lambda value about to be consumed for clean logging.
        epoch_lambda = self._lambda
        # Reset the per-epoch BC accumulators before the parent calls
        # calc_gradients repeatedly.
        self._bc_loss_running = 0.0
        self._bc_loss_running_count = 0

        result = super().train_epoch()

        # Log
        writer = getattr(self, "writer", None)
        if writer is not None:
            current_epoch = getattr(self, "epoch_num", 0)
            if self._bc_loss_running_count > 0:
                avg = self._bc_loss_running / self._bc_loss_running_count
                writer.add_scalar("dapg/bc_loss", avg, current_epoch)
            writer.add_scalar("dapg/lambda", epoch_lambda, current_epoch)

        # Decay AFTER use so the next train_epoch sees the decayed value.
        self._lambda *= self._lambda_decay
        return result


def register() -> None:
    """Register the DAPG agent with rl_games' algo factory."""
    if not _RL_GAMES_AVAILABLE:
        return
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
