"""Sanity tests for the VideoRLFollower training pipeline.

Runs 4 environments for ~50 sim steps and verifies:

  1. The env constructs from the cfg without raising.
  2. The object spawns at (or close to) the trajectory's start_pose.
  3. Sending non-zero arm joint targets actually moves the arm DOFs.
  4. Sending non-zero finger joint targets actually moves the hand DOFs.
  5. The hand body (palm) follows the arm — confirms the splice joint linking
     iiwa14_link_ee → sharpa_mount → right_hand_flange holds in PhysX.
  6. The reset_idx → first observation cycle produces obs of the declared
     ``num_observations`` dimensionality (catches the hand-goal append path).

Run with the conda env that has IsaacGym installed.  ``--gui`` opens the
viewer for visual inspection; default is headless.

Usage::

    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --gui --num-envs 1
    python scripts/test_pipeline.py --traj data/trajectories/g0.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Must be first IsaacGym import for binding registration.
import isaacgym  # noqa: F401

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_OK = "\033[92m✓\033[0m"
_FAIL = "\033[91m✗\033[0m"
_WARN = "\033[93m!\033[0m"


def _print_check(name: str, ok: bool, msg: str = "") -> bool:
    icon = _OK if ok else _FAIL
    print(f"  {icon} {name}{(': ' + msg) if msg else ''}")
    return ok


def _build_env(cfg_overrides: dict, headless: bool) -> object:
    """Construct a VideoRLFollower env directly via IsaacGymEnvs hydra cfg."""
    from hydra import compose, initialize
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

    cfg_path = "../isaacgymenvs/cfg"
    with initialize(config_path=cfg_path, version_base=None):
        cfg = compose(config_name="config", overrides=[
            "task=VideoRLFollower",
            f"num_envs={cfg_overrides['num_envs']}",
            f"headless={'true' if headless else 'false'}",
            "test=true",                                    # no checkpoint loading
            "seed=42",
        ])

    if "trajectoryPath" in cfg_overrides:
        cfg.task.env.trajectoryPath = cfg_overrides["trajectoryPath"]

    cfg_dict = omegaconf_to_dict(cfg.task)
    cfg_dict["env"]["numEnvs"] = cfg_overrides["num_envs"]

    creator = get_rlgames_env_creator(
        seed=42,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=headless,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=not headless,
    )
    return creator()


def _step_with_action(env, action: torch.Tensor, n_steps: int):
    """Step ``env`` with a constant action for ``n_steps``."""
    for _ in range(n_steps):
        obs, rew, done, info = env.step(action)
    return obs, rew, done, info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_construction(env) -> bool:
    """T1: env constructs and exposes expected attrs."""
    print("\n[T1] Construction & space declarations")
    ok = True
    ok &= _print_check(
        "obs_space size matches num_observations",
        env.observation_space.shape[0] == env.num_observations,
        f"obs_space={env.observation_space.shape[0]}, "
        f"num_observations={env.num_observations}",
    )
    ok &= _print_check(
        "action_space size matches num_actions",
        env.action_space.shape[0] == env.num_actions,
        f"action_space={env.action_space.shape[0]}, "
        f"num_actions={env.num_actions}",
    )
    ok &= _print_check(
        "trajectory loaded",
        hasattr(env, "trajectory") and env.trajectory.num_goals > 0,
        f"num_goals={env.trajectory.num_goals}",
    )
    ok &= _print_check(
        "right Sharpa hand active",
        env.use_right_sharpa,
        f"use_right_sharpa={env.use_right_sharpa}",
    )
    return ok


def test_object_spawn(env) -> bool:
    """T2: object spawns within tolerance of trajectory.start_pose."""
    print("\n[T2] Object spawn pose")
    # Reset every env to get fresh initial state.
    env.reset()
    # SimToolReal stores object state at self.object_state[:, 0:7].
    obj_xyz = env.object_state[:, 0:3]                          # (N, 3)
    expected = env.trajectory.object_start_pose[:3].to(env.device)  # (3,)
    delta = torch.norm(obj_xyz - expected[None], dim=-1)        # (N,)
    max_d = float(delta.max())
    # Spawn tolerance: 5 cm (procedural primitive paths randomize ±few cm).
    return _print_check(
        "object xyz close to trajectory.start_pose",
        max_d < 0.05,
        f"max distance {max_d * 100:.2f} cm (expected < 5 cm)",
    )


def test_arm_driven(env) -> bool:
    """T3: applying arm joint targets actually moves the arm DOFs."""
    print("\n[T3] Arm DOF response to action")
    env.reset()
    n_arm = 7
    n_total = env.num_robot_actions

    # Snapshot current arm DOF positions
    q0 = env.arm_hand_dof_pos[:, :n_arm].clone()

    # Build an action that nudges each arm joint by +0.5 rad (in normalized [-1,1]
    # action space — we go halfway to the upper bound).
    action = torch.zeros(env.num_envs, n_total, device=env.device)
    action[:, :n_arm] = 0.5

    _step_with_action(env, action, n_steps=20)

    q1 = env.arm_hand_dof_pos[:, :n_arm]
    diff = (q1 - q0).abs().sum(dim=-1)                          # (N,)
    moved = float(diff.mean())
    return _print_check(
        "arm DOFs moved when commanded",
        moved > 0.05,
        f"mean Σ|Δq_arm| = {moved:.4f} rad (expected > 0.05)",
    )


def test_fingers_driven(env) -> bool:
    """T4: applying finger joint targets actually moves the finger DOFs."""
    print("\n[T4] Finger DOF response to action")
    env.reset()
    n_arm = 7
    n_total = env.num_robot_actions
    n_hand = n_total - n_arm

    # Snapshot finger DOFs
    q0 = env.arm_hand_dof_pos[:, n_arm:].clone()

    # Close all fingers by issuing positive normalized targets.
    action = torch.zeros(env.num_envs, n_total, device=env.device)
    action[:, n_arm:] = 0.6

    _step_with_action(env, action, n_steps=20)

    q1 = env.arm_hand_dof_pos[:, n_arm:]
    diff = (q1 - q0).abs().sum(dim=-1)                          # (N,)
    moved = float(diff.mean())
    return _print_check(
        f"finger DOFs ({n_hand}) moved when commanded",
        moved > 0.1,
        f"mean Σ|Δq_finger| = {moved:.4f} rad (expected > 0.1)",
    )


def test_palm_follows_arm(env) -> bool:
    """T5: palm body translates when the arm joint targets change.

    Confirms the splice joint between iiwa14_link_ee → sharpa_mount →
    right_hand_flange is intact (not broken or detached).
    """
    print("\n[T5] Palm follows arm (splice joint integrity)")
    env.reset()
    n_arm = 7
    n_total = env.num_robot_actions
    palm_p0 = env.palm_center_pos.clone()

    action = torch.zeros(env.num_envs, n_total, device=env.device)
    # Move only joint 1 (shoulder) by a substantial amount.
    action[:, 1] = 0.8

    _step_with_action(env, action, n_steps=30)
    palm_p1 = env.palm_center_pos
    delta = torch.norm(palm_p1 - palm_p0, dim=-1).mean().item()
    return _print_check(
        "palm position changes when arm moves",
        delta > 0.02,
        f"mean ‖Δpalm‖ = {delta * 100:.2f} cm (expected > 2 cm)",
    )


def test_obs_round_trip(env) -> bool:
    """T6: obs returned by reset() has shape consistent with declared spaces."""
    print("\n[T6] Observation round-trip dimensionality")
    obs_dict = env.reset()
    # rl_games style returns dict with 'obs' (and optionally 'states').
    obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
    ok = True
    ok &= _print_check(
        "obs shape == (N, num_observations)",
        tuple(obs.shape) == (env.num_envs, env.num_observations),
        f"got {tuple(obs.shape)} expected ({env.num_envs}, {env.num_observations})",
    )
    if isinstance(obs_dict, dict) and "states" in obs_dict and obs_dict["states"] is not None:
        st = obs_dict["states"]
        ok &= _print_check(
            "states shape == (N, num_states)",
            tuple(st.shape) == (env.num_envs, env.num_states),
            f"got {tuple(st.shape)} expected ({env.num_envs}, {env.num_states})",
        )
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--gui", action="store_true",
                   help="open the IsaacGym viewer (interactive)")
    p.add_argument("--traj", type=str, default=None,
                   help="override trajectoryPath (relative to project root)")
    args = p.parse_args()

    overrides = {"num_envs": args.num_envs}
    if args.traj is not None:
        overrides["trajectoryPath"] = args.traj

    print(f"[setup] num_envs={args.num_envs}  gui={args.gui}")
    if args.traj:
        print(f"[setup] trajectoryPath={args.traj}")
    env = _build_env(overrides, headless=not args.gui)
    print("[setup] env constructed")

    results = {
        "T1 construction": test_construction(env),
        "T2 object spawn": test_object_spawn(env),
        "T3 arm driven":   test_arm_driven(env),
        "T4 fingers driven": test_fingers_driven(env),
        "T5 palm-arm splice": test_palm_follows_arm(env),
        "T6 obs round-trip": test_obs_round_trip(env),
    }

    print("\n" + "=" * 56)
    n_pass = sum(int(v) for v in results.values())
    for name, ok in results.items():
        print(f"  {(_OK if ok else _FAIL)} {name}")
    print(f"\n  {n_pass} / {len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
