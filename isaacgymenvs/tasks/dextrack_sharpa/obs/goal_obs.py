"""Goal observation builders for DexTrack-Sharpa residual policy.

Two distinct styles (paper-faithful) are provided; the env picks one via cfg.

  ManipTrans Stage-1 (dexhandimitator.py):
    For each future_frame Δ ∈ FUTURE_FRAMES_MANIPTRANS = [1] (paper default):
      - target wrist pos (3) + quat (4) in world
      - target wrist linear vel (3) + ang vel (3)
      - target finger-joint world pos (27 × 3 = 81)   [excludes wrist]
      - target finger-joint world vel (27 × 3 = 81)
      - target object pose (pos 3 + quat 4)
      - target object lin/ang vel (6)
      - object goal-frame mask (1, always 1.0 for our setup)
    → per-frame dim = 3+4+3+3+81+81+3+4+6+1 = 189
    With 1 future frame → 189

  DexTrack (allegro_hand_tracking_generalist.py compute_full_state L8361):
    `future_feats = cat([future_goal_pos, future_goal_rot, future_hand_qtars])`
    — absolute reference values, NOT deltas to current state.
    Default window FUTURE_FRAMES_DEXTRACK = [0, 1, 2, 3, 4] (history_length=5):
      - target obj_pos      (3)
      - target obj_quat     (4)
      - target dof_pos      (29)              — arm+hand qtars
      - target wrist_pos    (3)
      - target wrist_quat   (4)
    → per-frame = 43; with 5 future frames → 215

Both builders return (N, D) flat tensor for concatenation with proprio obs.
"""
from __future__ import annotations

from typing import Dict, List
import torch
from torch import Tensor


# Default lookahead windows (frames into the future) — match paper defaults.
#
# ManipTrans Stage-1 (dexhandimitator):
#   `obsFutureLength: 1` (DexHandImitator.yaml:21), code builds
#   `cur_idx + range(obs_future_length)` → 1 future frame.
#
# DexTrack tracking (allegro_hand_tracking_generalist):
#   `history_length: 5`, `history_freq: 1` →
#   `ranged_future_ws = arange(5) * 1 = [0, 1, 2, 3, 4]`
#   i.e. 5 consecutive future frames including the current one.
FUTURE_FRAMES_MANIPTRANS: List[int] = [1]
FUTURE_FRAMES_DEXTRACK:   List[int] = [0, 1, 2, 3, 4]


def _gather_future(traj, t_now: Tensor, deltas: List[int]) -> List[Dict[str, Tensor]]:
    """Return list of state dicts at t_now + Δ, clamped at T-1."""
    T = traj.T
    refs = []
    for d in deltas:
        t_fut = torch.clamp(t_now + d, max=T - 1)
        refs.append(traj.get(t_fut))
    return refs


def build_maniptrans_goal_obs(
    state: Dict[str, Tensor],
    traj,
    progress_buf: Tensor,                       # (N,) long — current frame idx per env
    future_frames: List[int] = None,
) -> Tensor:
    """ManipTrans Stage-1-style goal obs (~570d with 3 future frames).

    The env conventionally subtracts current state from the target to make
    the obs translation-invariant — but ManipTrans paper passes both absolute
    and relative; here we follow paper L1370 verbatim: target absolutes.
    """
    future_frames = future_frames or FUTURE_FRAMES_MANIPTRANS
    refs = _gather_future(traj, progress_buf, future_frames)
    N = progress_buf.shape[0]
    parts = []
    for ref in refs:
        # Wrist 6DOF + vel (3+4+3+3 = 13)
        parts.append(ref["wrist_pos"])                          # (N, 3)
        parts.append(ref["wrist_quat"])                         # (N, 4)
        parts.append(ref["wrist_vel"])                          # (N, 3)
        parts.append(ref["wrist_ang_vel"])                      # (N, 3)
        # Finger joints (27 × 3 each, excludes wrist idx 0)
        parts.append(ref["link_pos"][:, 1:].reshape(N, -1))     # (N, 81)
        parts.append(ref["link_vel"][:, 1:].reshape(N, -1))     # (N, 81)
        # Object pose + vel (3+4+3+3 = 13)
        parts.append(ref["obj_pos"])                            # (N, 3)
        parts.append(ref["obj_quat"])                           # (N, 4)
        parts.append(ref["obj_lin_vel"])                        # (N, 3)
        parts.append(ref["obj_ang_vel"])                        # (N, 3)
        # Goal-frame mask (always 1 for our fixed-trajectory setup)
        parts.append(torch.ones(N, 1, device=progress_buf.device))
    return torch.cat(parts, dim=-1)


def build_dextrack_goal_obs(
    state: Dict[str, Tensor],
    traj,
    progress_buf: Tensor,
    future_frames: List[int] = None,
) -> Tensor:
    """DexTrack-style goal obs.

    Paper convention (compute_full_state, allegro_hand_tracking_generalist.py
    L8361):
        `future_feats = cat([future_goal_pos, future_goal_rot, future_hand_qtars])`
    — absolute reference values, NOT deltas to current state. The policy
    learns to map (current obs + absolute future ref) → residual.
    """
    future_frames = future_frames or FUTURE_FRAMES_DEXTRACK
    refs = _gather_future(traj, progress_buf, future_frames)
    N = progress_buf.shape[0]
    parts = []
    for ref in refs:
        parts.append(ref["obj_pos"])     # (N, 3)
        parts.append(ref["obj_quat"])    # (N, 4)
        parts.append(ref["dof_pos"])     # (N, 29)  — hand qtars (DexTrack: arm+hand)
        parts.append(ref["wrist_pos"])   # (N, 3)
        parts.append(ref["wrist_quat"])  # (N, 4)
    return torch.cat(parts, dim=-1)


def get_goal_obs_dim(style: str, future_frames: List[int] = None) -> int:
    """Return flat dimension of goal_obs for a given style + window."""
    if style == "maniptrans":
        ff = future_frames or FUTURE_FRAMES_MANIPTRANS
        return len(ff) * (3 + 4 + 3 + 3 + 81 + 81 + 3 + 4 + 3 + 3 + 1)
    elif style == "dextrack":
        ff = future_frames or FUTURE_FRAMES_DEXTRACK
        return len(ff) * (29 + 3 + 4 + 3 + 4)
    raise ValueError(f"unknown style: {style}")


def _smoke_test():
    """Build a fake trajectory + state and verify the dims match get_goal_obs_dim."""
    device = "cuda:0"
    N = 4

    class FakeTraj:
        def __init__(self, T=300):
            self.T = T
            self.dof_pos      = torch.zeros(T, 29,    device=device)
            self.dof_vel      = torch.zeros(T, 29,    device=device)
            self.obj_pos      = torch.zeros(T, 3,     device=device)
            self.obj_quat     = torch.tensor([[0,0,0,1.]]*T, device=device)
            self.obj_lin_vel  = torch.zeros(T, 3,     device=device)
            self.obj_ang_vel  = torch.zeros(T, 3,     device=device)
            self.link_pos     = torch.zeros(T, 28, 3, device=device)
            self.link_vel     = torch.zeros(T, 28, 3, device=device)
            self.wrist_pos    = torch.zeros(T, 3,     device=device)
            self.wrist_quat   = torch.tensor([[0,0,0,1.]]*T, device=device)
            self.wrist_lin_vel = torch.zeros(T, 3, device=device)
            self.wrist_ang_vel = torch.zeros(T, 3, device=device)

        def get(self, t):
            t = torch.clamp(t, 0, self.T - 1)
            return {
                "dof_pos":       self.dof_pos[t],
                "dof_vel":       self.dof_vel[t],
                "obj_pos":       self.obj_pos[t],
                "obj_quat":      self.obj_quat[t],
                "obj_lin_vel":   self.obj_lin_vel[t],
                "obj_ang_vel":   self.obj_ang_vel[t],
                "link_pos":      self.link_pos[t],
                "link_vel":      self.link_vel[t],
                "wrist_pos":     self.wrist_pos[t],
                "wrist_quat":    self.wrist_quat[t],
                "wrist_vel":     self.wrist_lin_vel[t],
                "wrist_ang_vel": self.wrist_ang_vel[t],
            }

    traj = FakeTraj(T=300)
    state = traj.get(torch.zeros(N, dtype=torch.long, device=device))
    progress = torch.zeros(N, dtype=torch.long, device=device)

    mt_obs = build_maniptrans_goal_obs(state, traj, progress)
    dt_obs = build_dextrack_goal_obs(state, traj, progress)

    mt_dim = get_goal_obs_dim("maniptrans")
    dt_dim = get_goal_obs_dim("dextrack")

    print(f"ManipTrans goal_obs: {mt_obs.shape}  (expect ({N}, {mt_dim}))")
    print(f"DexTrack   goal_obs: {dt_obs.shape}  (expect ({N}, {dt_dim}))")

    assert mt_obs.shape == (N, mt_dim), "ManipTrans dim mismatch"
    assert dt_obs.shape == (N, dt_dim), "DexTrack dim mismatch"

    # Late-progress wrap test
    progress2 = torch.tensor([0, 50, 200, 295], dtype=torch.long, device=device)
    mt2 = build_maniptrans_goal_obs(state, traj, progress2)
    dt2 = build_dextrack_goal_obs(state, traj, progress2)
    print(f"Late-progress shapes: MT {mt2.shape}  DT {dt2.shape}")
    print(f"  per-style breakdown: MT={mt_dim//3} per frame ×3, DT={dt_dim//3} per frame ×3")
    print("✓ goal_obs smoke test passed")


if __name__ == "__main__":
    _smoke_test()
