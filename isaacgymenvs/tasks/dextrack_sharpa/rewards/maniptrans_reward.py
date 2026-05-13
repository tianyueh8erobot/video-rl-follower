"""ManipTrans Stage-1 imitation reward (13 terms, paper coefficients).

Adapted from `ManipTrans/maniptrans_envs/lib/envs/tasks/dexhandimitator.py:1066`.
The structure follows the paper verbatim:

  exp-decay terms (paper alpha values):
    r_eef_pos        = exp(-40  * ||cur - tgt||)          weight 0.1
    r_eef_rot        = exp(-1   * geodesic angle)          weight 0.6
    r_thumb_tip_pos  = exp(-100 * ||cur - tgt||)           weight 0.9
    r_index_tip_pos  = exp(-90  * ||cur - tgt||)           weight 0.8
    r_middle_tip_pos = exp(-80  * ||cur - tgt||)           weight 0.75
    r_pinky_tip_pos  = exp(-60  * ||cur - tgt||)           weight 0.6
    r_ring_tip_pos   = exp(-60  * ||cur - tgt||)           weight 0.6
    r_level_1_pos    = exp(-50  * mean per-joint err)      weight 0.5
    r_level_2_pos    = exp(-40  * mean per-joint err)      weight 0.3
    r_eef_vel        = exp(-1   * |delta vel|.mean)        weight 0.1
    r_eef_ang_vel    = exp(-1   * |delta ang_vel|.mean)    weight 0.05
    r_joints_vel     = exp(-1   * |delta vel|.mean)        weight 0.1
    r_power          = exp(-10  * power)                    weight 0.5
    r_wrist_power    = exp(-2   * wrist_power)             weight 0.5

  failed_execute (early terminate if any tip drifts beyond threshold and progress >= 20)

We expose a Sharpa-specific weight_idx (matches sharpa.py).
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
from torch import Tensor

# Sharpa weight_idx (from /home/intel/Codes/ManipTrans/maniptrans_envs/lib/envs/dexhands/sharpa.py L198-206)
# Indices reference SHARPA_BODY_NAMES (28 entries, 0 = wrist).
# For per-finger reward we exclude wrist → use idx-1 to index into [1:] of body links.
SHARPA_WEIGHT_IDX = {
    "thumb_tip":     [27],
    "index_tip":     [5],
    "middle_tip":    [10],
    "ring_tip":      [21],
    "pinky_tip":     [16],
    "level_1_joints":[1, 6, 11, 17, 22],
    "level_2_joints":[2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26],
}


def _quat_geodesic_angle_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Angle between two unit xyzw quaternions, in radians (N,)."""
    dot = (a * b).sum(-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def compute_maniptrans_reward(
    state: Dict[str, Tensor],
    target: Dict[str, Tensor],
    actions: Tensor,                # (N, 29)
    progress_buf: Tensor,           # (N,) int — current step in episode
    failed_execute_enabled: bool = True,
    failed_execute_progress_threshold: int = 20,
    scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Return (reward, failed_execute, succeeded, components).

    state / target each have keys:
      dof_pos / dof_vel  : (N, 29)
      obj_pos / obj_quat / obj_lin_vel / obj_ang_vel
      link_pos / link_vel: (N, 28, 3)
      wrist_pos / wrist_quat / wrist_vel / wrist_ang_vel
    """
    N = state["dof_pos"].shape[0]
    device = state["dof_pos"].device

    # --- EEF (wrist) tracking ---
    diff_eef_pos = target["wrist_pos"] - state["wrist_pos"]
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)
    reward_eef_pos = torch.exp(-40.0 * diff_eef_pos_dist)

    diff_eef_rot_angle = _quat_geodesic_angle_xyzw(target["wrist_quat"], state["wrist_quat"])
    reward_eef_rot = torch.exp(-1.0 * diff_eef_rot_angle.abs())

    diff_eef_vel     = target["wrist_vel"]     - state["wrist_vel"]
    diff_eef_ang_vel = target["wrist_ang_vel"] - state["wrist_ang_vel"]
    reward_eef_vel     = torch.exp(-1.0 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1.0 * diff_eef_ang_vel.abs().mean(dim=-1))

    # --- Per-finger link tracking ---
    # link_pos: (N, 28, 3); exclude wrist (idx 0) by taking [:, 1:] → (N, 27, 3).
    cur_links  = state["link_pos"][:, 1:]
    tgt_links  = target["link_pos"][:, 1:]
    diff_links = torch.norm(tgt_links - cur_links, dim=-1)            # (N, 27)
    cur_links_vel = state["link_vel"][:, 1:]
    tgt_links_vel = target["link_vel"][:, 1:]
    diff_links_vel = (tgt_links_vel - cur_links_vel).abs().mean(-1)   # (N, 27)

    def _group_mean(group_name: str) -> Tensor:
        # body_names index → (idx - 1) into the [1:] tensor
        idxs = [k - 1 for k in SHARPA_WEIGHT_IDX[group_name]]
        return diff_links[:, idxs].mean(dim=-1)

    d_thumb  = _group_mean("thumb_tip")
    d_index  = _group_mean("index_tip")
    d_middle = _group_mean("middle_tip")
    d_pinky  = _group_mean("pinky_tip")
    d_ring   = _group_mean("ring_tip")
    d_lvl1   = _group_mean("level_1_joints")
    d_lvl2   = _group_mean("level_2_joints")

    reward_thumb_tip_pos  = torch.exp(-100.0 * d_thumb)
    reward_index_tip_pos  = torch.exp(- 90.0 * d_index)
    reward_middle_tip_pos = torch.exp(- 80.0 * d_middle)
    reward_pinky_tip_pos  = torch.exp(- 60.0 * d_pinky)
    reward_ring_tip_pos   = torch.exp(- 60.0 * d_ring)
    reward_level_1_pos    = torch.exp(- 50.0 * d_lvl1)
    reward_level_2_pos    = torch.exp(- 40.0 * d_lvl2)

    reward_joints_vel = torch.exp(-1.0 * diff_links_vel.mean(-1))

    # --- Power penalties (real effort × velocity, paper-faithful) ---
    # Paper formula (dexhandimitator.py L579-589):
    #   power       = |dof_force × dof_vel|.sum()                   (all DOFs)
    #   wrist_power = |F_lin · v_lin| + |F_ang · v_ang|             (wrist body)
    # F_lin / F_ang are the EXTERNAL forces applied to the wrist body
    # (`apply_forces[wrist]`).  We don't apply external perturbation forces in
    # our setup, so wrist_power is effectively 0 → reward_wrist_power saturates
    # at exp(0)=1.  This matches DexTrack/ManipTrans tracking eval where no
    # random perturbations are active either.  If you turn on apply_forces in
    # future, plumb them in here.
    if "dof_force" in state:
        power = torch.abs(state["dof_force"] * state["dof_vel"]).sum(dim=-1)
    else:
        # Fallback for callers that haven't wired dof_force yet (e.g. unit tests).
        power = torch.zeros(N, device=device)
    wrist_power = torch.zeros(N, device=device)
    reward_power       = torch.exp(-10.0 * power)
    reward_wrist_power = torch.exp(- 2.0 * wrist_power)

    # --- Failed execute ---
    # If any tip / level drifts beyond paper threshold AND progress >= 20 steps,
    # the env terminates with no positive reward.
    if failed_execute_enabled:
        failed = (
            (d_thumb  > 0.04 / 0.7 * scale_factor)
            | (d_index  > 0.045 / 0.7 * scale_factor)
            | (d_middle > 0.05 / 0.7 * scale_factor)
            | (d_pinky  > 0.06 / 0.7 * scale_factor)
            | (d_ring   > 0.06 / 0.7 * scale_factor)
            | (d_lvl1   > 0.07 / 0.7 * scale_factor)
            | (d_lvl2   > 0.08 / 0.7 * scale_factor)
        ) & (progress_buf >= failed_execute_progress_threshold)
    else:
        failed = torch.zeros(N, dtype=torch.bool, device=device)

    # --- Total reward (paper sum) ---
    reward = (
        0.1   * reward_eef_pos
      + 0.6   * reward_eef_rot
      + 0.9   * reward_thumb_tip_pos
      + 0.8   * reward_index_tip_pos
      + 0.75  * reward_middle_tip_pos
      + 0.6   * reward_pinky_tip_pos
      + 0.6   * reward_ring_tip_pos
      + 0.5   * reward_level_1_pos
      + 0.3   * reward_level_2_pos
      + 0.1   * reward_eef_vel
      + 0.05  * reward_eef_ang_vel
      + 0.1   * reward_joints_vel
      + 0.5   * reward_power
      + 0.5   * reward_wrist_power
    )

    # Zero out reward on failed steps (paper convention: terminate w/o positive credit)
    reward = torch.where(failed, torch.zeros_like(reward), reward)

    succeeded = torch.zeros(N, dtype=torch.bool, device=device)

    components = {
        "r_eef_pos":         reward_eef_pos,
        "r_eef_rot":         reward_eef_rot,
        "r_eef_vel":         reward_eef_vel,
        "r_eef_ang_vel":     reward_eef_ang_vel,
        "r_thumb_tip_pos":   reward_thumb_tip_pos,
        "r_index_tip_pos":   reward_index_tip_pos,
        "r_middle_tip_pos":  reward_middle_tip_pos,
        "r_pinky_tip_pos":   reward_pinky_tip_pos,
        "r_ring_tip_pos":    reward_ring_tip_pos,
        "r_level_1_pos":     reward_level_1_pos,
        "r_level_2_pos":     reward_level_2_pos,
        "r_joints_vel":      reward_joints_vel,
        "r_power":           reward_power,
        "r_wrist_power":     reward_wrist_power,
        "d_thumb":           d_thumb,
        "d_index":           d_index,
        "d_middle":          d_middle,
        "d_pinky":           d_pinky,
        "d_ring":            d_ring,
    }
    return reward, failed, succeeded, components


def _smoke_test():
    """Run with two states: target == state (perfect tracking → reward ~ sum of weights)."""
    N = 4
    device = "cuda:0"
    fake = {
        "dof_pos":       torch.zeros(N, 29, device=device),
        "dof_vel":       torch.zeros(N, 29, device=device),
        "obj_pos":       torch.zeros(N, 3, device=device),
        "obj_quat":      torch.tensor([[0,0,0,1.]]*N, device=device),
        "obj_lin_vel":   torch.zeros(N, 3, device=device),
        "obj_ang_vel":   torch.zeros(N, 3, device=device),
        "link_pos":      torch.zeros(N, 28, 3, device=device),
        "link_vel":      torch.zeros(N, 28, 3, device=device),
        "wrist_pos":     torch.zeros(N, 3, device=device),
        "wrist_quat":    torch.tensor([[0,0,0,1.]]*N, device=device),
        "wrist_vel":     torch.zeros(N, 3, device=device),
        "wrist_ang_vel": torch.zeros(N, 3, device=device),
    }
    target = {k: v.clone() for k, v in fake.items()}
    actions = torch.zeros(N, 29, device=device)
    progress = torch.zeros(N, dtype=torch.long, device=device)

    reward, failed, success, comp = compute_maniptrans_reward(fake, target, actions, progress)
    # Perfect tracking: most exp(-..)→1, so reward ≈ sum of weights
    paper_max = 0.1+0.6+0.9+0.8+0.75+0.6+0.6+0.5+0.3+0.1+0.05+0.1+0.5+0.5
    print(f"reward (perfect): {reward[0]:.3f}  (paper max sum = {paper_max:.3f})")
    print(f"failed: {failed[0].item()}, succeeded: {success[0].item()}")
    print(f"components: r_eef_pos={comp['r_eef_pos'][0]:.3f}  r_thumb_tip={comp['r_thumb_tip_pos'][0]:.3f}")
    assert abs(reward[0].item() - paper_max) < 0.01, "perfect tracking should saturate to weight sum"
    print("✓ ManipTrans reward smoke test passed")


if __name__ == "__main__":
    _smoke_test()
