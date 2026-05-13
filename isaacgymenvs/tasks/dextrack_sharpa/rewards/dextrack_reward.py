"""DexTrack tracking reward — paper-faithful port of `compute_hand_reward_tracking`.

Adapted line-for-line from
  /home/intel/Codes/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py
  lines ~12671-12978 (`compute_hand_reward_tracking`).

Formula (excerpt from DexTrack source L12976):

    reward = (-rew_delta_hand_pose_coef) * delta_value
           + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)
           + goal_hand_rew
           + bonus

where:
  delta_value = glb_trans_coef * |Δwrist_pos|_1
              + glb_rot_coef   * |Δwrist_rot|_1            (rotvec L1)
              + fingerpose_coef * |Δhand_qpos|_1

  right_hand_finger_dist  = Σ_i ||fingertip_i - obj||_2     (DexTrack uses 4 fingers:
                                                              thumb/index/middle/ring;
                                                              pinky commented out at L12698)
                            clamped to ≤ 0.6 × 4 = 2.4

  right_hand_dist         = ||wrist - obj||_2,  clamped to ≤ 0.5

  WRIST-GATING (L12962-12963):
    if right_hand_dist > 0.12:  right_hand_finger_dist  ←  0
                                (no finger-obj penalty when wrist is far)

  goal_hand_rew  = obj_pos_coef * (0.9 - 2 × goal_dist)
                 + obj_rot_coef * (π   - rot_dist)         (gated by close-contact)
                 = 0 when not in contact (flag != 5)

  bonus          = 1/(1 + 10*goal_dist)   when goal_dist ≤ 0.05    (object near goal)
                 + 1/(1 + 5*rot_dist)     when rot_dist  ≤ 5°       (orient near goal)
                 + reach_goal_bonus       on full success (cfg flag)
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import math
from torch import Tensor


def _quat_geodesic_angle_xyzw(a: Tensor, b: Tensor) -> Tensor:
    dot = (a * b).sum(-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def _quat_rotvec_xyzw(q: Tensor) -> Tensor:
    """xyzw quaternion → rotation vector (axis × angle), shape (N, 3)."""
    qx, qy, qz, qw = q.unbind(-1)
    sin_half = torch.sqrt((qx * qx + qy * qy + qz * qz).clamp(min=1e-12))
    angle = 2.0 * torch.atan2(sin_half, qw.abs())
    axis = torch.stack([qx, qy, qz], dim=-1) / sin_half.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


def compute_dextrack_reward(
    state: Dict[str, Tensor],
    target: Dict[str, Tensor],
    actions: Tensor,                # (N, 29) — raw policy action; unused by DexTrack tracking
    progress_buf: Tensor,           # (N,)
    obj_pos_coef:         float = 1.0,    # (DexTrack `inhand_obj_pos_ornt_rew` coef on pos)
    obj_rot_coef:         float = 0.33,   # coef on rot
    glb_trans_coef:       float = 0.6,    # `hand_pose_guidance_glb_trans_coef`
    glb_rot_coef:         float = 0.1,
    fingerpose_coef:      float = 0.1,
    finger_obj_dist_coef: float = 0.3,    # `rew_finger_obj_dist_coef`
    delta_hand_pose_coef: float = 0.5,    # `rew_delta_hand_pose_coef`
    wrist_gate_thresh:    float = 0.12,   # finger_dist counts only if wrist ≤ this
    finger_dist_cap:      float = 0.6,    # per-finger L2 cap (×4 fingers)
    wrist_dist_cap:       float = 0.5,    # wrist L2 cap
    success_obj_pos_thresh: float = 0.05, # 5cm
    success_obj_rot_thresh: float = 5.0 / 180.0 * math.pi,  # 5°
    reach_goal_bonus: float = 0.0,
    early_terminate_obj_dist: float = 0.0,  # 0 disables; DexTrack uses 0.2m drift
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Return (reward, failed_execute, succeeded, components).

    State / target tensor keys: dof_pos, dof_vel, wrist_pos, wrist_quat, wrist_vel,
    wrist_ang_vel, link_pos, link_vel, obj_pos, obj_quat, obj_lin_vel, obj_ang_vel.
    """
    N = state["dof_pos"].shape[0]
    device = state["dof_pos"].device

    # ─── Hand pose guidance (delta_value, L1 norms) ─────────────────────────
    d_wrist_trans = (target["wrist_pos"] - state["wrist_pos"]).abs().sum(-1)
    # geodesic angle of orientation error (radians).  DexTrack uses an L1 of
    # an Euler-like rotation vector, but |rotvec|_1 ≥ |rotvec|_2 = angle, so
    # the geodesic angle is a clean lower bound; pick whichever matches our
    # FK / data convention.  We follow the geodesic since orientation in
    # SimToolReal/Isaac is also reported as a rotation vector magnitude.
    d_wrist_rot   = _quat_geodesic_angle_xyzw(target["wrist_quat"], state["wrist_quat"])
    # Hand-only DOF diff (skip Franka arm 0..6)
    d_fingerpose  = (target["dof_pos"][:, 7:] - state["dof_pos"][:, 7:]).abs().sum(-1)

    delta_value = (glb_trans_coef * d_wrist_trans
                   + glb_rot_coef * d_wrist_rot
                   + fingerpose_coef * d_fingerpose)
    r_hand_guidance = -delta_hand_pose_coef * delta_value

    # ─── Finger / wrist distances to object ─────────────────────────────────
    # DexTrack uses 4 fingers (thumb / index / middle / ring); pinky commented
    # out in source (L12698).  Body indices follow SHARPA_BODY_NAMES.
    fingertip_idxs = [27, 5, 10, 21]            # thumb, index, middle, ring
    tip_pos = state["link_pos"][:, fingertip_idxs]              # (N, 4, 3)
    obj_pos_b = state["obj_pos"][:, None, :]                    # (N, 1, 3)
    d_tip_obj_each = torch.norm(tip_pos - obj_pos_b, dim=-1)    # (N, 4)
    d_tip_obj = d_tip_obj_each.sum(-1)                          # (N,)
    finger_dist_cap_total = finger_dist_cap * len(fingertip_idxs)
    d_tip_obj = torch.minimum(d_tip_obj, torch.full_like(d_tip_obj, finger_dist_cap_total))

    d_wrist_obj = torch.norm(state["wrist_pos"] - state["obj_pos"], dim=-1)
    d_wrist_obj = torch.minimum(d_wrist_obj, torch.full_like(d_wrist_obj, wrist_dist_cap))

    # Wrist gating: finger-obj penalty only counts when wrist is close enough
    finger_dist_gated = torch.where(d_wrist_obj <= wrist_gate_thresh,
                                     d_tip_obj, torch.zeros_like(d_tip_obj))
    r_finger_obj = -finger_obj_dist_coef * (finger_dist_gated + 2.0 * d_wrist_obj)

    # ─── Object 6D tracking (goal_hand_rew, gated on contact) ──────────────
    d_obj_pos = torch.norm(target["obj_pos"] - state["obj_pos"], dim=-1)
    d_obj_rot = _quat_geodesic_angle_xyzw(target["obj_quat"], state["obj_quat"])

    inhand_rew = (obj_pos_coef * (0.9 - 2.0 * d_obj_pos)
                  + obj_rot_coef * (math.pi - d_obj_rot))
    # DexTrack `flag = finger_close + wrist_close + target_flag` (==3 for our
    # always-on target_flag); we use 2-of-2 (finger+wrist close) since we
    # don't carry the L1 sanity-check flag from DexTrack.
    contact_flag = ((d_tip_obj <= 0.12 * len(fingertip_idxs))
                     & (d_wrist_obj <= 0.12))
    goal_hand_rew = torch.where(contact_flag, inhand_rew,
                                 torch.zeros_like(inhand_rew))

    # ─── Success bonus (DexTrack `bonus`) ──────────────────────────────────
    success_pos = d_obj_pos < success_obj_pos_thresh
    success_rot = d_obj_rot < success_obj_rot_thresh
    succeeded = success_pos & success_rot

    bonus = torch.zeros_like(d_obj_pos)
    bonus = torch.where(success_pos, 1.0 / (1.0 + 10.0 * d_obj_pos), bonus)
    bonus = bonus + torch.where(success_rot,
                                 1.0 / (1.0 + 5.0 * d_obj_rot),
                                 torch.zeros_like(d_obj_rot))
    bonus = bonus + reach_goal_bonus * succeeded.float()

    # ─── Total ─────────────────────────────────────────────────────────────
    reward = r_hand_guidance + r_finger_obj + goal_hand_rew + bonus

    # ─── Early termination on object loss (DexTrack `early_terminate`) ────
    if early_terminate_obj_dist > 0:
        failed = d_obj_pos >= early_terminate_obj_dist
        # DexTrack penalizes the failure step with reward=-10
        reward = torch.where(failed, torch.full_like(reward, -10.0), reward)
    else:
        failed = torch.zeros(N, dtype=torch.bool, device=device)

    components = {
        "r_hand_guidance": r_hand_guidance,
        "r_finger_obj":    r_finger_obj,
        "r_goal_hand":     goal_hand_rew,
        "r_bonus":         bonus,
        "delta_value":     delta_value,
        "d_obj_pos":       d_obj_pos,
        "d_obj_rot":       d_obj_rot,
        "d_wrist_trans":   d_wrist_trans,
        "d_wrist_rot":     d_wrist_rot,
        "d_fingerpose":    d_fingerpose,
        "d_tip_obj":       d_tip_obj,
        "d_wrist_obj":     d_wrist_obj,
        "contact_flag":    contact_flag.float(),
    }
    return reward, failed, succeeded, components


def _smoke_test():
    """Two scenarios: perfect tracking with hand on object, vs hand far from object."""
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

    # Scenario 1: perfect tracking, hand AT object
    reward, failed, ok, comp = compute_dextrack_reward(fake, target, actions, progress,
                                                       reach_goal_bonus=10.0)
    print(f"[1] perfect + hand-at-object:")
    print(f"    reward[0]={reward[0]:+.4f}  hand_guide={comp['r_hand_guidance'][0]:+.4f}  "
          f"finger_obj={comp['r_finger_obj'][0]:+.4f}  goal_hand={comp['r_goal_hand'][0]:+.4f}  "
          f"bonus={comp['r_bonus'][0]:+.4f}")
    print(f"    contact_flag={comp['contact_flag'][0]:.0f}  succeeded={ok[0].item()}")

    # Scenario 2: object 10cm off, wrist 20cm away → r_finger_obj kicks in
    state2 = {k: v.clone() for k, v in fake.items()}
    state2["obj_pos"][:, 2] = 0.1                              # object 10cm up
    state2["wrist_pos"][:, 0] = 0.2                            # wrist 20cm in x → far
    reward2, failed2, ok2, comp2 = compute_dextrack_reward(state2, target, actions, progress,
                                                            reach_goal_bonus=10.0)
    print(f"\n[2] obj 10cm off, wrist 20cm far (gated finger should = 0):")
    print(f"    reward[0]={reward2[0]:+.4f}  hand_guide={comp2['r_hand_guidance'][0]:+.4f}  "
          f"finger_obj={comp2['r_finger_obj'][0]:+.4f}  goal_hand={comp2['r_goal_hand'][0]:+.4f}")
    print(f"    d_wrist_obj={comp2['d_wrist_obj'][0]:.3f}  d_tip_obj={comp2['d_tip_obj'][0]:.3f}  "
          f"contact_flag={comp2['contact_flag'][0]:.0f}")

    # Scenario 3: object 10cm off, wrist 8cm away → gating ON, finger penalty engaged
    state3 = {k: v.clone() for k, v in fake.items()}
    state3["obj_pos"][:, 2] = 0.1                              # obj at z=10cm
    state3["wrist_pos"][:, 2] = 0.05                           # wrist 5cm below obj (within 12cm gate)
    reward3, failed3, ok3, comp3 = compute_dextrack_reward(state3, target, actions, progress)
    print(f"\n[3] wrist within gate, object 10cm off:")
    print(f"    d_wrist_obj={comp3['d_wrist_obj'][0]:.3f}  gating ON (≤0.12)")
    print(f"    finger_obj={comp3['r_finger_obj'][0]:+.4f}  contact_flag={comp3['contact_flag'][0]:.0f}")

    print("\n✓ DexTrack reward smoke test passed")


if __name__ == "__main__":
    _smoke_test()
