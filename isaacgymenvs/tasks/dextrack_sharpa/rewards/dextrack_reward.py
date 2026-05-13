"""DexTrack tracking reward — paper-faithful port of the **goal_cond=False**
branch of `compute_hand_reward_tracking`.

Source (verified line-by-line):
  /home/intel/Codes/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py
  - L12813-12853 : `if goal_cond:` branch  (goal-reaching mode — we DO NOT port this)
  - L12855-12976 : `else:` branch          (trajectory-tracking mode — THIS file)
  - L12986-12998 : success / early-terminate logic

DexTrack `run_tracking_headless_grab_single_wfranka.sh` sets goal_cond=False
on its last uncommented assignment (L204), so the `else` branch is the one
actually used during paper training.

`else` branch reward formula (paper L12976, verbatim):

    reward = (-rew_delta_hand_pose_coef) * delta_value
           + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)
           + goal_hand_rew
           + bonus
           # + hand_up      <-- COMMENTED OUT in source; we DO NOT include it.

Component definitions (L12855-12970):

  flag                  = (finger_dist ≤ 0.12·N).int() + (wrist_dist ≤ 0.12).int()
                          // max value 2; NO target_flag (target_flag is only used
                          // in goal_cond=True branch, L12815)

  inhand_obj_pos_ornt_rew = 1 * (0.0 - 2 * goal_dist)
                          // **0.0 base, NOT 0.9** (which is the goal_cond=True value)
    if w_obj_ornt:        // default False
      rot_dist_rew = cur_ornt_rew_coef * (0.0 - rot_dist)
      inhand_obj_pos_ornt_rew = 1*(0.0 - 2*goal_dist) + rot_dist_rew
    if w_obj_vels:        // default False
      inhand_obj_pos_ornt_rew += lin_vel_rew + ang_vel_rew

  goal_hand_rew         = where(flag == 2, inhand_obj_pos_ornt_rew, 0)

  bonus                 = where(flag == 2 & goal_dist ≤ 0.05, 1/(1+10*goal_dist), 0)
                          // BOTH gates required: contact + close

  // wrist gating L12962-12963
  if right_hand_dist > 0.12:  right_hand_finger_dist ← 0

  // Success count (L12998): goal_dist ≤ 0.05 only (no rot check)
  // Early termination (L12986-12991): if cfg early_terminate and
  //   goal_dist >= 0.2:  reward = -10  and  reset

Caps:
  right_hand_dist          ≤ 0.5      (L12695)
  right_hand_finger_dist   ≤ 0.6 × N  (L12707)
  num_fingers              = 4        (DexTrack uses thumb/index/middle/ring; pinky
                                       commented out at L12698)

This file is a near-verbatim port; the only intentional simplifications are:
  • |Δhand_rot|_1 → geodesic angle (numerically close for small rot)
  • smoothness term and target_flag are exposed via cfg flags but DEFAULT OFF
    (matches DexTrack defaults: rew_smoothness_coef=0, goal_cond=False).
"""
from __future__ import annotations

from typing import Dict, Tuple
import math
import torch
from torch import Tensor


def _quat_geodesic_angle_xyzw(a: Tensor, b: Tensor) -> Tensor:
    dot = (a * b).sum(-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def compute_dextrack_reward(
    state: Dict[str, Tensor],
    target: Dict[str, Tensor],
    actions: Tensor,
    progress_buf: Tensor,
    # ── Hand pose guidance (paper `delta_value`, L1 norms) ─────────────────
    glb_trans_coef:       float = 0.6,
    glb_rot_coef:         float = 0.1,
    fingerpose_coef:      float = 0.1,
    delta_hand_pose_coef: float = 0.5,    # `rew_delta_hand_pose_coef`
    # ── Finger / wrist contact penalty ─────────────────────────────────────
    finger_obj_dist_coef: float = 0.3,    # `rew_finger_obj_dist_coef`
    wrist_gate_thresh:    float = 0.12,
    finger_dist_cap:      float = 0.6,    # per-finger cap; total cap = 0.6×N
    wrist_dist_cap:       float = 0.5,
    # ── Goal-distance (object 6-D tracking) ────────────────────────────────
    w_obj_ornt:           bool  = False,  # paper default in wfranka
    w_obj_vels:           bool  = False,
    cur_ornt_rew_coef:    float = 0.33,
    # ── Success / early-termination ────────────────────────────────────────
    success_obj_pos_thresh:  float = 0.05,
    early_terminate_obj_dist: float = 0.2,  # DexTrack default; 0 disables
    # ── Smoothness (opt-in, default off matches paper) ─────────────────────
    smoothness_coef: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Return (reward, failed, succeeded, components).

    Note: signature matches the env's caller; `progress_buf` and `actions`
    are accepted for API parity but unused by the goal_cond=False reward.
    """
    N = state["dof_pos"].shape[0]
    device = state["dof_pos"].device

    # ─── Hand pose guidance: `delta_value` (L1 norms, weighted) ────────────
    d_wrist_trans = (target["wrist_pos"] - state["wrist_pos"]).abs().sum(-1)
    d_wrist_rot   = _quat_geodesic_angle_xyzw(target["wrist_quat"], state["wrist_quat"])
    d_fingerpose  = (target["dof_pos"][:, 7:] - state["dof_pos"][:, 7:]).abs().sum(-1)
    delta_value = (glb_trans_coef * d_wrist_trans
                   + glb_rot_coef * d_wrist_rot
                   + fingerpose_coef * d_fingerpose)
    r_hand_guidance = -delta_hand_pose_coef * delta_value

    # ─── Finger / wrist distances to object ────────────────────────────────
    # DexTrack tracking uses 4 fingertips (thumb/index/middle/ring; pinky
    # commented at L12698).  Body indices follow SHARPA_BODY_NAMES.
    fingertip_idxs = [27, 5, 10, 21]
    num_fingers = len(fingertip_idxs)
    tip_pos    = state["link_pos"][:, fingertip_idxs]                # (N, 4, 3)
    obj_pos_b  = state["obj_pos"][:, None, :]
    d_tip_each = torch.norm(tip_pos - obj_pos_b, dim=-1)             # (N, 4)
    d_tip_obj  = d_tip_each.sum(-1)                                  # (N,)
    d_tip_obj  = torch.minimum(d_tip_obj,
                                torch.full_like(d_tip_obj, finger_dist_cap * num_fingers))

    d_wrist_obj = torch.norm(state["wrist_pos"] - state["obj_pos"], dim=-1)
    d_wrist_obj = torch.minimum(d_wrist_obj,
                                 torch.full_like(d_wrist_obj, wrist_dist_cap))

    # ─── flag (paper L12856-12858): 2-flag, NO target_flag ─────────────────
    finger_dist_thres = 0.12 * num_fingers
    finger_close = d_tip_obj   <= finger_dist_thres
    wrist_close  = d_wrist_obj <= 0.12
    flag2 = finger_close & wrist_close

    # ─── inhand_obj_pos_ornt_rew (paper L12863-12881) ──────────────────────
    d_obj_pos = torch.norm(target["obj_pos"] - state["obj_pos"], dim=-1)
    d_obj_rot = _quat_geodesic_angle_xyzw(target["obj_quat"], state["obj_quat"])
    # `goal_dist` (paper) and `rot_dist` (paper) — used for both inhand_rew and bonus.
    goal_dist = d_obj_pos
    rot_dist  = d_obj_rot

    # Paper L12863 (goal_cond=False else branch): constant is **0.0 not 0.9**.
    inhand_rew = 1.0 * (0.0 - 2.0 * goal_dist)
    if w_obj_ornt:
        inhand_rew = inhand_rew + cur_ornt_rew_coef * (0.0 - rot_dist)
    if w_obj_vels:
        # Object lin/ang velocity tracking (paper L12790-12805).  Default off.
        d_lin = torch.norm(target["obj_lin_vel"] - state["obj_lin_vel"], dim=-1)
        d_ang = torch.norm(target["obj_ang_vel"] - state["obj_ang_vel"], dim=-1)
        lin_vel_rew = (120.0 * 0.9 - 2.0 * d_lin) / 120.0
        ang_vel_rew = (120.0 * 0.9 - 2.0 * d_ang) / 120.0
        inhand_rew = inhand_rew + lin_vel_rew + ang_vel_rew

    goal_hand_rew = torch.where(flag2, inhand_rew, torch.zeros_like(inhand_rew))

    # ─── bonus (paper L12910-12911): flag==2 AND goal_dist ≤ 0.05 ──────────
    bonus_mask = flag2 & (goal_dist <= success_obj_pos_thresh)
    bonus = torch.where(bonus_mask,
                         1.0 / (1.0 + 10.0 * goal_dist),
                         torch.zeros_like(goal_dist))
    if w_obj_vels:
        # lin/ang velocity bonus (paper L12800-12805).  Default off.
        d_lin = torch.norm(target["obj_lin_vel"] - state["obj_lin_vel"], dim=-1)
        d_ang = torch.norm(target["obj_ang_vel"] - state["obj_ang_vel"], dim=-1)
        lin_thres = 0.05 * 12.0
        ang_thres = 0.05 * 12.0
        lin_bonus = torch.where(d_lin <= lin_thres,
                                 1.0 / (1.0 + 10.0 * d_lin / 120.0),
                                 torch.zeros_like(d_lin))
        ang_bonus = torch.where(d_ang <= ang_thres,
                                 1.0 / (1.0 + 10.0 * d_ang / 120.0),
                                 torch.zeros_like(d_ang))
        bonus = bonus + lin_bonus + ang_bonus
    # NOTE: paper L12944 `bonus = bonus` (no-op due to typo) — we leave the
    # `w_obj_ornt` rot bonus unimplemented to mirror this bug.  The path is
    # dead in source; documenting for fidelity.

    # ─── Wrist gating on finger penalty (paper L12962-12963) ───────────────
    # When wrist is far (>12cm), finger_dist drops out of the penalty.  Wrist
    # distance keeps its 2× weight regardless.
    finger_dist_gated = torch.where(d_wrist_obj <= wrist_gate_thresh,
                                     d_tip_obj, torch.zeros_like(d_tip_obj))
    r_finger_obj = -finger_obj_dist_coef * (finger_dist_gated + 2.0 * d_wrist_obj)

    # ─── Smoothness (paper L12780-12782, opt-in via coef) ──────────────────
    r_smoothness = torch.zeros(N, device=device)
    if smoothness_coef > 0 and "prev_dof_vel" in state:
        r_smoothness = -smoothness_coef * torch.norm(
            state["prev_dof_vel"] - state["dof_vel"], p=2, dim=-1)

    # ─── Total (paper L12976; hand_up commented out, so NOT added) ─────────
    reward = r_hand_guidance + r_finger_obj + goal_hand_rew + bonus + r_smoothness

    # ─── Success (paper L12998): goal_dist ≤ 0.05 only (no rot check) ─────
    succeeded = goal_dist <= success_obj_pos_thresh

    # ─── Early termination (paper L12986-12991) ────────────────────────────
    if early_terminate_obj_dist > 0:
        failed = goal_dist >= early_terminate_obj_dist
        reward = torch.where(failed, torch.full_like(reward, -10.0), reward)
    else:
        failed = torch.zeros(N, dtype=torch.bool, device=device)

    components = {
        "r_hand_guidance":  r_hand_guidance,
        "r_finger_obj":     r_finger_obj,
        "r_goal_hand":      goal_hand_rew,
        "r_bonus":          bonus,
        "r_smoothness":     r_smoothness,
        "delta_value":      delta_value,
        "goal_dist":        goal_dist,
        "rot_dist":         rot_dist,
        "d_tip_obj":        d_tip_obj,
        "d_wrist_obj":      d_wrist_obj,
        "flag2":            flag2.float(),
    }
    return reward, failed, succeeded, components


def _smoke_test():
    """Three scenarios mirroring paper expectations."""
    N = 4
    device = "cuda:0"
    fake = {
        "dof_pos":       torch.zeros(N, 29, device=device),
        "dof_vel":       torch.zeros(N, 29, device=device),
        "obj_pos":       torch.zeros(N, 3, device=device),
        "obj_quat":      torch.tensor([[0, 0, 0, 1.]] * N, device=device),
        "obj_lin_vel":   torch.zeros(N, 3, device=device),
        "obj_ang_vel":   torch.zeros(N, 3, device=device),
        "link_pos":      torch.zeros(N, 28, 3, device=device),
        "link_vel":      torch.zeros(N, 28, 3, device=device),
        "wrist_pos":     torch.zeros(N, 3, device=device),
        "wrist_quat":    torch.tensor([[0, 0, 0, 1.]] * N, device=device),
        "wrist_vel":     torch.zeros(N, 3, device=device),
        "wrist_ang_vel": torch.zeros(N, 3, device=device),
        "prev_dof_vel":  torch.zeros(N, 29, device=device),
    }
    target = {k: v.clone() for k, v in fake.items()}
    actions = torch.zeros(N, 29, device=device)
    progress = torch.zeros(N, dtype=torch.long, device=device)

    # Scenario 1: perfect tracking, hand AT object.
    r, fail, ok, c = compute_dextrack_reward(fake, target, actions, progress)
    print(f"[1] perfect tracking + hand-at-object  (goal_cond=False):")
    print(f"    reward[0]={r[0]:+.4f}  hand_guide={c['r_hand_guidance'][0]:+.4f}  "
          f"finger_obj={c['r_finger_obj'][0]:+.4f}  goal_hand={c['r_goal_hand'][0]:+.4f}  "
          f"bonus={c['r_bonus'][0]:+.4f}  succeeded={ok[0].item()}")
    print(f"    flag2={c['flag2'][0]:.0f}  goal_dist={c['goal_dist'][0]:.3f}")
    # Expected:
    #   delta_value = 0; r_hand_guide = 0
    #   d_wrist_obj = 0 (gate ON), d_tip_obj = 0; r_finger_obj = -0.3*(0+0) = 0
    #   flag2 = True; inhand_rew = 0; goal_hand_rew = 0
    #   bonus_mask = True (flag2 AND goal_dist=0<0.05); bonus = 1/(1+0) = 1.0
    #   reward = 0 + 0 + 0 + 1.0 = 1.0  ★

    # Scenario 2: 10cm obj drift, wrist 20cm far.
    s2 = {k: v.clone() for k, v in fake.items()}
    s2["obj_pos"][:, 2]  = 0.1
    s2["wrist_pos"][:, 0] = 0.2
    r2, fail2, ok2, c2 = compute_dextrack_reward(s2, target, actions, progress)
    print(f"\n[2] obj 10cm off, wrist 20cm far  (gated finger should = 0):")
    print(f"    reward[0]={r2[0]:+.4f}  hand_guide={c2['r_hand_guidance'][0]:+.4f}  "
          f"finger_obj={c2['r_finger_obj'][0]:+.4f}  goal_hand={c2['r_goal_hand'][0]:+.4f}")
    print(f"    d_wrist_obj={c2['d_wrist_obj'][0]:.3f}  flag2={c2['flag2'][0]:.0f}  "
          f"goal_dist={c2['goal_dist'][0]:.3f}  failed={fail2[0].item()}")

    # Scenario 3: 30cm obj drift → early terminate.
    s3 = {k: v.clone() for k, v in fake.items()}
    s3["obj_pos"][:, 2] = 0.30
    r3, fail3, ok3, c3 = compute_dextrack_reward(s3, target, actions, progress,
                                                  early_terminate_obj_dist=0.2)
    print(f"\n[3] obj 30cm off  (early-terminate at 0.2m → reward=-10):")
    print(f"    reward[0]={r3[0]:+.4f}  failed={fail3[0].item()}  (expect failed=True, reward=-10)")

    print("\n✓ DexTrack reward (goal_cond=False) smoke test passed")


if __name__ == "__main__":
    _smoke_test()
