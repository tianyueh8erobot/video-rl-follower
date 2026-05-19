"""DexTrack tracking reward — faithful port of `compute_hand_reward_tracking_warm`,
the reward function the LEAP+Franka run that reached reward ~210 actually used.

Source (verified line-by-line against DexTrack_success_snapshot):
  isaacgymenvs/tasks/allegro_hand_tracking_generalist.py
  - compute_hand_reward_tracking_warm     L13396-13744  (the w-Franka reward fn)
  - delta_value / w_finger_pos_rew block  L13465-13516
  - goal_cond=False reward branch         L13609-13744
  - input prep (palm/fingertip link pos)  compute_reward_warm L6151-6300
  - delta_qpos = current - reference hand dof   L6770/6780

The successful run config (dextrack_success_run_config.yaml) has:
  goal_cond=False, w_finger_pos_rew=True, w_obj_ornt=False, w_obj_vels=False.

NOTE: our earlier port copied the NON-warm `compute_hand_reward_tracking`
(no w_finger_pos_rew); this rewrite fixes that — the warm fn is the one the
w-Franka path runs.

Reward (DexTrack L13742):
    reward = (-rew_delta_hand_pose_coef) * delta_value
           + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2*right_hand_dist)
           + goal_hand_rew + bonus
           # + hand_up   <- commented out in source; NOT included.

delta_value, w_finger_pos_rew=True (DexTrack L13516) — uses LINK POSITIONS, not
joint angles:
    delta_value = glb_trans_coef  * ||palm_ref   - palm_cur||
                + glb_rot_coef    * ||thumb_ref  - thumb_cur||
                + fingerpose_coef * (||index..|| + ||middle..|| + ||ring..||)
                + hand_qpos_rew_coef_real * ||hand_qpos_ref - hand_qpos_cur||_1
  (per-finger + qpos terms are zeroed past `compute_hand_rew_buf_threshold`;
   that is 900 and our episode is 300 frames, so it never triggers — the code
   is kept only for fidelity.)

flag (DexTrack L13611): flag==2 iff finger-close AND wrist-close.
goal_hand_rew = where(flag==2, 1*(0 - 2*goal_dist), 0)        (object tracking)
bonus         = where(flag==2 & goal_dist<=0.05, 1/(1+10*goal_dist), 0)

SHARPA robot-geometry calibration (these MUST differ from LEAP — see below):
  * palm body: LEAP uses `palm_lower`.  Sharpa's palm `right_hand_C_MC` is
    merged into `panda_link7` by collapse_fixed_joints, so the palm world pose
    is reconstructed as  wrist_pos + R(wrist_quat) @ palm_offset , where
    palm_offset = right_hand_C_MC expressed in the panda_link7 frame
    = [0.020868, -0.020851, 0.107]  (|.|=0.111 m; verified rigid constant).
  * wrist contact gate: LEAP uses `right_hand_dist <= 0.12`.  Measured on the
    reference trajectory (a valid grasp+lift), the Sharpa palm stays
    ~0.13-0.21 m from the object even at the grasp, so 0.12 is geometrically
    unreachable.  Widened to `wrist_gate_thresh` (cfg = 0.2 for Sharpa).
  * fingertips: thumb/index/middle/ring distal phalanges — SHARPA_BODY_NAMES
    indices [21, 3, 7, 16] (the collapsed *_DP bodies).
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
from torch import Tensor


def _quat_geodesic_angle_xyzw(a: Tensor, b: Tensor) -> Tensor:
    dot = (a * b).sum(-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def _quat_rotate_xyzw(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) v by quaternion q (xyzw).  q:(N,4); v:(3,) or (N,3) -> (N,3)."""
    if v.dim() == 1:
        v = v.unsqueeze(0).expand(q.shape[0], 3)
    qvec = q[:, :3]
    qw = q[:, 3:4]
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + qw * t + torch.cross(qvec, t, dim=-1)


# thumb / index / middle / ring distal-phalanx bodies, in SHARPA_BODY_NAMES order
_FINGERTIP_IDXS = [21, 3, 7, 16]


def compute_dextrack_reward(
    state: Dict[str, Tensor],
    target: Dict[str, Tensor],
    actions: Tensor,
    progress_buf: Tensor,
    # ── delta_value hand-pose-guidance coefs (DexTrack L13485-13516) ───────
    glb_trans_coef:          float = 1.0,   # hand_pose_guidance_glb_trans_coef -> diff_palm
    glb_rot_coef:            float = 1.0,   # hand_pose_guidance_glb_rot_coef   -> diff_thumb
    fingerpose_coef:         float = 0.2,   # hand_pose_guidance_fingerpose_coef-> idx+mid+ring
    hand_qpos_rew_coef_real: float = 0.01,  # DexTrack hand_qpos_rew_coef (qpos L1 term)
    delta_hand_pose_coef:    float = 0.3,   # rew_delta_hand_pose_coef
    # ── finger / wrist contact penalty ────────────────────────────────────
    finger_obj_dist_coef:    float = 0.5,   # rew_finger_obj_dist_coef
    wrist_gate_thresh:       float = 0.2,   # SHARPA right_hand_dist gate (LEAP: 0.12)
    finger_dist_cap:         float = 0.6,   # per-finger L2 cap (x4 fingers)
    wrist_dist_cap:          float = 0.5,   # right_hand_dist cap (DexTrack L13423)
    compute_hand_rew_buf_threshold: int = 900,   # delta_value finger-term zeroing
    palm_offset = (0.020868, -0.020851, 0.107),  # right_hand_C_MC in panda_link7 frame
    # ── object 6-D extras (default OFF, matches wfranka success config) ───
    w_obj_ornt:           bool  = False,
    w_obj_vels:           bool  = False,
    cur_ornt_rew_coef:    float = 0.33,
    # ── success / early-termination ───────────────────────────────────────
    success_obj_pos_thresh:   float = 0.05,
    early_terminate_obj_dist: float = 0.0,  # 0 disables (success run resolves early_terminate=False)
    # ── smoothness (opt-in; default off matches paper) ────────────────────
    smoothness_coef: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Return (reward, failed, succeeded, components).  `actions` accepted for
    API parity (the goal_cond=False warm reward has no action-penalty term)."""
    N = state["dof_pos"].shape[0]
    device = state["dof_pos"].device

    # ─── Palm world pose: right_hand_C_MC = panda_link7 (+) rigid offset ────
    off = torch.as_tensor(list(palm_offset), dtype=torch.float32, device=device)
    state_palm  = state["wrist_pos"]  + _quat_rotate_xyzw(state["wrist_quat"],  off)
    target_palm = target["wrist_pos"] + _quat_rotate_xyzw(target["wrist_quat"], off)

    # ─── Fingertip world positions (thumb/index/middle/ring *_DP) ──────────
    cur_tip = state["link_pos"][:, _FINGERTIP_IDXS]      # (N, 4, 3)
    ref_tip = target["link_pos"][:, _FINGERTIP_IDXS]     # (N, 4, 3)

    # ─── delta_value, w_finger_pos_rew=True  (DexTrack L13494-13516) ───────
    diff_palm = torch.norm(target_palm - state_palm, p=2, dim=-1)              # (N,)
    diff_tip = torch.norm(ref_tip - cur_tip, p=2, dim=-1)                      # (N, 4)
    diff_thumb, diff_index, diff_middle, diff_ring = diff_tip.unbind(-1)
    # delta_qpos = current - reference hand dof (DexTrack L6770); L1 norm of the
    # 22 hand DOF (delta_qpos[:, 7:]).  Sign-independent under |.|.
    delta_qpos_value = (target["dof_pos"][:, 7:] - state["dof_pos"][:, 7:]).abs().sum(-1)
    # zero per-finger + qpos terms past the buffer threshold (DexTrack L13509-13513).
    # threshold=900 >> 300-frame episode -> never fires; kept for fidelity.
    past = (progress_buf >= compute_hand_rew_buf_threshold)
    zero = torch.zeros_like(diff_thumb)
    diff_thumb       = torch.where(past, zero, diff_thumb)
    diff_index       = torch.where(past, zero, diff_index)
    diff_middle      = torch.where(past, zero, diff_middle)
    diff_ring        = torch.where(past, zero, diff_ring)
    delta_qpos_value = torch.where(past, torch.zeros_like(delta_qpos_value), delta_qpos_value)
    # NOTE: glb_rot_coef multiplies diff_thumb — this odd mapping is verbatim
    # from DexTrack L13516 (hand_rot_rew_coef * diff_thumb_link_pos).
    delta_value = (glb_trans_coef * diff_palm
                   + glb_rot_coef * diff_thumb
                   + fingerpose_coef * (diff_index + diff_middle + diff_ring)
                   + hand_qpos_rew_coef_real * delta_qpos_value)
    r_hand_guidance = -delta_hand_pose_coef * delta_value

    # ─── Object goal distance (DexTrack L13419) ────────────────────────────
    obj_pos = state["obj_pos"]
    goal_dist = torch.norm(target["obj_pos"] - obj_pos, p=2, dim=-1)
    rot_dist = _quat_geodesic_angle_xyzw(target["obj_quat"], state["obj_quat"])

    # ─── right_hand_dist: palm <-> object, capped (DexTrack L13422-13423) ──
    right_hand_dist = torch.norm(obj_pos - state_palm, p=2, dim=-1)
    right_hand_dist = torch.minimum(right_hand_dist,
                                    torch.full_like(right_hand_dist, wrist_dist_cap))

    # ─── right_hand_finger_dist: sum of 4 tip<->object, capped (L13425-13435)─
    num_fingers = 4
    finger_dist = torch.norm(cur_tip - obj_pos[:, None, :], p=2, dim=-1).sum(-1)   # (N,)
    finger_dist = torch.minimum(finger_dist,
                                torch.full_like(finger_dist, finger_dist_cap * num_fingers))

    # ─── flag==2 iff finger-close AND wrist-close (DexTrack L13611) ────────
    # finger gate keeps DexTrack's 0.12*num_fingers; wrist gate is the
    # SHARPA-calibrated wrist_gate_thresh (LEAP value 0.12 is unreachable).
    finger_close = finger_dist     <= 0.12 * num_fingers
    wrist_close  = right_hand_dist <= wrist_gate_thresh
    flag2 = finger_close & wrist_close

    # ─── inhand_obj_pos_ornt_rew (DexTrack L13617; goal_cond=False -> base 0.0)─
    inhand_rew = 1.0 * (0.0 - 2.0 * goal_dist)
    if w_obj_ornt:
        inhand_rew = inhand_rew + cur_ornt_rew_coef * (0.0 - rot_dist)
    if w_obj_vels:
        d_lin = torch.norm(target["obj_lin_vel"] - state["obj_lin_vel"], p=2, dim=-1)
        d_ang = torch.norm(target["obj_ang_vel"] - state["obj_ang_vel"], p=2, dim=-1)
        inhand_rew = inhand_rew + (120.0 * 0.9 - 2.0 * d_lin) / 120.0 \
                                + (120.0 * 0.9 - 2.0 * d_ang) / 120.0

    # ─── goal_hand_rew: object tracking, gated on contact (DexTrack L13654-55)─
    goal_hand_rew = torch.where(flag2, inhand_rew, torch.zeros_like(inhand_rew))

    # ─── bonus: flag==2 AND goal_dist<=0.05 (DexTrack L13678-13679) ────────
    bonus = torch.where(flag2 & (goal_dist <= success_obj_pos_thresh),
                        1.0 / (1.0 + 10.0 * goal_dist),
                        torch.zeros_like(goal_dist))
    if w_obj_vels:
        d_lin = torch.norm(target["obj_lin_vel"] - state["obj_lin_vel"], p=2, dim=-1)
        d_ang = torch.norm(target["obj_ang_vel"] - state["obj_ang_vel"], p=2, dim=-1)
        thr = 0.05 * 12.0
        bonus = bonus + torch.where(d_lin <= thr, 1.0 / (1.0 + 10.0 * d_lin / 120.0),
                                    torch.zeros_like(d_lin)) \
                      + torch.where(d_ang <= thr, 1.0 / (1.0 + 10.0 * d_ang / 120.0),
                                    torch.zeros_like(d_ang))

    # ─── wrist-gating of the finger penalty (DexTrack L13730-13731) ────────
    # when the wrist is far, the finger-distance term drops out of the penalty.
    finger_dist_gated = torch.where(right_hand_dist <= wrist_gate_thresh,
                                    finger_dist, torch.zeros_like(finger_dist))
    r_finger_obj = -finger_obj_dist_coef * (finger_dist_gated + 2.0 * right_hand_dist)

    # ─── smoothness (DexTrack default off) ────────────────────────────────
    r_smoothness = torch.zeros(N, device=device)
    if smoothness_coef > 0 and "prev_dof_vel" in state:
        r_smoothness = -smoothness_coef * torch.norm(
            state["prev_dof_vel"] - state["dof_vel"], p=2, dim=-1)

    # ─── Total (DexTrack L13742; hand_up commented out -> NOT added) ───────
    reward = r_hand_guidance + r_finger_obj + goal_hand_rew + bonus + r_smoothness

    # ─── Success / early termination ──────────────────────────────────────
    succeeded = goal_dist <= success_obj_pos_thresh
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
        "d_tip_obj":        finger_dist,        # sum of 4 fingertip<->object dists
        "d_wrist_obj":      right_hand_dist,    # palm (right_hand_C_MC) <-> object
        "flag2":            flag2.float(),
    }
    return reward, failed, succeeded, components


def _smoke_test():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    N = 4
    fake = {
        "dof_pos":      torch.zeros(N, 29, device=device),
        "dof_vel":      torch.zeros(N, 29, device=device),
        "prev_dof_vel": torch.zeros(N, 29, device=device),
        "wrist_pos":    torch.zeros(N, 3, device=device),
        "wrist_quat":   torch.tensor([[0, 0, 0, 1.0]] * N, device=device),
        "link_pos":     torch.zeros(N, 22, 3, device=device),
        "obj_pos":      torch.zeros(N, 3, device=device),
        "obj_quat":     torch.tensor([[0, 0, 0, 1.0]] * N, device=device),
        "obj_lin_vel":  torch.zeros(N, 3, device=device),
        "obj_ang_vel":  torch.zeros(N, 3, device=device),
    }
    target = {k: v.clone() for k, v in fake.items()}
    progress = torch.zeros(N, dtype=torch.long, device=device)

    # Scenario: perfect tracking, object at the palm (within the 0.2 gate).
    fake2 = {k: v.clone() for k, v in fake.items()}
    fake2["obj_pos"] = fake2["wrist_pos"] + torch.tensor([0.0209, -0.0209, 0.107], device=device)
    target2 = {k: v.clone() for k, v in fake2.items()}
    r, fail, ok, c = compute_dextrack_reward(fake2, target2, torch.zeros(N, 29, device=device), progress)
    print(f"[perfect track, obj at palm] reward={r[0]:+.4f}  flag2={c['flag2'][0]:.0f}  "
          f"d_wrist_obj={c['d_wrist_obj'][0]:.4f}  bonus={c['r_bonus'][0]:+.4f}  delta_value={c['delta_value'][0]:.4f}")
    print("  expect flag2=1, d_wrist_obj~0, bonus~1.0, delta_value~0")
    print("✓ dextrack_reward (warm goal_cond=False, w_finger_pos_rew) smoke ok")


if __name__ == "__main__":
    _smoke_test()
