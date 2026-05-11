"""VideoRLFollower env (ManipTrans × SimToolReal hybrid).

A SimToolReal-based environment that tracks a single reference trajectory
(object 6D pose + MANO hand keypoints + Sharpa retargeted DOF) loaded from a
JSON trajectory file produced by ``tools/process_maniptrans_trajectory.py`` +
``tools/merge_retarget_into_trajectory.py``.

Reward design (final after Codex review):

  r_total = R_imit              # ManipTrans Stage-1, 9-term position/orientation
                                # subset of the 14-term paper reward (coverage
                                # 5.15 / 6.40 ≈ 80.5% of paper's coefficient sum;
                                # vel + power terms not yet implemented)
          + R_goal_sparse       # SimToolReal sparse bonus (keypoints+dwell)
          + R_act_penalty       # SimToolReal action regularization (kept)
          + R_obj_vel_pen       # SimToolReal object-velocity penalty (cfg
                                # default 0.0 — set objectLinVelPenaltyScale +
                                # objectAngVelPenaltyScale > 0 to enable as
                                # anti-yeet; off by default to follow ManipTrans)

Reset (ManipTrans Stage-2 style):

  on episode reset:
    seq_idx = random in [0, T)  (or 0 if randomStateInit=False)
    sub_goal_idx[env]   = seq_idx                  # ★ goal aligned to reset frame
    arm_hand_dof_pos[env, 7:7+22] = opt_dof_pos[seq_idx]   # warm-start hand
    (arm DOFs left at SimToolReal-randomized default — no IK to opt_wrist_pos)

Sub-goal advancement (★ TRACKING BRANCH ★):

  Every physics step:
    sub_goal_idx[env]   = min(sub_goal_idx[env] + 1, T - 1)  # time-driven
    (no longer success-driven — paper's progress_buf-equivalent)

  This matches ManipTrans paper (dexhandmanip_sh.py:1333 `progress_buf += 1`).
  Policy MUST keep up with trajectory's pace — no "sit still and dwell" plateau.

The base class still runs its compute_kuka_reward(), but ``cfg.env`` zeroes
out the dense pickup terms (liftingRewScale=0, liftingBonus=0,
keypointRewScale=0, distanceDeltaRewScale=0) so what remains from super is the
sparse bonus + action/vel penalties — which is exactly what we want.

Single-hand only.  Bimanual support is out of scope for this codebase fork.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from gym import spaces
from torch import Tensor

from isaacgymenvs.tasks.simtoolreal.env import SimToolReal

from .trajectory import ReferenceTrajectory


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw layout, matches SimToolReal's keypoint rotation use)
# ---------------------------------------------------------------------------


def _quat_inverse_xyzw(q: Tensor) -> Tensor:
    """Conjugate of a unit quaternion."""
    out = q.clone()
    out[..., :3] *= -1.0
    return out


def _quat_rotate_xyzw(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector ``v`` by quaternion ``q`` (xyzw)."""
    qv = q[..., :3]
    qw = q[..., 3:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def _quat_geodesic_angle_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Returns the angle (radians) between two unit quaternions."""
    dot = (a * b).sum(-1).abs().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return 2.0 * torch.acos(dot)


def _quat_conjugate_xyzw(q: Tensor) -> Tensor:
    """Conjugate of quaternion (xyzw)."""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def _quat_mul_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Quaternion multiplication a*b for xyzw layout."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def _rotvec_to_quat_xyzw(rotvec: Tensor) -> Tensor:
    """Convert rotation vector (T, 3) to quaternion (T, 4) xyzw."""
    angle = rotvec.norm(dim=-1, keepdim=True)                      # (T, 1)
    half = angle * 0.5
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    # Avoid division by zero when angle=0; axis becomes [0,0,1] then sin=0 so xyz=0.
    safe_norm = torch.where(angle < 1e-9, torch.ones_like(angle), angle)
    axis = rotvec / safe_norm                                       # (T, 3)
    return torch.cat([axis * sin_half, cos_half], dim=-1)           # (T, 4) xyzw


# ---------------------------------------------------------------------------
# Sharpa right-hand body / weight definitions, replicated from
# maniptrans_envs/lib/envs/dexhands/sharpa.py to avoid a runtime ManipTrans
# package dependency.  KEEP IN SYNC if sharpa.py changes.
# ---------------------------------------------------------------------------

# 28 link names, in the same order as Sharpa.body_names (after right_ prefix).
SHARPA_BODY_NAMES = [
    "right_hand_C_MC",
    "right_index_MCP_VL", "right_index_PP", "right_index_MP", "right_index_DP", "right_index_fingertip",
    "right_middle_MCP_VL", "right_middle_PP", "right_middle_MP", "right_middle_DP", "right_middle_fingertip",
    "right_pinky_MC", "right_pinky_MCP_VL", "right_pinky_PP", "right_pinky_MP", "right_pinky_DP", "right_pinky_fingertip",
    "right_ring_MCP_VL", "right_ring_PP", "right_ring_MP", "right_ring_DP", "right_ring_fingertip",
    "right_thumb_CMC_VL", "right_thumb_MC", "right_thumb_MCP_VL", "right_thumb_PP", "right_thumb_DP", "right_thumb_fingertip",
]

# Per-Sharpa-link → MANO joint name (Sharpa's dex2hand_mapping reversed).
# Matches sharpa.py:122-149 hand2dex_mapping; verified by tools/validate_sharpa_mapping.py.
SHARPA_DEX2MANO_NAME = {
    0: "wrist",
    1: "index_proximal", 2: "index_proximal", 3: "index_intermediate", 4: "index_distal", 5: "index_tip",
    6: "middle_proximal", 7: "middle_proximal", 8: "middle_intermediate", 9: "middle_distal", 10: "middle_tip",
    11: "pinky_proximal", 12: "pinky_proximal", 13: "pinky_proximal", 14: "pinky_intermediate", 15: "pinky_distal", 16: "pinky_tip",
    17: "ring_proximal", 18: "ring_proximal", 19: "ring_intermediate", 20: "ring_distal", 21: "ring_tip",
    22: "thumb_proximal", 23: "thumb_proximal", 24: "thumb_proximal", 25: "thumb_intermediate", 26: "thumb_distal", 27: "thumb_tip",
}

# Sharpa.weight_idx (matches sharpa.py:186-194).  Indices reference SHARPA_BODY_NAMES.
SHARPA_WEIGHT_IDX = {
    "thumb_tip":      [27],
    "index_tip":      [5],
    "middle_tip":     [10],
    "ring_tip":       [21],
    "pinky_tip":      [16],
    "level_1_joints": [1, 6, 11, 17, 22],
    "level_2_joints": [2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26],
}

# 21 MANO keypoint names → index, matches trajectory.py MANO_JOINT_ORDER.
MANO_NAME_TO_IDX = {
    "wrist": 0,
    "index_proximal": 1,  "index_intermediate": 2,  "index_distal": 3,
    "middle_proximal": 4, "middle_intermediate": 5, "middle_distal": 6,
    "pinky_proximal": 7,  "pinky_intermediate": 8,  "pinky_distal": 9,
    "ring_proximal": 10,  "ring_intermediate": 11,  "ring_distal": 12,
    "thumb_proximal": 13, "thumb_intermediate": 14, "thumb_distal": 15,
    "index_tip": 16, "middle_tip": 17, "pinky_tip": 18, "ring_tip": 19, "thumb_tip": 20,
}


class VideoRLFollower(SimToolReal):
    """Trajectory-following variant of SimToolReal.

    Configuration additions (see ``cfg/task/VideoRLFollower.yaml``):

    ``env.trajectoryPath``                     str   path to the .json trajectory
    ``env.handPoseRewardScale.wristPos``       float weight on wrist-position term
    ``env.handPoseRewardScale.wristRot``       float weight on wrist-rotation term
    ``env.handPoseRewardScale.fingertip``      float weight on fingertip term
    ``env.handPoseRewardScale.lambdaWristPos`` float decay rate (m^-1)
    ``env.handPoseRewardScale.lambdaWristRot`` float decay rate (rad^-1)
    ``env.handPoseRewardScale.lambdaFingertip`` float decay rate (m^-1)
    ``env.exposeHandGoalToPolicy``             bool  append hand goal to obs
    """

    def __init__(self, cfg, *args, **kwargs):
        # ----- Resolve trajectory path -----
        traj_path = cfg["env"].get("trajectoryPath", None)
        if traj_path is None:
            raise ValueError(
                "VideoRLFollower requires env.trajectoryPath to point at a "
                "JSON trajectory produced by tools/process_maniptrans_trajectory.py"
            )
        if not os.path.isabs(traj_path):
            project_root = Path(__file__).resolve().parents[3]
            traj_path = str(project_root / traj_path)
        self._trajectory_path = traj_path
        self._trajectory = ReferenceTrajectory.from_file(self._trajectory_path)

        # Hand-reward weights (with sensible defaults).
        h_cfg = cfg["env"].get("handPoseRewardScale", {}) or {}
        self.w_wrist_pos = float(h_cfg.get("wristPos", 1.0))
        self.w_wrist_rot = float(h_cfg.get("wristRot", 0.5))
        self.w_fingertip = float(h_cfg.get("fingertip", 2.0))
        self.lambda_wrist_pos = float(h_cfg.get("lambdaWristPos", 30.0))
        self.lambda_wrist_rot = float(h_cfg.get("lambdaWristRot", 5.0))
        self.lambda_fingertip = float(h_cfg.get("lambdaFingertip", 50.0))
        self.expose_hand_goal_to_policy = bool(
            cfg["env"].get("exposeHandGoalToPolicy", True)
        )
        # ── ManipTrans Stage-1 imitation reward (R_imit) ──────────────────
        # All defaults verbatim from dexhandimitator.py:1115-1172.  Set
        # ``imitRewardScale.enable=0`` to disable the entire R_imit term.
        ir = cfg["env"].get("imitRewardScale", {}) or {}
        self.w_imit              = float(ir.get("enable",      1.0))   # 0 → off
        self.w_imit_eef_pos      = float(ir.get("eefPos",      0.1))
        self.w_imit_eef_rot      = float(ir.get("eefRot",      0.6))
        self.w_imit_thumb_tip    = float(ir.get("thumbTip",    0.9))
        self.w_imit_index_tip    = float(ir.get("indexTip",    0.8))
        self.w_imit_middle_tip   = float(ir.get("middleTip",   0.75))
        self.w_imit_ring_tip     = float(ir.get("ringTip",     0.6))
        self.w_imit_pinky_tip    = float(ir.get("pinkyTip",    0.6))
        self.w_imit_level_1      = float(ir.get("level1",      0.5))
        self.w_imit_level_2      = float(ir.get("level2",      0.3))
        # Coarse wrist-position term (lambda~4) so policy gets gradient signal
        # at >10cm from goal — paper's lambdaEefPos=40 saturates beyond ~10cm
        # because their floating-base setup teleports the wrist to opt_wrist_pos
        # at reset, while we have a Kuka arm with no IK warm-start.
        self.w_imit_eef_pos_wide      = float(ir.get("eefPosWide",        0.0))
        self.lambda_imit_eef_pos_wide = float(ir.get("lambdaEefPosWide",  4.0))
        # Decay rates (paper):
        self.lambda_imit_eef_pos    = float(ir.get("lambdaEefPos",     40.0))
        self.lambda_imit_eef_rot    = float(ir.get("lambdaEefRot",      1.0))
        self.lambda_imit_thumb_tip  = float(ir.get("lambdaThumbTip",  100.0))
        self.lambda_imit_index_tip  = float(ir.get("lambdaIndexTip",   90.0))
        self.lambda_imit_middle_tip = float(ir.get("lambdaMiddleTip",  80.0))
        self.lambda_imit_ring_tip   = float(ir.get("lambdaRingTip",    60.0))
        self.lambda_imit_pinky_tip  = float(ir.get("lambdaPinkyTip",   60.0))
        self.lambda_imit_level_1    = float(ir.get("lambdaLevel1",     50.0))
        self.lambda_imit_level_2    = float(ir.get("lambdaLevel2",     40.0))

        # Optional ManipTrans Stage-2 reset: warm-start hand DOFs from
        # opt_dof_pos[seq_idx] of the retarget pkl so RL doesn't have to
        # discover the pre-grasp pose from scratch.  When enabled, we hard-
        # require the trajectory JSON to have a `dex.dof_pos` block matching
        # the env's hand DOF count — silent truncation breaks the warm-start.
        self.use_retarget_dof_init = bool(
            cfg["env"].get("useRetargetDofInit", True)
        )
        if self.use_retarget_dof_init:
            if self._trajectory.dex_dof_pos is None:
                raise ValueError(
                    "[VideoRLFollower] useRetargetDofInit=True but the "
                    f"trajectory '{traj_path}' has no `dex.dof_pos` block.  "
                    "Either run tools/merge_retarget_into_trajectory.py to "
                    "add a Sharpa retarget pkl, or set useRetargetDofInit=False."
                )
        # Random-state init: episode reset can pick a random frame from the
        # trajectory rather than always frame 0 (helps cover the entire
        # state-space during training).  ★ sub_goal_idx will be aligned to
        # this seq_idx so the goal does not jump back to frame 0.
        self.random_state_init = bool(
            cfg["env"].get("randomStateInit", True)
        )
        # Optional ceiling on the random-init seq_idx (≥ 0; -1 = no ceiling).
        # Combined with the trajectory's reachable_frames mask: init seq_idx
        # ∈ {f : reachable[f] AND f ≤ randomInitMaxIdx}.  Use this as a
        # curriculum knob — narrow the init range to the early lift phase
        # while the policy is learning, then widen.
        self.random_init_max_idx = int(
            cfg["env"].get("randomInitMaxIdx", -1)
        )
        # Relax the dex DOF warm-start toward the open-hand neutral pose by
        # this factor (1.0 = paper-exact closed grasp, <1.0 backs fingers off).
        # Mitigates IK-error-induced contact penetration: the kuka can't land
        # the wrist at the trajectory's exact target (~1cm error even for
        # reachable frames), so a closed-on-the-target grasp ends up slightly
        # offset from the actual object → finger overlap → PhysX ejection.
        # 0.9 ≈ 10% relaxation gives finger clearance without spoiling the
        # warm-start signal.  Applied only when useRetargetDofInit=True.
        self.dof_init_relax_scale = float(
            cfg["env"].get("dofInitRelaxScale", 1.0)
        )

        # ----- Force fixed-goal mode -----
        # The base class only ever reads trajectory_states[..., 0:7] for the
        # object goal; we override the tensor below with a richer one.
        cfg["env"]["useFixedGoalStates"] = True
        cfg["env"]["fixedGoalStates"] = [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        ]
        cfg["env"]["maxConsecutiveSuccesses"] = self._trajectory.num_goals

        # Seed cfg.env.objectStartPose from the trajectory's frame-0 pose so
        # the object spawns where the trajectory expects it (already on the
        # table at the trajectory's table_z).  Without this, SimToolReal's
        # default `tableResetZ + tableObjectZOffset` puts the object 11cm
        # above the trajectory's expected pose, leaving a constant ~20cm
        # d_obj_pos gap the policy can't close (object doesn't reach goal
        # without manipulation, but hand can't manipulate without first
        # closing arm).  SimToolReal §III.B's "random table object" applies
        # to their grasping task; ours is trajectory tracking.
        if self._trajectory.object_start_pose is not None:
            cfg["env"]["objectStartPose"] = [
                float(v) for v in self._trajectory.object_start_pose.tolist()
            ]
        cfg["env"]["useFixedInitObjectPose"] = True  # don't add SimToolReal noise

        # ----- Trajectory-driven object asset (URDF) selection -----
        self._use_trajectory_object = bool(
            cfg["env"].get("useTrajectoryObject", True)
        ) and self._trajectory.object_urdf_path is not None
        if self._use_trajectory_object:
            urdf = self._trajectory.object_urdf_path
            if not os.path.isabs(urdf):
                project_root = Path(__file__).resolve().parents[3]
                urdf = str(project_root / urdf)
            if not os.path.isfile(urdf):
                raise FileNotFoundError(
                    f"Trajectory points at object URDF '{urdf}' but it doesn't "
                    "exist on disk.  Either update the trajectory's "
                    "object.urdf_path or download the corresponding ManipTrans "
                    "demo data (see ManipTrans README §Prerequisites)."
                )
            self._trajectory_object_urdf_abs = urdf
            # Pin objectName to a sentinel so the base method falls into our
            # subclass override branch instead of NAME_TO_OBJECT lookup.
            cfg["env"]["objectName"] = "video_rl_follower_trajectory_object"

        super().__init__(cfg, *args, **kwargs)

        # ----- Move trajectory to env device, install dense state tensor -----
        self._trajectory.to(self.device)
        self.trajectory_states = self._trajectory.stacked_goals()
        self.max_consecutive_successes = self._trajectory.num_goals
        self._traj_K = self._trajectory.num_fingertips

        # Per-env hand goal buffers.
        self._wrist_goal = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )
        self._fingertip_goal_local = torch.zeros(
            (self.num_envs, self._traj_K, 3), device=self.device, dtype=torch.float32
        )
        self._fingertip_curr_local = torch.zeros_like(self._fingertip_goal_local)

        # ── Sub-goal index per env (ManipTrans-aligned, replaces successes-mod) ──
        # Allocate BEFORE _set_hand_goal_from_trajectory which dereferences it.
        # reset_idx will overwrite per-env to seq_idx at every episode reset.
        self.sub_goal_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._set_hand_goal_from_trajectory(env_ids)

        if self.expose_hand_goal_to_policy:
            self._extend_obs_state_spaces()

        # ── ManipTrans-style R_imit setup ────────────────────────────────────
        self._imit_active = (
            self.w_imit > 0.0
            and self._trajectory.joints_world is not None
        )
        if self._imit_active:
            self._setup_imit_link_indices()
            # Pre-build the (27,) gather index that maps each non-wrist Sharpa
            # body to its MANO-keypoint target index (paper to_hand mapping).
            non_wrist_sharpa_idx = [i for i in range(28) if i != 0]
            mano_target_idx = [
                MANO_NAME_TO_IDX[SHARPA_DEX2MANO_NAME[i]] for i in non_wrist_sharpa_idx
            ]
            self._imit_target_mano_idx = torch.tensor(
                mano_target_idx, device=self.device, dtype=torch.long
            )                                                # (27,)
            # Per-group index list (NOTE: subtract 1 because we skip wrist=body[0])
            def _grp(name):
                return torch.tensor(
                    [k - 1 for k in SHARPA_WEIGHT_IDX[name]],
                    device=self.device, dtype=torch.long,
                )
            self._imit_grp_thumb_tip  = _grp("thumb_tip")
            self._imit_grp_index_tip  = _grp("index_tip")
            self._imit_grp_middle_tip = _grp("middle_tip")
            self._imit_grp_ring_tip   = _grp("ring_tip")
            self._imit_grp_pinky_tip  = _grp("pinky_tip")
            self._imit_grp_level_1    = _grp("level_1_joints")
            self._imit_grp_level_2    = _grp("level_2_joints")
            print(f"[VideoRLFollower] R_imit ON: 27 target points expanded "
                  f"from MANO via to_hand mapping; {len(self._imit_link_handles)} "
                  f"of 28 Sharpa links matched in URDF.")
        else:
            self._imit_link_handles = []
            self._imit_target_mano_idx = None

        # Legacy hand-pose fields (kept zero so old log keys still exist).
        self._dex_goal_world = None
        self._dex_link_handles = []
        self._dex_link_weights = None
        self._full_hand_active = False
        self.w_full_hand = 0.0

    # ------------------------------------------------------------------
    # Space + queue resize so rl_games sees the correct shapes
    # ------------------------------------------------------------------

    def _extend_obs_state_spaces(self) -> None:
        """Extend gym observation_space, state_space and obs/state buffers to
        account for the hand-goal channels appended in
        ``populate_obs_and_states_buffers``.

        The base ``obs_queue`` is INTENTIONALLY left at its original width.
        Hand-goal channels are not subjected to the action-loop observation
        delay simulation (goals are control inputs, not noisy sensor reads),
        so they are appended **after** the base class's delay sampling step.
        Widening the queue would also break ``update_queue``'s shape assertion
        which compares ``current_values.shape[-1]`` against ``queue.shape[-1]``.
        """
        extra = self.hand_goal_obs_dim                           # 7 + K*3

        # 1) gym spaces — rl_games reads these to size the network heads.
        new_obs_size = self.num_observations + extra
        new_state_size = self.num_states + extra
        self.num_observations = new_obs_size
        self.num_states = new_state_size
        self.obs_space = spaces.Box(
            np.ones(new_obs_size) * -np.inf,
            np.ones(new_obs_size) * np.inf,
        )
        self.state_space = spaces.Box(
            np.ones(new_state_size) * -np.inf,
            np.ones(new_state_size) * np.inf,
        )
        self.cfg["env"]["numObservations"] = new_obs_size
        self.cfg["env"]["numStates"] = new_state_size

        # 2) re-allocate obs_buf and states_buf so rl_games sees the right
        #    shape on the very first ``reset()`` (before any compute step).
        self.obs_buf = torch.zeros(
            self.num_envs, new_obs_size, dtype=torch.float32, device=self.device
        )
        self.states_buf = torch.zeros(
            self.num_envs, new_state_size, dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # Object asset (URDF) selection
    # ------------------------------------------------------------------

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        """If the trajectory specifies a URDF, return that single asset and
        skip SimToolReal's procedural primitive generation entirely.

        We still honour the base ``useFixedGoalStates`` block by inlining its
        ``trajectory_states`` write here — we cannot call ``super()`` cleanly
        because the base method's ``object_name`` branch would raise on our
        sentinel name.
        """
        if not self._use_trajectory_object:
            return super()._main_object_assets_and_scales(
                object_asset_root, tmp_assets_dir
            )

        object_asset_files = [self._trajectory_object_urdf_abs]

        # The base ``object_asset_scales`` contract is normalised by
        # ``object_base_size``: ``_handle_head_primitives`` returns scales as
        # ``(x / object_base_size, y / object_base_size, z / object_base_size)``
        # (env.py L1740-1747) and the keypoint code multiplies them back by
        # ``object_base_size`` (L2091-2092).  We therefore need to do the same
        # division to keep the geometric scale correct.  The trajectory JSON
        # stores the grasp bbox in **metres** for human readability.
        bbox_m = self._trajectory.object_grasp_bbox_scale            # (m, m, m)
        base = float(self.object_base_size)
        if base <= 0.0:
            raise ValueError(
                f"objectBaseSize must be > 0, got {base}"
            )
        normalised = (bbox_m[0] / base, bbox_m[1] / base, bbox_m[2] / base)
        object_asset_scales = [normalised]
        need_vhacds = [self._trajectory.object_need_vhacd]

        # Replicate the base's useFixedGoalStates → trajectory_states wiring.
        if self.cfg["env"]["useFixedGoalStates"]:
            fixed = self.cfg["env"].get("fixedGoalStates")
            fixed_path = self.cfg["env"].get("fixedGoalStatesJsonPath")
            if fixed is not None:
                self.trajectory_states = torch.tensor(
                    fixed, device=self.device
                )
            elif fixed_path is not None:
                with open(fixed_path, "r") as f:
                    import json as _json
                    self.trajectory_states = torch.tensor(
                        _json.load(f)["goals"], device=self.device
                    )
            self.max_consecutive_successes = len(self.trajectory_states)

        return object_asset_files, object_asset_scales, need_vhacds

    # ------------------------------------------------------------------
    # Object asset gravity / damping override
    # ------------------------------------------------------------------

    def _load_main_object_asset(self):
        """Override SimToolReal's loader to mirror ManipTrans's
        ``_create_obj_assets`` (dexhandmanip_sh.py:572-616).  The MaipTrans
        recipe is essential for stability under untrained-policy contacts:

          - ``override_com=True, override_inertia=True``: PhysX recomputes
            the inertia tensor from the (VHACD'd) geometry, avoiding the
            brittle defaults that come with un-curated URDFs.
          - ``max_linear_velocity=50, max_angular_velocity=100``: HARD
            velocity caps that prevent the object from being launched into
            CUDA-illegal-memory-access territory when the hand flails.
            **This is the main missing ingredient that causes our segfaults.**
          - ``vhacd_resolution=200000``: high-quality convex decomposition.
          - ``thickness=0.001``: thin contact layer.
          - density=200 / friction=2.0 / rolling=0.05 / torsion=0.05: the
            paper's "low-fill 3D-print" object physics.

        We still expose ``objectDisableGravity`` / damping via cfg as
        debugging knobs but DEFAULT them to the ManipTrans values
        (gravity ON, no extra damping).
        """
        from isaacgym import gymapi
        import os as _os

        # Cfg overrides — ManipTrans defaults (gravity ON, no damping).
        disable_g = bool(self.cfg["env"].get("objectDisableGravity", False))
        ang_damp  = float(self.cfg["env"].get("objectAngularDamping", 0.0))
        lin_damp  = float(self.cfg["env"].get("objectLinearDamping", 0.0))

        object_assets = []
        for object_asset_file, need_vhacd in zip(
            self.object_asset_files, self.object_need_vhacds
        ):
            opts = gymapi.AssetOptions()
            # ── ManipTrans paper recipe ──────────────────────────────────
            opts.override_com               = True
            opts.override_inertia           = True
            opts.convex_decomposition_from_submeshes = True
            opts.mesh_normal_mode           = gymapi.COMPUTE_PER_VERTEX
            opts.thickness                  = 0.001
            opts.max_linear_velocity        = 50.0   # ★ anti-explosion cap
            opts.max_angular_velocity       = 100.0  # ★ anti-explosion cap
            opts.fix_base_link              = False
            opts.vhacd_enabled              = True
            opts.vhacd_params               = gymapi.VhacdParams()
            opts.vhacd_params.resolution    = 200000
            opts.density                    = 200.0  # 3D-printed equivalent
            # ── Debugging knobs (cfg-overridable) ────────────────────────
            opts.disable_gravity            = disable_g
            opts.angular_damping            = ang_damp
            opts.linear_damping             = lin_damp
            # ── kept for compatibility with SimToolReal pipeline ─────────
            opts.collapse_fixed_joints      = True
            opts.replace_cylinder_with_capsule = True

            asset = self.gym.load_asset(
                self.sim,
                _os.path.dirname(object_asset_file),
                _os.path.basename(object_asset_file),
                opts,
            )
            # ★ Match paper friction (compensates missing skin friction).
            shape_props = self.gym.get_asset_rigid_shape_properties(asset)
            for el in shape_props:
                el.friction = 2.0
                el.rolling_friction = 0.05
                el.torsion_friction = 0.05
            self.gym.set_asset_rigid_shape_properties(asset, shape_props)
            object_assets.append(asset)

        rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])
        sh_count = self.gym.get_asset_rigid_shape_count(object_assets[0])
        print(f"[VideoRLFollower] object asset (ManipTrans recipe): "
              f"disable_gravity={disable_g}, lin_damp={lin_damp}, "
              f"ang_damp={ang_damp}, max_lin_vel=50, max_ang_vel=100, "
              f"density=200, friction=2.0, vhacd_res=200000")
        return object_assets, rb_count, sh_count

    # ------------------------------------------------------------------
    # Trajectory-aware goal handling
    # ------------------------------------------------------------------

    def _setup_imit_link_indices(self) -> None:
        """Look up Sharpa link rigid-body indices in the SimToolReal robot.

        SimToolReal builds ``self.rigid_body_name_to_idx["robot/<name>"]``
        during actor creation (simtoolreal/env.py:2014-2019), giving the
        env-domain rigid-body slot for every Sharpa link by name.

        The shipped Kuka+Sharpa URDF
        (assets/urdf/kuka_sharpa_description/iiwa14_right_sharpa_adjusted_restricted.urdf)
        has ``collapse_fixed_joints=True`` applied at load time, which fuses
        6 fixed-joined links into their parents:
          - right_hand_C_MC          → fused into iiwa14_link_7
          - right_<finger>_fingertip → fused into right_<finger>_DP
        We accept these substitutions explicitly so R_imit can still run on
        the unmodified asset.  The DP→fingertip substitution is mildly
        biased (the DP origin is the proximal-end joint, not the tip vertex
        — ~3-5 cm off depending on finger), but the alternative is to
        disable R_imit entirely or rebuild the URDF.
        """
        # Remap missing-after-collapse names → present-in-URDF names.
        FUSED_FALLBACKS = {
            "right_hand_C_MC":         "iiwa14_link_7",
            "right_thumb_fingertip":   "right_thumb_DP",
            "right_index_fingertip":   "right_index_DP",
            "right_middle_fingertip":  "right_middle_DP",
            "right_ring_fingertip":    "right_ring_DP",
            "right_pinky_fingertip":   "right_pinky_DP",
        }
        idx_map = self.rigid_body_name_to_idx
        handles = []
        unresolvable = []
        substituted = []
        for name in SHARPA_BODY_NAMES:
            key = "robot/" + name
            if key in idx_map:
                handles.append(int(idx_map[key]))
                continue
            fb = FUSED_FALLBACKS.get(name)
            if fb is not None and ("robot/" + fb) in idx_map:
                handles.append(int(idx_map["robot/" + fb]))
                substituted.append((name, fb))
                continue
            unresolvable.append(name)
            handles.append(-1)
        if unresolvable:
            raise RuntimeError(
                "[VideoRLFollower] R_imit setup failed: the loaded robot "
                f"URDF is missing {len(unresolvable)} Sharpa body links required "
                f"by the dex2hand mapping: {unresolvable}.  Either fix the URDF "
                "or set imitRewardScale.enable=0 to disable R_imit."
            )
        if substituted:
            print(f"[VideoRLFollower] R_imit substituted "
                  f"{len(substituted)} fused-joint links: {substituted}")
        self._imit_link_handles = handles                  # length 28
        self._imit_link_handles_t = torch.tensor(
            handles, device=self.device, dtype=torch.long
        )

    def _set_hand_goal_from_trajectory(self, env_ids: Tensor) -> None:
        """Copy MANO wrist + 5-fingertip-local targets for the given envs from
        ``trajectory[sub_goal_idx[env]]`` into the per-env goal buffers.
        Caller is responsible for having updated ``sub_goal_idx`` first.
        """
        if env_ids.numel() == 0:
            return
        idx = self.sub_goal_idx[env_ids]
        self._wrist_goal[env_ids] = self._trajectory.wrist_goals[idx]
        self._fingertip_goal_local[env_ids] = self._trajectory.fingertip_local[idx]

    # ManipTrans Stage-2 style reset_object_pose: when the trajectory has
    # IK warm-start data (`kuka_dof_pos` + `wrist_pos_ik`), place the object
    # at trajectory[seq_idx].object_pose translated by the IK residual
    # (wrist_pos_ik - dex.wrist_pos), so the relative hand-object grasp
    # configuration matches the retarget exactly even when our 7-DoF Kuka
    # arm IK has 0-6cm error.  Without IK data, fall back to parent
    # SimToolReal default (object at cfg.objectStartPose).
    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        traj = self._trajectory
        use_traj_obj_init = (
            traj.kuka_dof_pos is not None
            and traj.wrist_pos_ik is not None
            and self._cached_reset_seq_idx is not None
            and tensor_reset
            and reset_buf_idxs is None
            and len(env_ids) > 0
        )
        if not use_traj_obj_init:
            return super().reset_object_pose(env_ids, reset_buf_idxs=reset_buf_idxs,
                                              tensor_reset=tensor_reset)
        # Per-env target object pose from trajectory
        seq_idx = self._cached_reset_seq_idx           # (E,)
        obj_pose = traj.object_goals[seq_idx]          # (E, 7) xyz+xyzw
        # IK residual: how far the kuka actually puts the wrist vs paper target
        ik_delta = traj.wrist_pos_ik[seq_idx] - traj.dex_wrist_pos[seq_idx]  # (E, 3)
        obj_pos = obj_pose[:, :3] + ik_delta           # (E, 3) compensated
        obj_quat = obj_pose[:, 3:7]                    # (E, 4) keep paper quat
        # Push into root_state_tensor for object actor only.
        obj_indices = self.object_indices[env_ids]
        self.root_state_tensor[obj_indices, 0:3] = obj_pos
        self.root_state_tensor[obj_indices, 3:7] = obj_quat
        # ★ Tier 0: paper-style object velocity init (was zero before).
        # Without this, object frozen at spawn — goal advances 1 frame per
        # physics step but object has no momentum to follow → instant lag.
        # Paper dexhandmanip_sh.py:1115-1121 sets obj_velocity + obj_angular_velocity.
        if (traj.obj_velocity is not None
                and traj.obj_angular_velocity is not None):
            self.root_state_tensor[obj_indices, 7:10] = traj.obj_velocity[seq_idx]
            self.root_state_tensor[obj_indices, 10:13] = traj.obj_angular_velocity[seq_idx]
        else:
            self.root_state_tensor[obj_indices, 7:10] = 0.0   # zero linvel
            self.root_state_tensor[obj_indices, 10:13] = 0.0  # zero angvel
        # Also update object_init_state cache so any downstream readers stay
        # consistent (e.g., reach-goal logic that diff-tracks against init).
        self.object_init_state[env_ids, 0:3] = obj_pos
        self.object_init_state[env_ids, 3:7] = obj_quat
        # Table init kept at cfg.tableResetZ (already paper-aligned).
        table_indices = self.table_indices[env_ids]
        self.root_state_tensor[table_indices] = self.table_init_state[env_ids].clone()
        # Defer the actor-root push to vec_task's deferred queue (parent
        # already owns this bookkeeping in reset_idx; we just need to make
        # sure both objects are in the next set_actor_root_state call).
        self.deferred_set_actor_root_state_tensor_indexed([obj_indices])
        self.deferred_set_actor_root_state_tensor_indexed([table_indices])

    def _reset_target(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        tensor_reset: bool = True,
        is_first_goal: bool = True,
    ) -> None:
        """Set ``goal_states[env, 0:7]`` from the current sub_goal_idx frame.

        Both call sites just COPY — they do NOT advance sub_goal_idx:
          • ``is_first_goal=True`` (called from reset_idx after our reset_idx
            override has already set sub_goal_idx[env] = seq_idx).
          • ``is_first_goal=False`` (called from pre_physics_step on success
            envs).  By that point compute_kuka_reward has ALREADY advanced
            sub_goal_idx for those envs (so the policy's previous obs already
            saw the new goal).  Re-advancing here would skip a frame.
        """
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            idx = self.sub_goal_idx[env_ids]
            self.goal_states[env_ids, 0:7] = self.trajectory_states[idx, 0:7]
            # Compensate the success-target object position by the kuka IK
            # residual: when our IK can't put the wrist exactly at the paper
            # opt_wrist_pos, the dex-carried object will land 0-6cm offset
            # from paper_obj_pose.  Shift the goal so a perfectly-tracking
            # policy gets d_obj=0 (otherwise succeeding requires a 7.5cm
            # window minus up to 6cm IK error = 1.5cm margin → reset/success
            # noise dominates).  Only applies when trajectory has IK data.
            traj = self._trajectory
            if traj.wrist_pos_ik is not None and traj.dex_wrist_pos is not None:
                ik_delta = traj.wrist_pos_ik[idx] - traj.dex_wrist_pos[idx]  # (E, 3)
                self.goal_states[env_ids, 0:3] = self.goal_states[env_ids, 0:3] + ik_delta
            # NOTE: we intentionally do NOT call _clip_goal_z() here.  That
            # method clamps z to a table-surface minimum, which would corrupt
            # the exact frame-aligned object pose the user explicitly asked
            # for.  The trajectory was preprocessed offline (mujoco2gym +
            # table_z transform); z is already valid above-table by
            # construction.
            self._set_hand_goal_from_trajectory(env_ids)

    # ------------------------------------------------------------------
    # Reset (ManipTrans Stage-2 style: align sub_goal_idx to seq_idx and
    # warm-start hand DOFs from opt_dof_pos[seq_idx])
    # ------------------------------------------------------------------

    def reset_idx(
        self,
        env_ids,
        reset_buf_idxs=None,
        episode_reset: bool = True,
        tensor_reset: bool = True,
    ) -> None:
        # Step 1: pick seq_idx for each reset env BEFORE super(), so super's
        # reset_target_pose (which we override) sees the right sub_goal_idx.
        if (len(env_ids) > 0 and tensor_reset and episode_reset
                and reset_buf_idxs is None):
            T = self._trajectory.num_goals
            if self.random_state_init and T > 1:
                # Sample only from frames where the kuka can land the wrist
                # within ~1cm of the trajectory's target (workspace boundary
                # frames drop the wrist below the table → finger penetration).
                # Frames outside the reachable set are still visited via
                # natural sub_goal advancement after success.
                reachable = self._trajectory.reachable_frames
                if reachable is not None and bool(reachable.any()):
                    choices = torch.nonzero(reachable, as_tuple=False).squeeze(-1)
                    if self.random_init_max_idx >= 0:
                        choices = choices[choices <= self.random_init_max_idx]
                    pick = torch.randint(
                        0, choices.numel(), (len(env_ids),),
                        device=self.device, dtype=torch.long,
                    )
                    seq_idx = choices[pick].long()
                else:
                    seq_idx = torch.randint(
                        0, T, (len(env_ids),), device=self.device, dtype=torch.long
                    )
            else:
                seq_idx = torch.zeros(
                    len(env_ids), device=self.device, dtype=torch.long
                )
            # ★ TRACKING BRANCH: warm-start state to trajectory[seq_idx],
            # set sub_goal_idx = seq_idx (matches env state).  Time-driven
            # advance in compute_kuka_reward will increment it next step,
            # forcing policy to move object forward by per-frame motion.
            # Mirrors paper's `progress_buf[env_ids] = seq_idx` at reset
            # (dexhandimitator.py:831, dexhandmanip_sh.py:1145).
            self.sub_goal_idx[env_ids] = seq_idx
            self._cached_reset_seq_idx = seq_idx     # consumed below
        else:
            self._cached_reset_seq_idx = None

        # Step 2: parent runs the standard reset (object pose, randomized DOFs,
        # zeros progress_buf/successes, then calls reset_target_pose with
        # is_first_goal=True → our override copies trajectory[sub_goal_idx]).
        super().reset_idx(
            env_ids,
            reset_buf_idxs=reset_buf_idxs,
            episode_reset=episode_reset,
            tensor_reset=tensor_reset,
        )

        # ★ KNOWN GAP (deferred to V2): ManipTrans Stage-2
        # (dexhandmanip_sh.py:1093-1100) sets the dexhand's floating-base
        # ROOT to opt_wrist_pos[seq_idx] / opt_wrist_rot[seq_idx] so the
        # hand starts gripping the object.  We have a Kuka arm — the
        # dexhand's "wrist" is iiwa14_link_7 — so directly setting the
        # wrist pose requires solving 7-DoF arm IK to opt_wrist_pos.  We
        # don't currently do this: the arm is left at SimToolReal's
        # randomized default DOFs from super().reset_idx, which puts the
        # wrist FAR from the trajectory's expected wrist pose.  Effects:
        #   • The hand's pre-grasp shape (from opt_dof_pos) is correct
        #     but at the wrong location → it can't grip the object.
        #   • The object falls under gravity from the trajectory pose.
        #   • RL must learn to drive the arm to within reach.
        # Mitigation in flight: object_max_velocity caps + ManipTrans
        # density/friction prevent PhysX explosion when object falls.
        # Step 3: ManipTrans-style hand-DOF warm-start.
        if (self.use_retarget_dof_init
                and self._cached_reset_seq_idx is not None
                and len(env_ids) > 0
                and tensor_reset and reset_buf_idxs is None):
            seq_idx = self._cached_reset_seq_idx
            n_arm = self.num_arm_dofs                                # 7
            n_hand_traj = self._trajectory.dex_dof_pos.shape[1]      # 22 (Sharpa)
            n_hand_env  = self.num_hand_dofs                         # also 22 expected
            if n_hand_traj != n_hand_env:
                raise ValueError(
                    f"[VideoRLFollower] dex_dof_pos has {n_hand_traj} DoF but "
                    f"the env's Sharpa hand has {n_hand_env}.  Silent truncation "
                    "would break the warm-start — fix the retarget pkl or env."
                )
            hand_dof = self._trajectory.dex_dof_pos[seq_idx]         # (E, n_hand)
            # Optional relaxation: scale toward 0 (open neutral) to give
            # finger clearance from the IK-shifted object — see field doc.
            if self.dof_init_relax_scale != 1.0:
                hand_dof = hand_dof * self.dof_init_relax_scale
            # Clamp to dexhand DOF limits (paper safety).
            lo = self.arm_hand_dof_lower_limits[n_arm:n_arm + n_hand_env]
            hi = self.arm_hand_dof_upper_limits[n_arm:n_arm + n_hand_env]
            hand_dof = torch.clamp(hand_dof, lo, hi)
            self.arm_hand_dof_pos[env_ids, n_arm:n_arm + n_hand_env] = hand_dof
            # ★ Tier 0: paper-style dex DOF velocity init (was zero before).
            # Matches dexhandmanip_sh.py:1086-1091 — clamped to joint speed limits.
            if self._trajectory.dex_dof_velocity is not None:
                hand_dof_vel = self._trajectory.dex_dof_velocity[seq_idx]
                # Speed-limit clamp (paper safety): use ±200 rad/s as proxy
                # (Sharpa joint speed limit; conservative).
                hand_dof_vel = torch.clamp(hand_dof_vel, -200.0, 200.0)
                self.arm_hand_dof_vel[env_ids, n_arm:n_arm + n_hand_env] = hand_dof_vel
            else:
                self.arm_hand_dof_vel[env_ids, n_arm:n_arm + n_hand_env] = 0.0
            self.prev_targets[env_ids, n_arm:n_arm + n_hand_env] = hand_dof
            self.cur_targets[env_ids, n_arm:n_arm + n_hand_env] = hand_dof

            # Kuka 7-DoF arm warm-start from offline IK (ManipTrans Stage-2's
            # floating-base equivalent for our arm-mounted robot).  Without
            # this the wrist starts ~50cm from the trajectory's expected
            # opt_wrist_pos and the policy plateaus on the slide-then-grab
            # phase (sub_goal_idx ≤ 6 of 14) because it can't drive the arm
            # close enough to follow the lift.
            if self._trajectory.kuka_dof_pos is not None:
                kuka_dof = self._trajectory.kuka_dof_pos[seq_idx]   # (E, 7)
                lo_arm = self.arm_hand_dof_lower_limits[:n_arm]
                hi_arm = self.arm_hand_dof_upper_limits[:n_arm]
                kuka_dof = torch.clamp(kuka_dof, lo_arm, hi_arm)
                self.arm_hand_dof_pos[env_ids, :n_arm] = kuka_dof
                # ★ Tier 0: paper-style kuka DOF velocity init.
                if self._trajectory.kuka_dof_velocity is not None:
                    kuka_dof_vel = self._trajectory.kuka_dof_velocity[seq_idx]
                    kuka_dof_vel = torch.clamp(kuka_dof_vel, -100.0, 100.0)
                    self.arm_hand_dof_vel[env_ids, :n_arm] = kuka_dof_vel
                else:
                    self.arm_hand_dof_vel[env_ids, :n_arm] = 0.0
                self.prev_targets[env_ids, :n_arm] = kuka_dof
                self.cur_targets[env_ids, :n_arm] = kuka_dof
            # NOTE: do NOT re-push set_dof_position_target_tensor_indexed
            # here.  Earlier we did (worried about super's already-pushed
            # random targets fighting our warm-start state), but the policy's
            # action loop overwrites position targets every sim step anyway,
            # and the duplicate call may race with super's deferred
            # set_dof_state_tensor_indexed leading to PhysX state corruption
            # during mass episode-reset (step ~episodeLength).
        # Don't leak the cache into next call.
        self._cached_reset_seq_idx = None

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def _current_fingertips_in_wrist_local(self) -> Tensor:
        """Return current fingertip positions expressed in the palm-center
        rotated frame, which is the SAME convention the env uses elsewhere
        (`fingertip_pos_rel_palm` is computed against ``palm_center_pos``,
        not against the wrist link).

        IMPORTANT alignment requirement: the trajectory file's
        ``fingertip_local`` field MUST be expressed in this same frame.  In
        practice that means the trajectory pipeline should compute the local
        offsets relative to the **palm-center proxy** (e.g. middle MCP joint)
        rather than the MANO wrist joint.  The trajectory exporter currently
        approximates the wrist origin via ``rhand_transl``; this introduces a
        small constant translation bias relative to palm-center.  Document
        this as a known limitation; it may be tightened later by fitting a
        wrist→palm-center offset on real Sharpa hand data.
        """
        offsets_world = self.fingertip_pos_rel_palm                 # (N, K, 3)
        palm_quat = self._palm_state[:, 3:7]                        # (N, 4)
        inv_q = _quat_inverse_xyzw(palm_quat).unsqueeze(1)          # (N, 1, 4)
        K = offsets_world.shape[1]
        return _quat_rotate_xyzw(inv_q.expand(-1, K, -1), offsets_world)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _maniptrans_imit_reward(self) -> Tensor:
        """ManipTrans Stage-1 13-term imitation reward (paper coefficients).

        Computes per-Sharpa-body-link distance to MANO-target (expanded via
        the to_hand mapping), then groups + exp-decays exactly as in
        ``ManipTrans/maniptrans_envs/lib/envs/tasks/dexhandimitator.py:1066-1172``.

        Velocity terms (eef vel/ang_vel, joint vel) and power terms in the
        paper require target velocities + dof_force tensors that this env
        does not currently expose; we emit zero contributions for those terms.
        The 9 retained position/orientation terms sum to coefficients = 5.15;
        the 5 missing terms (vel/ang_vel/joint_vel/power/wrist_power) sum to
        coefficients = 1.25 — i.e. the patch covers 5.15 / 6.40 ≈ **80.5%** of
        the paper's full coefficient sum.  Document the gap rather than
        silently dropping it.
        """
        if not self._imit_active:
            return torch.zeros(self.num_envs, device=self.device)

        # ── Current Sharpa body link world positions (N, 28, 3) ──
        rb_t = self.rigid_body_states                 # (N, num_bodies, 13)
        actual_28 = rb_t[:, self._imit_link_handles_t, :3]       # (N, 28, 3)
        actual_27 = actual_28[:, 1:, :]                           # skip wrist (handled by R_eef)

        # ── MANO target per non-wrist Sharpa link (N, 27, 3) via to_hand ──
        idx = self.sub_goal_idx                                   # (N,)
        joints_world = self._trajectory.joints_world[idx]         # (N, 21, 3)
        target_27 = joints_world[:, self._imit_target_mano_idx, :]  # (N, 27, 3)

        # ── Per-link distance ──
        diff_27 = (actual_27 - target_27).norm(dim=-1)            # (N, 27)

        # ── Per-group mean distance + exp-decay ──
        d_thumb_tip  = diff_27[:, self._imit_grp_thumb_tip].mean(dim=-1)
        d_index_tip  = diff_27[:, self._imit_grp_index_tip].mean(dim=-1)
        d_middle_tip = diff_27[:, self._imit_grp_middle_tip].mean(dim=-1)
        d_ring_tip   = diff_27[:, self._imit_grp_ring_tip].mean(dim=-1)
        d_pinky_tip  = diff_27[:, self._imit_grp_pinky_tip].mean(dim=-1)
        d_level_1    = diff_27[:, self._imit_grp_level_1].mean(dim=-1)
        d_level_2    = diff_27[:, self._imit_grp_level_2].mean(dim=-1)

        r_thumb_tip  = torch.exp(-self.lambda_imit_thumb_tip  * d_thumb_tip)
        r_index_tip  = torch.exp(-self.lambda_imit_index_tip  * d_index_tip)
        r_middle_tip = torch.exp(-self.lambda_imit_middle_tip * d_middle_tip)
        r_ring_tip   = torch.exp(-self.lambda_imit_ring_tip   * d_ring_tip)
        r_pinky_tip  = torch.exp(-self.lambda_imit_pinky_tip  * d_pinky_tip)
        r_level_1    = torch.exp(-self.lambda_imit_level_1    * d_level_1)
        r_level_2    = torch.exp(-self.lambda_imit_level_2    * d_level_2)

        # ── EEF (wrist) position + rotation ──
        # Use palm_center as the wrist proxy (consistent with rest of env).
        eef_pos = self.palm_center_pos                            # (N, 3)
        eef_pos_target = self._wrist_goal[:, :3]                  # (N, 3) (MANO wrist)
        d_eef_pos = (eef_pos - eef_pos_target).norm(dim=-1)
        r_eef_pos = torch.exp(-self.lambda_imit_eef_pos * d_eef_pos)
        r_eef_pos_wide = torch.exp(-self.lambda_imit_eef_pos_wide * d_eef_pos)

        eef_quat = self._palm_state[:, 3:7]
        eef_quat_target = self._wrist_goal[:, 3:7]
        eef_rot_angle = _quat_geodesic_angle_xyzw(eef_quat, eef_quat_target)
        r_eef_rot = torch.exp(-self.lambda_imit_eef_rot * eef_rot_angle)

        # ── Object position + rotation reward (paper's DOMINANT terms) ──
        # ★ Codex review fix: target shifted by per-frame ik_delta to match
        # reset's IK-shifted object placement.  Without this, spawn-state
        # has d_obj_pos = ik_delta_norm (~1-6cm) → policy starts off-reward.
        cur_obj_pos = self.object_state[:, :3]
        target_obj_pos = self._trajectory.object_goals[idx][:, :3]
        if (self._trajectory.wrist_pos_ik is not None
                and self._trajectory.dex_wrist_pos is not None):
            ik_delta = (self._trajectory.wrist_pos_ik[idx]
                        - self._trajectory.dex_wrist_pos[idx])
            target_obj_pos = target_obj_pos + ik_delta
        diff_obj_pos_dist = (cur_obj_pos - target_obj_pos).norm(dim=-1)
        r_obj_pos = torch.exp(-80.0 * diff_obj_pos_dist)
        r_obj_pos_wide = torch.exp(-5.0 * diff_obj_pos_dist)

        cur_obj_quat = self.object_state[:, 3:7]
        target_obj_quat = self._trajectory.object_goals[idx][:, 3:7]
        diff_obj_rot_angle = _quat_geodesic_angle_xyzw(cur_obj_quat, target_obj_quat)
        r_obj_rot = torch.exp(-3.0 * diff_obj_rot_angle)

        # ── ★ Tier 0: 5 paper velocity reward terms (eef_vel, eef_ang_vel,
        # joints_vel, obj_vel, obj_ang_vel) per dexhandmanip_sh.py:1472-1504.
        # All use exp(-1 * |target - actual|.abs().mean) with paper coefficients.
        traj = self._trajectory
        if traj.obj_velocity is not None:
            cur_obj_vel = self.object_state[:, 7:10]
            target_obj_vel = traj.obj_velocity[idx]
            diff_obj_vel = (target_obj_vel - cur_obj_vel).abs().mean(dim=-1)
            r_obj_vel = torch.exp(-1.0 * diff_obj_vel)
        else:
            r_obj_vel = torch.zeros(self.num_envs, device=self.device)
        if traj.obj_angular_velocity is not None:
            cur_obj_ang_vel = self.object_state[:, 10:13]
            target_obj_ang_vel = traj.obj_angular_velocity[idx]
            diff_obj_ang_vel = (target_obj_ang_vel - cur_obj_ang_vel).abs().mean(dim=-1)
            r_obj_ang_vel = torch.exp(-1.0 * diff_obj_ang_vel)
        else:
            r_obj_ang_vel = torch.zeros(self.num_envs, device=self.device)
        if traj.dex_wrist_velocity is not None:
            cur_eef_vel = self._palm_state[:, 7:10]
            target_eef_vel = traj.dex_wrist_velocity[idx]
            diff_eef_vel = (target_eef_vel - cur_eef_vel).abs().mean(dim=-1)
            r_eef_vel = torch.exp(-1.0 * diff_eef_vel)
        else:
            r_eef_vel = torch.zeros(self.num_envs, device=self.device)
        if traj.dex_wrist_angular_velocity is not None:
            cur_eef_ang_vel = self._palm_state[:, 10:13]
            target_eef_ang_vel = traj.dex_wrist_angular_velocity[idx]
            diff_eef_ang_vel = (target_eef_ang_vel - cur_eef_ang_vel).abs().mean(dim=-1)
            r_eef_ang_vel = torch.exp(-1.0 * diff_eef_ang_vel)
        else:
            r_eef_ang_vel = torch.zeros(self.num_envs, device=self.device)
        if traj.joints_velocity is not None:
            # actual joint velocity from rigid_body_states
            rb_t = self.rigid_body_states
            actual_joint_vel = rb_t[:, self._imit_link_handles_t[1:], 7:10]  # skip wrist (28→27)
            # subset to MANO joints via to_mano mapping (use same target_27)
            target_joint_vel = traj.joints_velocity[idx][:, self._imit_target_mano_idx, :]
            diff_joints_vel = (target_joint_vel - actual_joint_vel).abs().mean(dim=-1).mean(dim=-1)
            r_joints_vel = torch.exp(-1.0 * diff_joints_vel)
        else:
            r_joints_vel = torch.zeros(self.num_envs, device=self.device)

        # ★ Tier 0: pure paper reward (removed non-paper r_obj_pos_wide and
        # r_eef_pos_wide hacks that I added earlier for arm slip; now that we
        # have velocity init + tighter rate match, those wides are no longer
        # needed and they break strict paper fidelity).
        r_imit = (
            self.w_imit_eef_pos      * r_eef_pos
          + self.w_imit_eef_rot      * r_eef_rot
          + self.w_imit_thumb_tip  * r_thumb_tip
          + self.w_imit_index_tip  * r_index_tip
          + self.w_imit_middle_tip * r_middle_tip
          + self.w_imit_ring_tip   * r_ring_tip
          + self.w_imit_pinky_tip  * r_pinky_tip
          + self.w_imit_level_1    * r_level_1
          + self.w_imit_level_2    * r_level_2
          + 5.0 * r_obj_pos                   # paper coefficient
          + 1.0 * r_obj_rot                   # paper coefficient
          # ★ Tier 0: 5 velocity terms (paper-faithful weights)
          + 0.1  * r_eef_vel                  # paper 0.1
          + 0.05 * r_eef_ang_vel              # paper 0.05
          + 0.1  * r_joints_vel               # paper 0.1
          + 0.1  * r_obj_vel                  # paper 0.1
          + 0.1  * r_obj_ang_vel              # paper 0.1
        )

        # Cache for episode logging + termination logic in compute_kuka_reward.
        self._last_imit_components = dict(
            r_eef_pos=r_eef_pos,    r_eef_rot=r_eef_rot,
            r_thumb_tip=r_thumb_tip, r_index_tip=r_index_tip,
            r_middle_tip=r_middle_tip, r_ring_tip=r_ring_tip, r_pinky_tip=r_pinky_tip,
            r_level_1=r_level_1, r_level_2=r_level_2,
            r_obj_pos=r_obj_pos, r_obj_rot=r_obj_rot,
        )
        # Cache distances for paper-style failed_execute termination.
        self._last_imit_dists = dict(
            d_obj_pos=diff_obj_pos_dist,
            d_obj_rot=diff_obj_rot_angle,
            d_thumb_tip=d_thumb_tip, d_index_tip=d_index_tip,
            d_middle_tip=d_middle_tip, d_ring_tip=d_ring_tip, d_pinky_tip=d_pinky_tip,
            d_level_1=d_level_1, d_level_2=d_level_2,
        )
        return r_imit

    def _write_diagnostic_metrics(self, is_success: Tensor, r_imit: Tensor) -> None:
        """Push per-step training/eval diagnostics into ``self.extras`` so
        rl_games surfaces them via wandb/tensorboard.  Cheap (no extra
        forward pass — reuses tensors already computed for R_imit)."""
        # Object 6D distance to current sub-goal
        cur_obj_pos  = self.object_state[:, :3]                            # (N, 3)
        cur_obj_quat = self.object_state[:, 3:7]
        idx          = self.sub_goal_idx
        goal_obj     = self._trajectory.object_goals[idx]                  # (N, 7)
        d_obj_pos = (cur_obj_pos - goal_obj[:, :3]).norm(dim=-1)
        d_obj_rot = _quat_geodesic_angle_xyzw(cur_obj_quat, goal_obj[:, 3:7])

        # Wrist (palm-center) distance to MANO wrist target
        d_wrist_pos = (self.palm_center_pos - self._wrist_goal[:, :3]).norm(dim=-1)
        d_wrist_rot = _quat_geodesic_angle_xyzw(
            self._palm_state[:, 3:7], self._wrist_goal[:, 3:7]
        )

        # Mean fingertip-to-MANO-tip distance (Sharpa thumb/index/.../pinky)
        if self._imit_active:
            rb_t = self.rigid_body_states                 # (N, num_bodies, 13)
            actual_28 = rb_t[:, self._imit_link_handles_t, :3]             # (N, 28, 3)
            tip_idx_in_27 = torch.tensor(
                # subtract 1 because we skip wrist=body[0]
                [SHARPA_WEIGHT_IDX[k][0] - 1 for k in
                 ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")],
                device=self.device, dtype=torch.long,
            )
            sharpa_tips = actual_28[:, 1:][:, tip_idx_in_27, :]            # (N, 5, 3)
            joints_world = self._trajectory.joints_world[idx]              # (N, 21, 3)
            mano_tips = joints_world[:, [20, 16, 17, 19, 18], :]            # thumb/index/mid/ring/pinky tip
            d_ftip = (sharpa_tips - mano_tips).norm(dim=-1).mean(dim=-1)   # (N,)
        else:
            d_ftip = torch.zeros(self.num_envs, device=self.device)

        # Sub-goal progress (mean idx + max so we know coverage)
        T = self._trajectory.num_goals
        progress_frac = self.sub_goal_idx.float() / max(T - 1, 1)

        # Stash scalars for logging — rl_games picks these up.
        self.extras.setdefault("scalars", {}).update(
            d_obj_pos_cm        = (d_obj_pos.mean() * 100.0).item(),
            d_obj_rot_deg       = (d_obj_rot.mean() * 180.0 / 3.14159265).item(),
            d_wrist_pos_cm      = (d_wrist_pos.mean() * 100.0).item(),
            d_wrist_rot_deg     = (d_wrist_rot.mean() * 180.0 / 3.14159265).item(),
            d_fingertip_cm      = (d_ftip.mean() * 100.0).item(),
            r_imit_mean         = r_imit.mean().item(),
            r_total_mean        = self.rew_buf.mean().item(),
            sub_goal_idx_mean   = self.sub_goal_idx.float().mean().item(),
            sub_goal_idx_max    = self.sub_goal_idx.max().item(),
            sub_goal_progress   = progress_frac.mean().item(),
            success_rate_step   = is_success.float().mean().item(),
            successes_per_env   = self.successes.float().mean().item(),
        )

    def _hand_pose_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute three hand-pose reward terms, each in [0, 1] before scaling.

        The "wrist" position is taken to be ``palm_center_pos`` (i.e. the same
        origin as ``fingertip_pos_rel_palm``).  The trajectory file should
        emit ``wrist_goals`` in the matching convention; see the comment in
        ``_current_fingertips_in_wrist_local``.
        """
        # 1) wrist position (palm-center convention)
        wrist_pos_curr = self.palm_center_pos
        wrist_pos_goal = self._wrist_goal[:, :3]
        d_pos = torch.norm(wrist_pos_curr - wrist_pos_goal, dim=-1)
        wrist_pos_rew = torch.exp(-self.lambda_wrist_pos * d_pos)

        # 2) wrist rotation (geodesic angle); palm body rotation is what
        #    drives the end-effector frame in SimToolReal.
        wrist_quat_curr = self._palm_state[:, 3:7]
        wrist_quat_goal = self._wrist_goal[:, 3:7]
        ang = _quat_geodesic_angle_xyzw(wrist_quat_curr, wrist_quat_goal)
        wrist_rot_rew = torch.exp(-self.lambda_wrist_rot * ang)

        # 3) fingertip positions in wrist-local frame
        ftip_local_curr = self._current_fingertips_in_wrist_local()
        K = min(ftip_local_curr.shape[1], self._fingertip_goal_local.shape[1])
        delta = ftip_local_curr[:, :K] - self._fingertip_goal_local[:, :K]
        d_ftip = torch.norm(delta, dim=-1).mean(dim=-1)               # mean over K
        fingertip_rew = torch.exp(-self.lambda_fingertip * d_ftip)

        # Cache for obs reuse so populate_obs_and_states_buffers doesn't
        # recompute the rotation.
        self._fingertip_curr_local = ftip_local_curr
        return wrist_pos_rew, wrist_rot_rew, fingertip_rew

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        """Hybrid reward:

            r_total = R_imit                                 (ManipTrans Stage-1)
                    + base_rew_buf                           (SimToolReal sparse goal
                                                              + action/vel penalties;
                                                              dense pickup terms must
                                                              be zeroed via cfg)

        Pre-condition (caller must enforce in cfg):
            liftingRewScale=0, liftingBonus=0, keypointRewScale=0,
            distanceDeltaRewScale=0   →  super's rew = action_pen + obj_vel_pen +
                                                       reach_goal_bonus only.
        """
        # 1) SimToolReal base: object-keypoint near-goal detection, success
        #    counter increment (used only for stats), and reset_goal_buf.
        rew_buf, is_success = super().compute_kuka_reward()

        # 2) Refresh hand goal cache so populate_obs_and_states_buffers sees
        #    the latest fingertip-local target for the (advanced) sub_goal_idx.
        #    super().compute_kuka_reward() set reset_goal_buf based on success;
        #    the actual sub_goal_idx advance happens in step() → _reset_target.
        #    Here we only need to ensure self._fingertip_curr_local is fresh.
        _ = self._hand_pose_reward()           # populates self._fingertip_curr_local

        # 3) ManipTrans Stage-1 R_imit on the current frame's MANO target.
        #    R_imit measures vs the goal we were CHASING this step (i.e. the
        #    sub-goal *before* any post-success advance), which gives high
        #    reward on the success step itself — that's the desired credit.
        r_imit = self._maniptrans_imit_reward()

        # 4) ★ TRACKING BRANCH: paper-style failed_execute termination.
        #    dexhandmanip_sh.py:1528-1542: bail out of an episode if any
        #    tracking distance exceeds a threshold (after an 8-step grace
        #    period).  This forces the policy to actually stay on trajectory;
        #    without it, the env keeps running far off-trajectory and the
        #    R_imit signal saturates near zero (no useful gradient).
        # ★ TRACKING BRANCH: paper-style failed_execute, knob-controlled.
        # Set cfg.env.failedExecuteEnable=False to disable (let env run full
        # episodeLength regardless of how far off-trajectory).  When enabled,
        # the per-component thresholds below are applied AFTER a configurable
        # grace period (default 8 steps, paper-equivalent).  Defaults are
        # already loosened ~2x vs paper for our IK error budget.
        if (hasattr(self, "_last_imit_dists")
                and bool(self.cfg["env"].get("failedExecuteEnable", True))):
            d = self._last_imit_dists
            grace_steps = int(self.cfg["env"].get("failedExecuteGraceSteps", 8))
            grace = (self.progress_buf >= grace_steps)
            t_obj_pos = float(self.cfg["env"].get("failedObjPos", 0.10))
            t_obj_rot_deg = float(self.cfg["env"].get("failedObjRotDeg", 90.0))
            t_finger = float(self.cfg["env"].get("failedFingerPos", 0.10))
            paper_failed = (
                (d["d_obj_pos"] > t_obj_pos)
                | (d["d_thumb_tip"]  > t_finger)
                | (d["d_index_tip"]  > t_finger)
                | (d["d_middle_tip"] > t_finger)
                | (d["d_ring_tip"]   > t_finger * 1.2)
                | (d["d_pinky_tip"]  > t_finger * 1.2)
                | (d["d_level_1"]    > t_finger * 1.2)
                | (d["d_level_2"]    > t_finger * 1.4)
                | (d["d_obj_rot"] * 180.0 / 3.14159265 > t_obj_rot_deg)
            ) & grace
            self.reset_buf = self.reset_buf | paper_failed

        # 5) ★ TRACKING BRANCH: paper-only reward path.  Codex review flagged
        #    that calling super's compute_kuka_reward injects SimToolReal
        #    success logic, sparse bonus, action penalties, and extra
        #    terminations — all non-paper.  Replace combined with PURE r_imit
        #    (matches paper's reward_execute at dexhandmanip_sh.py:1543).
        #    Action/velocity penalties from parent already accumulated in
        #    rew_buf; we discard them entirely on tracking branch.
        if bool(self.cfg["env"].get("paperOnlyReward", True)):
            combined = self.w_imit * r_imit
        else:
            combined = rew_buf + self.w_imit * r_imit
        # Likewise: override reset_buf with paper-style success | failed.
        # Paper success = progress_buf + 1 + 3 >= max_length.  We use
        # sub_goal_idx + 4 >= T (T = trajectory length) as equivalent.
        if bool(self.cfg["env"].get("paperOnlyReset", True)):
            T = self._trajectory.num_goals
            paper_succeeded = (self.sub_goal_idx + 4 >= T)
            if hasattr(self, "_last_imit_dists"):
                d = self._last_imit_dists
                grace_steps = int(self.cfg["env"].get("failedExecuteGraceSteps", 8))
                grace = (self.progress_buf >= grace_steps)
                t_obj_pos = float(self.cfg["env"].get("failedObjPos", 0.10))
                t_obj_rot_deg = float(self.cfg["env"].get("failedObjRotDeg", 90.0))
                t_finger = float(self.cfg["env"].get("failedFingerPos", 0.10))
                paper_failed = (
                    (d["d_obj_pos"] > t_obj_pos)
                    | (d["d_thumb_tip"] > t_finger)
                    | (d["d_index_tip"] > t_finger)
                    | (d["d_middle_tip"] > t_finger)
                    | (d["d_ring_tip"] > t_finger * 1.2)
                    | (d["d_pinky_tip"] > t_finger * 1.2)
                    | (d["d_level_1"] > t_finger * 1.2)
                    | (d["d_level_2"] > t_finger * 1.4)
                    | (d["d_obj_rot"] * 180.0 / 3.14159265 > t_obj_rot_deg)
                ) & grace
            else:
                paper_failed = torch.zeros_like(self.reset_buf)
            # Also keep parent's max_episode_length terminator (sanity cap).
            ep_done = (self.progress_buf >= self.max_episode_length - 1)
            self.reset_buf = (paper_succeeded | paper_failed | ep_done).long()

        # 4b) Strict-correctness check on BOTH r_imit and the combined reward.
        # A non-finite parent reward (e.g. norm-of-NaN-velocity from a sim
        # explosion) would otherwise silently poison the policy gradient.
        # Disable by cfg.env.allowNonfiniteReward=True (debug only).
        if not bool(self.cfg["env"].get("allowNonfiniteReward", False)):
            for name, t in (("R_imit", r_imit), ("R_total", combined)):
                if not torch.isfinite(t).all():
                    bad = (~torch.isfinite(t)).nonzero(as_tuple=False).squeeze(-1)
                    raise RuntimeError(
                        f"[VideoRLFollower] non-finite {name} at envs "
                        f"{bad.tolist()[:8]} "
                        f"(sub_goal_idx={self.sub_goal_idx[bad].tolist()[:8]}).  "
                        "Likely cause: corrupted trajectory frame, bad MANO "
                        "data, or sim state explosion (parent reward includes "
                        "norm-of-velocity terms).  Re-validate the JSON or "
                        "set cfg.env.allowNonfiniteReward=True if you really "
                        "want to tolerate this."
                    )

        self.rew_buf[:] = combined

        # 4c) Diagnostic metrics (logged via self.extras → rl_games → wandb).
        #     Cheap to compute; off via cfg.env.disableExtraMetrics if needed.
        if not bool(self.cfg["env"].get("disableExtraMetrics", False)):
            self._write_diagnostic_metrics(is_success, r_imit)

        # 5) ★ TRACKING BRANCH: time-driven sub_goal advancement.
        #    Every K physics steps (NOT just on success) we advance
        #    sub_goal_idx by 1, clamped at T-1.  K = subGoalAdvanceInterval
        #    matches the trajectory's recording rate vs physics rate ratio:
        #    e.g. trajectory at 3Hz + physics at 60Hz → K=20.  Without this
        #    rate-matching, goal would advance 1.6cm per physics step (96cm/s
        #    object motion required — physically impossible).
        #    Mirrors paper's `self.progress_buf += 1` in post_physics_step
        #    (dexhandmanip_sh.py:1333), but rate-corrected for our trajectory.
        T = self._trajectory.num_goals
        adv_interval = int(self.cfg["env"].get("subGoalAdvanceInterval", 20))
        # ★ Codex fix: per-env phase from progress_buf (which resets to 0
        # at episode reset and advances per step), NOT global gym frame count
        # (which would create random first-frame durations across envs).
        if adv_interval > 0:
            advance_mask = (
                (self.reset_buf == 0)
                & (self.progress_buf > 0)
                & (self.progress_buf % adv_interval == 0)
            )
            advance_envs = advance_mask.nonzero(as_tuple=False).squeeze(-1)
            if advance_envs.numel() > 0:
                self.sub_goal_idx[advance_envs] = torch.clamp(
                    self.sub_goal_idx[advance_envs] + 1, max=T - 1
                )
                # Refresh per-env hand goal cache + object goal_states (with
                # ik_delta consistency, matching reset's compensation) so
                # populate_obs and reward see the fresh frame.
                self._set_hand_goal_from_trajectory(advance_envs)
                new_idx = self.sub_goal_idx[advance_envs]
                new_goal = self.trajectory_states[new_idx, 0:7].clone()
                if (self._trajectory.wrist_pos_ik is not None
                        and self._trajectory.dex_wrist_pos is not None):
                    ik_d = (self._trajectory.wrist_pos_ik[new_idx]
                            - self._trajectory.dex_wrist_pos[new_idx])
                    new_goal[:, 0:3] = new_goal[:, 0:3] + ik_d
                self.goal_states[advance_envs, 0:7] = new_goal

        # 5) Bookkeeping.
        for key in (
            "raw_imit_rew",
            "raw_imit_eef_pos", "raw_imit_eef_rot",
            "raw_imit_thumb_tip", "raw_imit_index_tip",
            "raw_imit_middle_tip", "raw_imit_ring_tip", "raw_imit_pinky_tip",
            "raw_imit_level_1", "raw_imit_level_2",
        ):
            if key not in self.rewards_episode:
                self.rewards_episode[key] = torch.zeros_like(self.rew_buf)
        self.rewards_episode["raw_imit_rew"] += r_imit
        if self._imit_active and hasattr(self, "_last_imit_components"):
            c = self._last_imit_components
            self.rewards_episode["raw_imit_eef_pos"]    += c["r_eef_pos"]
            self.rewards_episode["raw_imit_eef_rot"]    += c["r_eef_rot"]
            self.rewards_episode["raw_imit_thumb_tip"]  += c["r_thumb_tip"]
            self.rewards_episode["raw_imit_index_tip"]  += c["r_index_tip"]
            self.rewards_episode["raw_imit_middle_tip"] += c["r_middle_tip"]
            self.rewards_episode["raw_imit_ring_tip"]   += c["r_ring_tip"]
            self.rewards_episode["raw_imit_pinky_tip"]  += c["r_pinky_tip"]
            self.rewards_episode["raw_imit_level_1"]    += c["r_level_1"]
            self.rewards_episode["raw_imit_level_2"]    += c["r_level_2"]

        return self.rew_buf, is_success

    # ------------------------------------------------------------------
    # Observation hook
    # ------------------------------------------------------------------

    def populate_obs_and_states_buffers(self) -> None:
        super().populate_obs_and_states_buffers()
        if not self.expose_hand_goal_to_policy:
            return

        # ★ Tier 0: paper-style `target` obs (K=1 lookahead) per
        # dexhandmanip_sh.py:913-1010.  Replaces the previous trivial
        # `_wrist_goal[7] + _fingertip_goal_local[K,3]` (30 dim) with a
        # ~256 dim block containing future wrist+joints+obj pose/vel/delta.
        # Falls back to legacy hand_goal if paper-target prerequisites aren't
        # in the trajectory (velocities missing).
        if (bool(self.cfg["env"].get("paperTargetObs", True))
                and self._trajectory.obj_velocity is not None):
            target_obs = self._build_paper_target_obs()
            # Critic also sees the target — paper makes it part of obs, not
            # privileged.  Asymmetric obs would expose more for critic but we
            # keep SimToolReal's actor/critic obs sharing.
            self.obs_buf = torch.cat([self.obs_buf, target_obs], dim=-1)
            self.states_buf = torch.cat([self.states_buf, target_obs], dim=-1)
            return

        # Legacy path (3Hz trajectories without velocities)
        wrist_goal = self._wrist_goal                                 # (N, 7)
        ftip_goal = self._fingertip_goal_local.reshape(self.num_envs, -1)  # (N, K*3)
        hand_goal = torch.cat([wrist_goal, ftip_goal], dim=-1)        # (N, 7+K*3)
        self.obs_buf = torch.cat([self.obs_buf, hand_goal], dim=-1)
        self.states_buf = torch.cat([self.states_buf, hand_goal], dim=-1)

    # ------------------------------------------------------------------
    # Paper-style target obs (Tier 0)
    # ------------------------------------------------------------------

    def _build_paper_target_obs(self) -> Tensor:
        """Construct paper's `target` obs block (K=1 lookahead) per
        dexhandmanip_sh.py:913-1010.

        Returns (N, dim) tensor where dim ≈ 256 for K=1.

        Layout (concat order, all flattened to (N, ·)):
          1. delta_wrist_pos       (N, K*3)  = target.wrist_pos - current.palm_pos
          2. wrist_vel             (N, K*3)  = target.wrist_vel
          3. delta_wrist_vel       (N, K*3)  = target.wrist_vel - current.palm_vel
          4. wrist_quat            (N, K*4)  = target.wrist_quat (xyzw)
          5. delta_wrist_quat      (N, K*4)  = current.palm_quat * target.wrist_quat^-1
          6. wrist_ang_vel         (N, K*3)
          7. delta_wrist_ang_vel   (N, K*3)
          8. delta_joints_pos      (N, K*21*3) MANO joints (target - current Sharpa)
          9. joints_vel            (N, K*21*3) target MANO joints vel
         10. delta_joints_vel      (N, K*21*3) target - current
         11. delta_obj_pos         (N, K*3)
         12. obj_vel               (N, K*3)
         13. delta_obj_vel         (N, K*3)
         14. obj_quat              (N, K*4)
         15. delta_obj_quat        (N, K*4)
         16. obj_ang_vel           (N, K*3)
         17. delta_obj_ang_vel     (N, K*3)
         18. obj_to_joints         (N, 21) current obj to each MANO joint

        Total for K=1: 3*7 + 3*3 + 21*9 + 4*2 + 4*2 + 21 = 246 dims.
        """
        N = self.num_envs
        K = int(self.cfg["env"].get("obsFutureLength", 1))
        T = self._trajectory.num_goals

        traj = self._trajectory
        # Future indices: sub_goal_idx + 1, +2, ..., +K, clamped to T-1.
        cur_idx = (self.sub_goal_idx + 1).clamp(max=T - 1)            # (N,)
        idxs = torch.stack(
            [torch.clamp(cur_idx + t, max=T - 1) for t in range(K)],
            dim=-1,
        )                                                              # (N, K)

        # Helper: gather along T axis for tensor (T, ...) → (N, K, ...)
        def gather_kt(data, idxs_NK):
            # data: (T, ...); idxs_NK: (N, K)
            return data[idxs_NK]                                       # (N, K, ...)

        # --- Current proprioceptive state ---
        cur_palm_pos = self.palm_center_pos                            # (N, 3)
        cur_palm_quat = self._palm_state[:, 3:7]                       # (N, 4) xyzw
        cur_palm_vel = self._palm_state[:, 7:10]                       # (N, 3)
        cur_palm_ang_vel = self._palm_state[:, 10:13]                  # (N, 3)
        cur_obj_pos = self.object_state[:, :3]                         # (N, 3)
        cur_obj_quat = self.object_state[:, 3:7]                       # (N, 4) xyzw
        cur_obj_vel = self.object_state[:, 7:10]                       # (N, 3)
        cur_obj_ang_vel = self.object_state[:, 10:13]                  # (N, 3)

        # --- Wrist (palm proxy) target block ---
        # Target wrist pose comes from trajectory.dex_wrist_pos/rot at idxs.
        # Note: traj.dex_wrist_pos is (T, 3), traj.dex_wrist_rot is (T, 3) rotvec.
        tw_pos = gather_kt(traj.dex_wrist_pos, idxs)                   # (N, K, 3)
        tw_rot_rotvec = gather_kt(traj.dex_wrist_rot, idxs)            # (N, K, 3)
        tw_quat = _rotvec_to_quat_xyzw(tw_rot_rotvec.reshape(-1, 3))   # (N*K, 4)
        tw_quat = tw_quat.reshape(N, K, 4)
        tw_vel = (
            gather_kt(traj.dex_wrist_velocity, idxs)
            if traj.dex_wrist_velocity is not None
            else torch.zeros_like(tw_pos)
        )                                                              # (N, K, 3)
        tw_ang_vel = (
            gather_kt(traj.dex_wrist_angular_velocity, idxs)
            if traj.dex_wrist_angular_velocity is not None
            else torch.zeros_like(tw_pos)
        )                                                              # (N, K, 3)

        delta_wrist_pos = (tw_pos - cur_palm_pos[:, None]).reshape(N, -1)
        delta_wrist_vel = (tw_vel - cur_palm_vel[:, None]).reshape(N, -1)
        delta_wrist_ang_vel = (tw_ang_vel - cur_palm_ang_vel[:, None]).reshape(N, -1)
        # Quat delta: cur * target.conj
        # (matching paper line 946-949 ordering)
        tw_quat_flat = tw_quat.reshape(N * K, 4)
        cur_quat_rep = cur_palm_quat[:, None].repeat(1, K, 1).reshape(N * K, 4)
        delta_wrist_quat = _quat_mul_xyzw(cur_quat_rep, _quat_conjugate_xyzw(tw_quat_flat))
        delta_wrist_quat = delta_wrist_quat.reshape(N, -1)

        # --- MANO joints block ---
        # Target joints (T, 21, 3); current Sharpa joints from rigid_body_states
        # via the same dex2mano mapping used in R_imit (27 non-wrist links).
        rb_t = self.rigid_body_states
        actual_28 = rb_t[:, self._imit_link_handles_t, :3]             # (N, 28, 3)
        actual_27 = actual_28[:, 1:, :]                                # skip wrist (N, 27, 3)
        if traj.joints_world is not None:
            target_joints = gather_kt(traj.joints_world, idxs)         # (N, K, 21, 3)
            target_joints_subset = target_joints[:, :, self._imit_target_mano_idx, :]  # (N, K, 27, 3)
            delta_joints_pos = (target_joints_subset - actual_27[:, None]).reshape(N, -1)
        else:
            delta_joints_pos = torch.zeros(N, K * 27 * 3, device=self.device)
        if traj.joints_velocity is not None:
            target_joints_vel = gather_kt(traj.joints_velocity, idxs)  # (N, K, 21, 3)
            target_joints_vel_subset = target_joints_vel[:, :, self._imit_target_mano_idx, :]
            joints_vel = target_joints_vel_subset.reshape(N, -1)
            # current Sharpa joint vel: rigid_body_states[..., 7:10] for the 27 links
            actual_27_vel = rb_t[:, self._imit_link_handles_t[1:], 7:10]  # (N, 27, 3)
            delta_joints_vel = (target_joints_vel_subset - actual_27_vel[:, None]).reshape(N, -1)
        else:
            joints_vel = torch.zeros(N, K * 27 * 3, device=self.device)
            delta_joints_vel = torch.zeros(N, K * 27 * 3, device=self.device)

        # --- Object block ---
        to_pos = gather_kt(traj.object_goals, idxs)[..., 0:3]          # (N, K, 3)
        to_quat = gather_kt(traj.object_goals, idxs)[..., 3:7]         # (N, K, 4) xyzw
        to_vel = (
            gather_kt(traj.obj_velocity, idxs)
            if traj.obj_velocity is not None
            else torch.zeros_like(to_pos)
        )                                                               # (N, K, 3)
        to_ang_vel = (
            gather_kt(traj.obj_angular_velocity, idxs)
            if traj.obj_angular_velocity is not None
            else torch.zeros_like(to_pos)
        )                                                               # (N, K, 3)

        delta_obj_pos = (to_pos - cur_obj_pos[:, None]).reshape(N, -1)
        delta_obj_vel = (to_vel - cur_obj_vel[:, None]).reshape(N, -1)
        delta_obj_ang_vel = (to_ang_vel - cur_obj_ang_vel[:, None]).reshape(N, -1)
        to_quat_flat = to_quat.reshape(N * K, 4)
        cur_obj_quat_rep = cur_obj_quat[:, None].repeat(1, K, 1).reshape(N * K, 4)
        delta_obj_quat = _quat_mul_xyzw(cur_obj_quat_rep, _quat_conjugate_xyzw(to_quat_flat))
        delta_obj_quat = delta_obj_quat.reshape(N, -1)

        # --- obj_to_joints (current state, not target-indexed) ---
        # Distance from current obj to each MANO joint position (21 joints in
        # actual hand frame).  Use the same 27 Sharpa link positions plus wrist.
        all_28_pos = rb_t[:, self._imit_link_handles_t, :3]            # (N, 28, 3)
        # subset to the 21 MANO equivalents
        mano_subset = all_28_pos[:, [0] + (self._imit_target_mano_idx + 1).tolist()[:20], :]  # heuristic
        if mano_subset.shape[1] < 21:
            pad = torch.zeros(N, 21 - mano_subset.shape[1], 3, device=self.device)
            mano_subset = torch.cat([mano_subset, pad], dim=1)
        else:
            mano_subset = mano_subset[:, :21, :]
        obj_to_joints = torch.norm(
            cur_obj_pos[:, None] - mano_subset, dim=-1
        )                                                               # (N, 21)

        # --- Concatenate (paper order) ---
        target_obs = torch.cat(
            [
                delta_wrist_pos,                                       # N, K*3
                tw_vel.reshape(N, -1),                                 # N, K*3
                delta_wrist_vel,                                       # N, K*3
                tw_quat.reshape(N, -1),                                # N, K*4
                delta_wrist_quat,                                      # N, K*4
                tw_ang_vel.reshape(N, -1),                             # N, K*3
                delta_wrist_ang_vel,                                   # N, K*3
                delta_joints_pos,                                      # N, K*27*3
                joints_vel,                                            # N, K*27*3
                delta_joints_vel,                                      # N, K*27*3
                delta_obj_pos,                                         # N, K*3
                to_vel.reshape(N, -1),                                 # N, K*3
                delta_obj_vel,                                         # N, K*3
                to_quat.reshape(N, -1),                                # N, K*4
                delta_obj_quat,                                        # N, K*4
                to_ang_vel.reshape(N, -1),                             # N, K*3
                delta_obj_ang_vel,                                     # N, K*3
                obj_to_joints,                                         # N, 21
            ],
            dim=-1,
        )
        return target_obs

    # ------------------------------------------------------------------
    # Properties for downstream code
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> ReferenceTrajectory:
        return self._trajectory

    @property
    def hand_goal_obs_dim(self) -> int:
        """Dimension of the extra goal-target block appended to obs.

        - Legacy: 7 (wrist pose xyz+xyzw) + 3K (fingertip-local), K=5 → 22.
        - Paper target (Tier 0): K_future × (23 wrist + 243 joints + 23 obj) + 21
          (obj_to_joints).  For K_future=1 → 310 dims.
        """
        if bool(self.cfg["env"].get("paperTargetObs", True)):
            Kf = int(self.cfg["env"].get("obsFutureLength", 1))
            # Wrist deltas: K*(3+3+3+4+4+3+3) = K*23
            # Joints deltas: K*(27*3 + 27*3 + 27*3) = K*243
            # Obj deltas:    K*(3+3+3+4+4+3+3) = K*23
            # Plus obj_to_joints (current state, not lookahead-indexed): 21
            return Kf * (23 + 243 + 23) + 21
        return 7 + 3 * self._traj_K

    @property
    def total_num_goals(self) -> int:
        return self._trajectory.num_goals
