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

Sub-goal advancement:

  on success (keypoints_max_dist <= tolerance for N consecutive sim steps):
    sub_goal_idx[env]   = (sub_goal_idx[env] + 1) % T      # ★ next frame
    successes[env]     += 1                                  # stats only

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

        # ----- Force fixed-goal mode -----
        # The base class only ever reads trajectory_states[..., 0:7] for the
        # object goal; we override the tensor below with a richer one.
        cfg["env"]["useFixedGoalStates"] = True
        cfg["env"]["fixedGoalStates"] = [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        ]
        cfg["env"]["maxConsecutiveSuccesses"] = self._trajectory.num_goals

        # ----- Trajectory-driven object start pose (always honoured) -----
        # The trajectory's first object pose is the intended initial state of
        # the manipulated object, irrespective of whether we load its URDF or
        # fall back to procedural primitives.  Set this BEFORE super().__init__
        # so the base ``_object_start_pose`` honours it.
        cfg["env"]["objectStartPose"] = list(
            self._trajectory.object_start_pose.tolist()
        )

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
        """Override SimToolReal's loader so the user can flip
        ``objectDisableGravity`` / damping per-task without touching code.

        Defaults match ManipTrans's _create_obj_assets
        (dexhandmanip_sh.py:572): ``fix_base_link=False``, gravity ENABLED,
        no extra damping.  The paper's "G=0 → restore" relaxation is
        sim-level (gym.set_sim_params) and is NOT applied here.
        """
        from isaacgym import gymapi
        import os as _os

        disable_g  = bool(self.cfg["env"].get("objectDisableGravity", False))
        ang_damp   = float(self.cfg["env"].get("objectAngularDamping", 0.0))
        lin_damp   = float(self.cfg["env"].get("objectLinearDamping", 0.0))

        object_assets = []
        for object_asset_file, need_vhacd in zip(
            self.object_asset_files, self.object_need_vhacds
        ):
            opts = gymapi.AssetOptions()
            opts.vhacd_enabled = need_vhacd
            opts.collapse_fixed_joints = True
            opts.replace_cylinder_with_capsule = True
            opts.disable_gravity = disable_g
            opts.angular_damping = ang_damp
            opts.linear_damping = lin_damp

            asset = self.gym.load_asset(
                self.sim,
                _os.path.dirname(object_asset_file),
                _os.path.basename(object_asset_file),
                opts,
            )
            object_assets.append(asset)

        rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])
        sh_count = self.gym.get_asset_rigid_shape_count(object_assets[0])
        print(f"[VideoRLFollower] object asset loaded with "
              f"disable_gravity={disable_g}, lin_damp={lin_damp}, "
              f"ang_damp={ang_damp}")
        return object_assets, rb_count, sh_count

    # ------------------------------------------------------------------
    # Trajectory-aware goal handling
    # ------------------------------------------------------------------

    def _setup_imit_link_indices(self) -> None:
        """Look up Sharpa link rigid-body indices in the SimToolReal robot.

        SimToolReal builds ``self.rigid_body_name_to_idx["robot/<name>"]``
        during actor creation (simtoolreal/env.py:2014-2019), giving the
        env-domain rigid-body slot for every Sharpa link by name.  We use that
        directly — no actor handles needed.

        We REQUIRE all 28 Sharpa body_names to exist in the URDF (otherwise
        the to_hand mapping is broken for the missing link and the resulting
        distance term would be biased).  Raises a clear error rather than
        silently substituting the wrist as a proxy.
        """
        idx_map = self.rigid_body_name_to_idx          # built by SimToolReal
        handles = []
        missing = []
        for name in SHARPA_BODY_NAMES:
            key = "robot/" + name
            if key not in idx_map:
                missing.append(name)
                handles.append(-1)
            else:
                handles.append(int(idx_map[key]))
        if missing:
            raise RuntimeError(
                "[VideoRLFollower] R_imit setup failed: the loaded robot "
                f"URDF is missing {len(missing)} Sharpa body links required "
                f"by the dex2hand mapping: {missing}.  Either fix the URDF "
                "or set imitRewardScale.enable=0 to disable R_imit."
            )
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
                seq_idx = torch.randint(
                    0, T, (len(env_ids),), device=self.device, dtype=torch.long
                )
            else:
                seq_idx = torch.zeros(
                    len(env_ids), device=self.device, dtype=torch.long
                )
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
            # Clamp to dexhand DOF limits (paper safety).
            lo = self.arm_hand_dof_lower_limits[n_arm:n_arm + n_hand_env]
            hi = self.arm_hand_dof_upper_limits[n_arm:n_arm + n_hand_env]
            hand_dof = torch.clamp(hand_dof, lo, hi)
            self.arm_hand_dof_pos[env_ids, n_arm:n_arm + n_hand_env] = hand_dof
            self.arm_hand_dof_vel[env_ids, n_arm:n_arm + n_hand_env] = 0.0
            self.prev_targets[env_ids, n_arm:n_arm + n_hand_env] = hand_dof
            self.cur_targets[env_ids, n_arm:n_arm + n_hand_env] = hand_dof
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
        rb_t = self._rigid_body_state_tensor.view(self.num_envs, -1, 13)
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

        eef_quat = self._palm_state[:, 3:7]
        eef_quat_target = self._wrist_goal[:, 3:7]
        eef_rot_angle = _quat_geodesic_angle_xyzw(eef_quat, eef_quat_target)
        r_eef_rot = torch.exp(-self.lambda_imit_eef_rot * eef_rot_angle)

        r_imit = (
            self.w_imit_eef_pos    * r_eef_pos
          + self.w_imit_eef_rot    * r_eef_rot
          + self.w_imit_thumb_tip  * r_thumb_tip
          + self.w_imit_index_tip  * r_index_tip
          + self.w_imit_middle_tip * r_middle_tip
          + self.w_imit_ring_tip   * r_ring_tip
          + self.w_imit_pinky_tip  * r_pinky_tip
          + self.w_imit_level_1    * r_level_1
          + self.w_imit_level_2    * r_level_2
        )

        # Cache for episode logging.
        self._last_imit_components = dict(
            r_eef_pos=r_eef_pos,    r_eef_rot=r_eef_rot,
            r_thumb_tip=r_thumb_tip, r_index_tip=r_index_tip,
            r_middle_tip=r_middle_tip, r_ring_tip=r_ring_tip, r_pinky_tip=r_pinky_tip,
            r_level_1=r_level_1, r_level_2=r_level_2,
        )
        return r_imit

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

        # 4) Combine.
        combined = rew_buf + self.w_imit * r_imit

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

        # 5) ★ Phase fix (Codex round 2): advance sub_goal_idx for SUCCESS envs
        #    NOW so populate_obs_and_states_buffers (which runs immediately
        #    after this in post_physics_step) packs the FRESH goal into obs.
        #    Without this the policy would see the stale (just-achieved) goal
        #    for one full control step.  _reset_target (called next tick from
        #    pre_physics_step) is now a pure copy and will not double-advance.
        success_env_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if success_env_ids.numel() > 0:
            T = self._trajectory.num_goals
            self.sub_goal_idx[success_env_ids] = (
                self.sub_goal_idx[success_env_ids] + 1
            ) % T
            # Refresh per-env hand goal cache so populate_obs sees the new
            # frame's wrist + fingertip-local targets.
            self._set_hand_goal_from_trajectory(success_env_ids)
            # Also refresh object goal_states so any consumer (e.g. logging)
            # sees the fresh frame.
            new_idx = self.sub_goal_idx[success_env_ids]
            self.goal_states[success_env_ids, 0:7] = self.trajectory_states[
                new_idx, 0:7
            ]

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

        wrist_goal = self._wrist_goal                                 # (N, 7)
        ftip_goal = self._fingertip_goal_local.reshape(self.num_envs, -1)  # (N, K*3)
        hand_goal = torch.cat([wrist_goal, ftip_goal], dim=-1)        # (N, 7+K*3)

        # Concatenate to obs_buf and states_buf.  The space sizes were already
        # bumped in __init__, so rl_games's network heads agree with these
        # shapes.
        self.obs_buf = torch.cat([self.obs_buf, hand_goal], dim=-1)
        self.states_buf = torch.cat([self.states_buf, hand_goal], dim=-1)

    # ------------------------------------------------------------------
    # Properties for downstream code
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> ReferenceTrajectory:
        return self._trajectory

    @property
    def hand_goal_obs_dim(self) -> int:
        return 7 + 3 * self._traj_K

    @property
    def total_num_goals(self) -> int:
        return self._trajectory.num_goals
