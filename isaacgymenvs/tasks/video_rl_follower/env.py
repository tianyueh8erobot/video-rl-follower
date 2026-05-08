"""VideoRLFollower env.

A SimToolReal-based environment that tracks a *single* reference trajectory
(object 6D pose + hand wrist 6D pose + fingertip keypoints) loaded from a JSON
trajectory file produced by ``tools/process_maniptrans_trajectory.py``.

Differences from the base ``SimToolReal``:

* ``useFixedGoalStates`` is forced to True; the ``trajectory_states`` buffer is
  filled from the loaded reference instead of from cfg's ``fixedGoalStates``.
* The reward adds two hand-pose terms (wrist-pose + fingertip-position) on top
  of the original object-keypoint reward, with configurable weights.
* The observation/state buffers are extended with two tail blocks
  ``(wrist_goal, fingertip_goal_local)`` after the base class assembles
  ``obs_buf``/``states_buf``.  We RESIZE the gym observation_space, state_space
  and the obs delay queue to match, so that rl_games observes the correct
  shapes and the action-latency simulation continues to work.

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

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._set_hand_goal_from_trajectory(env_ids)

        if self.expose_hand_goal_to_policy:
            self._extend_obs_state_spaces()

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
    # Trajectory-aware goal handling
    # ------------------------------------------------------------------

    def _set_hand_goal_from_trajectory(self, env_ids: Tensor) -> None:
        if env_ids.numel() == 0:
            return
        T = self._trajectory.num_goals
        idx = (self.successes[env_ids].long() % T)
        self._wrist_goal[env_ids] = self._trajectory.wrist_goals[idx]
        self._fingertip_goal_local[env_ids] = self._trajectory.fingertip_local[idx]

    def _reset_target(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        tensor_reset: bool = True,
        is_first_goal: bool = True,
    ) -> None:
        # Delegate the object-goal half (column 0:7) to the base class —
        # because we forced useFixedGoalStates=True, the base writes
        # trajectory_states[idx, 0:7] into goal_states[..., 0:7].
        super()._reset_target(
            env_ids=env_ids,
            reset_buf_idxs=reset_buf_idxs,
            tensor_reset=tensor_reset,
            is_first_goal=is_first_goal,
        )
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            self._set_hand_goal_from_trajectory(env_ids)

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
        # Base class returns (rew_buf, is_success).  Capture both — we rely on
        # is_success for downstream stats and resets rather than self.reset_buf.
        rew_buf, is_success = super().compute_kuka_reward()

        wrist_pos_rew, wrist_rot_rew, fingertip_rew = self._hand_pose_reward()
        hand_extra = (
            self.w_wrist_pos * wrist_pos_rew
            + self.w_wrist_rot * wrist_rot_rew
            + self.w_fingertip * fingertip_rew
        )
        self.rew_buf[:] = rew_buf + hand_extra

        # Bookkeeping for logs
        if "raw_wrist_pos_rew" not in self.rewards_episode:
            self.rewards_episode["raw_wrist_pos_rew"] = torch.zeros_like(
                self.rew_buf
            )
            self.rewards_episode["raw_wrist_rot_rew"] = torch.zeros_like(
                self.rew_buf
            )
            self.rewards_episode["raw_fingertip_rew"] = torch.zeros_like(
                self.rew_buf
            )
        self.rewards_episode["raw_wrist_pos_rew"] += wrist_pos_rew
        self.rewards_episode["raw_wrist_rot_rew"] += wrist_rot_rew
        self.rewards_episode["raw_fingertip_rew"] += fingertip_rew

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
