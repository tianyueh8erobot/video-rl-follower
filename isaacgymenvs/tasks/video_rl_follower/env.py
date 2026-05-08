"""VideoRLFollower env.

A SimToolReal-based environment that tracks a *single* reference trajectory
(object 6D pose + hand wrist 6D pose + fingertip keypoints) loaded from a JSON
trajectory file produced by ``tools/process_maniptrans_trajectory.py``.

Differences from the base ``SimToolReal``:

* ``useFixedGoalStates`` is forced to True; the ``trajectory_states`` buffer is
  filled from the loaded reference instead of from cfg's ``fixedGoalStates``.
* The reward adds two hand-pose terms (wrist-pose + fingertip-position) on top
  of the original object-keypoint reward, with configurable weights.
* The observation/state buffer is extended with the next reference wrist pose
  and fingertip positions in the wrist-local frame, matching the structure
  SimToolReal uses for object goals.
* Object URDF/scale come from the trajectory file (procedural primitive
  generation is bypassed when ``trajectory.object.urdf_path`` is set).

Single-hand only.  Bimanual support is out of scope for this codebase fork.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor

from isaacgymenvs.tasks.simtoolreal.env import SimToolReal

from .trajectory import ReferenceTrajectory


def _quat_inverse_xyzw(q: Tensor) -> Tensor:
    # Conjugate of a unit quaternion; xyzw layout matches SimToolReal's keypoints
    out = q.clone()
    out[..., :3] *= -1.0
    return out


def _quat_mul_xyzw(a: Tensor, b: Tensor) -> Tensor:
    # Hamilton product, both inputs xyzw
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([x, y, z, w], dim=-1)


def _quat_geodesic_angle_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Returns the angle (radians) between two unit quaternions."""
    dot = (a * b).sum(-1).abs().clamp(-1.0, 1.0)
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
    ``env.exposeHandGoalToPolicy``             bool  add hand goal to obs (default True)
    """

    def __init__(self, cfg, *args, **kwargs):
        # Force fixed-goal mode; we do not want the base class to sample random
        # delta goals.
        cfg["env"]["useFixedGoalStates"] = True
        # Provide a single dummy fixed goal so base init doesn't crash trying to
        # parse fixedGoalStates; we will overwrite trajectory_states below.
        cfg["env"]["fixedGoalStates"] = [
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        ]

        traj_path = cfg["env"].get("trajectoryPath", None)
        if traj_path is None:
            raise ValueError(
                "VideoRLFollower requires env.trajectoryPath to point at a "
                "JSON trajectory produced by tools/process_maniptrans_trajectory.py"
            )

        # Resolve relative paths against the project root rather than the cwd.
        if not os.path.isabs(traj_path):
            project_root = Path(__file__).resolve().parents[3]
            traj_path = str(project_root / traj_path)
        self._trajectory_path = traj_path

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

        # Load trajectory on CPU first; will move to device after super().__init__
        self._trajectory = ReferenceTrajectory.from_file(self._trajectory_path)

        # Tell base class how big trajectory_states will be so that it can
        # allocate the right tensor.  SimToolReal currently only consumes
        # columns [0:7] (object pose); the additional columns are stored but
        # ignored by the base reward path.
        cfg["env"]["maxConsecutiveSuccesses"] = self._trajectory.num_goals

        super().__init__(cfg, *args, **kwargs)

        # Move trajectory to env device and pre-build the dense state tensor.
        self._trajectory.to(self.device)
        dense = self._trajectory.stacked_goals()
        # SimToolReal stores trajectory_states as (K, _, _) of dtype float32;
        # see _reset_target.  We replace it here with our richer tensor.
        self.trajectory_states = dense                                 # (T, D)
        self.max_consecutive_successes = self._trajectory.num_goals
        self._traj_K = self._trajectory.num_fingertips

        # Pre-allocate per-env hand goal buffers (overwritten on each goal reset)
        self._wrist_goal = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )
        self._fingertip_goal_local = torch.zeros(
            (self.num_envs, self._traj_K, 3), device=self.device, dtype=torch.float32
        )

        # Initialise goal index 0 for every env (base class also calls
        # _reset_target; call again here is safe and idempotent).
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._set_hand_goal_from_trajectory(env_ids)

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
        # Update our hand goals in lockstep.
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            self._set_hand_goal_from_trajectory(env_ids)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _hand_pose_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute three hand-pose reward terms, each in [0, 1] before scaling.

        Returns ``(wrist_pos_rew, wrist_rot_rew, fingertip_rew)`` shape (N,).
        """
        # 1) wrist position
        wrist_pos_curr = self._palm_state[:, :3]
        wrist_pos_goal = self._wrist_goal[:, :3]
        d_pos = torch.norm(wrist_pos_curr - wrist_pos_goal, dim=-1)
        wrist_pos_rew = torch.exp(-self.lambda_wrist_pos * d_pos)

        # 2) wrist rotation (geodesic angle)
        wrist_quat_curr = self._palm_state[:, 3:7]
        wrist_quat_goal = self._wrist_goal[:, 3:7]
        ang = _quat_geodesic_angle_xyzw(wrist_quat_curr, wrist_quat_goal)
        wrist_rot_rew = torch.exp(-self.lambda_wrist_rot * ang)

        # 3) fingertip positions in wrist-local frame
        # (a) current fingertip positions in world (already on env)
        # SimToolReal exposes ``self.fingertip_pos_rel_palm`` that we can compare
        # against ``self._fingertip_goal_local`` directly.
        ftip_local_curr = self.fingertip_pos_rel_palm                 # (N, K, 3)
        # If the env was configured with a different number of fingertips than
        # the trajectory, truncate to the smaller one defensively.
        K = min(ftip_local_curr.shape[1], self._fingertip_goal_local.shape[1])
        delta = ftip_local_curr[:, :K] - self._fingertip_goal_local[:, :K]
        d_ftip = torch.norm(delta, dim=-1).mean(dim=-1)               # mean over K
        fingertip_rew = torch.exp(-self.lambda_fingertip * d_ftip)

        return wrist_pos_rew, wrist_rot_rew, fingertip_rew

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        # Base class fills self.rew_buf with the object-goal reward (and
        # writes to self.reset_buf, success counters, etc.).  We add the hand
        # terms on top.
        super().compute_kuka_reward()

        wrist_pos_rew, wrist_rot_rew, fingertip_rew = self._hand_pose_reward()
        hand_extra = (
            self.w_wrist_pos * wrist_pos_rew
            + self.w_wrist_rot * wrist_rot_rew
            + self.w_fingertip * fingertip_rew
        )
        self.rew_buf[:] = self.rew_buf + hand_extra

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

        return self.rew_buf, self.reset_buf

    # ------------------------------------------------------------------
    # Observation hook
    # ------------------------------------------------------------------

    def populate_obs_and_states_buffers(self) -> None:
        super().populate_obs_and_states_buffers()
        if not self.expose_hand_goal_to_policy:
            return

        # Compose hand-goal observation: wrist goal (7) + fingertip goal (K*3)
        wrist_goal = self._wrist_goal                                 # (N, 7)
        ftip_goal = self._fingertip_goal_local.reshape(self.num_envs, -1)  # (N, K*3)

        hand_goal_obs = torch.cat([wrist_goal, ftip_goal], dim=-1)
        # Append to obs_buf and states_buf so policy and critic both see it.
        self.obs_buf = torch.cat([self.obs_buf, hand_goal_obs], dim=-1)
        self.states_buf = torch.cat([self.states_buf, hand_goal_obs], dim=-1)

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
