"""DexTrack-Sharpa: residual policy environment for tracking a fixed kinematic trajectory.

Inherits directly from `VecTask` (NOT SimToolReal) to keep the dependency surface
small — SimToolReal carries a lot of curriculum/lifting/keypoint logic we don't
need for single-trajectory imitation tracking.

Control mode: cumulative residual on the kinematic reference, identical to
DexTrack `use_kinematics_bias_wdelta=True` and ManipTrans Stage-2:

    target_dof[t] = ref.dof_pos[t+1] + action_scale * actions

Reward style: switched at cfg time via `task.env.reward_style`.
  - "maniptrans"  -> 13-term ManipTrans Stage-1 imitation reward
  - "dextrack"    -> 7-term DexTrack tracking reward

Goal observation: matching style (see obs/goal_obs.py).
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch

from isaacgymenvs.tasks.base.vec_task import VecTask

from .trajectory   import DexTrackTrajectory, SHARPA_BODY_NAMES, WRIST_LINK
from .rewards      import compute_maniptrans_reward, compute_dextrack_reward
from .obs.goal_obs import (
    build_maniptrans_goal_obs, build_dextrack_goal_obs, get_goal_obs_dim,
    FUTURE_FRAMES_MANIPTRANS, FUTURE_FRAMES_DEXTRACK,
)


NUM_ARM_DOFS  = 7              # Franka panda joints
NUM_HAND_DOFS = 22             # Sharpa right hand
NUM_DOFS      = NUM_ARM_DOFS + NUM_HAND_DOFS                  # 29
NUM_BODIES_HAND = len(SHARPA_BODY_NAMES)                       # 28 (idx 0 = wrist)
N_FINGERTIPS = 5


class DexTrackSharpa(VecTask):
    """Single fixed trajectory tracking task; residual policy on top of kinematic reference."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # ---- top-level config (env block) ----
        env_cfg = cfg["env"]
        self.max_episode_length = env_cfg.get("episodeLength", 300)
        self.reward_style       = env_cfg.get("reward_style", "dextrack")
        assert self.reward_style in ("maniptrans", "dextrack"), \
            f"reward_style must be 'maniptrans' or 'dextrack', got {self.reward_style}"

        # Residual control parameters — DexTrack accumulating-residual semantics.
        # delta_delta_targets = speed_scale_per_joint × dt × action × 2 × ctlFreqInv
        # cur_delta_targets_warm += delta_delta_targets         (running sum)
        # target = ref + cur_delta_targets_warm
        # speed_scale_per_joint (wfranka):
        #   arm 7-DOF : dofSpeedScale × frankaDeltaDeltaMultCoef = 20 × 2 = 40
        #   hand 22-DOF: dofSpeedScale = 20
        self.dof_speed_scale   = env_cfg.get("dofSpeedScale", 20.0)
        self.franka_delta_mult = env_cfg.get("frankaDeltaDeltaMultCoef", 2.0)
        self.action_moving_avg = env_cfg.get("actionMovingAverage", 1.0)

        # DexTrack-style one-shot progress scattering on first reset
        # (cfg key `randomTime` follows DexTrack `random_time` semantics: cfg-True →
        # 8192 envs scatter to random frames on initial reset, then auto-off so every
        # subsequent reset returns to frame 0).  This gives the value-fn a diverse
        # critic warm-start without requiring true mid-trajectory training.
        self._random_time_pending = bool(env_cfg.get("randomTime", False))

        # Trajectory cfg
        traj_cfg = env_cfg["trajectory"]
        self.trajectory_npy  = traj_cfg["npy_path"]
        self.urdf_path       = env_cfg["asset"]["robot"]
        self.control_dt      = traj_cfg.get("dt", 1.0 / 60.0)

        # Reward cfg knobs
        rew_cfg = env_cfg.get("reward", {})
        self.dextrack_rew_kwargs   = rew_cfg.get("dextrack",   {})
        self.maniptrans_rew_kwargs = rew_cfg.get("maniptrans", {})

        # Object / table.  DexTrack supplies rigid_obj_density=500 and lets
        # IsaacGym compute mass from density × mesh volume (paper L471).
        # For cubesmall (5cm cube, vol=1.25e-4 m³): mass = 0.0625 kg.
        obj_cfg = env_cfg["object"]
        self.object_size    = obj_cfg.get("size",    [0.05, 0.05, 0.05])
        self.object_density = obj_cfg.get("density", 500.0)
        self.object_friction = obj_cfg.get("friction", 1.0)

        table_cfg = env_cfg.get("table", {})
        # DexTrack wfranka geometry (allegro_hand_tracking_generalist.py L4990/L5053):
        #   table_dims = (1, 1, table_z_dim);  default table_z_dim=0.5
        #   table_pose.p = (0.70, 0, 0.5 * table_z_dim)   ← centre of table
        # Our retargeted trajectory uses the same convention:
        #   target wrist mean at world (0.4, 0, 0.7), Franka base at origin.
        self.table_size  = table_cfg.get("size",  [1.0, 1.0, 0.5])
        self.table_pose  = table_cfg.get("pose",  [0.70, 0.0, 0.25])
        self.table_friction = table_cfg.get("friction", 1.0)

        # PD gains (Franka panda + Sharpa) — same defaults as SimToolReal class.
        self.arm_stiffness   = env_cfg.get("armStiffness",   400.0)
        self.arm_damping     = env_cfg.get("armDamping",     40.0)
        self.hand_stiffness  = env_cfg.get("handStiffness",  20.0)
        self.hand_damping    = env_cfg.get("handDamping",    2.0)

        # ---- obs / state dims ----
        future_frames = (FUTURE_FRAMES_MANIPTRANS if self.reward_style == "maniptrans"
                                                else FUTURE_FRAMES_DEXTRACK)
        self.future_frames = future_frames
        self.goal_obs_dim  = get_goal_obs_dim(self.reward_style, future_frames)
        # Proprio obs (SimToolReal subset, no object shape):
        #   joint_pos(29) + joint_vel(29) + prev_targets(29)
        # + palm_pos(3) + palm_rot(4) + palm_vel(6)
        # + object_pos(3) + object_rot(4) + object_vel(6)
        # + fingertip_pos_rel_palm(5*3=15)
        # + progress(1) + reward(1)              == 130
        self.proprio_obs_dim = 29 + 29 + 29 + 3 + 4 + 6 + 3 + 4 + 6 + 15 + 1 + 1
        self.cfg["env"]["numObservations"] = self.proprio_obs_dim + self.goal_obs_dim
        self.cfg["env"]["numStates"]       = self.cfg["env"]["numObservations"]  # symmetric AC
        self.cfg["env"]["numActions"]      = NUM_DOFS

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # ---- Acquire sim tensors ----
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state        = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_state  = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states     = gymtorch.wrap_tensor(actor_root_state)        # (n_actors_total, 13)
        self.dof_state       = gymtorch.wrap_tensor(dof_state).view(self.num_envs, NUM_DOFS, 2)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_rigid_bodies_per_env, 13)
        self.dof_force       = gymtorch.wrap_tensor(dof_force_state).view(self.num_envs, NUM_DOFS)

        self.arm_hand_dof_pos = self.dof_state[..., 0]
        self.arm_hand_dof_vel = self.dof_state[..., 1]
        # Track previous-step dof_vel for the DexTrack smoothness reward term
        # (smoothness_rew = -coef × |prev_dof_vel - cur_dof_vel|, paper L12780).
        self.prev_dof_vel = torch.zeros_like(self.arm_hand_dof_vel)

        # Per-env actor indices (robot + object + table)
        self.robot_actor_idx_global   = torch.tensor(self.robot_actor_idx_global_list,
                                                     dtype=torch.int32, device=self.device)
        self.object_actor_idx_global  = torch.tensor(self.object_actor_idx_global_list,
                                                     dtype=torch.int32, device=self.device)

        # Target buffers
        self.prev_targets = torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        self.cur_targets  = torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        # Raw policy action from the last step (used by DexTrack r_delta_hand)
        self.last_actions = torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        # Accumulating residual on top of the kinematic reference (DexTrack
        # cur_delta_targets_warm, L12220-12225).
        self.cur_delta_targets_warm = torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        # Per-joint speed scale (wfranka path): arm × mult, hand × 1.
        speed_scale = [self.dof_speed_scale * self.franka_delta_mult] * NUM_ARM_DOFS \
                    + [self.dof_speed_scale] * NUM_HAND_DOFS
        self.dof_speed_scale_tsr = torch.tensor(speed_scale, dtype=torch.float32,
                                                 device=self.device)
        # Control-frequency inverse (multiplied into delta_delta_targets, paper L12219).
        self.control_freq_inv = env_cfg.get("controlFrequencyInv", 1)
        # Sim dt (used in the residual formula).  Default 1/60 matches DexTrack.
        self.sim_dt = self.cfg.get("sim", {}).get("dt", 1.0 / 60.0)

        # Trajectory loader (single fixed trajectory, shared across envs)
        self.traj = DexTrackTrajectory(self.trajectory_npy, self.urdf_path,
                                       dt=self.control_dt, device=self.device)
        self.episode_T = min(self.traj.T - 1, self.max_episode_length)

        # progress_buf in VecTask is num_envs long (step counter); we use it as ref-frame index.
        self.last_reward = torch.zeros(self.num_envs, device=self.device)
        self.successes   = torch.zeros(self.num_envs, device=self.device)

        # Cache wrist body index (in rigid_body_state per-env)
        self.wrist_body_idx = self.robot_body_name_to_idx[WRIST_LINK]
        # Fingertip body indices (matches reward fingertip_idxs convention)
        self.fingertip_body_idxs = torch.tensor([
            self.robot_body_name_to_idx[SHARPA_BODY_NAMES[i]] for i in [27, 5, 10, 21, 16]
        ], dtype=torch.long, device=self.device)
        # All Sharpa link indices (28 entries, including wrist at index 0)
        self.all_hand_body_idxs = torch.tensor([
            self.robot_body_name_to_idx[n] for n in SHARPA_BODY_NAMES
        ], dtype=torch.long, device=self.device)
        # Per-env object initial-frame z (= obj_pos[0, 2] from the trajectory).
        # Used by the DexTrack hand_up reward term (paper L12810-12811
        # `lift_z = object_init_z + (hand_up_thresh_1 - 0.030) + 0.003`).
        self.obj_init_z = self.traj.obj_pos[0, 2].expand(self.num_envs).clone()

        # Initial reset to ref-frame 0 for every env
        self.reset_buf[:] = 1
        self._do_reset_initial()

    # =======================================================================
    # Sim creation
    # =======================================================================
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0.0
        self.sim_params.gravity.y = 0.0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3( spacing,  spacing, spacing)

        # --- Robot asset ---
        urdf_dir = os.path.dirname(self.urdf_path)
        urdf_file = os.path.basename(self.urdf_path)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.use_mesh_materials = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        robot_asset = self.gym.load_asset(self.sim, urdf_dir, urdf_file, asset_options)

        self.num_robot_dofs   = self.gym.get_asset_dof_count(robot_asset)
        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        assert self.num_robot_dofs == NUM_DOFS, \
            f"URDF has {self.num_robot_dofs} DOF; expected {NUM_DOFS}"

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        # PD gains: arm vs hand
        for i in range(self.num_robot_dofs):
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if i < NUM_ARM_DOFS:
                robot_dof_props["stiffness"][i] = self.arm_stiffness
                robot_dof_props["damping"][i]   = self.arm_damping
            else:
                robot_dof_props["stiffness"][i] = self.hand_stiffness
                robot_dof_props["damping"][i]   = self.hand_damping

        self.robot_dof_lower = torch.tensor(
            [robot_dof_props["lower"][i] for i in range(self.num_robot_dofs)],
            device=self.device)
        self.robot_dof_upper = torch.tensor(
            [robot_dof_props["upper"][i] for i in range(self.num_robot_dofs)],
            device=self.device)

        # Body name → idx populated AFTER first actor creation (uses DOMAIN_ENV).
        # Keep asset names cached in case we need them later.
        self._robot_asset_body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.robot_body_name_to_idx = None

        # --- Object asset (box, free body) ---
        # Density-based mass (DexTrack convention); for cubesmall (5cm cube)
        # the geometry is equivalent to the GRAB cubesmall.obj mesh.
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = self.object_density
        object_asset = self.gym.create_box(self.sim,
            self.object_size[0], self.object_size[1], self.object_size[2], obj_opts)
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)

        # --- Table asset (static) ---
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim,
            self.table_size[0], self.table_size[1], self.table_size[2], table_opts)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)

        # Total bodies per env (used to view rigid_body_state correctly)
        self.num_rigid_bodies_per_env = (self.num_robot_bodies + self.num_object_bodies
                                          + self.num_table_bodies)

        # --- Per-env spawn ---
        # DexTrack wfranka: Franka base at WORLD ORIGIN (allegro_hand_tracking_generalist.py L5017
        # `shadow_hand_start_pose.p = (0, 0, 0)`).  Our retargeted trajectory follows the
        # same convention: target_center=(0.4, 0, 0.7) wrist mean → Franka base = origin.
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        # Table pose: centre at cfg.table.pose = (0.70, 0, 0.25) per DexTrack.
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(self.table_pose[0], self.table_pose[1], self.table_pose[2])

        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(self.table_pose[0], 0.0,
                                     self.table_pose[2] + self.object_size[2] / 2)
        object_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.envs = []
        self.robot_actor_idx_global_list = []
        self.object_actor_idx_global_list = []

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            robot_actor = self.gym.create_actor(env, robot_asset, robot_pose,
                                                 "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env, robot_actor, robot_dof_props)
            # Enable DOF force sensors so we can later read per-DOF joint torques
            # via acquire_dof_force_tensor (needed for ManipTrans r_power that
            # uses |dof_force × dof_vel| — paper dexhandimitator.py L579).
            self.gym.enable_actor_dof_force_sensors(env, robot_actor)
            self.robot_actor_idx_global_list.append(
                self.gym.get_actor_index(env, robot_actor, gymapi.DOMAIN_SIM))

            # Populate canonical body-name → env-local idx map from the FIRST env only.
            if self.robot_body_name_to_idx is None:
                self.robot_body_name_to_idx = {
                    n: self.gym.find_actor_rigid_body_index(
                        env, robot_actor, n, gymapi.DOMAIN_ENV)
                    for n in self._robot_asset_body_names
                }

            table_actor = self.gym.create_actor(env, table_asset, table_pose,
                                                 "table", i, 0, 0)

            object_actor = self.gym.create_actor(env, object_asset, object_pose,
                                                  "object", i, 0, 0)
            obj_rigid_props = self.gym.get_actor_rigid_shape_properties(env, object_actor)
            for p in obj_rigid_props:
                p.friction = self.object_friction
            self.gym.set_actor_rigid_shape_properties(env, object_actor, obj_rigid_props)
            self.object_actor_idx_global_list.append(
                self.gym.get_actor_index(env, object_actor, gymapi.DOMAIN_SIM))

            self.envs.append(env)

        print(f"[DexTrackSharpa] Created {num_envs} envs; "
              f"rigid_bodies/env = {self.num_rigid_bodies_per_env} "
              f"(robot {self.num_robot_bodies} + object {self.num_object_bodies} + table {self.num_table_bodies})")

    # =======================================================================
    # Buffers (override VecTask defaults if needed)
    # =======================================================================
    def allocate_buffers(self):
        super().allocate_buffers()
        # Extra: keep last_reward for obs feedback
        # (Already created in __init__ AFTER super().__init__, that's fine.)

    # =======================================================================
    # Step loop
    # =======================================================================
    def pre_physics_step(self, actions: Tensor, joint_pos_targets=None):
        """Cumulative residual on kinematic reference.

        actions ∈ [-1, 1] (clipped by VecTask), shape (N, 29).
        target = ref.dof_pos[t+1] + action_scale * actions
        """
        actions = actions.to(self.device).clamp(-1.0, 1.0)
        self.last_actions[:] = actions
        # Reset envs that are flagged
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # ── DexTrack accumulating residual (paper L12219-12225) ────────────
        # delta_delta_targets = speed_scale × dt × action × 2 × ctlFreqInv
        # cur_delta_targets_warm += delta_delta_targets
        # target = ref + cur_delta_targets_warm
        delta_delta = (self.dof_speed_scale_tsr.unsqueeze(0)
                        * self.sim_dt * actions * 2.0 * self.control_freq_inv)
        self.cur_delta_targets_warm[:] = self.cur_delta_targets_warm + delta_delta

        # Reference (kinematic bias) target for next frame
        t_next   = torch.clamp(self.progress_buf + 1, max=self.traj.T - 1)
        ref_dof  = self.traj.dof_pos[t_next]                   # (N, 29)
        target   = ref_dof + self.cur_delta_targets_warm

        # Clip to joint limits
        target = torch.max(torch.min(target, self.robot_dof_upper), self.robot_dof_lower)
        # Optional action moving-average smoothing
        if self.action_moving_avg < 1.0:
            target = (self.action_moving_avg * target
                      + (1.0 - self.action_moving_avg) * self.prev_targets)
        self.cur_targets[:] = target
        self.prev_targets[:] = target

        self.gym.set_dof_position_target_tensor(self.sim,
            gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        """Refresh sim, compute reward & obs, increment progress."""
        # Cache last-step dof_vel BEFORE refreshing — smoothness reward uses
        # |prev_dof_vel - cur_dof_vel|.
        self.prev_dof_vel[:] = self.arm_hand_dof_vel

        self.progress_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self._compute_reward_and_termination()
        self._compute_obs()

        # Bookkeeping for logging (VecTask uses self.extras)
        self.extras["successes"] = self.successes.clone()

    # =======================================================================
    # Reset
    # =======================================================================
    def _do_reset_initial(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids: Tensor):
        """Reset selected envs to the trajectory.

        Default: every env starts at frame 0 (pre-grasp, physically feasible).
        DexTrack `random_time` first-reset scattering: on the FIRST reset call
        after env construction, envs are scattered across the trajectory
        (random per-env start frame in [0, T-1)); the flag then auto-clears so
        every subsequent reset returns to frame 0.  This matches DexTrack's
        L11535-11542 behaviour.
        """
        if len(env_ids) == 0:
            return
        E = len(env_ids)

        # Per-env start frame (default 0; randomized once if random_time was set).
        if self._random_time_pending:
            start_frames = torch.randint(0, self.traj.T - 1, (E,),
                                          device=self.device, dtype=torch.long)
            self._random_time_pending = False
            print(f"[DexTrackSharpa] random_time scattering: "
                  f"{E} envs spread over frames [0, {self.traj.T - 2}]")
        else:
            start_frames = torch.zeros(E, dtype=torch.long, device=self.device)

        # 1) DOF state: pos = traj.dof_pos[start_frames], vel = 0
        dof_pos_init = self.traj.dof_pos[start_frames]              # (E, 29)
        dof_vel_init = torch.zeros_like(dof_pos_init)
        self.dof_state[env_ids, :, 0] = dof_pos_init
        self.dof_state[env_ids, :, 1] = dof_vel_init
        self.prev_targets[env_ids] = dof_pos_init
        self.cur_targets[env_ids]  = dof_pos_init

        # 2) Object root state: position+orientation from traj, vel = 0
        # (DexTrack L11510 sets root_state_tensor[obj_idx, 7:13]=0 even though
        # they pre-computed `goal_obj_lin_vels`; we follow the same convention.)
        obj_pos_init  = self.traj.obj_pos[start_frames]              # (E, 3)
        obj_quat_init = self.traj.obj_quat[start_frames]             # (E, 4)
        obj_state = torch.zeros(E, 13, device=self.device)
        obj_state[:, 0:3] = obj_pos_init
        obj_state[:, 3:7] = obj_quat_init
        self.root_states[self.object_actor_idx_global[env_ids].long(), :] = obj_state

        # Apply tensors back to sim
        env_ids_i32_obj = self.object_actor_idx_global[env_ids].contiguous()
        env_ids_i32_rob = self.robot_actor_idx_global[env_ids].contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_i32_obj), E)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.dof_state.view(-1, 2)),
            gymtorch.unwrap_tensor(env_ids_i32_rob), E)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.cur_targets),
            gymtorch.unwrap_tensor(env_ids_i32_rob), E)

        # Counters
        self.progress_buf[env_ids] = start_frames
        self.reset_buf[env_ids]    = 0
        self.successes[env_ids]    = 0
        self.last_reward[env_ids]  = 0
        # Reset accumulating residual (paper resets cur_delta_targets_warm per episode)
        self.cur_delta_targets_warm[env_ids] = 0
        self.prev_dof_vel[env_ids] = 0

    # =======================================================================
    # Reward + termination
    # =======================================================================
    def _build_state_dict(self) -> Dict[str, Tensor]:
        # Robot hand links: per-env (28, 13)
        body = self.rigid_body_state                                   # (E, B, 13)
        wrist = body[:, self.wrist_body_idx]                           # (E, 13)
        hand_links = body[:, self.all_hand_body_idxs]                  # (E, 28, 13)

        obj_root = self.root_states[self.object_actor_idx_global.long(), :]  # (E, 13)

        return {
            "dof_pos":       self.arm_hand_dof_pos,
            "dof_vel":       self.arm_hand_dof_vel,
            "dof_force":     self.dof_force,
            "prev_dof_vel":  self.prev_dof_vel,
            "wrist_pos":     wrist[:, 0:3],
            "wrist_quat":    wrist[:, 3:7],
            "wrist_vel":     wrist[:, 7:10],
            "wrist_ang_vel": wrist[:, 10:13],
            "link_pos":      hand_links[:, :, 0:3],
            "link_vel":      hand_links[:, :, 7:10],
            "obj_pos":       obj_root[:, 0:3],
            "obj_quat":      obj_root[:, 3:7],
            "obj_lin_vel":   obj_root[:, 7:10],
            "obj_ang_vel":   obj_root[:, 10:13],
            "obj_init_z":    self.obj_init_z,
        }

    def _compute_reward_and_termination(self):
        state = self._build_state_dict()
        target = self.traj.get(torch.clamp(self.progress_buf, max=self.traj.T - 1))

        if self.reward_style == "maniptrans":
            # ManipTrans reward doesn't use `actions` directly (power term uses dof_vel);
            # we still pass last_actions for interface consistency.
            reward, failed, succeeded, comp = compute_maniptrans_reward(
                state, target, self.last_actions,
                self.progress_buf, **self.maniptrans_rew_kwargs)
        else:
            # DexTrack r_delta_hand = -coef * action² — must be the RAW policy action,
            # not the absolute DOF position target.
            reward, failed, succeeded, comp = compute_dextrack_reward(
                state, target, self.last_actions, self.progress_buf,
                **self.dextrack_rew_kwargs)

        self.rew_buf[:] = reward
        self.last_reward[:] = reward
        self.successes += succeeded.float()

        # Termination: failed_execute OR episode timeout OR end of trajectory
        timeout = self.progress_buf >= self.episode_T
        out_of_traj = self.progress_buf >= (self.traj.T - 1)
        self.reset_buf[:] = (failed | timeout | out_of_traj).to(self.reset_buf.dtype)

        # Store components for logging
        self.extras["reward_components"] = {k: v.mean().item() for k, v in comp.items()
                                            if v.dtype == torch.float}

    # =======================================================================
    # Observations
    # =======================================================================
    def _compute_obs(self):
        state = self._build_state_dict()

        palm_pos     = state["wrist_pos"]
        palm_rot     = state["wrist_quat"]
        palm_lin_vel = state["wrist_vel"]
        palm_ang_vel = state["wrist_ang_vel"]

        # Fingertip pos relative to palm
        body = self.rigid_body_state
        fingertip_pos_world = body[:, self.fingertip_body_idxs, 0:3]   # (E, 5, 3)
        fingertip_pos_rel_palm = (fingertip_pos_world - palm_pos[:, None, :]).reshape(self.num_envs, -1)

        progress_obs = torch.log(self.progress_buf.float().unsqueeze(-1) / 10.0 + 1.0)
        reward_obs   = self.last_reward.unsqueeze(-1)

        proprio = torch.cat([
            self.arm_hand_dof_pos,                                     # 29
            self.arm_hand_dof_vel,                                     # 29
            self.prev_targets,                                         # 29
            palm_pos,                                                  # 3
            palm_rot,                                                  # 4
            torch.cat([palm_lin_vel, palm_ang_vel], dim=-1),           # 6
            state["obj_pos"],                                          # 3
            state["obj_quat"],                                         # 4
            torch.cat([state["obj_lin_vel"], state["obj_ang_vel"]], -1),  # 6
            fingertip_pos_rel_palm,                                    # 15
            progress_obs,                                              # 1
            reward_obs,                                                # 1
        ], dim=-1)

        if self.reward_style == "maniptrans":
            goal_obs = build_maniptrans_goal_obs(state, self.traj, self.progress_buf,
                                                  self.future_frames)
        else:
            goal_obs = build_dextrack_goal_obs(state, self.traj, self.progress_buf,
                                                self.future_frames)

        self.obs_buf[:]    = torch.cat([proprio, goal_obs], dim=-1)
        self.states_buf[:] = self.obs_buf
