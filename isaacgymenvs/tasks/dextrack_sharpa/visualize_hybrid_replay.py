"""Hybrid replay: OUR Sharpa hand qpos + LEAP-shipped object trajectory.

Same env + viewer + collision-filter trick as visualize_kinematic_replay.py,
but at each frame we write hand DOFs from OUR retargeted npy and object
root state from the DexTrack-shipped LEAP npy.  Frame indices match 1:1
(both files were retargeted from the same GRAB s2_cubesmall_inspect_1
clip into 300 frames).

This lets you visually judge whether LEAP's object trajectory is geometrically
consistent with the hand grasp:
  - if the cube tracks the fingers cleanly  → LEAP data is fine, our
    earlier suspicion was wrong
  - if the cube floats away from the fingers / pokes through them
    → LEAP data is misaligned with the hand mocap

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/visualize_hybrid_replay.py

Keys: SPACE step  R reset  P auto-play  ESC quit
"""
import os
import isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import numpy as np
import torch

from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa

ROOT     = os.environ.get("VIDEO_RL_FOLLOWER_ROOT", "/home/intel/Codes/video-rl-follower")
LEAP_NPY = "/home/intel/下载/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data/leap_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"

cfg = OmegaConf.create({
    "name": "DexTrackSharpa", "physics_engine": "physx",
    "env": {
        "numEnvs": 1, "envSpacing": 1.5, "episodeLength": 300,
        "clampAbsObservations": 50.0, "controlFrequencyInv": 1,
        "dofSpeedScale": 0.0, "frankaDeltaDeltaMultCoef": 0.0,
        "actionMovingAverage": 1.0, "randomTime": False, "reward_style": "dextrack",
        "reward": {"dextrack": {"early_terminate_obj_dist": 0.0},
                   "maniptrans": {"failed_execute_enabled": False}},
        "armStiffness": 400.0, "armDamping": 80.0, "handStiffness": 100.0, "handDamping": 4.0,
        "trajectory": {
            "npy_path": f"{ROOT}/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy",
            "dt": 1.0/60.0,
        },
        "object": {"size": [0.05, 0.05, 0.05], "density": 500.0, "friction": 1.0},
        "table":  {"size":  [1.0, 1.0, 0.5], "pose": [0.70, 0.0, 0.25], "friction": 1.0},
        "asset":  {"robot": f"{ROOT}/assets/urdf/franka_sharpa_description/franka_panda_sharpa.urdf"},
        "enableCameraSensors": False,
    },
    "sim": {
        "dt": 1.0/60.0, "substeps": 2, "up_axis": "z",
        "use_gpu_pipeline": True, "gravity": [0.0, 0.0, 0.0],
        "physx": {
            "num_threads": 4, "solver_type": 1, "use_gpu": True,
            "num_position_iterations": 8, "num_velocity_iterations": 0,
            "contact_offset": 0.002, "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2, "max_depenetration_velocity": 1000.0,
            "default_buffer_size_multiplier": 5.0, "max_gpu_contact_pairs": 8388608,
            "num_subscenes": 4, "contact_collection": 0,
        },
    },
    "task": {"randomize": False},
})

env = DexTrackSharpa(cfg=OmegaConf.to_container(cfg, resolve=True),
                     rl_device="cuda:0", sim_device="cuda:0",
                     graphics_device_id=0, headless=False,
                     virtual_screen_capture=False, force_render=True)

# ─── Load LEAP-shipped object trajectory ────────────────────────────────
leap = np.load(LEAP_NPY, allow_pickle=True).item()
leap_obj_pos  = torch.tensor(np.asarray(leap["object_transl"]),    device=env.device)  # (T, 3)
leap_obj_quat = torch.tensor(np.asarray(leap["object_rot_quat"]),  device=env.device)  # (T, 4) xyzw

T = min(env.traj.T, leap_obj_pos.shape[0])
print(f"[hybrid-viz] OUR hand traj T={env.traj.T},  LEAP obj traj T={leap_obj_pos.shape[0]},  using min={T}")
print(f"[hybrid-viz] LEAP obj_pos[0]  = {leap_obj_pos[0].cpu().numpy()}")
print(f"[hybrid-viz] LEAP obj_quat[0] = {leap_obj_quat[0].cpu().numpy()}")
print(f"[hybrid-viz] OUR  obj_pos[0]  = {env.traj.obj_pos[0].cpu().numpy()}")
print(f"[hybrid-viz] OUR  obj_quat[0] = {env.traj.obj_quat[0].cpu().numpy()}")
print(f"[hybrid-viz] Δpos (OUR-LEAP at t=0) = {(env.traj.obj_pos[0] - leap_obj_pos[0]).cpu().numpy()}")

# ─── Camera + keys ──────────────────────────────────────────────────────
env.gym.viewer_camera_look_at(env.viewer, env.envs[0],
    gymapi.Vec3(+0.10, -0.70, +0.95), gymapi.Vec3(+0.45, -0.05, +0.55))
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "step")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R,     "reset")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P,     "play")

# ─── Disable cube ↔ robot contact (replay-only) ─────────────────────────
NO_COLLIDE_BIT = 1 << 30
for actor_name in ("object", "robot"):
    h = env.gym.find_actor_handle(env.envs[0], actor_name)
    props = env.gym.get_actor_rigid_shape_properties(env.envs[0], h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    env.gym.set_actor_rigid_shape_properties(env.envs[0], h, props)
print(f"[hybrid-viz] cube↔robot contact disabled (filter {NO_COLLIDE_BIT:#x})")


def set_hybrid_frame(t: int):
    """Hand = OUR Sharpa traj.dof_pos[t], Object = LEAP traj at same t."""
    t = max(0, min(t, T - 1))
    ref_dof = env.traj.dof_pos[t]

    # Hand DOFs from OUR retargeted Sharpa trajectory
    env.dof_state[0, :, 0] = ref_dof
    env.dof_state[0, :, 1] = 0.0
    env.cur_targets[0, :]  = ref_dof
    env.prev_targets[0, :] = ref_dof
    env.gym.set_dof_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    env.gym.set_dof_position_target_tensor(env.sim,
        gymtorch.unwrap_tensor(env.cur_targets))

    # Object pose from LEAP-shipped trajectory
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = leap_obj_pos[t]
    env.root_states[obj_idx, 3:7] = leap_obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)


# Warmup
set_hybrid_frame(0)
env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True); env.gym.step_graphics(env.sim)

print(f"\n[hybrid-viz] keys: SPACE step  R reset  P auto-play  ESC quit\n")

t = 0
auto = False
while not env.gym.query_viewer_has_closed(env.viewer):
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0: continue
        if evt.action == "step":
            t = (t + 1) % T
            d_pos = (env.traj.obj_pos[t] - leap_obj_pos[t]).cpu().numpy()
            print(f"[hybrid-viz] t={t}  Δpos(OUR-LEAP)={d_pos}")
        elif evt.action == "reset":
            t = 0; print("[hybrid-viz] reset → t=0")
        elif evt.action == "play":
            auto = not auto; print(f"[hybrid-viz] auto-play = {auto}")
    if auto:
        t = (t + 1) % T
    set_hybrid_frame(t)
    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim); env.gym.draw_viewer(env.viewer, env.sim, True)
    env.gym.sync_frame_time(env.sim)
