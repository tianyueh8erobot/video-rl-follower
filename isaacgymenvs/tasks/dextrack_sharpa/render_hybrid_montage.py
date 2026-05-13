"""Headless montage: OUR hand + LEAP cube traj, 6 frames side-by-side.

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/render_hybrid_montage.py
Saved to /tmp/hybrid_montage.png
"""
import os
import isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import numpy as np
import torch
import imageio.v2 as imageio
import cv2

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
        "enableCameraSensors": True,
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
                     graphics_device_id=0, headless=True,
                     virtual_screen_capture=False, force_render=False)
gym, sim, env_ptr = env.gym, env.sim, env.envs[0]

leap = np.load(LEAP_NPY, allow_pickle=True).item()
leap_obj_pos  = torch.tensor(np.asarray(leap["object_transl"]),    device=env.device)
leap_obj_quat = torch.tensor(np.asarray(leap["object_rot_quat"]),  device=env.device)
T = min(env.traj.T, leap_obj_pos.shape[0])

# Disable cube↔robot contact
NO_COLLIDE_BIT = 1 << 30
for actor_name in ("object", "robot"):
    h = gym.find_actor_handle(env_ptr, actor_name)
    props = gym.get_actor_rigid_shape_properties(env_ptr, h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    gym.set_actor_rigid_shape_properties(env_ptr, h, props)

# Camera close to hand+object cluster
cam_props = gymapi.CameraProperties(); cam_props.width = 800; cam_props.height = 600
cam = gym.create_camera_sensor(env_ptr, cam_props)
gym.set_camera_location(cam, env_ptr,
    gymapi.Vec3(+0.30, -0.30, +0.85), gymapi.Vec3(+0.45, -0.05, +0.70))


def set_state(t, mode):
    """mode in {'ours', 'leap_quat'}.
       ours      : hand = OUR,  obj_pos = OUR,  obj_quat = OUR
       leap_quat : hand = OUR,  obj_pos = OUR,  obj_quat = LEAP   ← isolate orientation
    """
    t = max(0, min(t, T - 1))
    ref_dof = env.traj.dof_pos[t]
    env.dof_state[0, :, 0] = ref_dof; env.dof_state[0, :, 1] = 0.0
    env.cur_targets[0, :] = ref_dof; env.prev_targets[0, :] = ref_dof
    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    if mode == "ours":
        env.root_states[obj_idx, 3:7] = env.traj.obj_quat[t]
    elif mode == "leap_quat":
        env.root_states[obj_idx, 3:7] = leap_obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)


def render_frame(t, mode):
    set_state(t, mode)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    img = np.asarray(gym.get_camera_image(sim, env_ptr, cam, gymapi.IMAGE_COLOR))
    img = img.reshape(cam_props.height, cam_props.width, 4)[..., :3].copy()
    label = f"t={t}  [{mode}]"
    cv2.putText(img, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


# Warmup
set_state(0, "ours"); gym.simulate(sim); gym.fetch_results(sim, True)

frames = [0, 80, 150, 220]
imgs_ours   = [render_frame(t, "ours")   for t in frames]
imgs_hybrid = [render_frame(t, "leap_quat") for t in frames]

# Vertically pair OUR / HYBRID at same frame; horizontally lay out frames.
cols = []
for t, a, b in zip(frames, imgs_ours, imgs_hybrid):
    pair = np.vstack([a, b])
    cv2.putText(pair, "[OURS] cube_pos=OUR  cube_quat=OUR",
                (15, cam_props.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(pair, "[QUAT] cube_pos=OUR  cube_quat=LEAP",
                (15, 2*cam_props.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cols.append(pair)
montage = np.hstack(cols)
imageio.imwrite("/tmp/hybrid_montage.png", montage)
print(f"\n→ /tmp/hybrid_montage.png  shape={montage.shape}")
print("  row 1: OUR hand + OUR cube  (the canonical replay)")
print("  row 2: OUR hand + LEAP cube (same frame index; cube uses LEAP-shipped pose)")
print("  if LEAP cube tracks the fingertips like row 1, LEAP data is consistent with GRAB.")
print("  if LEAP cube floats away from the fingers, LEAP data is misaligned.")
