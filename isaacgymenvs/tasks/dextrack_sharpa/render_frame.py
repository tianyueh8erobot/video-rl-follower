"""Offscreen-render a single frame of the DexTrackSharpa env, save as PNG.
Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/render_frame.py

Output: /tmp/dextrack_frame0.png  (robot + hand + table + object at trajectory frame 0)
"""
import os
import numpy as np
import imageio.v2 as imageio
import isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import torch

from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa


ROOT = os.environ.get("VIDEO_RL_FOLLOWER_ROOT", "/home/intel/Codes/video-rl-follower")

cfg = OmegaConf.create({
    "name": "DexTrackSharpa", "physics_engine": "physx",
    "env": {
        "numEnvs": 1, "envSpacing": 1.5,
        "episodeLength": 50, "clampAbsObservations": 50.0, "controlFrequencyInv": 1,
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
        "enableCameraSensors": True,                      # required for offscreen render
    },
    "sim": {
        "dt": 1.0/60.0, "substeps": 2, "up_axis": "z",
        "use_gpu_pipeline": True, "gravity": [0.0, 0.0, -9.81],
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
                     graphics_device_id=0, headless=True,            # no viewer
                     virtual_screen_capture=False, force_render=False)

env_ptr = env.envs[0]
gym = env.gym
sim = env.sim

# Force frame 0 trajectory state (kinematic set, no physics step).
gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)
env.dof_state[0, :, 0] = env.traj.dof_pos[0]
env.dof_state[0, :, 1] = 0.0
gym.set_dof_state_tensor_indexed(sim,
    gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
    gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
obj_idx = env.object_actor_idx_global[:1].long()
env.root_states[obj_idx, 0:3] = env.traj.obj_pos[0]
env.root_states[obj_idx, 3:7] = env.traj.obj_quat[0]
env.root_states[obj_idx, 7:13] = 0.0
gym.set_actor_root_state_tensor_indexed(sim,
    gymtorch.unwrap_tensor(env.root_states),
    gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)
gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)

# Camera sensor: bird's-eye behind-left of robot.
cam_props = gymapi.CameraProperties()
cam_props.width  = 1280
cam_props.height = 800
cam_props.enable_tensors = False
cam_handle = gym.create_camera_sensor(env_ptr, cam_props)
# Near-top-down view (tilted ~15° so the camera up-vector is well defined).
cam_pos    = gymapi.Vec3(+0.45, -0.20, +1.00)
cam_target = gymapi.Vec3(+0.45, -0.05, +0.53)
gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)

# Step graphics + render
gym.simulate(sim)
gym.fetch_results(sim, True)
gym.step_graphics(sim)
gym.render_all_camera_sensors(sim)

img_rgba = gym.get_camera_image(sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
# IsaacGym returns flat uint8 array (H, W*4) when fetched this way
img = np.asarray(img_rgba).reshape(cam_props.height, cam_props.width, 4)
img_rgb = img[..., :3]

out_path = "/tmp/dextrack_frame0.png"
imageio.imwrite(out_path, img_rgb)
print(f"\n✓ saved {out_path}  shape={img_rgb.shape}  mean_brightness={img_rgb.mean():.1f}")
print(f"  camera pos = (-1.0, -1.5, 1.8)  → target = (+0.5, 0, +0.4)")
print(f"  robot base = (0,0,0)  table = (0.70, 0, 0.25)  obj0 = (0.478,-0.061,0.533)")
