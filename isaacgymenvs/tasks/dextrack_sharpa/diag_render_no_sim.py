"""Render same env at two trajectory frames WITHOUT calling simulate().
If both PNGs show the same scene → step_graphics doesn't pull from sim
state until simulate() runs. If they differ → set_*_tensor is enough.

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/diag_render_no_sim.py
"""
import os
import isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import torch, numpy as np, imageio.v2 as imageio

from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa

ROOT = os.environ.get("VIDEO_RL_FOLLOWER_ROOT", "/home/intel/Codes/video-rl-follower")

cfg = OmegaConf.create({
    "name": "DexTrackSharpa", "physics_engine": "physx",
    "env": {
        "numEnvs": 1, "envSpacing": 1.5,
        "episodeLength": 300, "clampAbsObservations": 50.0, "controlFrequencyInv": 1,
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
cam_props = gymapi.CameraProperties(); cam_props.width=1280; cam_props.height=800
cam = gym.create_camera_sensor(env_ptr, cam_props)
gym.set_camera_location(cam, env_ptr,
    gymapi.Vec3(+0.10, -0.70, +0.95), gymapi.Vec3(+0.45, -0.05, +0.55))


def set_state(t):
    ref_dof = env.traj.dof_pos[t]
    env.dof_state[0, :, 0] = ref_dof
    env.dof_state[0, :, 1] = 0.0
    env.cur_targets[0, :]  = ref_dof
    env.prev_targets[0, :] = ref_dof
    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    env.root_states[obj_idx, 3:7] = env.traj.obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)


def render_png(name, with_simulate=False):
    if with_simulate:
        gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    img = np.asarray(gym.get_camera_image(sim, env_ptr, cam, gymapi.IMAGE_COLOR))
    img = img.reshape(cam_props.height, cam_props.width, 4)[..., :3]
    imageio.imwrite(f"/tmp/diag_{name}.png", img)
    print(f"  → /tmp/diag_{name}.png  brightness={img.mean():.1f}")
    return img


print("\n=== A. Without simulate() — only set_*_tensor + step_graphics ===")
set_state(0)
img_a0 = render_png("noSim_frame0")
set_state(100)
img_a100 = render_png("noSim_frame100")
delta_a = np.abs(img_a0.astype(np.int32) - img_a100.astype(np.int32)).mean()
print(f"  pixel diff frame 0 vs 100 = {delta_a:.2f}  (>5 means real visual change)")

print("\n=== B. With simulate() each frame + PD-sync to ref ===")
# Reset to frame 0 state first
set_state(0)
gym.simulate(sim); gym.fetch_results(sim, True)
set_state(0)        # over-write any drift from the very first simulate
img_b0 = render_png("withSim_frame0", with_simulate=True)
set_state(100)
img_b100 = render_png("withSim_frame100", with_simulate=True)
delta_b = np.abs(img_b0.astype(np.int32) - img_b100.astype(np.int32)).mean()
print(f"  pixel diff frame 0 vs 100 = {delta_b:.2f}")

print("\n=== C. With simulate() but NO PD target sync (worst case) ===")
# Skip PD set so PD drives joints to 0
def set_state_no_pd(t):
    ref_dof = env.traj.dof_pos[t]
    env.dof_state[0, :, 0] = ref_dof
    env.dof_state[0, :, 1] = 0.0
    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    env.root_states[obj_idx, 3:7] = env.traj.obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)

set_state_no_pd(100)
img_c100 = render_png("noPDsync_frame100", with_simulate=True)

print("\n=== Done.  Compare images in /tmp/diag_*.png ===")
