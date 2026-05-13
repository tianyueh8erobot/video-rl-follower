"""Pixel-precise comparison: render the same cube at OUR quat[t] vs LEAP quat[t]
for several frames.  If cube is truly symmetric, images should match
exactly modulo lighting.  If they differ visually, lighting/shading breaks
the symmetry and that explains why visual replay differs.

Run:
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/diag_cube_orientation_render.py
"""
import os, isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import numpy as np, torch, imageio.v2 as imageio, cv2

from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa

ROOT     = "/home/intel/Codes/video-rl-follower"
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
        "table":  {"size": [1.0, 1.0, 0.5], "pose": [0.70, 0.0, 0.25], "friction": 1.0},
        "asset":  {"robot": f"{ROOT}/assets/urdf/franka_sharpa_description/franka_panda_sharpa.urdf"},
        "enableCameraSensors": True,
    },
    "sim": {
        "dt": 1.0/60.0, "substeps": 2, "up_axis": "z",
        "use_gpu_pipeline": True, "gravity": [0.0, 0.0, 0.0],
        "physx": {"num_threads": 4, "solver_type": 1, "use_gpu": True,
            "num_position_iterations": 8, "num_velocity_iterations": 0,
            "contact_offset": 0.002, "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2, "max_depenetration_velocity": 1000.0,
            "default_buffer_size_multiplier": 5.0, "max_gpu_contact_pairs": 8388608,
            "num_subscenes": 4, "contact_collection": 0,}},
    "task": {"randomize": False},
})

env = DexTrackSharpa(cfg=OmegaConf.to_container(cfg, resolve=True),
                     rl_device="cuda:0", sim_device="cuda:0",
                     graphics_device_id=0, headless=True,
                     virtual_screen_capture=False, force_render=False)
gym, sim, env_ptr = env.gym, env.sim, env.envs[0]

leap_q = torch.tensor(np.asarray(np.load(LEAP_NPY, allow_pickle=True).item()["object_rot_quat"]), device=env.device)

NO_COLLIDE_BIT = 1 << 30
for actor_name in ("object", "robot"):
    h = gym.find_actor_handle(env_ptr, actor_name)
    props = gym.get_actor_rigid_shape_properties(env_ptr, h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    gym.set_actor_rigid_shape_properties(env_ptr, h, props)

# Camera tight on cube
cam_props = gymapi.CameraProperties(); cam_props.width = 480; cam_props.height = 360
cam = gym.create_camera_sensor(env_ptr, cam_props)


def set_state(t, use_leap_quat=False):
    t = max(0, min(t, env.traj.T - 1))
    ref = env.traj.dof_pos[t]
    env.dof_state[0, :, 0] = ref; env.dof_state[0, :, 1] = 0.0
    env.cur_targets[0, :] = ref; env.prev_targets[0, :] = ref
    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    env.root_states[obj_idx, 3:7] = leap_q[t] if use_leap_quat else env.traj.obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)


def render_close_to_cube(t):
    cube_pos = env.traj.obj_pos[t].cpu().numpy()
    cam_pos = gymapi.Vec3(float(cube_pos[0] - 0.15), float(cube_pos[1] - 0.20), float(cube_pos[2] + 0.10))
    cam_tgt = gymapi.Vec3(float(cube_pos[0]),         float(cube_pos[1]),         float(cube_pos[2]))
    gym.set_camera_location(cam, env_ptr, cam_pos, cam_tgt)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    img = np.asarray(gym.get_camera_image(sim, env_ptr, cam, gymapi.IMAGE_COLOR))
    return img.reshape(cam_props.height, cam_props.width, 4)[..., :3].copy()


# Warmup
set_state(0); gym.simulate(sim); gym.fetch_results(sim, True)

frames = [30, 100, 150, 200, 270]
cols = []
for t in frames:
    set_state(t, use_leap_quat=False); img_ours = render_close_to_cube(t)
    set_state(t, use_leap_quat=True);  img_leap = render_close_to_cube(t)
    diff = np.abs(img_ours.astype(np.int32) - img_leap.astype(np.int32))
    # Per-pixel difference visualisation (10× amplified)
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
    cv2.putText(img_ours, f"t={t} OUR ours-quat", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.putText(img_leap, f"t={t} LEAP-quat",    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    cv2.putText(diff_vis, f"diff x10 max={diff.max()}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    col = np.vstack([img_ours, img_leap, diff_vis])
    cols.append(col)
    print(f"t={t}: max pixel diff = {diff.max()}  mean = {diff.mean():.2f}")

montage = np.hstack(cols)
imageio.imwrite("/tmp/cube_orientation_diff.png", montage)
print(f"\n→ /tmp/cube_orientation_diff.png  shape={montage.shape}")
print("Rows: OUR quat  /  LEAP quat  /  10× pixel difference")
