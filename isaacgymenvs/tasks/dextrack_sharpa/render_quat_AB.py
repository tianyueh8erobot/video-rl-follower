"""A/B compare: OUR quat (R_obj2world from GRAB) vs OUR quat inverted.
Same cube position (OUR, retargeting-shifted) in both rows.
If row B (inverted) matches what user calls "correct hybrid",
IsaacGym is interpreting the quat differently than scipy active rotation.

Run:
  cd /home/intel/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/render_quat_AB.py
"""
import os, isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import numpy as np, torch, imageio.v2 as imageio, cv2
from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa

ROOT = "/home/intel/Codes/video-rl-follower"
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

NO_COLLIDE_BIT = 1 << 30
for actor_name in ("object", "robot"):
    h = gym.find_actor_handle(env_ptr, actor_name)
    props = gym.get_actor_rigid_shape_properties(env_ptr, h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    gym.set_actor_rigid_shape_properties(env_ptr, h, props)

cam_props = gymapi.CameraProperties(); cam_props.width = 800; cam_props.height = 600
cam = gym.create_camera_sensor(env_ptr, cam_props)
gym.set_camera_location(cam, env_ptr,
    gymapi.Vec3(+0.30, -0.30, +0.85), gymapi.Vec3(+0.45, -0.05, +0.70))


def quat_inv_xyzw(q):
    """For unit quat in xyzw: inverse = conjugate = (-x, -y, -z, w)."""
    return torch.stack([-q[0], -q[1], -q[2], q[3]])


def set_state(t, invert_quat: bool):
    t = max(0, min(t, env.traj.T - 1))
    ref_dof = env.traj.dof_pos[t]
    env.dof_state[0, :, 0] = ref_dof; env.dof_state[0, :, 1] = 0.0
    env.cur_targets[0, :] = ref_dof; env.prev_targets[0, :] = ref_dof
    gym.set_dof_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env.cur_targets))

    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    q = env.traj.obj_quat[t]
    if invert_quat:
        q = quat_inv_xyzw(q)
    env.root_states[obj_idx, 3:7] = q
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)


def render(t, invert_quat, label):
    set_state(t, invert_quat)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    img = np.asarray(gym.get_camera_image(sim, env_ptr, cam, gymapi.IMAGE_COLOR))
    img = img.reshape(cam_props.height, cam_props.width, 4)[..., :3].copy()
    cv2.putText(img, label, (15, cam_props.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return img


set_state(0, False); gym.simulate(sim); gym.fetch_results(sim, True)

frames = [0, 80, 150, 220]
cols = []
for t in frames:
    a = render(t, invert_quat=False, label=f"t={t}  OUR quat (R_obj2world)")
    b = render(t, invert_quat=True,  label=f"t={t}  INVERTED (R_world2obj = LEAP-style)")
    cols.append(np.vstack([a, b]))
montage = np.hstack(cols)
imageio.imwrite("/tmp/quat_AB.png", montage)
print(f"\n→ /tmp/quat_AB.png  shape={montage.shape}")
print("Row 1: current convention (OUR quat directly).")
print("Row 2: inverted convention (same as hybrid replay used).")
print("Tell me which row's cube grasp matches the hand correctly.")
