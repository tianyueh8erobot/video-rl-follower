"""Force the cube to a known rotation in xyzw, then read back its world
pose via the rigid_body_state tensor.  If IsaacGym interprets the quat as
xyzw the recovered axes will match scipy's interpretation of the same
quat; if it interprets it as wxyz the axes will be wrong.

Run:
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/diag_quat_order.py
"""
import os, isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import numpy as np, torch
from scipy.spatial.transform import Rotation as R

from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa

ROOT = "/home/intel/Codes/video-rl-follower"
cfg = OmegaConf.create({
    "name": "DexTrackSharpa", "physics_engine": "physx",
    "env": {
        "numEnvs": 1, "envSpacing": 1.5, "episodeLength": 50,
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
        "enableCameraSensors": False,
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

# Disable contact so simulate() doesn't perturb anything
NO_COLLIDE_BIT = 1 << 30
for actor_name in ("object", "robot"):
    h = gym.find_actor_handle(env_ptr, actor_name)
    props = gym.get_actor_rigid_shape_properties(env_ptr, h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    gym.set_actor_rigid_shape_properties(env_ptr, h, props)

obj_idx = int(env.object_actor_idx_global[0].item())

# Test rotations: known quat in xyzw, compute expected axis directions
def test_quat(label, q_xyzw):
    q = np.asarray(q_xyzw, dtype=np.float32)
    env.root_states[obj_idx, 0:3] = torch.tensor([0.5, 0.0, 0.7], device=env.device)
    env.root_states[obj_idx, 3:7] = torch.tensor(q, device=env.device)
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    # Find object body index
    # First env, object is the 3rd actor.  Object has 1 body.
    # Easier: just read root state back.
    rb_quat = env.root_states[obj_idx, 3:7].cpu().numpy()

    # Read object rigid-body world transform via rigid_body_state tensor.
    gym.refresh_rigid_body_state_tensor(sim)
    # Object body is the second to last actor (robot, table, object).
    # Find object body index via rigid_body_state shape: (num_envs, num_bodies_per_env, 13)
    # We already have env.rigid_body_state.  Need cube's body idx in rigid_body_state.
    # Total bodies: 44 (robot) + 1 (table) + 1 (object) = 46.  Cube is body idx 45.
    cube_body_idx = env.num_robot_bodies + env.num_table_bodies   # 44 + 1 = 45
    rb_pos = env.rigid_body_state[0, cube_body_idx, 0:3].cpu().numpy()
    rb_quat_api = env.rigid_body_state[0, cube_body_idx, 3:7].cpu().numpy()

    # Expected: scipy from_quat(xyzw) tells us where the body axes go in world
    if hasattr(R, "from_quat"):
        r = R.from_quat(q)
        x_world = r.apply([1, 0, 0])
        y_world = r.apply([0, 1, 0])
        z_world = r.apply([0, 0, 1])
    print(f"\n=== {label} ===")
    print(f"  set quat (xyzw)        = {q}")
    print(f"  refresh tensor q       = {rb_quat}")
    print(f"  rigid_body_state q     = {rb_quat_api}  [whatever IsaacGym's convention is]")
    print(f"  expected (scipy xyzw): body_x →world = {x_world.round(3)}")
    print(f"                         body_y →world = {y_world.round(3)}")
    print(f"                         body_z →world = {z_world.round(3)}")


# 1) Identity
test_quat("Identity", [0.0, 0.0, 0.0, 1.0])
# 2) 90° around world z  (xyzw: (0, 0, sin45, cos45))
test_quat("90° around Z (xyzw)", [0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
# 3) 90° around world x  (xyzw: (sin45, 0, 0, cos45))
test_quat("90° around X (xyzw)", [np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4)])
# 4) GRAB frame 0 quat  ((+0.71, +0.01, +0.01, -0.70))
ours = np.load(f"{ROOT}/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy", allow_pickle=True).item()
test_quat("OUR obj_quat[0]", ours["object_rot_quat"][0])
