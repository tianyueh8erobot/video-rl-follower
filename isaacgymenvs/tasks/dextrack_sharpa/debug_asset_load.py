"""Diagnostic: create the env and check whether robot / object / table assets
loaded their meshes correctly.  Prints every actor's body count + body names.

If `robot 44 bodies` is shown, the Franka+Sharpa URDF loaded fully.
If `robot 0/1 bodies`, the URDF likely failed silently — meshes missing.
"""
import os
import isaacgym
from isaacgym import gymapi
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
        "enableCameraSensors": False,
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
n_actors = env.gym.get_actor_count(env_ptr)
print(f"\n[diagnose] actors in env: {n_actors}")
for i in range(n_actors):
    name = env.gym.get_actor_name(env_ptr, i)
    n_bodies = env.gym.get_actor_rigid_body_count(env_ptr, i)
    print(f"  actor [{i}] name={name!r:<10s} bodies={n_bodies}")
    if n_bodies <= 5:
        for b in range(n_bodies):
            bn = env.gym.get_actor_rigid_body_names(env_ptr, i)[b]
            print(f"        body {b}: {bn}")

print(f"\n[diagnose] env.num_rigid_bodies_per_env: {env.num_rigid_bodies_per_env}")
print(f"[diagnose] robot {env.num_robot_bodies}  object {env.num_object_bodies}  table {env.num_table_bodies}")

if env.num_robot_bodies == 44:
    print("\n✓ robot URDF loaded fully (44 bodies = 8 Franka + 28 Sharpa + 8 fixed adapters)")
elif env.num_robot_bodies < 10:
    print("\n✗ robot URDF mesh load FAILED — only root link visible.  Check IsaacGym stderr.")
else:
    print(f"\n? robot has {env.num_robot_bodies} bodies — verify expected count")
