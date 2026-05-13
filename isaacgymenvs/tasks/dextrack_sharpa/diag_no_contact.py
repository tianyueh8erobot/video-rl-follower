"""Repeat diag_long_play.py but disable cube↔robot contact.  If the
remaining drift drops to ~0, the "spinning cube" artifact is fully
explained by contact reaction forces on finger-cube interpenetration.

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/diag_no_contact.py
"""
import os
import isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import torch, numpy as np

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
                     graphics_device_id=0, headless=True,
                     virtual_screen_capture=False, force_render=False)

gym, sim = env.gym, env.sim
env_ptr = env.envs[0]

# === Disable cube ↔ robot contact via shared collision filter bit ===
# IsaacGym semantics: two shapes whose filter share ANY bit do NOT collide.
NO_COLLIDE_BIT = 1 << 30   # within signed int32 range

for actor_name in ("object", "robot"):
    h = gym.find_actor_handle(env_ptr, actor_name)
    props = gym.get_actor_rigid_shape_properties(env_ptr, h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    gym.set_actor_rigid_shape_properties(env_ptr, h, props)
    print(f"[no-contact] {actor_name}: set bit {NO_COLLIDE_BIT:#010x} on {len(props)} shapes")


def set_kinematic_frame(t):
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


set_kinematic_frame(0); gym.simulate(sim); gym.fetch_results(sim, True)

max_dof_err = max_obj_pos_err = max_obj_quat_err = 0.0
print(f"\n{'t':>4} {'dof_err':>10} {'obj_pos':>10} {'obj_quat':>10}")
for t in range(env.traj.T):
    set_kinematic_frame(t)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim); gym.refresh_actor_root_state_tensor(sim)
    obj_idx = env.object_actor_idx_global[:1].long()
    dof_err = (env.dof_state[0, :, 0] - env.traj.dof_pos[t]).abs().max().item()
    obj_pos_err = (env.root_states[obj_idx, 0:3] - env.traj.obj_pos[t]).abs().max().item()
    obj_quat_err = (env.root_states[obj_idx, 3:7] - env.traj.obj_quat[t]).abs().max().item()
    max_dof_err = max(max_dof_err, dof_err)
    max_obj_pos_err = max(max_obj_pos_err, obj_pos_err)
    max_obj_quat_err = max(max_obj_quat_err, obj_quat_err)
    if t % 25 == 0 or t == env.traj.T - 1:
        print(f"{t:>4} {dof_err:>10.4f} {obj_pos_err:>10.4f} {obj_quat_err:>10.4f}")

print(f"\n=== Worst-case drift with cube↔robot contact DISABLED ===")
print(f"  max DOF error      : {max_dof_err:.4f} rad")
print(f"  max obj pos error  : {max_obj_pos_err:.4f} m")
print(f"  max obj quat error : {max_obj_quat_err:.4f}")
print()
print(f"=== Compare to PREVIOUS run (contact enabled) ===")
print(f"  prev DOF error  was: 0.5219 rad")
print(f"  prev obj pos    was: 0.0762 m   (this run: {max_obj_pos_err:.4f})")
print(f"  prev obj quat   was: 0.7526     (this run: {max_obj_quat_err:.4f})")
