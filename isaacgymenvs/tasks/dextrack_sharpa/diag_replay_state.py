"""Diagnostic: write trajectory state into sim, refresh, read back, compare.
This isolates whether the bug is:
  (A) state writes are not applied (set_* doesn't reach gym internal state)
  (B) DOF index mismatch (we write to wrong joints)
  (C) render-side issue (state is correct in tensor but renderer shows old state)

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/diag_replay_state.py
"""
import os
import isaacgym
from isaacgym import gymapi, gymtorch
from omegaconf import OmegaConf
import torch
import numpy as np

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
        "use_gpu_pipeline": True, "gravity": [0.0, 0.0, 0.0],     # zero gravity
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

print("\n=== Joint name → DOF index mapping ===")
for name, idx in env.robot_dof_name_to_idx.items() if hasattr(env, "robot_dof_name_to_idx") else []:
    print(f"  [{idx:2d}] {name}")

print(f"\n[traj] T = {env.traj.T}")
print(f"[traj] dof_pos shape  = {tuple(env.traj.dof_pos.shape)}")
print(f"[traj] obj_pos shape  = {tuple(env.traj.obj_pos.shape)}")
print(f"[traj] obj_quat shape = {tuple(env.traj.obj_quat.shape)}")

# Print joint names from the trajectory's pytorch_kinematics chain
print("\n=== Traj chain joint names (input DOF order assumed) ===")
for i, n in enumerate(env.traj.joint_names):
    print(f"  [{i:2d}] {n}")

# ── Test sequence: write frame 0, 50, 100; refresh, compare ─────────────────
test_frames = [0, 50, 100, 150]

for t in test_frames:
    print(f"\n===== Frame t = {t} =====")
    ref_dof = env.traj.dof_pos[t]
    ref_obj_pos = env.traj.obj_pos[t]
    ref_obj_quat = env.traj.obj_quat[t]
    print(f"[in]  ref dof[:7] (arm)  = {ref_dof[:7].cpu().numpy()}")
    print(f"[in]  ref dof[7:14]      = {ref_dof[7:14].cpu().numpy()}")
    print(f"[in]  ref dof[14:22]     = {ref_dof[14:22].cpu().numpy()}")
    print(f"[in]  ref dof[22:29]     = {ref_dof[22:29].cpu().numpy()}")
    print(f"[in]  obj pos            = {ref_obj_pos.cpu().numpy()}")
    print(f"[in]  obj quat           = {ref_obj_quat.cpu().numpy()}")

    # Write into the tensor view
    env.dof_state[0, :, 0] = ref_dof
    env.dof_state[0, :, 1] = 0.0
    env.gym.set_dof_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = ref_obj_pos
    env.root_states[obj_idx, 3:7] = ref_obj_quat
    env.root_states[obj_idx, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)

    # Refresh WITHOUT simulate — does set_* actually write to sim internal state?
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    rb_after_set = env.dof_state[0, :, 0].clone()
    obj_after_set = env.root_states[obj_idx, 0:3].clone()
    obj_q_after_set = env.root_states[obj_idx, 3:7].clone()
    print(f"[refresh-no-sim] dof[:7]  = {rb_after_set[:7].cpu().numpy()}")
    print(f"[refresh-no-sim] obj pos  = {obj_after_set.cpu().numpy()}")
    print(f"[refresh-no-sim] obj quat = {obj_q_after_set.cpu().numpy()}")

    diff_dof_no_sim = (rb_after_set - ref_dof).abs().max().item()
    diff_obj_pos_no_sim = (obj_after_set - ref_obj_pos).abs().max().item()
    print(f"[refresh-no-sim] |dof - ref| max = {diff_dof_no_sim:.4f}")
    print(f"[refresh-no-sim] |obj - ref| max = {diff_obj_pos_no_sim:.4f}")

    # Now call simulate() once and refresh
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    rb_after_sim = env.dof_state[0, :, 0].clone()
    obj_after_sim = env.root_states[obj_idx, 0:3].clone()
    obj_q_after_sim = env.root_states[obj_idx, 3:7].clone()
    print(f"[refresh-w-sim ] dof[:7]  = {rb_after_sim[:7].cpu().numpy()}")
    print(f"[refresh-w-sim ] obj pos  = {obj_after_sim.cpu().numpy()}")
    print(f"[refresh-w-sim ] obj quat = {obj_q_after_sim.cpu().numpy()}")

    diff_dof_w_sim = (rb_after_sim - ref_dof).abs().max().item()
    diff_obj_pos_w_sim = (obj_after_sim - ref_obj_pos).abs().max().item()
    print(f"[refresh-w-sim ] |dof - ref| max = {diff_dof_w_sim:.4f}")
    print(f"[refresh-w-sim ] |obj - ref| max = {diff_obj_pos_w_sim:.4f}")

    # Check wrist body world pose via rigid_body_state
    wrist_idx = env.wrist_body_idx
    wrist_pos = env.rigid_body_state[0, wrist_idx, 0:3].cpu().numpy()
    wrist_quat = env.rigid_body_state[0, wrist_idx, 3:7].cpu().numpy()
    print(f"[fk-wrist     ] world pos  = {wrist_pos}")
    print(f"[fk-wrist     ] world quat = {wrist_quat}")
    # Compare to trajectory wrist_pos (precomputed FK)
    if hasattr(env.traj, "wrist_pos"):
        traj_wrist = env.traj.wrist_pos[t].cpu().numpy()
        print(f"[fk-wrist     ] traj wrist pos = {traj_wrist}  delta = {np.abs(wrist_pos - traj_wrist).max():.4f}")

print("\n=== Diagnostic complete ===")
