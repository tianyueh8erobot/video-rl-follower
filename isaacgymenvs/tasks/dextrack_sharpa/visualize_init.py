"""Spawn 1 env, render initial pose forever — verify robot/table/object geometry.

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/visualize_init.py

Keyboard:
  SPACE  step physics 1 frame (zero-action)
  R      reset to frame 0
  ESC    quit
"""
import os, sys, time
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
        "episodeLength": 300, "clampAbsObservations": 50.0, "controlFrequencyInv": 1,
        "actionScale": 0.1, "actionMovingAverage": 1.0, "randomTime": False,
        "reward_style": "dextrack",
        "reward": {
            "dextrack":   {"reach_goal_bonus": 0.0, "early_terminate_obj_dist": 0.0},  # disable for viz
            "maniptrans": {"failed_execute_enabled": False},
        },
        "armStiffness": 400.0, "armDamping": 40.0,
        "handStiffness": 20.0, "handDamping": 2.0,
        "trajectory": {
            "npy_path": f"{ROOT}/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy",
            "dt": 1.0/60.0,
        },
        "object": {"size": [0.05, 0.05, 0.05], "mass": 0.18, "friction": 1.0},
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
                     graphics_device_id=0, headless=False,
                     virtual_screen_capture=False, force_render=True)
print(f"[viz] env created with 1 env; trajectory T={env.traj.T}")
print(f"[viz] robot at (0,0,0), table CENTRE (0.70, 0, 0.25), object init (",
      f"{env.traj.obj_pos[0,0].item():.3f},",
      f"{env.traj.obj_pos[0,1].item():.3f},",
      f"{env.traj.obj_pos[0,2].item():.3f})")
print(f"[viz] wrist init pos (FK from traj.dof_pos[0]):",
      env.traj.wrist_pos[0].cpu().numpy().round(3))
print(f"[viz] SPACE=step  R=reset  ESC=quit")

env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "step")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R,     "reset")

zero_act = torch.zeros(env.num_envs, env.num_actions, device="cuda:0")
while not env.gym.query_viewer_has_closed(env.viewer):
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.action == "step" and evt.value > 0:
            obs, rew, done, info = env.step(zero_act)
            print(f"[viz] step → progress={env.progress_buf[0].item()}  "
                  f"reward={rew[0].item():+.3f}  obj_z={env.root_states[env.object_actor_idx_global[0].long(), 2].item():.3f}")
        elif evt.action == "reset" and evt.value > 0:
            env.reset_buf[:] = 1
            env.reset_idx(torch.arange(env.num_envs, device="cuda:0"))
            print(f"[viz] reset → progress={env.progress_buf[0].item()}  obj_z={env.root_states[env.object_actor_idx_global[0].long(), 2].item():.3f}")
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)
    env.gym.sync_frame_time(env.sim)
