"""Pure kinematic playback of the reference trajectory inside the DexTrackSharpa
env. NO physics, NO policy — at each step we DIRECTLY set hand DOFs + object
root state from `traj.dof_pos[t]` / `traj.obj_pos[t]` / `traj.obj_quat[t]`.

Use this to confirm:
  - robot is placed at the right pose (origin, facing +X)
  - table is in the right place (in front of the robot, 70cm out)
  - object initial pose matches the trajectory frame 0
  - the trajectory itself is feasible / reasonable in the env coordinate frame

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/visualize_kinematic_replay.py

Keys:
  SPACE  step 1 frame forward
  R      reset to frame 0
  P      auto-play loop (toggle)
  ESC    quit
"""
import os
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
        "episodeLength": 300, "clampAbsObservations": 50.0, "controlFrequencyInv": 1,
        "dofSpeedScale": 0.0, "frankaDeltaDeltaMultCoef": 0.0,  # kill residual
        "actionMovingAverage": 1.0, "randomTime": False,
        "reward_style": "dextrack",
        "reward": {
            "dextrack":   {"early_terminate_obj_dist": 0.0},   # never terminate during viz
            "maniptrans": {"failed_execute_enabled": False},
        },
        "armStiffness": 400.0, "armDamping": 80.0,
        "handStiffness": 100.0, "handDamping": 4.0,
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
                     graphics_device_id=0, headless=False,
                     virtual_screen_capture=False, force_render=True)

print(f"[viz-kinematic] T={env.traj.T}  obj_init z={env.traj.obj_pos[0,2].item():.3f}")
print("[viz-kinematic] keys: SPACE=step  R=reset  P=auto-play  ESC=quit\n")

# Close-up of the robot + hand + object cluster — verified by offscreen render.
# Earlier camera at (1.8,-1.5,1.5) had the robot looking tiny and easily hidden.
cam_pos    = gymapi.Vec3(+0.10, -0.70, +0.95)
cam_target = gymapi.Vec3(+0.45, -0.05, +0.55)
env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)
print(f"[viz] camera (0.10, -0.70, 0.95) → target (0.45, -0.05, 0.55)")
print("[viz] mouse: right-drag=rotate  scroll=zoom  left-drag=pan")

env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "step")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R,     "reset")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P,     "play")


def set_kinematic_frame(t: int):
    """Direct write of dof_pos + object root state from trajectory, NO physics step."""
    t = max(0, min(t, env.traj.T - 1))
    # 1) DOFs
    env.dof_state[0, :, 0] = env.traj.dof_pos[t]
    env.dof_state[0, :, 1] = 0.0
    env.gym.set_dof_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)
    # 2) Object root state
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    env.root_states[obj_idx, 3:7] = env.traj.obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)
    # CRITICAL: after writing tensors, must call simulate + fetch_results so
    # IsaacGym propagates the new state into the rendering pipeline.  Without
    # this the viewer shows whatever scene existed before the writes (only
    # the table, since that was placed at env construction).
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


t = 0
auto = False
set_kinematic_frame(t)
while not env.gym.query_viewer_has_closed(env.viewer):
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue
        if evt.action == "step":
            t = (t + 1) % env.traj.T
            set_kinematic_frame(t)
            print(f"[viz] t={t}  obj=({env.traj.obj_pos[t,0]:.3f}, {env.traj.obj_pos[t,1]:.3f}, {env.traj.obj_pos[t,2]:.3f})")
        elif evt.action == "reset":
            t = 0
            set_kinematic_frame(t)
            print(f"[viz] reset → t=0")
        elif evt.action == "play":
            auto = not auto
            print(f"[viz] auto-play = {auto}")
    if auto:
        t = (t + 1) % env.traj.T
        set_kinematic_frame(t)
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)
    env.gym.sync_frame_time(env.sim)
