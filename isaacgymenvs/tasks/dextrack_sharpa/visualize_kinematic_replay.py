"""Pure kinematic playback of the reference trajectory in the DexTrackSharpa env.
Forces the robot DOFs + object root state to traj frame t every step.  NO policy.

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/visualize_kinematic_replay.py

Keys:
  SPACE  step 1 frame forward
  R      reset to frame 0
  P      auto-play loop (toggle)
  ESC    quit

Background — root cause discovered 2026-05-13:
  IsaacGym's `step_graphics(sim)` ONLY reflects the state from the last
  `simulate(sim)` call.  Writing through `set_dof_state_tensor_indexed` /
  `set_actor_root_state_tensor_indexed` updates the simulator's internal
  state, but the render pipeline ignores those writes unless simulate()
  is called afterwards.  Therefore kinematic replay MUST call simulate()
  every frame.  To prevent simulate() from disturbing our forced state:

    (1) zero gravity in sim cfg
    (2) write zero velocity for DOFs and object root
    (3) sync PD targets to ref_dof every frame (otherwise PD wakes up
        with target=0 and yanks every hand joint to zero in one dt,
        producing the "twisted hand" artifact)

  This matches ManipTrans `DexManipNet/dexmanip_sh.py::play()` which
  uses the same set→simulate→render pattern.

  Verified via diag_render_no_sim.py (pixel diff = 0 without simulate,
  = 11.78 with simulate) and diag_set_after_sim.py (set-after-simulate
  doesn't reach the renderer).
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
        "dofSpeedScale": 0.0, "frankaDeltaDeltaMultCoef": 0.0,
        "actionMovingAverage": 1.0, "randomTime": False,
        "reward_style": "dextrack",
        "reward": {
            "dextrack":   {"early_terminate_obj_dist": 0.0},
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
        # Zero gravity + zero velocities → simulate() is (near-)idempotent
        # under the forced state we write each frame.  Combined with PD
        # targets synced to ref_dof, simulate() does not disturb DOFs.
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
                     graphics_device_id=0, headless=False,
                     virtual_screen_capture=False, force_render=True)

print(f"[viz-kinematic] T={env.traj.T}  obj_init z={env.traj.obj_pos[0,2].item():.3f}")
print("[viz-kinematic] keys: SPACE=step  R=reset  P=auto-play  ESC=quit\n")

cam_pos    = gymapi.Vec3(+0.10, -0.70, +0.95)
cam_target = gymapi.Vec3(+0.45, -0.05, +0.55)
env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "step")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R,     "reset")
env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P,     "play")


def set_kinematic_frame(t: int):
    """Force the entire scene to trajectory frame t.

    Writes BOTH the state and the PD target (synced) so that the upcoming
    simulate() does not drag any joint via stale PD targets.  Velocities
    are zeroed so the integrator has nothing to advance.
    """
    t = max(0, min(t, env.traj.T - 1))
    ref_dof = env.traj.dof_pos[t]

    # 1) DOF state (position + zero velocity)
    env.dof_state[0, :, 0] = ref_dof
    env.dof_state[0, :, 1] = 0.0
    env.gym.set_dof_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.dof_state.view(-1, 2)),
        gymtorch.unwrap_tensor(env.robot_actor_idx_global[:1].contiguous()), 1)

    # 2) PD targets in sync with the desired DOF state.  Without this,
    #    the PD controller (stiffness 100, damping 4 on the hand) drives
    #    each hand joint toward 0 inside simulate(), causing visible
    #    finger twisting within a single frame.
    env.cur_targets[0, :]  = ref_dof
    env.prev_targets[0, :] = ref_dof
    env.gym.set_dof_position_target_tensor(env.sim,
        gymtorch.unwrap_tensor(env.cur_targets))

    # 3) Object root state (pos + orient + zero linear/angular velocity)
    obj_idx = env.object_actor_idx_global[:1].long()
    env.root_states[obj_idx, 0:3] = env.traj.obj_pos[t]
    env.root_states[obj_idx, 3:7] = env.traj.obj_quat[t]
    env.root_states[obj_idx, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env.object_actor_idx_global[:1].contiguous()), 1)


t = 0
auto = False
# Warmup: prime the sim once so the renderer has valid state to display.
set_kinematic_frame(t)
env.gym.simulate(env.sim)
env.gym.fetch_results(env.sim, True)
env.gym.step_graphics(env.sim)

while not env.gym.query_viewer_has_closed(env.viewer):
    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue
        if evt.action == "step":
            t = (t + 1) % env.traj.T
            print(f"[viz] t={t}  obj=({env.traj.obj_pos[t,0]:.3f}, {env.traj.obj_pos[t,1]:.3f}, {env.traj.obj_pos[t,2]:.3f})")
        elif evt.action == "reset":
            t = 0
            print(f"[viz] reset → t=0")
        elif evt.action == "play":
            auto = not auto
            print(f"[viz] auto-play = {auto}")
    if auto:
        t = (t + 1) % env.traj.T

    # ── set → simulate → render loop (per ManipTrans dexmanip_sh.play()) ──
    set_kinematic_frame(t)
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)
    env.gym.sync_frame_time(env.sim)
