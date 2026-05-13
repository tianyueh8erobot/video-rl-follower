"""End-to-end smoke test: instantiate DexTrackSharpa with 4 envs, step 10 times."""
import os
import sys
import isaacgym                                       # MUST come before torch
from omegaconf import OmegaConf
import torch

from isaacgymenvs.tasks.dextrack_sharpa.env import DexTrackSharpa


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "VIDEO_RL_FOLLOWER_ROOT",
    os.path.abspath(os.path.join(HERE, "..", "..", "..")),
)


def main(reward_style: str = "dextrack"):
    cfg = OmegaConf.create({
        "name": "DexTrackSharpa",
        "physics_engine": "physx",
        "env": {
            "numEnvs":             4,
            "envSpacing":          1.2,
            "episodeLength":       50,
            "clampAbsObservations": 50.0,
            "controlFrequencyInv": 1,
            "actionScale":         0.1,
            "actionMovingAverage": 1.0,
            "reward_style":        reward_style,
            "randomTime":          False,           # smoke uses frame-0 only (toggle to True to exercise scatter)
            "reward": {
                "dextrack":   {"reach_goal_bonus": 0.0},
                "maniptrans": {"failed_execute_enabled": False},   # no early termination
            },
            "armStiffness":  400.0,
            "armDamping":    40.0,
            "handStiffness": 20.0,
            "handDamping":   2.0,
            "trajectory": {
                "npy_path": f"{ROOT}/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy",
                "dt":       1.0 / 60.0,
            },
            "object": {"size": [0.05, 0.05, 0.05], "mass": 0.18, "friction": 1.0},
            "table":  {"size": [1.0, 1.0, 0.4], "pose": [0.4, 0.0, 0.4], "friction": 1.0},
            "asset":  {"robot": f"{ROOT}/assets/urdf/franka_sharpa_description/franka_panda_sharpa.urdf"},
            "enableCameraSensors": False,
        },
        "sim": {
            "dt":                1.0 / 60.0,
            "substeps":          2,
            "up_axis":           "z",
            "use_gpu_pipeline":  True,
            "gravity":           [0.0, 0.0, -9.81],
            "physx": {
                "num_threads":               4,
                "solver_type":               1,
                "use_gpu":                   True,
                "num_position_iterations":   8,
                "num_velocity_iterations":   0,
                "contact_offset":            0.002,
                "rest_offset":               0.0,
                "bounce_threshold_velocity": 0.2,
                "max_depenetration_velocity":1000.0,
                "default_buffer_size_multiplier": 5.0,
                "max_gpu_contact_pairs":     8388608,
                "num_subscenes":             4,
                "contact_collection":        0,
            },
        },
        "task": {"randomize": False},
    })

    print(f"[smoke_env] Creating DexTrackSharpa (reward_style={reward_style}) ...")
    env = DexTrackSharpa(cfg=OmegaConf.to_container(cfg, resolve=True),
                         rl_device="cuda:0", sim_device="cuda:0",
                         graphics_device_id=0, headless=True,
                         virtual_screen_capture=False, force_render=False)
    print(f"[smoke_env] num_envs={env.num_envs}  num_obs={env.num_obs}  num_actions={env.num_actions}")
    print(f"[smoke_env] goal_obs_dim={env.goal_obs_dim}  proprio_obs_dim={env.proprio_obs_dim}")
    print(f"[smoke_env] trajectory T={env.traj.T}  episode_T={env.episode_T}")

    # Step 10 times with zero action (residual=0 → pure kinematic replay)
    actions = torch.zeros(env.num_envs, env.num_actions, device="cuda:0")
    for step in range(10):
        obs, reward, done, info = env.step(actions)
        if step == 0:
            print(f"[smoke_env] step 0: obs.shape={obs['obs'].shape} reward.shape={reward.shape}")
        print(f"  step {step:2d}: reward={reward.mean().item():+.4f}  "
              f"progress={env.progress_buf.float().mean().item():.1f}  "
              f"done={done.sum().item()}/{env.num_envs}")

    print("\n✓ end-to-end smoke test passed")


if __name__ == "__main__":
    style = sys.argv[1] if len(sys.argv) > 1 else "dextrack"
    main(style)
