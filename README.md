# video_rl_follower

A SimToolReal-derived IsaacGym environment that **follows a single reference
trajectory** (object 6D pose + hand wrist 6D pose + 5 fingertip keypoints)
extracted from MANO-style human demonstrations (e.g. ManipTrans GRAB demo).

The training task reduces to: given a 3 Hz sequence of `(object_goal, hand_goal)`
keypoints, drive the Sharpa hand mounted on a KUKA iiwa14 to satisfy them in
order, advancing through goals when the per-step keypoint error falls inside
the success tolerance (same dwell-then-advance mechanism as SimToolReal).

## What's different from SimToolReal

| | SimToolReal | video_rl_follower |
|---|---|---|
| Task | one universal goal-pose-reaching policy across procedurally generated tools | per-trajectory specialist that follows a fixed reference |
| Goal | object 6D | object 6D **+ wrist 6D + 5 fingertip keypoints** |
| Reward | object-keypoint reward only | object-keypoint reward **+ wrist-pose reward + fingertip reward** (configurable weights) |
| Object | procedurally generated handle+head primitives | trajectory-specified URDF (or any single object) |
| Goal source | random delta sampling / coin-flip | the trajectory file's `object.goals` |
| Optimization | PPO / SAPG | PPO **and** DAPG (PPO + BC on expert demos) |

## Repo layout

```
video_rl_follower/
├── isaacgymenvs/                      # forked from SimToolReal
│   ├── tasks/
│   │   ├── simtoolreal/               # original env, kept for reference
│   │   └── video_rl_follower/         # new env (subclass of SimToolReal)
│   │       ├── env.py
│   │       └── trajectory.py
│   ├── algos/
│   │   └── dapg.py                    # rl_games DAPG-augmented PPO
│   ├── cfg/
│   │   ├── task/VideoRLFollower.yaml
│   │   └── train/VideoRLFollower{PPO,LSTMAsymmetric{PPO,DAPG}}.yaml
│   └── ...                            # train.py, launch_training.py, utils
├── tools/
│   ├── process_maniptrans_trajectory.py    # MANO → 3 Hz JSON
│   └── visualize_reference.py              # GUI viewer
├── data/trajectories/                       # processed JSONs go here
├── assets/urdf/kuka_sharpa_description/     # copied from SimToolReal
└── rl_games/, assets/, ...                  # rest of SimToolReal carrying over
```

## Robot

Sharpa Robotics' official URDF repo (`sharpa-robotics/sharpa-urdf-usd-xml`)
ships hand-only meshes — no arm.  We therefore reuse SimToolReal's bundled
**KUKA iiwa14 + left Sharpa Wave hand** description in
`assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`.

If you later want to swap to right hand, copy the official
`right_sharpa_wave_with_flange.urdf` from `sharpa-urdf-usd-xml` and re-attach
it to the KUKA flange.

## Setup

Same dependency tree as SimToolReal (IsaacGym Preview 4 + IsaacGymEnvs +
rl_games + warp-lang etc.); see SimToolReal's main README for install steps.

```bash
pip install -e .
```

## Quick start

### 1) Process a ManipTrans demo into a trajectory

```bash
# With ManipTrans data on disk:
python tools/process_maniptrans_trajectory.py \
    --maniptrans-root /home/intel/Codes/ManipTrans \
    --data-idx g0 \
    --src-fps 60 --target-fps 3 \
    --out data/trajectories/g0.json

# Without ManipTrans data (smoke test):
python tools/process_maniptrans_trajectory.py \
    --mock --out data/trajectories/example_g0.json
```

### 2) Visualise the cleaned trajectory

```bash
python tools/visualize_reference.py --traj data/trajectories/example_g0.json
# slider on the bottom; coordinate axes show object & wrist orientation
```

### 3) Train PPO

```bash
python isaacgymenvs/train.py \
    task=VideoRLFollower train=VideoRLFollowerLSTMAsymmetricPPO \
    headless=True num_envs=8192 \
    ++task.env.trajectoryPath=data/trajectories/example_g0.json
```

### 4) Train DAPG

You'll need a demo file `data/demos/<name>.pt` containing
`{"observations": (M, obs_dim), "actions": (M, act_dim)}` — typically collected
by rolling out an oracle policy or replaying a teleop session against the same
trajectory.

```bash
python isaacgymenvs/train.py \
    task=VideoRLFollower train=VideoRLFollowerLSTMAsymmetricDAPG \
    headless=True num_envs=4096 \
    ++task.env.trajectoryPath=data/trajectories/g0.json \
    ++train.params.config.dapg.demo_path=data/demos/g0_demo.pt
```

## Reward

```
r = r_object_keypoint                       (inherited from SimToolReal)
  + w_wrist_pos  * exp(-λ_pos  * ||wrist_pos - wrist_pos_goal||)
  + w_wrist_rot  * exp(-λ_rot  * geodesic_angle(wrist_quat, wrist_quat_goal))
  + w_fingertip  * exp(-λ_ftip * mean_k ||ftip_local_k - ftip_goal_local_k||)
```

Weights live in `cfg/task/VideoRLFollower.yaml::env.handPoseRewardScale`.
Defaults: `wristPos=1.0, wristRot=0.5, fingertip=2.0`.

## Trajectory JSON schema

```jsonc
{
  "meta": { "source": "...", "fps": 3.0, "n_fingertips": 5,
            "fingertip_order": ["thumb", "index", "middle", "ring", "pinky"] },
  "object": {
    "urdf_path": "data/objects/.../102_obj.urdf",
    "scale": 1.0,
    "start_pose": [x, y, z, qx, qy, qz, qw],
    "goals":      [[x, y, z, qx, qy, qz, qw], ...]                 // (T, 7)
  },
  "hand": {
    "wrist_goals":     [[x, y, z, qx, qy, qz, qw], ...],            // (T, 7)
    "fingertip_goals": [[[fx,fy,fz]*5], ...],                       // (T, 5, 3) world
    "fingertip_local": [[[fx,fy,fz]*5], ...]                        // (T, 5, 3) wrist-local
  }
}
```

## Limitations / known gaps

* Object asset loading still goes through SimToolReal's procedural primitive
  generator unless you also adapt `_load_main_object_asset` to honour
  `trajectory.object.urdf_path`.  The generated trajectory's `object.urdf_path`
  is currently informational only.
* MANO joint regressor isn't redistributed here; the trajectory pipeline
  approximates the wrist position via `data["rhand_transl"]` from the GRAB
  demo, which is close but not identical to ManipTrans's
  `j0 - 0.25*(j4 - j0)` shift.
* DAPG demo collection is **not** automated.  Provide your own .pt file or
  write a small `tools/collect_demo.py` that runs an oracle policy and dumps
  `(obs, action)` pairs.
* Single-hand only; bimanual support intentionally not added.
