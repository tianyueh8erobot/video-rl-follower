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

KUKA iiwa14 (7-DoF) + Sharpa Wave hand.  Sharpa's official URDF repo
(`sharpa-robotics/sharpa-urdf-usd-xml`) ships **hand-only** meshes/URDF — no
arm.  We splice the official ``right_sharpa_wave_with_flange.urdf`` onto
SimToolReal's KUKA iiwa14 chain via the helper script
``tools/build_iiwa14_right_sharpa_urdf.py``; the result lives at
``assets/urdf/kuka_sharpa_description/iiwa14_right_sharpa_adjusted_restricted.urdf``
and is the default loaded by ``cfg/task/VideoRLFollower.yaml::asset.robot``.

The base SimToolReal env auto-detects chirality from the substring
``right_sharpa`` / ``left_sharpa`` in the URDF path
(``env.use_right_sharpa`` / ``env.use_left_sharpa``); the right-hand
fingertip names (``right_index_DP``, …) and right-hand collision filter
(``RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS`` in ``adjacent_links.py``) are
already wired in upstream, so flipping the cfg is enough.

To rebuild the right-hand assembly URDF:

```bash
git clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml /tmp/sharpa_urdf
python tools/build_iiwa14_right_sharpa_urdf.py \
    --left-urdf  assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf \
    --right-urdf /tmp/sharpa_urdf/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.urdf \
    --right-meshes-src /tmp/sharpa_urdf/wave_01/right_sharpa_wave/meshes \
    --out-urdf  assets/urdf/kuka_sharpa_description/iiwa14_right_sharpa_adjusted_restricted.urdf \
    --out-meshes assets/urdf/kuka_sharpa_description/right_sharpa_meshes
```

> ⚠️  Visual validation needed: the wrist mount rpy is set to `(0, 0, +π/2)` as
> a mirror of the left hand's `(0, 0, -π/2)`.  Load the assembled URDF in
> IsaacGym at the all-zero joint pose and verify that the palm faces the
> intended direction; tweak the `sharpa_mount_to_right_flange` joint origin if
> not.

## Docs

- **[`docs/RETARGETING_AND_ENV_SETUP.md`](docs/RETARGETING_AND_ENV_SETUP.md)** —
  runbook for retargeting a GRAB clip, building the IsaacGym env, and the
  coordinate-frame definitions. Follow it when adding a new trajectory or
  swapping the robot/hand.
- **[`docs/ISAACGYM_GOTCHAS.md`](docs/ISAACGYM_GOTCHAS.md)** — read this
  before touching the IsaacGym task code. The top entry — IsaacGym sorts
  URDF DOFs alphabetically while PyBullet / pytorch_kinematics / MuJoCo
  preserve URDF DFS order — silently corrupts both visualisation AND
  training when a trajectory authored against PK order is fed straight
  into `set_dof_state`. Cost us a day; future-you will thank present-you.

## Setup

Same dependency tree as SimToolReal (IsaacGym Preview 4 + IsaacGymEnvs +
rl_games + warp-lang etc.); see SimToolReal's main README for install steps.

```bash
pip install -e .
```

## Quick start

### 1) Process a ManipTrans demo into a trajectory

You first need the upstream ManipTrans GRAB demo data on disk:

```bash
# 1. Clone ManipTrans (one-time)
git clone https://github.com/ManipTrans/ManipTrans /home/intel/Codes/ManipTrans

# 2. Download the GRAB demo from
#    https://1drv.ms/f/s!AgSPtac7QUbHgVE5vMBOAUPzxxsV
#    (linked from ManipTrans README §Prerequisites)
# 3. Extract under
#    /home/intel/Codes/ManipTrans/data/grab_demo/102/
#    so that data/grab_demo/102/102_sv_dict.npy and 102_obj.obj exist.
# 4. Copy the example URDF:
cp /home/intel/Codes/ManipTrans/assets/obj_urdf_example.urdf \
   /home/intel/Codes/ManipTrans/data/grab_demo/102/102_obj.urdf
```

Then convert to a 3 Hz JSON the env can consume:

```bash
# With ManipTrans data on disk → produces a real teapot trajectory + URDF link:
python tools/process_maniptrans_trajectory.py \
    --maniptrans-root /home/intel/Codes/ManipTrans \
    --data-idx g0 \
    --src-fps 60 --target-fps 3 \
    --out data/trajectories/g0.json

# Without ManipTrans data (smoke test, no real URDF):
python tools/process_maniptrans_trajectory.py \
    --mock --out data/trajectories/example_g0.json
```

> The env reads ``object.urdf_path`` from the JSON.  When it is set (real-data
> case), the env spawns that exact ManipTrans object and uses
> ``object.start_pose`` as the initial pose.  When it is null (mock case),
> the env falls back to SimToolReal's procedural primitives.

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

* MANO joint regressor isn't redistributed here; the trajectory pipeline
  approximates the wrist position via `data["rhand_transl"]` from the GRAB
  demo, which is close but not identical to ManipTrans's
  `j0 - 0.25*(j4 - j0)` shift.
* **Palm-center vs wrist origin**: SimToolReal's `fingertip_pos_rel_palm`
  is computed against `palm_center_pos`, which is the palm body position
  PLUS an offset (the "grip center").  The trajectory exporter currently
  expresses `wrist_goals` and `fingertip_local` relative to the MANO wrist
  joint (or `rhand_transl`), not the palm-center proxy.  This introduces a
  small constant translation bias in the wrist-position and fingertip-local
  rewards.  When you have a matching Sharpa hand mesh, the cleanest fix is
  to pick a hand-relative origin (e.g. middle MCP joint for the goal side
  and a measured offset for the env side) and align both before training.
* DAPG demo collection is **not** automated.  Provide your own .pt file or
  write a small `tools/collect_demo.py` that runs an oracle policy and dumps
  `(obs, action)` pairs.
* Single-hand only; bimanual support intentionally not added.
