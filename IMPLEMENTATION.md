# video-rl-follower — Implementation notes

End-to-end pipeline for training a Sharpa Wave dexterous hand to follow a
human-demonstration reference trajectory (object 6D pose + MANO 21
keypoints), built on top of `SimToolReal` and ManipTrans.

This document explains **what was built and why**, organised around the
questions that came up during development.  The audit/review trail is in
`AUTO_REVIEW.md`.

## Contents

1.  Architecture overview
2.  Data pipeline (OakInk-V2 → MANO → Sharpa retarget → JSON)
3.  ManipTrans vs SimToolReal — what each contributes
4.  Sharpa hand mapping (the `to_hand` correspondence problem)
5.  Reward design (final hybrid)
6.  Trajectory time-axis & sub-goal advancement
7.  Reset & hand DOF warm-start
8.  Reference data and training trajectories
9.  Visualisation
10. Validation history (the OakInk quaternion bug)

---

## 1. Architecture overview

```
                ┌────────────────────────────┐
                │ OakInk-V2 anno_preview pkl │  (HF: kelvin34501/OakInk-v2)
                │ raw_mano + obj_transf      │
                └───────────────┬────────────┘
                                │
                                ▼
              tools/oakink_extract_pickup.py  (on /tmp)
              + Y-up → Z-up + segment selection
                                │
                                ▼
              data/grab_demo/102/  (hot-swap into ManipTrans GRAB loader slot)
                                │
                                ▼
       ManipTrans/main/dataset/mano2dexhand.py --dexhand sharpa
              (Adam IK over Sharpa wrist 6D + 22 DoF, 4000 iter)
                                │
                                ▼
       data/oakink_subset/<seq>/sharpa_rh_retarget.pkl
                                │
                                ▼
       tools/process_maniptrans_trajectory.py   (mujoco2gym + table_z transform)
                                │
                                ▼
       tools/merge_retarget_into_trajectory.py  (paste pkl into JSON)
                                │
                                ▼
       data/trajectories/<seq>_with_dex.json    (single source of truth for env)
                                │
                                ▼
       isaacgymenvs/tasks/video_rl_follower/env.py  (RL env)
              + ManipTrans Stage-1 R_imit
              + SimToolReal sparse goal bonus
              + ManipTrans Stage-2 reset (opt_dof_pos warm-start)
              + sub_goal_idx aligned to reset frame
                                │
                                ▼
       PPO via rl_games  →  policy.pth
                                │
                                ▼
       scripts/visualize_trajectory_in_isaacgym.py  (offline replay overlay)
```

## 2. Data pipeline

**OakInk-V2 anno_preview pkl** (we earlier mistook it for "preview-quality"
MANO; actually contains the *full* `raw_mano`, `raw_smplx`, `obj_transf`,
camera intrinsics — only the RGB videos are excluded).  Each scene pkl is
~100 MB; HF mirror has 627 scenes.

The MANO data per frame:
```
rh__pose_coeffs : (1, 16, 4)   # quaternions in [w, x, y, z] format ★
rh__tsl         : (1, 3)        # wrist position (matches J_regressor[0] when
                                  ManoLayer is built with center_idx=0)
rh__betas       : (1, 10)       # shape coefficients
```

★ The quaternion order is `wxyz`, NOT `xyzw`.  scipy's `Rotation.from_quat`
expects `xyzw`, so any extraction code MUST permute first
(`pose_coeffs[..., [1,2,3,0]]`).  This was the root cause of an early bug
that made fingertip-to-mesh distance look ~25 cm; the correct value for a
held object is ~1.6 cm.

The extractor (`/tmp/oakink_extract_pickup.py`):
- Picks an `(rh_main, primitive)` segment from `program_info/<seq>.json`
- Sub-samples by SKIP (default 2) and clips to `FRAME_LIMIT`
- Converts `pose_coeffs` (wxyz quat) → axis-angle
- Runs `ManoLayer(center_idx=0, use_pca=False, ncomps=45, flat_hand_mean=True)`
  to recover 778 verts in world frame (verts = MANO + rh_tsl)
- Rotates the whole scene (verts + obj 6D) by R_x(+π/2) to convert
  MANO Y-up → ManipTrans Z-up
- Writes a `sv_dict.npy` in QuasiSim/GRAB format that the unmodified
  ManipTrans GRAB loader (`grab_dataset_dexhand.py`) can read

For each new sequence we hot-swap the OakInk-derived `sv_dict.npy` +
`102_obj.{obj,urdf,npy}` into `ManipTrans/data/grab_demo/102/`, run
`mano2dexhand.py`, then restore the original GRAB g0 files.

The processed JSON has three top-level blocks:
```
object  : urdf_path, start_pose, goals[T,7], grasp_bbox_scale
hand    : wrist_goals[T,7], joints_world[T,21,3], joints_local, fingertip_*
dex     : link_names[28], links_world[T,28,3], dof_pos[T,22], wrist_pos/rot
```

## 3. ManipTrans vs SimToolReal — what each contributes

We built `VideoRLFollower(SimToolReal)`.  SimToolReal provides the IsaacGym
infrastructure (Kuka 7-DoF arm + multi-finger hand, asymmetric critic with
state augmentation, action delay simulation, sparse-goal-with-dwell
mechanics).  ManipTrans provides the imitation-learning pieces (per-link
MANO target distance reward, opt_dof_pos warm-start at reset).

| feature | SimToolReal | ManipTrans Stage-1 / 2 | new env |
|---|---|---|---|
| **Embodiment** | Kuka + Inspire/Sharpa | Floating-base hand only | Kuka + Sharpa (kept) |
| **Goal recipe** | Sub-goal pose + dwell N + tolerance curriculum | Frame-locked time-driven | sub-goal pose + dwell (SimToolReal style) |
| **Dense MANO tracking** | none | wrist + 5 fingertips + 17 mid-joints + power, 14 terms | 9 of 14 (position/orientation only) |
| **Dense object shaping** | lifting + keypoint + ftip-delta | none | none (deleted per user) |
| **Sparse success bonus** | reach_goal_bonus | none | yes (kept) |
| **Reset DOF init** | default + noise | opt_dof_pos[seq_idx] | opt_dof_pos[seq_idx] (kept) |
| **Reset wrist init** | random | opt_wrist_pos[seq_idx] | SimToolReal random (no IK to opt_wrist) |
| **Action regularization** | action²-penalty | power-penalty (≈ same purpose) | SimToolReal (kept) |
| **Object velocity penalty** | yes (default 0) | no | yes (kept, default 0, opt-in) |

ManipTrans does NOT have a published Sharpa baseline; the original paper
covers Inspire, Shadow, Allegro, ArtiMano, XHand, InspireFtp.  Sharpa is
new for this codebase; the `sharpa.py` config file mirrors Inspire/Shadow's
design conventions for the `to_hand` mapping and `weight_idx` grouping.

## 4. Sharpa hand mapping — the `to_hand` correspondence problem

Sharpa Wave has 28 rigid-body links and 22 actuated joints.  MANO has 21
keypoints (16 J_regressor joints + 5 fingertip vertices).  These are
**not 1-to-1** — Sharpa's thumb has 6 segments where MANO has 4 keypoints,
and the pinky has an extra CMC joint.

The `hand2dex_mapping` in `sharpa.py` resolves this by mapping 1 MANO
keypoint to *multiple* Sharpa links when needed:

```
wrist                ← right_hand_C_MC                                    (1:1)
thumb_proximal       ← right_thumb_CMC_VL + right_thumb_MC + right_thumb_MCP_VL (1:3)
thumb_intermediate   ← right_thumb_PP                                     (1:1)
thumb_distal         ← right_thumb_DP                                     (1:1)
thumb_tip            ← right_thumb_fingertip                              (1:1)
pinky_proximal       ← right_pinky_MC + right_pinky_MCP_VL + right_pinky_PP (1:3)
... (other 4 fingers: 1:2 for proximal, 1:1 elsewhere)
```

This is the same convention Inspire/Shadow use for their multi-link thumb
chains.  Validation (`tools/validate_sharpa_mapping.py` against GRAB g0 +
OakInk b7853 retarget pkls) shows:

- 16 / 28 links land within nearest-MANO-keypoint distance
- 6–7 / 28 ambiguous (within 5 mm of nearest)
- 5–6 / 28 are 1–2 cm "off" — **structural**, not a mapping bug:
  - `right_thumb_CMC_VL` and `right_thumb_MC` physically sit near the
    wrist (MANO has no thumb-CMC keypoint), so they end up ~2 cm short of
    `thumb_proximal`.  Trying to remap them to `wrist` triggers an
    IndexError in the retargeter (it hard-assumes exactly 1 wrist link).
  - In curled grasp poses, adjacent finger joints are physically next to
    each other, so the "nearest MANO keypoint" can be a different finger.
    Pose-specific, not mapping-specific.

The `weight_idx` grouping for the reward:
```
thumb_tip      = [27]                                # body_names[27] = thumb_fingertip
index_tip      = [5]
middle_tip     = [10]
ring_tip       = [21]
pinky_tip      = [16]
level_1_joints = [1, 6, 11, 17, 22]                  # MCP_VL of each finger
level_2_joints = [2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26]
```

These integers index into the 28-link `body_names` (0-based with root).
The reward implementation subtracts 1 to convert to indices into the
27-non-wrist target tensor (the paper does the same trick because
`joints_state[:, 1:, ...]` skips the root).

## 5. Reward design (final hybrid)

```
r_total(t)  =  R_imit(t)              # ManipTrans Stage-1, paper coefficients
            +  R_goal_sparse(t)       # SimToolReal sparse bonus (kept)
            +  R_act_penalty(t)       # SimToolReal action regularization (kept)
            +  R_obj_vel_pen(t)       # SimToolReal anti-yeet (cfg default 0)
```

**R_imit** (9 of 14 paper terms, coverage 5.15 / 6.40 ≈ 80.5%):
```
R_imit  =  0.10 · exp(-40 · ‖p_eef          - p_eef_goal‖)
        +  0.60 · exp(- 1 · angle(q_eef     , q_eef_goal))
        +  0.90 · exp(-100 · d̄_thumb_tip)
        +  0.80 · exp(- 90 · d̄_index_tip)
        +  0.75 · exp(- 80 · d̄_middle_tip)
        +  0.60 · exp(- 60 · d̄_ring_tip)
        +  0.60 · exp(- 60 · d̄_pinky_tip)
        +  0.50 · exp(- 50 · d̄_level_1)
        +  0.30 · exp(- 40 · d̄_level_2)
        # NOT YET implemented (need target velocities + dof_force tensor):
        # + 0.10·exp(-1·‖v_eef-v̂‖)
        # + 0.05·exp(-1·‖ω_eef-ω̂‖)
        # + 0.10·exp(-1·‖q̇-q̂̇‖)
        # + 0.50·exp(-10·P_total)
        # + 0.50·exp(-2·P_wrist)

# d̄_<group> = mean per-link distance over the group
# eef = palm_center (the env's wrist proxy, consistent with fingertip_pos_rel_palm)
```

The MANO target for each non-wrist Sharpa link comes from the `to_hand`
mapping: for body index i in [1, 27], target = `joints_world[t, MANO_NAME_TO_IDX[SHARPA_DEX2MANO_NAME[i]]]`.
Multiple Sharpa links can share the same MANO target (1-to-many mapping).

**R_goal_sparse** (SimToolReal):
```
near_goal      = (keypoints_max_dist(t) ≤ tolerance(t))
near_goal_steps = (near_goal_steps + near_goal) · near_goal     # consecutive
is_success      = (near_goal_steps ≥ N)
R_goal_sparse(t) = is_success(t) · reach_goal_bonus
# tolerance shrinks via curriculum (toleranceCurriculumIncrement, Interval)
```

**R_act_penalty + R_obj_vel_pen**: standard SimToolReal `_action_penalties`
and `object_lin/ang_vel_penalty`, weights from cfg.

**Strict-correctness assertions** (no silent NaN swallowing):
- raises if r_imit OR combined reward is non-finite (escape: cfg.env.allowNonfiniteReward=True)
- raises at __init__ if useRetargetDofInit=True AND dex_dof_pos is None
- raises at reset_idx if dex_dof_pos.shape[1] != num_hand_dofs
- raises at _setup_imit_link_indices if any of 28 Sharpa body_names is missing in the URDF

**Removed (per user request — drop pickup dense shaping):**
- liftingRewScale (was 20.0)
- liftingBonus (was 300.0)
- keypointRewScale (was 200.0)
- distanceDeltaRewScale (was 50.0)

These cfg keys are zeroed in `VideoRLFollower.yaml`, so the parent
`compute_kuka_reward()` still runs but contributes only the sparse bonus
+ action penalties + (optional) velocity penalties.

## 6. Trajectory time-axis & sub-goal advancement

This is the most subtle design decision in the env.  The two parents
disagree on how trajectory progress is driven:

- **ManipTrans Stage-1**: time-locked.  `progress_buf` increments by 1
  every sim step; the MANO target for step t is `trajectory[t]`.
- **SimToolReal**: success-driven.  Each sub-goal stays active until it
  is reached (within tolerance for N consecutive steps); the sub-goal
  index then advances by 1.

We picked SimToolReal style (Option A in the design doc): a single
per-env buffer `sub_goal_idx[env]` is the **sole source of truth** for
which trajectory frame is currently the goal — both `R_imit` and
`R_goal_sparse` read it.  `successes` is decoupled from goal indexing
and is a pure stats counter.

```
on episode reset (reset_idx override):
  seq_idx[env]      = randint(0, T)  if randomStateInit else 0
  sub_goal_idx[env] = seq_idx[env]                         # ★ aligned
  arm_hand_dof_pos[env, 7:7+22] = opt_dof_pos[seq_idx]     # warm-start

on sub-goal success (in compute_kuka_reward, AFTER R_imit + reward write,
                     BEFORE populate_obs runs in the same post_physics_step):
  sub_goal_idx[env] = (sub_goal_idx[env] + 1) % T          # ★ advance
  successes[env]   += 1                                     # stats
  refresh _wrist_goal, _fingertip_goal_local, goal_states[env, 0:7]
                                                            # so policy obs sees fresh goal
```

The advance is placed in `compute_kuka_reward` (not in `_reset_target`)
so that the very next `populate_obs_and_states_buffers` call in the same
post_physics_step packs the *new* hand-goal into the obs.  Without this,
the policy would receive the just-achieved goal in its obs for one
control step, decide its action against the stale goal, and only then
see the new goal — a 1-step "off-by-one" that costs PPO a lot of wasted
exploration.

`_reset_target` was simplified to a pure copy: it reads `sub_goal_idx[env]`
and writes `goal_states[env, 0:7]`.  Both call sites (episode-reset path
with `is_first_goal=True`, success-path with `is_first_goal=False`)
behave identically.

## 7. Reset & hand DOF warm-start

ManipTrans Stage-2 (`dexhandmanip_sh.py:1080-1100`) initializes the
dexhand at episode reset by reading `opt_dof_pos[env, seq_idx]` from the
retarget pkl.  We replicate this in `reset_idx`:

```
n_arm = 7
hand_dof = clamp(opt_dof_pos[seq_idx], dof_lower, dof_upper)
arm_hand_dof_pos[env, 7:7+22] = hand_dof
arm_hand_dof_vel[env, 7:7+22] = 0
prev/cur_targets[env, 7:7+22] = hand_dof
```

The arm DOFs (slots 0:7) are NOT IK-solved to `opt_wrist_pos[seq_idx]` —
ManipTrans uses a floating-base hand and can directly set the wrist pose,
but our env keeps the Kuka arm so the wrist pose is reachable only via
joint-space IK that we don't currently run.  The arm starts at
SimToolReal's randomized default + noise.  This is a documented limitation;
RL will learn to reach the wrist target via the arm.

`useRetargetDofInit=True` HARD-REQUIRES the trajectory to have a
`dex.dof_pos` block of the right shape — silent shape truncation would
break the warm-start invisibly.  Disable via cfg if no retarget pkl is
available.

## 8. Reference data and training trajectories

| trajectory | source | frames @ Hz | contact | Sharpa retarget loss | use case |
|---|---|---|---|---|---|
| `data/grab_demo/102/` (g0) | QuasiSim GRAB demo (rotating mouse) | 60 @ 60 Hz | n/a (rotation) | ~2.5 cm | quick-start, paper baseline |
| `oakink_b7853_3_rh_with_dex.json` | OakInk-V2 b7853 scene, rh#3 "rearrange" segment | 60 @ 60 Hz | 1.6 cm fingertip-mesh | 2.5 cm | main ref, paper-aligned |

OakInk-V2 anno_preview pkls live at
`/home/intel/Codes/ManipTrans/data/OakInk-v2/anno_preview/`; we have 2
scenes locally (~190 MB total) but the HF mirror has 627 scenes (~60 GB
full).  Download more via:
```
curl -OL "https://huggingface.co/datasets/kelvin34501/OakInk-v2/resolve/main/anno_preview/<scene_pkl>"
```

The DexManipNet release (`data/DexManipNet/`, ~8 GB) contains
*pre-trained Inspire policy rollouts* for 77 OakInk-V2 sequences — useful
as paper baselines but **NOT raw MANO source** (no `raw_mano` field, just
executed `q_rh` and `state_rh`).  ManipTrans paper trains by reading
OakInk-V2 anno_preview directly, not DexManipNet.

## 9. Visualisation

`scripts/visualize_trajectory_in_isaacgym.py` is a kinematic-only viewer
that supports overlaying multiple hands on the same trajectory:

```
python scripts/visualize_trajectory_in_isaacgym.py \
    --traj data/trajectories/oakink_b7853_3_rh_with_dex.json \
    --hz 30 \
    --overlay-pkl /path/to/inspire_rh_retarget.pkl \
    --overlay-urdf /path/to/inspire_hand_right.urdf
```

Renders:
- orange object mesh from `object.urdf_path`
- 21 colour-coded MANO joint markers + 20 bones (`hand.joints_world`)
- light-blue Sharpa hand (28 links, 22 DoF) driven by `dex.dof_pos`
- light-green overlay hand (e.g. Inspire 18 links, 12 DoF) from `--overlay-pkl`

CLI flags:
- `--no-mano`     hide MANO ground-truth markers
- `--no-sharpa`   hide Sharpa, useful for comparing other dexhands only
- `--auto`        headless self-checks (no viewer window)
- `--hz <Hz>`     playback rate
- `--no-loop`     don't restart at frame 0 after the last frame

## 10. Validation history (the OakInk quaternion bug)

Early extraction code parsed `rh__pose_coeffs` with `Rotation.from_quat`
(scipy default `xyzw`).  OakInk-V2 stores quaternions as `wxyz`.  The
silent reorder produced rotations off by an arbitrary roll, which made:

- fingertip-to-mesh distance show as ~25 cm at all times (hand floating
  away from the held object)
- "wrist drift" between MANO and Sharpa retarget show as 90+ cm
- OakInk anno_preview look like it had "loose contact data"

After fixing (`pose_coeffs[..., [1,2,3,0]]`) and rebuilding the
ManoLayer with `center_idx=0` (so `rh_tsl` becomes the wrist position
exactly), the fingertip-to-mesh distance dropped to **1.6 cm during
motion**, confirming OakInk-V2's MANO is contact-fitted and is the right
training signal.

The buggy intermediate trajectory products
(`oakink_take_outside_spoon[_with_dex].json`,
`oakink_grip_pickup[_with_dex].json`) have been deleted; only the
quaternion-correct `oakink_b7853_3_rh_with_dex.json` remains.

---

## File map

```
isaacgymenvs/
├── tasks/video_rl_follower/
│   ├── env.py                     # the RL env (this is where the reward lives)
│   └── trajectory.py              # JSON loader + dataclass
└── cfg/task/VideoRLFollower.yaml  # cfg defaults

tools/
├── process_maniptrans_trajectory.py  # MANO + obj → JSON (mujoco2gym + table_z)
├── merge_retarget_into_trajectory.py # paste retarget pkl into JSON
├── validate_sharpa_mapping.py        # check hand2dex correspondence
└── build_iiwa14_right_sharpa_urdf.py # robot URDF assembly

scripts/
└── visualize_trajectory_in_isaacgym.py

assets/urdf/kuka_sharpa_description/
└── iiwa14_right_sharpa_adjusted_restricted.urdf

data/trajectories/
├── oakink_b7853_3_rh_with_dex.json   # paper-aligned, contact-validated (★ recommended)
├── grab_g0_validate_with_dex.json    # GRAB g0 baseline
└── ...
```

External references (not in this repo):
```
/home/intel/Codes/ManipTrans/                    # ManipTrans codebase
├── maniptrans_envs/lib/envs/dexhands/sharpa.py  # Sharpa config
├── main/dataset/mano2dexhand.py                 # retargeting entry point
├── main/dataset/grab_dataset_dexhand.py         # the GRAB loader we hot-swap into
└── data/
    ├── OakInk-v2/anno_preview/                  # downloaded scene pkls
    ├── grab_demo/102/                           # GRAB g0 + hot-swap slot
    ├── oakink_subset/<seq>/                     # extracted OakInk segments
    └── retargeting/grab_demo/mano2sharpa_rh/    # working dir for mano2dexhand output
```
