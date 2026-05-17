# Retargeting & Environment Setup — runbook

How to take a GRAB demonstration, retarget it onto a robot+hand, and build
the IsaacGym tracking environment.  Follow this when **adding a new
trajectory** or **swapping the robot/hand**.

Read alongside `docs/ISAACGYM_GOTCHAS.md` — every pitfall referenced below
(`§1`..`§4`) is documented there in detail.

---

## 0. The pipeline at a glance

```
GRAB official .npz                         (human MANO hand + object 6-D pose)
   │
   │  tools/dextrack_sharpa/joint_29dof_ik.py
   │    - MANO forward kinematics → 24 keypoint world positions / frame
   │    - gradient IK: optimise N robot DOFs to track those keypoints
   │    - translate the whole clip into the robot workspace (xyz_offset)
   ▼
<subj>_<seq>_joint29_replay.npy            (robot DOF trajectory + object pose)
   │
   │  isaacgymenvs/tasks/dextrack_sharpa/trajectory.py  (DexTrackTrajectory)
   │    - load npy, FK every frame for link/wrist poses
   │    - DOF reorder  PK→IsaacGym   (§1)
   │    - object quat invert  R_obj→world → R_world→obj   (§4)
   ▼
DexTrackSharpa env  (env.py)               (residual-policy tracking task)
   │
   │  isaacgymenvs/train.py + cfg/task/DexTrackSharpa.yaml
   ▼
trained policy
```

Three artefact families, three things that can break: **the data
(`.npy`), the asset (`.urdf`), the conventions that glue them to
IsaacGym**.  Sections 2–4 cover each.

---

## 1. Coordinate frames — the single source of truth

There are **three** frames in play.  Getting one wrong is the most common
failure mode.

### 1.1 GRAB world frame
The raw `.npz` lives here.  `rhand.params` (MANO axis-angle + translation)
and `object.params` (`transl`, `global_orient` axis-angle) are both
expressed in this frame.  The numbers are wherever the mocap subject
happened to be — not aligned to any robot.

### 1.2 Robot workspace frame (IsaacGym world)
The IsaacGym scene is fixed:

| Actor | Pose | Set in |
|---|---|---|
| Robot base | `(0, 0, 0)`, identity rot | `env.py::_create_envs` `robot_pose` |
| Table centre | `(0.70, 0, 0.25)` → top surface at z=0.5 | `cfg` `env.table.pose` |
| Object (spawn) | table-centre + ½cube; **overwritten every frame** by the trajectory | `env.py::_create_envs` `object_pose` |

These are hard constants.  The robot base and table never move between
trajectories.

### 1.3 The bridge: `xyz_offset`
`joint_29dof_ik.py` translates the entire GRAB clip (hand **and** object,
by the *same* offset) so it lands in the robot workspace:

```python
wrist_mean   = joints16[:, 0].mean(0)        # MANO wrist averaged over all T frames
target_center = np.array([0.4, 0.0, 0.7])    # fixed anchor in robot frame
xyz_offset    = target_center - wrist_mean
mano_targets += xyz_offset                   # hand IK targets
obj_pos      += xyz_offset                   # object positions
```

So the **alignment anchor is the hand-wrist trajectory centroid → fixed
point `(0.4, 0, 0.7)`**.  The hand↔object *relative* geometry is the GRAB
original (same offset applied to both).  `xyz_offset` is stored in the npy
as `_xyz_offset` for traceability.

> ⚠️ **Multi-trajectory caveat.** The anchor is the *wrist centroid*, not
> the object and not the object-table relationship.  Across trajectories
> the table is fixed but the object's height is whatever GRAB recorded, so
> the object may float above / intersect the table differently per clip.
> If you train a generalist policy across many trajectories, see §6.

### 1.4 Rotation conventions (the subtle ones)

**DOF order** (`§1`): IsaacGym sorts a URDF's revolute joints
*alphabetically* within each subtree; PyBullet / pytorch_kinematics /
MuJoCo keep URDF DFS order.  `joint_29dof_ik.py` produces DOF in URDF DFS
order; `trajectory.py::_reorder_traj_to_isaacgym` permutes to IsaacGym
order on load.  Verify with the printout `DOF reorder PK→IG: arm
moved=…, hand moved=…`.

**Object quaternion** (`§4`): the npy stores `object_rot_quat` in the
GRAB convention (`R.from_rotvec(GRAB_axis_angle).as_quat()`, xyzw).
Empirically IsaacGym's root-state slot needs the *inverse*
(`R_world→obj`) for the cube to render the way the hand mocap expects.
`trajectory.py` inverts on load:

```python
obj_quat_np[:, :3] *= -1.0      # conjugate of a unit xyzw quat = its inverse
```

`obj_ang_vel` is still computed from the *raw* (non-inverted) GRAB
sequence — it is the cube's physical angular velocity in the world frame.

**Quaternion layout**: xyzw everywhere — scipy default, IsaacGym
root-state slot, `gymapi.Quat(x,y,z,w)`.  No wxyz anywhere in this repo.

---

## 2. Retargeting a NEW trajectory

### 2.1 Prerequisites (one-time per robot)
- Robot+hand URDF (see §4)
- Keypoint table `sharpa_mano_keypoints.npy` — maps 24 MANO keypoints to
  Sharpa mesh-vertex indices, built by
  `tools/dextrack_sharpa/build_keypoints.py`.  Robot-specific; rebuild
  when you swap the hand.
- MANO assets at `/home/intel/Codes/ManipTrans/data/mano_v1_2`.

### 2.2 Run the retargeter

```bash
cd /home/intel/Codes/video-rl-follower
/home/intel/miniconda3/envs/factory/bin/python \
  tools/dextrack_sharpa/joint_29dof_ik.py \
    --grab_npz   /home/intel/Codes/GRAB/grab__s2/s2/<object>_<action>_<idx>.npz \
    --out_dir    data/sharpa_retarget_dextrack \
    --urdf       assets/urdf/franka_sharpa_description/franka_panda_sharpa.urdf \
    --kp_table   tools/dextrack_sharpa/sharpa_mano_keypoints.npy \
    --start_frame 0 --skip 2 --max_frames 300 --iters 4000
```

Key arguments:

| Arg | Meaning | Notes |
|---|---|---|
| `--grab_npz` | raw GRAB clip | the **GRAB official** npz, not the grab_wuji repack |
| `--start_frame` | first GRAB raw frame | trims dead lead-in; DexTrack used 80 for cubesmall_inspect_1 |
| `--skip` | frame stride | `2` → 120 Hz mocap down to 60 Hz |
| `--max_frames` | cap after skip | 300 = our episode length |
| `--iters` | Adam IK iterations | 4000 is enough for <1 cm keypoint error |
| `--auto_xyz_offset` | on by default | anchors wrist centroid to `(0.4,0,0.7)` (§1.3) |
| `--xyz_offset x y z` | manual override | use if auto anchor pushes the clip out of reach |

Watch the log: final `kp_err mean` should be **< 1 cm**, `max` **< 3 cm**.
Higher means the hand can't reach the human pose — pick a different
`xyz_offset` or check the URDF.

### 2.3 Output `.npy` schema
`<subj>_<seq>_joint29_replay.npy` (a pickled dict):

| Key | Shape | Meaning |
|---|---|---|
| `robot_delta_states_weights_np` / `hand_qs` | (T, N) | per-frame robot DOF, **URDF-DFS order** |
| `object_transl` | (T, 3) | object position, robot-workspace frame |
| `object_rot_quat` | (T, 4) | object orientation, xyzw, **GRAB convention** (`R_obj→world`) |
| `_xyz_offset` | (3,) | the translation applied (provenance only) |
| `_per_kp_err_cm` | (T, 24) | per-keypoint IK error in cm |

> The npy keeps the GRAB convention untouched so it stays interoperable
> with other consumers; the IsaacGym-specific reorder + quat-invert happen
> in `trajectory.py` at load time, not in the file.

### 2.4 Point the env at it
Edit `cfg/task/DexTrackSharpa.yaml`:
```yaml
env:
  trajectory:
    npy_path: ".../data/sharpa_retarget_dextrack/<subj>_<seq>_joint29_replay.npy"
  episodeLength: <T>          # match the trajectory length
```
Object size also needs to match the real GRAB object (currently a 5 cm
cube hard-coded in `env.object.size`); for a non-cube object you must
load its mesh (see §6).

### 2.5 Sanity check before training
```bash
PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/visualize_kinematic_replay.py
```
Press `P`.  The hand should track the object cleanly with no twist and
the object should follow the fingers without jitter.  If it doesn't, the
diagnostics in `docs/ISAACGYM_GOTCHAS.md §5` isolate which stage broke.

---

## 3. How the environment is built

`env.py::DexTrackSharpa` inherits `VecTask`.  Build order:

1. **`create_sim`** → ground plane → `_create_envs`.
2. **`_create_envs`**: load robot asset (`DOF_MODE_POS`, force sensors on),
   `create_box` for object (density-based mass) and table (fixed base).
   Per-env: spawn robot at origin, table at `(0.70,0,0.25)`, object above
   the table.  Cache global actor indices.
3. **Acquire sim tensors**: `root_states`, `dof_state`, `rigid_body_state`,
   `dof_force` — wrapped as torch views over GPU buffers.
4. **Load trajectory** (`DexTrackTrajectory`), then
   `_reorder_traj_to_isaacgym` (§1.4).
5. **Initial reset** to frame 0.

Control loop (`pre_physics_step`): accumulating residual on the kinematic
reference —
`cur_delta_warm += speed_scale·dt·action·2; target = ref[t+1] + cur_delta_warm`.

Key cfg knobs (`cfg/task/DexTrackSharpa.yaml`), all aligned to DexTrack's
`run_tracking_headless_grab_single_wfranka.sh`:

| Group | Values |
|---|---|
| Residual | `dofSpeedScale 20`, `frankaDeltaDeltaMultCoef 2` (arm scale = 40, hand = 20) |
| PD gains | arm `400/80`, hand `100/4` |
| Object | `density 500` → mass = density × mesh volume |
| Table | size `(1,1,0.5)`, centre `(0.70,0,0.25)` |
| Reset | `randomTime: True` (one-shot scatter on first reset only) |

PPO hyperparameters live in `cfg/train/DexTrackSharpaPPO.yaml` (mirrors
DexTrack `HumanoidPPOSupervised.yaml`: 7-layer MLP `[8192…128]`,
`lr 5e-4`, `kl 0.008`, `horizon 32`).

---

## 4. Swapping the robot / hand

When the URDF changes, redo this checklist:

1. **Assemble the URDF.** Splice the hand onto the arm.  For the current
   Franka+Sharpa rig the helper is `tools/build_iiwa14_right_sharpa_urdf.py`
   (KUKA variant) / the franka equivalent.  ⚠️ Verify the wrist mount
   `rpy` — load at zero pose in IsaacGym and check the palm faces the
   intended direction.
2. **Count DOFs.** Update `NUM_ARM_DOFS` / `NUM_HAND_DOFS` / `NUM_DOFS` at
   the top of `env.py`.  `joint_29dof_ik.py` and `trajectory.py` assert
   the count — they will fail loudly if it's wrong.
3. **Rebuild the keypoint table.** `build_keypoints.py` picks hand
   mesh-vertex indices that correspond to the 24 MANO keypoints.  Hand-
   specific — must be redone for a new hand. ★ visually verify the
   keypoint placement on the mesh.
4. **Check the DOF order.** Run any viz script and read the
   `DOF reorder PK→IG` line.  IsaacGym's alphabetic sort (§1) is
   robot-specific; the permutation is rebuilt automatically from joint
   names, but confirm `set(pk_names) == set(ig_names)` doesn't assert.
5. **Body-name constants.** `SHARPA_BODY_NAMES`, `WRIST_LINK`,
   fingertip indices in `trajectory.py` are hand-specific — update them.
6. **PD gains & joint limits.** New arm/hand → new `armStiffness` etc. in
   the cfg; joint limits are read from the URDF automatically.
7. **Re-retarget every trajectory.** Keypoint targets are the same (MANO),
   but the IK solves against the new chain — all npys must be regenerated.

---

## 5. Quick reference — "I want to …"

| Goal | Do this |
|---|---|
| Add a trajectory, same robot | §2.2 retarget → §2.4 point cfg → §2.5 verify |
| Swap the robot/hand | §4 checklist (then re-retarget everything) |
| Replay-check a trajectory | `visualize_kinematic_replay.py` |
| Debug a broken replay | `docs/ISAACGYM_GOTCHAS.md §5` diagnostic scripts |
| Compare against LEAP data | `visualize_hybrid_replay.py` |

---

## 6. Known limitations & open questions

These are **not solved** — flag them before relying on the pipeline
beyond a single trajectory.

1. **Multi-trajectory semantic alignment (§1.3).** The alignment anchor is
   the wrist centroid, so the object's pose relative to the (fixed) table
   varies per clip.  For a generalist policy across many trajectories the
   "table" is not a semantically consistent entity.  Options: anchor on
   the object's initial pose → a fixed table-surface point instead of the
   wrist centroid; or drop the table if it's not load-bearing for the
   task.  **Decide before multi-trajectory training.**

2. **Observation contains absolute coordinates.** The proprio obs includes
   `object_pos(3)` in absolute world coordinates.  Across trajectories the
   object spans a wide absolute range — a generalisation risk.  Prefer
   fully relative obs (object-relative-to-palm, `delta_qpos`, future
   reference frames).  Cross-check against DexTrack's
   `run_tracking_headless_grab_multiple*` before finalising.

3. **Object asset is a hard-coded 5 cm cube.** `env.py` calls
   `create_box`.  Non-cube GRAB objects (apple, mug, …) need real mesh
   loading.  The symmetric cube also *hides* orientation bugs — see
   point 4.

4. **`object_rot_quat` invert — root cause unconfirmed.** The fix (§1.4,
   `§4`) is empirically verified (A/B test in the viewer) but the
   *reason* IsaacGym wants `R_world→obj` here is still inferred.  Most
   likely the GRAB `object.global_orient` axis-angle is not the
   `R_obj→world` active rotation we assumed.  To close this, read GRAB's
   official object-pose loading code.  Until then: the cube is symmetric
   so this never bites visually for cube clips, but a marked / asymmetric
   object would expose any residual error — re-verify when you add one.
