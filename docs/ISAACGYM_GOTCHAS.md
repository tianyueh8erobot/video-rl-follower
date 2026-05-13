# IsaacGym gotchas (learned the hard way)

## 1. IsaacGym reorders DOFs alphabetically — PyBullet / pytorch_kinematics / MuJoCo do not

**The trap.** Three otherwise interchangeable URDF consumers ALL load Sharpa
in URDF DFS order (thumb → index → middle → ring → pinky), while IsaacGym
silently sorts joints within each subtree **alphabetically** (index →
middle → pinky → ring → thumb). 18 of 22 hand DOFs end up at different
indices.

|                      | DOF order                                       |
|----------------------|-------------------------------------------------|
| URDF file            | DFS (thumb, index, middle, ring, pinky)         |
| PyBullet             | DFS — matches URDF file                         |
| pytorch_kinematics   | DFS — matches URDF file                         |
| MuJoCo               | DFS — matches URDF file                         |
| **IsaacGym**         | **alphabetic** (index, middle, pinky, ring, thumb) |

**Why it's invisible.** If you do the *whole* pipeline in PyBullet (or
pytorch_kinematics for FK, or `pybullet_replay.py` for video), the data
order matches the consumer order, so nothing ever looks wrong. Only when
you cross the boundary — e.g. retargeting in pytorch_kinematics, then
running the policy in IsaacGym — do you get "twisted hand" artifacts
that look like FK bugs or contact instability but are pure index mismatch.

**Concrete symptom we hit.** Replay in IsaacGym showed contorted finger
poses + the cube spinning wildly. Same `hand_qs` rendered cleanly by
`/home/intel/Codes/DexTrack/tools/pybullet_replay.py`. Root cause was 18
swapped joint indices feeding into `set_dof_state_tensor`.

**Fix pattern.** Right after the trajectory loads and after IsaacGym actor
creation, build a `pk_to_ig` permutation by joint name and reorder once:

```python
ig_name_to_idx = self.gym.get_actor_dof_dict(env_ptr, robot_actor)
pk_names = self.traj.joint_names           # PK / URDF DFS order
pk_name_to_idx = {n: i for i, n in enumerate(pk_names)}
ig_idx_to_name = {v: k for k, v in ig_name_to_idx.items()}
pk_to_ig = [pk_name_to_idx[ig_idx_to_name[i]]
            for i in range(len(ig_name_to_idx))]
perm = torch.tensor(pk_to_ig, dtype=torch.long, device=self.device)
self.traj.dof_pos = self.traj.dof_pos[:, perm].contiguous()
self.traj.dof_vel = self.traj.dof_vel[:, perm].contiguous()
```

See `isaacgymenvs/tasks/dextrack_sharpa/env.py::_reorder_traj_to_isaacgym`.

**Sanity check on every new robot.** Print both lists side-by-side and
assert each IsaacGym index maps to an existing PK index. The reorder
function above logs how many joints moved; if `arm moved=0/7` but
`hand moved=18/22` you know it kicked in.

**This bug breaks training too**, not just visualization. The PD targets
written to IsaacGym were thumb angles landing on index joints, etc., so
the policy was learning a residual on top of completely wrong reference
DOFs. Always verify reorder before claiming training is "working".


## 2. IsaacGym kinematic replay needs `simulate()` every frame

`step_graphics(sim)` ONLY reflects state from the most recent
`simulate(sim)` call. `set_dof_state_tensor_indexed` / `set_actor_root_state_tensor_indexed`
update the simulator's internal state but the renderer ignores those
writes until simulate runs.

**Verified empirically:**
- `set_state(0); render(); set_state(100); render()` without simulate:
  pixel diff = 0.00 (renderer shows frame 0 both times)
- Same with `simulate()` between set and render: pixel diff ≈ 11.8 ✓
- `simulate(); set_state(150); render()`: renders frame 0 (set-after-simulate
  doesn't reach the renderer)

So a correct per-frame kinematic-replay loop is **set → simulate → render**:

```python
def set_state(t):
    # 1) DOF: position + ZERO velocity
    env.dof_state[0, :, 0] = ref_dof
    env.dof_state[0, :, 1] = 0.0
    gym.set_dof_state_tensor_indexed(...)
    # 2) PD targets in sync with state — without this, PD with target=0
    #    yanks every hand joint to zero in one dt = "twisted hand" artifact
    env.cur_targets[0, :] = ref_dof
    gym.set_dof_position_target_tensor(...)
    # 3) Object root: pos + quat + ZERO lin/ang velocity
    env.root_states[obj_idx, 7:13] = 0.0
    gym.set_actor_root_state_tensor_indexed(...)

# Per frame:
set_state(t)
gym.simulate(sim)       # near-noop because vel=0, gravity=0, PD synced
gym.fetch_results(sim, True)
gym.step_graphics(sim)
gym.draw_viewer(viewer, sim, True)
```

Also set sim `gravity = [0, 0, 0]` in the replay config so simulate stays
idempotent under the forced state.

Reference: `isaacgymenvs/tasks/dextrack_sharpa/visualize_kinematic_replay.py`
and ManipTrans `DexManipNet/dexmanip_sh.py::play()` (same pattern).


## 3. IsaacGym has no "kinematic body" — disable contact for replay

PyBullet's `createMultiBody(baseMass=0.0, ...)` makes a true kinematic
body that ignores physics forces.  Combined with `setGravity(0,0,0)` and
**never calling `stepSimulation()`**, you can `resetBasePositionAndOrientation`
every frame and the renderer shows exactly that pose.  This is how
`pybullet_replay.py` produces clean trajectory videos.

IsaacGym has no equivalent flag.  Any actor with finite density is a
dynamic body, and `simulate()` (which we MUST call per §2) runs the
contact solver every frame.  Retargeted finger grasps typically have
1-5 mm of interpenetration with the object, and the contact solver
turns that into a large reaction impulse — within one dt=1/60 s the
cube can rotate >40°.  Same `set_actor_root_state` written next frame
gets overwritten by physics again, so visually the cube wobbles
continuously.

**Fix for replay: disable cube↔robot collision via filter bit.**
IsaacGym's per-shape `filter` is a 32-bit mask; if two shapes share
ANY bit, the contact solver skips that pair.

```python
NO_COLLIDE_BIT = 1 << 30     # stay within signed int32 range
for actor_name in ("object", "robot"):
    h = gym.find_actor_handle(env_ptr, actor_name)
    props = gym.get_actor_rigid_shape_properties(env_ptr, h)
    for p in props:
        p.filter = int(p.filter) | NO_COLLIDE_BIT
    gym.set_actor_rigid_shape_properties(env_ptr, h, props)
```

Verified with `diag_no_contact.py`: max obj-quat drift drops from
0.7526 (contact on) → **0.0000** (contact off) across the full 300-frame
replay.  Cube↔table contact is preserved (table.filter still 0).

**Do NOT do this in training.**  This trick is replay-only — the policy
needs real finger-object contact for grasping.  Training-time
trajectory bursts (e.g. small retargeting overlaps causing a brief
contact spike at frame 0) are handled instead by the `randomTime`
scattering and PD smoothing in `pre_physics_step`.

## 4. DexTrack-shipped LEAP `object_rot_quat` is inverted vs GRAB

(Not strictly an IsaacGym pitfall, but the bug surfaced during a sanity
check against the LEAP reference and is worth recording so future devs
don't conclude "our data is broken" by comparing to LEAP.)

For `s2_cubesmall_inspect_1`:

| Source | Conversion | Δ vs GRAB rotation |
|---|---|---|
| GRAB original `obj.params.global_orient` (axis-angle, `|aa|=4.70 rad`) | — | 0° (reference) |
| Our packed `object_rot_quat` (scipy `R.from_rotvec(aa).as_quat()` → xyzw) | direct | **0.08°** (float32 rounding only) |
| `leap_passive_active_info_*_nf_300.npy::object_rot_quat` | DexTrack-shipped | **179°** |

LEAP's quat has the right `xyz` signs and the **wrong `w` sign**.
Reproducible by wrapping `|aa| > π` to `2π − |aa|` without flipping the
axis sign — a classic quat-conversion bug.  Invisible in LEAP replay
videos because the GRAB cubesmall is a symmetric 5 cm box and the
fingertip retargeting was fit with the inverted reference, so contacts
still land on cube faces.  If you ever try to use LEAP's quat directly
on a non-symmetric object you'll get a 180°-off pose.

Verification: `isaacgymenvs/tasks/dextrack_sharpa/diag_obj_quat_provenance.py`.

Implication for our env: `set_actor_root_state_tensor_indexed` writes
xyzw, IsaacGym reads xyzw — no transformation in our env code.
`traj.obj_quat == GRAB.global_orient.as_quat()` end-to-end.

## 5. Diagnostic helpers for future trips into this swamp

Located under `isaacgymenvs/tasks/dextrack_sharpa/`, but the principles
generalise to any IsaacGym task:

- `diag_replay_state.py` — set DOFs + obj root, refresh, compare to ref.
  Confirms set_*_tensor actually reaches sim internal state.
- `diag_render_no_sim.py` — proves `step_graphics` doesn't pull from
  `set_*_tensor` writes; quantifies pixel diff with vs without simulate.
- `diag_set_after_sim.py` — proves set-after-simulate doesn't reach the
  renderer in the same frame.
- `diag_long_play.py` — runs the full replay and reports max DOF / obj
  drift across all frames. Per-joint breakdown of worst offenders is
  useful for spotting joint-limit-clamped DOFs that are a retargeting
  artifact rather than a viz bug.
- `render_replay_montage.py` — 6-frame side-by-side PNG for quick visual
  sanity check without an interactive viewer.
