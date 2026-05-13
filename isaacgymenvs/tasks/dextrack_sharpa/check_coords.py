"""Numerically verify that all coordinate frames line up.

For frame 0 of the trajectory, prints:
  - robot base pose (should be (0,0,0))
  - table centre / top z (should match cfg)
  - obj_pos[0]                (from trajectory data)
  - wrist FK pos at dof_pos[0]  (should be at wrist target, far above table)
  - 5 fingertip world positions
  - distance(wrist, obj)        (≤ 10 cm for grasping prep)
  - distance(fingertip_i, obj)  (small if hand is around obj)
  - obj_bottom_z vs table_top_z (≥ 0 means obj sits on / above table)

If any of these are obviously wrong (e.g. wrist 50 cm below table, obj at
negative z), the coordinate setup is broken.

Run: cd ~/Codes/video-rl-follower
     PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/check_coords.py
"""
import os
import isaacgym  # noqa: F401  (must precede torch)
import torch

from isaacgymenvs.tasks.dextrack_sharpa.trajectory import DexTrackTrajectory

ROOT = os.environ.get("VIDEO_RL_FOLLOWER_ROOT", "/home/intel/Codes/video-rl-follower")
NPY  = f"{ROOT}/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy"
URDF = f"{ROOT}/assets/urdf/franka_sharpa_description/franka_panda_sharpa.urdf"

# Hard-coded cfg defaults (matches DexTrackSharpa.yaml)
ROBOT_POSE      = (0.0, 0.0, 0.0)
TABLE_CENTRE    = (0.70, 0.0, 0.25)
TABLE_DIMS      = (1.0, 1.0, 0.5)
OBJ_SIZE        = (0.05, 0.05, 0.05)

# Sharpa body indices in 28-link list (env.py SHARPA_BODY_NAMES order)
WRIST_IDX        = 0
FINGERTIP_IDXS   = [27, 5, 10, 21, 16]    # thumb, index, middle, ring, pinky
FINGER_NAMES     = ["thumb", "index", "middle", "ring", "pinky"]


def main():
    print("─" * 70)
    print(f"Trajectory: {NPY}")
    print(f"URDF:       {URDF}")
    print("─" * 70)

    traj = DexTrackTrajectory(NPY, URDF, dt=1.0/60.0, device="cuda:0")

    # Convert tensors to numpy for printing
    obj_pos0  = traj.obj_pos[0].cpu().numpy()
    obj_quat0 = traj.obj_quat[0].cpu().numpy()
    wrist_p0  = traj.wrist_pos[0].cpu().numpy()
    wrist_q0  = traj.wrist_quat[0].cpu().numpy()
    fingertip_w = traj.link_pos[0, FINGERTIP_IDXS].cpu().numpy()  # (5, 3)
    panda_dof   = traj.dof_pos[0, :7].cpu().numpy()
    sharpa_dof  = traj.dof_pos[0, 7:].cpu().numpy()

    table_top_z = TABLE_CENTRE[2] + TABLE_DIMS[2] / 2
    obj_bottom_z = obj_pos0[2] - OBJ_SIZE[2] / 2
    obj_table_gap = obj_bottom_z - table_top_z

    print("\n[FRAME 0]")
    print(f"  Robot base pose (cfg) : ({ROBOT_POSE[0]:+.3f}, {ROBOT_POSE[1]:+.3f}, {ROBOT_POSE[2]:+.3f})")
    print(f"  Table centre (cfg)    : ({TABLE_CENTRE[0]:+.3f}, {TABLE_CENTRE[1]:+.3f}, {TABLE_CENTRE[2]:+.3f})"
          f"   dims ({TABLE_DIMS[0]}, {TABLE_DIMS[1]}, {TABLE_DIMS[2]})  → top z = {table_top_z:.3f}")
    print(f"  Object init pos       : ({obj_pos0[0]:+.3f}, {obj_pos0[1]:+.3f}, {obj_pos0[2]:+.3f})"
          f"   (cube {OBJ_SIZE[0]:.2f})  bottom z = {obj_bottom_z:.3f}")
    print(f"  Object init quat xyzw : ({obj_quat0[0]:+.3f}, {obj_quat0[1]:+.3f}, {obj_quat0[2]:+.3f}, {obj_quat0[3]:+.3f})")
    print(f"  Object↔Table gap (z)  : {obj_table_gap*100:+.1f} cm   "
          f"{'(✓ on table)' if 0 <= obj_table_gap <= 0.05 else '(✗ floating)' if obj_table_gap > 0.05 else '(✗ INSIDE table)'}")

    print(f"\n  Panda joints (rad)    : {panda_dof.round(3).tolist()}")
    print(f"  Sharpa joints (rad)   : {sharpa_dof.round(3).tolist()}")

    print(f"\n  Wrist FK world pos    : ({wrist_p0[0]:+.3f}, {wrist_p0[1]:+.3f}, {wrist_p0[2]:+.3f})")
    print(f"  Wrist FK world quat   : ({wrist_q0[0]:+.3f}, {wrist_q0[1]:+.3f}, {wrist_q0[2]:+.3f}, {wrist_q0[3]:+.3f})")
    wrist_obj = ((wrist_p0 - obj_pos0) ** 2).sum() ** 0.5
    print(f"  ||wrist - obj||       : {wrist_obj*100:.1f} cm   "
          f"{'(✓ near obj for grasp prep)' if wrist_obj < 0.30 else '(✗ wrist too far)'}")

    print(f"\n  Fingertip world positions (and distance to obj):")
    for name, p in zip(FINGER_NAMES, fingertip_w):
        d = ((p - obj_pos0) ** 2).sum() ** 0.5
        marker = "✓" if d < 0.10 else ("·" if d < 0.20 else "✗")
        print(f"    {name:8s}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  "
              f"d_obj = {d*100:5.1f} cm  {marker}")

    print("\n[Trajectory range over T={} frames]".format(traj.T))
    obj_pos_np = traj.obj_pos.cpu().numpy()
    print(f"  obj x: [{obj_pos_np[:,0].min():+.3f}, {obj_pos_np[:,0].max():+.3f}]   "
          f"y: [{obj_pos_np[:,1].min():+.3f}, {obj_pos_np[:,1].max():+.3f}]   "
          f"z: [{obj_pos_np[:,2].min():+.3f}, {obj_pos_np[:,2].max():+.3f}]")
    wrist_np = traj.wrist_pos.cpu().numpy()
    print(f"  wrist x: [{wrist_np[:,0].min():+.3f}, {wrist_np[:,0].max():+.3f}]   "
          f"y: [{wrist_np[:,1].min():+.3f}, {wrist_np[:,1].max():+.3f}]   "
          f"z: [{wrist_np[:,2].min():+.3f}, {wrist_np[:,2].max():+.3f}]")

    print("\n[Sanity checks]")
    checks = [
        ("obj never below table top (z > 0.5)", (obj_pos_np[:,2] > table_top_z - 0.01).all(),
            f"min obj z = {obj_pos_np[:,2].min():.3f}, table top = {table_top_z:.3f}"),
        ("wrist FK above table at frame 0 (z > 0.5)", wrist_p0[2] > table_top_z,
            f"wrist z = {wrist_p0[2]:.3f}"),
        ("obj x within table extent ([0.20, 1.20])", 0.20 < obj_pos_np[:,0].min() and obj_pos_np[:,0].max() < 1.20,
            f"obj x range = [{obj_pos_np[:,0].min():.3f}, {obj_pos_np[:,0].max():.3f}]"),
        ("obj y within table extent ([-0.5, 0.5])", -0.5 < obj_pos_np[:,1].min() and obj_pos_np[:,1].max() < 0.5,
            f"obj y range = [{obj_pos_np[:,1].min():.3f}, {obj_pos_np[:,1].max():.3f}]"),
    ]
    all_ok = True
    for name, ok, detail in checks:
        sym = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {sym} {name}   ({detail})")

    print("\n" + ("✓ ALL COORDINATE CHECKS PASS — frames are consistent."
                  if all_ok else
                  "✗ SOME CHECKS FAILED — investigate before training."))


if __name__ == "__main__":
    main()
