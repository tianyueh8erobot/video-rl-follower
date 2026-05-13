"""Verify object orientation data lineage: GRAB original → our retargeted npy.
Also compares against the DexTrack-downloaded LEAP retargeted reference.

Run:
  cd ~/Codes/video-rl-follower
  PYTHONPATH=. python isaacgymenvs/tasks/dextrack_sharpa/diag_obj_quat_provenance.py

Background: a user-visible "spinning cube" in IsaacGym replay led to
suspicion that we mangled object orientation between data sources and
the simulator.  This script proves we did not.

  GRAB axis-angle (rotvec)
    │
    ├── R.from_rotvec(aa).as_quat()  (scipy default, xyzw)
    │   → our `object_rot_quat`              ← matches GRAB to 0.08°
    │
    └── ... LEAP retargeter's internal quat conversion
        → DexTrack-shipped LEAP npy           ← differs from GRAB by 179°
                                                (LEAP has a quat bug,
                                                hidden by cube symmetry)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

GRAB_ORIG = "/home/intel/Codes/GRAB/grab__s2/s2/cubesmall_inspect_1.npz"
LEAP_DL   = "/home/intel/下载/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data/leap_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
OUR_NPY   = "/home/intel/Codes/video-rl-follower/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy"

grab = np.load(GRAB_ORIG, allow_pickle=True)
obj_aa = grab["object"].item()["params"]["global_orient"][0]
print(f"GRAB axis-angle [0]: {obj_aa}  |aa| = {np.linalg.norm(obj_aa):.3f} rad "
      f"({np.linalg.norm(obj_aa)*180/np.pi:.1f}°)")

leap_q = np.asarray(np.load(LEAP_DL,  allow_pickle=True).item()["object_rot_quat"])[0]
our_q  = np.asarray(np.load(OUR_NPY, allow_pickle=True).item()["object_rot_quat"])[0]
print(f"\nQuat (xyzw):")
print(f"  GRAB → scipy.from_rotvec().as_quat() = {R.from_rotvec(obj_aa).as_quat()}")
print(f"  OUR  ['object_rot_quat'][0]          = {our_q}")
print(f"  LEAP ['object_rot_quat'][0]          = {leap_q}")

r_grab = R.from_rotvec(obj_aa)
r_our  = R.from_quat(our_q)
r_leap = R.from_quat(leap_q)
diff_our  = np.linalg.norm((r_our.inv()  * r_grab).as_rotvec()) * 180 / np.pi
diff_leap = np.linalg.norm((r_leap.inv() * r_grab).as_rotvec()) * 180 / np.pi
print(f"\nAngular distance from GRAB-original rotation:")
print(f"  OUR  ↔ GRAB = {diff_our:7.3f}°   (expected near 0; only float32 rounding)")
print(f"  LEAP ↔ GRAB = {diff_leap:7.3f}°   (LEAP's quat is inverted vs GRAB!)")

if diff_our < 0.5 and diff_leap > 150:
    print(f"\n✓ Our packed object_rot_quat is the faithful conversion of GRAB axis-angle.")
    print(f"✗ LEAP's shipped quat is inverted — visually undetectable because the cube")
    print(f"  is symmetric (5 cm box).  Not our bug.")
else:
    print(f"\n⚠ Unexpected — inspect the data manually.")
