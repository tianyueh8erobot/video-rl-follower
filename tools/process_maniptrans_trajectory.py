"""Process a ManipTrans MANO demo into a SimToolReal-style 3 Hz goal trajectory.

This script does NOT run the IsaacGym retargeting pipeline (mano2dexhand).  It
operates purely on the MANO data dump that ManipTrans's GRAB demo provides
(the .npy file under data/grab_demo/<id>/<id>_obj.npy and the per-frame MANO
output).  The result is an embodiment-agnostic JSON of:

* object 6D pose @ src fps  →  downsampled to target fps
* hand wrist 6D pose
* fingertip keypoint trajectories (5 keypoints) in BOTH world frame and
  wrist-local frame

The downstream env (VideoRLFollower) tracks these targets directly; the
runtime IK from "fingertip-target → joint" is the policy's job.

Usage::

    python tools/process_maniptrans_trajectory.py \
        --maniptrans-root /home/intel/Codes/ManipTrans \
        --data-idx g0 \
        --src-fps 60 \
        --target-fps 3 \
        --out data/trajectories/example_g0.json

If --maniptrans-root is not importable (no IsaacGym available locally), pass
--mock to write a synthetic 60-frame teapot-rotation trajectory for env smoke
testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Subset of MANO joint indices used by ManipTrans (from grab_dataset_dexhand.py)
# joint 0 is wrist; we re-derive wrist position with a small shift:
#   wrist = j0 - 0.25 * (j4 - j0)   (matches ManipTrans's hack for Inspire)
# Fingertip vertex indices from the same file:
_FINGERTIP_VERTS = {
    "index": 353,
    "middle": 467,
    "pinky": 695,
    "ring": 576,
    "thumb": 766,
}
_FINGERTIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Generic single-link URDF template that wraps a Wavefront mesh.  This is a
# functional copy of ManipTrans/assets/obj_urdf_example.urdf with the mesh
# filename made into a placeholder.  We embed it here so the processor does
# not depend on a checked-out ManipTrans repo on disk just to grab one
# 18-line XML file.
_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="design">
  <material name="obj_color">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh_file}" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh_file}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


# Standard 21-keypoint MANO ordering used throughout the repo
# (see ReferenceTrajectory.MANO_JOINT_ORDER for the full table).
# Indices 0..15 come from MANO's joint regressor applied to verts;
# indices 16..20 come from the picked fingertip vertices above.
_MANO_JOINT_ORDER = [
    "wrist",
    "index_proximal",  "index_intermediate",  "index_distal",
    "middle_proximal", "middle_intermediate", "middle_distal",
    "pinky_proximal",  "pinky_intermediate",  "pinky_distal",
    "ring_proximal",   "ring_intermediate",   "ring_distal",
    "thumb_proximal",  "thumb_intermediate",  "thumb_distal",
    "index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip",
]
_MANO_BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 16),       # index
    (0, 4), (4, 5), (5, 6), (6, 17),       # middle
    (0, 7), (7, 8), (8, 9), (9, 18),       # pinky
    (0, 10), (10, 11), (11, 12), (12, 19), # ring
    (0, 13), (13, 14), (14, 15), (15, 20), # thumb
]

# Default search paths for MANO_RIGHT.pkl on this machine.  First hit wins.
_MANO_DEFAULT_PATHS = [
    "/home/intel/Codes/ASSET_HUB_GRAB_MANO_SMPLX/extracted/mano_v1_2/mano_v1_2/models/MANO_RIGHT.pkl",
    "/home/intel/Codes/V2P_manip/Manip/data/mano_v1_2/models/MANO_RIGHT.pkl",
    "/home/intel/Codes/ManipTrans/data/mano_v1_2/models/MANO_RIGHT.pkl",
]


def _load_mano_joint_regressor(mano_pkl: Optional[Path] = None) -> Optional[np.ndarray]:
    """Return the (16, 778) joint regressor matrix from MANO_RIGHT.pkl as a
    dense float32 numpy array.

    Returns None if the file can't be located OR the pickle contains chumpy
    objects we can't decode without the chumpy package installed (then the
    caller should fall back to fingertip-only mode).
    """
    candidates: List[Path] = []
    if mano_pkl is not None:
        candidates.append(Path(mano_pkl))
    candidates.extend(Path(p) for p in _MANO_DEFAULT_PATHS)

    pkl_path = next((c for c in candidates if c.is_file()), None)
    if pkl_path is None:
        print("[mano] MANO_RIGHT.pkl not found in any default location; "
              "skipping 21-joint skeleton extraction.")
        return None

    import pickle
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as exc:
        # Most commonly chumpy ImportError from very old MANO pickles.
        print(f"[mano] WARN: could not load {pkl_path}: {type(exc).__name__}: {exc}.\n"
              "       Skipping 21-joint skeleton extraction; the trajectory "
              "will only contain the 5 fingertip points.")
        return None

    j_reg = data.get("J_regressor")
    if j_reg is None:
        print(f"[mano] WARN: {pkl_path} has no 'J_regressor' key; skipping.")
        return None

    # J_regressor may be scipy.sparse or numpy or a chumpy.Ch wrapper.
    try:
        from scipy import sparse as _sp
        if _sp.issparse(j_reg):
            j_reg = j_reg.toarray()
    except ImportError:
        pass
    if hasattr(j_reg, "r"):  # chumpy
        j_reg = j_reg.r
    j_reg = np.asarray(j_reg, dtype=np.float32)
    if j_reg.shape != (16, 778):
        print(f"[mano] WARN: J_regressor has unexpected shape {j_reg.shape} "
              "(expected (16, 778)); skipping.")
        return None

    print(f"[mano] loaded J_regressor (16, 778) from {pkl_path}")
    return j_reg


def _compute_21_joints(
    verts: np.ndarray,                # (T, 778, 3)
    j_regressor: np.ndarray,          # (16, 778)
) -> np.ndarray:
    """Return (T, 21, 3): MANO 16 joints (from regressor) + 5 fingertips
    (from _FINGERTIP_VERTS).  Indices follow _MANO_JOINT_ORDER.
    """
    # joints[0..15] = J_reg @ verts
    joints_16 = np.einsum("jv,tvc->tjc", j_regressor, verts)         # (T, 16, 3)
    tip_idx = [_FINGERTIP_VERTS[name] for name in
               ("index", "middle", "pinky", "ring", "thumb")]
    tips_5 = verts[:, tip_idx, :]                                     # (T, 5, 3)
    return np.concatenate([joints_16, tips_5], axis=1)                # (T, 21, 3)


def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) → rotation matrix (..., 3, 3)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = aa / safe
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    I = np.broadcast_to(np.eye(3, dtype=aa.dtype), K.shape).copy()
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) → (..., 4) xyzw using a stable branch decision."""
    # Standard "Mike Day" decomposition.
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    out = np.zeros(m.shape[:-2] + (4,), dtype=m.dtype)

    cond = t > 0
    s_pos = 0.5 / np.sqrt(np.clip(t + 1.0, 1e-12, None))
    out[..., 3] = np.where(cond, 0.25 / s_pos, 0.0)
    out[..., 0] = np.where(cond, (m[..., 2, 1] - m[..., 1, 2]) * s_pos, 0.0)
    out[..., 1] = np.where(cond, (m[..., 0, 2] - m[..., 2, 0]) * s_pos, 0.0)
    out[..., 2] = np.where(cond, (m[..., 1, 0] - m[..., 0, 1]) * s_pos, 0.0)

    # Branches for non-positive trace
    not_cond = ~cond
    if np.any(not_cond):
        r = m[not_cond]
        # diagonal-max branches
        diag = np.stack([r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]], axis=-1)
        case = np.argmax(diag, axis=-1)
        q = np.zeros(r.shape[:-2] + (4,), dtype=m.dtype)
        for i in range(r.shape[0]):
            R_i = r[i]
            c = case[i]
            if c == 0:
                S = np.sqrt(1.0 + R_i[0, 0] - R_i[1, 1] - R_i[2, 2]) * 2
                q[i, 0] = 0.25 * S
                q[i, 1] = (R_i[0, 1] + R_i[1, 0]) / S
                q[i, 2] = (R_i[0, 2] + R_i[2, 0]) / S
                q[i, 3] = (R_i[2, 1] - R_i[1, 2]) / S
            elif c == 1:
                S = np.sqrt(1.0 + R_i[1, 1] - R_i[0, 0] - R_i[2, 2]) * 2
                q[i, 0] = (R_i[0, 1] + R_i[1, 0]) / S
                q[i, 1] = 0.25 * S
                q[i, 2] = (R_i[1, 2] + R_i[2, 1]) / S
                q[i, 3] = (R_i[0, 2] - R_i[2, 0]) / S
            else:
                S = np.sqrt(1.0 + R_i[2, 2] - R_i[0, 0] - R_i[1, 1]) * 2
                q[i, 0] = (R_i[0, 2] + R_i[2, 0]) / S
                q[i, 1] = (R_i[1, 2] + R_i[2, 1]) / S
                q[i, 2] = 0.25 * S
                q[i, 3] = (R_i[1, 0] - R_i[0, 1]) / S
        out[not_cond] = q
    # Normalise
    out /= np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-12)
    return out


def _quat_inverse_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (xyzw).  Both broadcastable."""
    qv = q[..., :3]
    qw = q[..., 3:]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _world_to_local_xyzw(
    points_w: np.ndarray, frame_pos: np.ndarray, frame_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Transform points_w into frame's local space."""
    delta = points_w - frame_pos[..., None, :]
    inv_q = _quat_inverse_xyzw(frame_quat_xyzw)[..., None, :]
    delta_local = _quat_rotate_xyzw(inv_q, delta)
    return delta_local


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_grab_demo(
    maniptrans_root: Path,
    data_idx: str,
    demo_dir: Optional[Path] = None,
) -> dict:
    """Return the raw ManipTrans GRAB demo as a dict.

    By default resolves the same path that grab_dataset_dexhand.py uses
    (``<maniptrans_root>/data/grab_demo/102/102_sv_dict.npy``).  Pass
    ``demo_dir`` to point at any directory that contains ``102_sv_dict.npy``
    directly — useful when the demo files were downloaded into a different
    location (e.g. ``data/trajectories/grab_102/``).
    """
    if data_idx != "g0":
        raise ValueError("Only g0 is shipped with the ManipTrans demo data.")
    if demo_dir is not None:
        npy_path = Path(demo_dir) / "102_sv_dict.npy"
    else:
        npy_path = maniptrans_root / "data" / "grab_demo" / "102" / "102_sv_dict.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"Expected ManipTrans GRAB demo at {npy_path}.  Either follow "
            "ManipTrans README §Prerequisites to download to the default "
            "location, or pass --demo-dir <path-to-102/-folder>."
        )
    return np.load(str(npy_path), allow_pickle=True).item()


def _build_full_trajectory(
    data: dict,
    j_regressor: Optional[np.ndarray] = None,
) -> dict:
    """Compute the full-rate (60 Hz) reference trajectory from raw MANO data.

    The MANO output gives:
      * data['object_global_orient']  axis-angle of the object   (T, 3)
      * data['object_transl']         translation                (T, 3)
      * data['rhand_global_orient_gt']  hand wrist axis-angle    (T, 3)
      * data['rhand_transl']          hand wrist translation    (T, 3)  (NB unused — wrist via joints)
      * data['rhand_verts']           MANO vertices              (T, 778, 3)

    For the wrist position we follow ManipTrans:
        wrist = j0 - 0.25 * (j4 - j0)
    where j_k is the k-th MANO joint.  We do not have direct access to MANO's
    joint regressor here, so we use the vertices-projected joint approximation
    by averaging neighbour vertices — this is a known coarse approximation and
    matches the behaviour the env expects for "wrist target".
    """
    T = data["object_global_orient"].shape[0]

    # Object pose
    obj_aa = data["object_global_orient"]
    obj_R = _aa_to_rotmat(obj_aa)                       # (T, 3, 3)
    # ManipTrans transposes here because GRAB's convention is row-vector.
    obj_R_proper = np.transpose(obj_R, (0, 2, 1))
    obj_quat = _rotmat_to_quat_xyzw(obj_R_proper)       # (T, 4) xyzw
    obj_pos = data["object_transl"]                     # (T, 3)
    obj_pose7 = np.concatenate([obj_pos, obj_quat], axis=-1)

    # Hand wrist
    hand_aa = data["rhand_global_orient_gt"]
    wrist_R = _aa_to_rotmat(hand_aa)                    # (T, 3, 3)
    wrist_quat = _rotmat_to_quat_xyzw(wrist_R)          # (T, 4)

    # MANO joint regressor isn't available without the manolayer; fall back to
    # vertex-based wrist (vertex 0 in MANO is approximately wrist) plus the
    # ManipTrans 25%-shift trick using the middle MCP (vertex 467 ≈ middle tip
    # is too far, use vertex 4? — without the regressor we use the rhand_transl
    # as the wrist position which is what GRAB ships).
    wrist_pos = data["rhand_transl"]                    # (T, 3)

    # Fingertips: pull vertex coordinates directly.
    verts = data["rhand_verts"]                         # (T, 778, 3)
    ftips_w = np.stack(
        [verts[:, _FINGERTIP_VERTS[name], :] for name in _FINGERTIP_ORDER],
        axis=1,
    )                                                    # (T, 5, 3)

    # Local-frame fingertip targets (relative to wrist)
    ftips_local = _world_to_local_xyzw(ftips_w, wrist_pos, wrist_quat)

    # Full 21-keypoint hand skeleton (MANO 16 joints + 5 fingertips)
    # only when the J_regressor was loadable.
    joints_w = None
    joints_local = None
    if j_regressor is not None:
        joints_w = _compute_21_joints(verts, j_regressor)            # (T, 21, 3) world
        joints_local = _world_to_local_xyzw(joints_w, wrist_pos, wrist_quat)

    return dict(
        T=T,
        object_pose=obj_pose7,
        wrist_pos=wrist_pos,
        wrist_quat=wrist_quat,
        fingertips_w=ftips_w,
        fingertips_local=ftips_local,
        joints_w=joints_w,
        joints_local=joints_local,
    )


def _downsample(traj: dict, src_fps: float, target_fps: float) -> dict:
    """Sub-sample the dense trajectory to ``target_fps``."""
    if target_fps >= src_fps:
        return traj
    stride = max(1, int(round(src_fps / target_fps)))
    sl = slice(None, None, stride)
    return dict(
        T=int(np.ceil(traj["T"] / stride)),
        object_pose=traj["object_pose"][sl],
        wrist_pos=traj["wrist_pos"][sl],
        wrist_quat=traj["wrist_quat"][sl],
        fingertips_w=traj["fingertips_w"][sl],
        fingertips_local=traj["fingertips_local"][sl],
        joints_w=traj.get("joints_w")[sl] if traj.get("joints_w") is not None else None,
        joints_local=traj.get("joints_local")[sl] if traj.get("joints_local") is not None else None,
    )


def _to_json(
    traj: dict,
    *,
    src: str,
    fps: float,
    urdf_path: str | None,
    grasp_bbox_scale: tuple[float, float, float] | None = None,
    need_vhacd: bool = False,
) -> dict:
    obj_pose = traj["object_pose"]                         # (T, 7)
    wrist_pos = traj["wrist_pos"]                          # (T, 3)
    wrist_quat = traj["wrist_quat"]                        # (T, 4)
    wrist_pose = np.concatenate([wrist_pos, wrist_quat], axis=-1)   # (T, 7)
    obj_block = {
        "urdf_path": urdf_path,
        "scale": 1.0,
        "start_pose": obj_pose[0].tolist(),
        "goals": obj_pose.tolist(),
        "need_vhacd": bool(need_vhacd),
    }
    if grasp_bbox_scale is not None:
        obj_block["grasp_bbox_scale"] = list(grasp_bbox_scale)
    hand_block = {
        "wrist_goals": wrist_pose.tolist(),
        "fingertip_goals": traj["fingertips_w"].tolist(),
        "fingertip_local": traj["fingertips_local"].tolist(),
    }
    # Optional 21-keypoint skeleton when available.
    if traj.get("joints_w") is not None and traj.get("joints_local") is not None:
        hand_block["joints_world"] = traj["joints_w"].tolist()
        hand_block["joints_local"] = traj["joints_local"].tolist()
    return {
        "meta": {
            "source": src,
            "fps": float(fps),
            "n_fingertips": 5,
            "fingertip_order": _FINGERTIP_ORDER,
            "n_joints": (21 if traj.get("joints_w") is not None else 0),
            "joint_order": (_MANO_JOINT_ORDER if traj.get("joints_w") is not None else []),
            "bone_pairs":  (_MANO_BONE_PAIRS  if traj.get("joints_w") is not None else []),
            "wrist_pose_format": "[x, y, z, qx, qy, qz, qw]",
            "object_pose_format": "[x, y, z, qx, qy, qz, qw]",
        },
        "object": obj_block,
        "hand": hand_block,
    }


# ---------------------------------------------------------------------------
# Mock trajectory (for smoke-testing without ManipTrans data)
# ---------------------------------------------------------------------------


def _mock_trajectory(num_goals: int = 36) -> dict:
    """Synthetic 1-hand cylindrical-rotation trajectory for testing."""
    T_full = num_goals * 20                                 # pretend 60 Hz source
    t = np.linspace(0, 1, T_full)
    # Object rotates about z while staying still in xy
    obj_pos = np.tile(np.array([0.5, 0.0, 0.5]), (T_full, 1))
    angles = t * 2.0 * np.pi
    half = angles / 2
    obj_quat = np.stack(
        [np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)],
        axis=-1,
    )                                                       # xyzw
    object_pose = np.concatenate([obj_pos, obj_quat], axis=-1)

    # Hand hovers above the object, wrist horizontal
    wrist_pos = obj_pos + np.array([0.0, 0.0, 0.10])
    wrist_quat = np.tile(np.array([0.0, 0.7071, 0.0, 0.7071]), (T_full, 1))

    # 5 fingertips around the object axis
    radii = np.array([0.04, 0.05, 0.05, 0.05, 0.05])
    phase = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi  # thumb opposed
    ftips_local = np.zeros((T_full, 5, 3))
    for k in range(5):
        ftips_local[:, k, 0] = radii[k] * np.cos(phase[k] + angles)
        ftips_local[:, k, 1] = radii[k] * np.sin(phase[k] + angles)
        ftips_local[:, k, 2] = -0.05

    # Express in world frame for completeness
    ftips_w = ftips_local + wrist_pos[:, None, :]

    # Mock 21-keypoint skeleton: arrange MCPs in a small arc above the
    # fingertips, PIPs / DIPs interpolated.  Produces a "reasonable looking"
    # but synthetic right hand with a curl that varies over time.
    joints_local = np.zeros((T_full, 21, 3))
    # joint 0 = wrist origin (zero in local frame)
    joints_local[:, 0, :] = 0.0
    # For each finger (index, middle, pinky, ring, thumb), generate
    # MCP / PIP / DIP / tip along a line from wrist to fingertip
    finger_groups = [
        ("index",  [1, 2, 3, 16]),
        ("middle", [4, 5, 6, 17]),
        ("pinky",  [7, 8, 9, 18]),
        ("ring",   [10, 11, 12, 19]),
        ("thumb",  [13, 14, 15, 20]),
    ]
    # Map name → fingertip index in our 5-tuple (thumb, index, middle, ring, pinky)
    name_to_ftip = {n: i for i, n in enumerate(_FINGERTIP_ORDER)}
    for name, idxs in finger_groups:
        ftip_pos = ftips_local[:, name_to_ftip[name], :]                  # (T, 3)
        # interpolate 4 points: at fractions 0.30, 0.55, 0.80, 1.00 of the way
        for j_local_idx, frac in zip(idxs, [0.30, 0.55, 0.80, 1.00]):
            joints_local[:, j_local_idx, :] = ftip_pos * frac
    joints_w = joints_local + wrist_pos[:, None, :]

    return dict(
        T=T_full,
        object_pose=object_pose,
        wrist_pos=wrist_pos,
        wrist_quat=wrist_quat,
        fingertips_w=ftips_w,
        fingertips_local=ftips_local,
        joints_w=joints_w,
        joints_local=joints_local,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--maniptrans-root", default="/home/intel/Codes/ManipTrans",
                   help="ONLY used when --demo-dir is not given.  Looks for "
                        "<maniptrans_root>/data/grab_demo/102/.  No longer "
                        "required for URDF generation — that is now handled by "
                        "an embedded URDF template.")
    p.add_argument("--demo-dir", default=None,
                   help="Direct path to the folder containing 102_sv_dict.npy "
                        "and 102_obj.obj.  Overrides --maniptrans-root.")
    p.add_argument("--obj-urdf", default=None,
                   help="Path to the object URDF.  If omitted, defaults to "
                        "<demo-dir>/102_obj.urdf and will be auto-generated "
                        "from the embedded URDF template if missing.")
    p.add_argument("--obj-mesh-name", default="102_obj.obj",
                   help="Mesh filename (relative to the URDF dir) referenced "
                        "by the auto-generated URDF.  Default 102_obj.obj.")
    p.add_argument("--data-idx", default="g0")
    p.add_argument("--src-fps", type=float, default=60.0)
    p.add_argument("--target-fps", type=float, default=3.0)
    p.add_argument(
        "--out",
        default="data/trajectories/example_g0.json",
        help="Path to write JSON (relative to --project-root)",
    )
    p.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Write a synthetic test trajectory (no ManipTrans data needed)",
    )
    p.add_argument(
        "--mano-pkl",
        default=None,
        help="Path to MANO_RIGHT.pkl.  If omitted the script searches a few "
             "default locations on this machine.  When found, the JSON will "
             "include the full 21-keypoint hand skeleton (16 MANO joints + 5 "
             "fingertips) under hand.joints_world / hand.joints_local.  When "
             "not found, the JSON only contains the 5 fingertip points.",
    )
    p.add_argument(
        "--no-skeleton", action="store_true",
        help="Skip 21-joint extraction even if MANO_RIGHT.pkl is available.",
    )
    args = p.parse_args()

    out_path = Path(args.project_root) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mock:
        full = _mock_trajectory()
        # mock trajectory has no real URDF; leave urdf_path null so the env
        # falls back to procedural primitives.
        urdf = None
        bbox = (0.05, 0.05, 0.05)
        src = "mock/cylinder_rotation"
    else:
        demo_dir = Path(args.demo_dir) if args.demo_dir else None
        data = _load_grab_demo(Path(args.maniptrans_root), args.data_idx, demo_dir)
        # Attempt to load the MANO joint regressor for a full 21-keypoint
        # skeleton; gracefully falls back to fingertips only if not found.
        j_regressor = None
        if not args.no_skeleton:
            j_regressor = _load_mano_joint_regressor(
                Path(args.mano_pkl) if args.mano_pkl else None
            )
        full = _build_full_trajectory(data, j_regressor=j_regressor)
        # Resolve URDF path: explicit override → <demo_dir>/102_obj.urdf →
        # <maniptrans_root>/data/grab_demo/102/102_obj.urdf
        if args.obj_urdf is not None:
            urdf = args.obj_urdf
        elif demo_dir is not None:
            urdf = str(demo_dir / "102_obj.urdf")
        else:
            urdf = str(
                Path(args.maniptrans_root) / "data" / "grab_demo" / "102" / "102_obj.urdf"
            )
        # If the URDF is missing, generate one in-place from our embedded
        # template, pointing at the .obj mesh that lives next to the URDF.
        # Default mesh filename is 102_obj.obj (matches GRAB demo); user can
        # override with --obj-mesh-name.
        if not Path(urdf).is_file():
            urdf_path = Path(urdf)
            mesh_file = args.obj_mesh_name
            mesh_full = urdf_path.parent / mesh_file
            if not mesh_full.is_file():
                raise FileNotFoundError(
                    f"Object URDF will be auto-generated but the referenced "
                    f"mesh {mesh_full} does not exist.  Either drop the .obj "
                    f"file next to the URDF or pass --obj-mesh-name "
                    f"<filename> if your mesh has a different name."
                )
            urdf_path.parent.mkdir(parents=True, exist_ok=True)
            urdf_path.write_text(_URDF_TEMPLATE.format(mesh_file=mesh_file))
            print(f"[info] generated URDF → {urdf} (mesh: {mesh_file})")
        # GRAB g0 is a teapot mesh — a sensible default grasp bbox is ~0.10 m
        # along the long axis.  User can override post-hoc by editing the JSON.
        bbox = (0.10, 0.05, 0.05)
        src = f"ManipTrans/grab_demo/{args.data_idx}"

    sub = _downsample(full, args.src_fps, args.target_fps)
    j = _to_json(
        sub,
        src=src,
        fps=args.target_fps,
        urdf_path=urdf,
        grasp_bbox_scale=bbox,
        need_vhacd=False if args.mock else True,
    )
    with open(out_path, "w") as f:
        json.dump(j, f, indent=2)

    n = len(j["object"]["goals"])
    print(f"Wrote {out_path} with {n} goals (target fps {args.target_fps})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
