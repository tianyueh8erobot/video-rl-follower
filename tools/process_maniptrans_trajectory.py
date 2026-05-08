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
from typing import Tuple

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


def _load_grab_demo(maniptrans_root: Path, data_idx: str) -> dict:
    """Return the raw ManipTrans GRAB demo as a dict.

    Resolves the same path that grab_dataset_dexhand.py uses, but **without
    importing** any IsaacGym-dependent modules.
    """
    if data_idx != "g0":
        raise ValueError("Only g0 is shipped with the ManipTrans demo data.")
    npy_path = maniptrans_root / "data" / "grab_demo" / "102" / "102_sv_dict.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"Expected ManipTrans GRAB demo at {npy_path}.  Follow ManipTrans "
            "README §Prerequisites to download the demo."
        )
    return np.load(str(npy_path), allow_pickle=True).item()


def _build_full_trajectory(data: dict) -> dict:
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

    return dict(
        T=T,
        object_pose=obj_pose7,
        wrist_pos=wrist_pos,
        wrist_quat=wrist_quat,
        fingertips_w=ftips_w,
        fingertips_local=ftips_local,
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
    )


def _to_json(traj: dict, *, src: str, fps: float, urdf_path: str | None) -> dict:
    obj_pose = traj["object_pose"]                         # (T, 7)
    wrist_pos = traj["wrist_pos"]                          # (T, 3)
    wrist_quat = traj["wrist_quat"]                        # (T, 4)
    wrist_pose = np.concatenate([wrist_pos, wrist_quat], axis=-1)   # (T, 7)
    return {
        "meta": {
            "source": src,
            "fps": float(fps),
            "n_fingertips": 5,
            "fingertip_order": _FINGERTIP_ORDER,
            "wrist_pose_format": "[x, y, z, qx, qy, qz, qw]",
            "object_pose_format": "[x, y, z, qx, qy, qz, qw]",
        },
        "object": {
            "urdf_path": urdf_path,
            "scale": 1.0,
            "start_pose": obj_pose[0].tolist(),
            "goals": obj_pose.tolist(),
        },
        "hand": {
            "wrist_goals": wrist_pose.tolist(),
            "fingertip_goals": traj["fingertips_w"].tolist(),
            "fingertip_local": traj["fingertips_local"].tolist(),
        },
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

    return dict(
        T=T_full,
        object_pose=object_pose,
        wrist_pos=wrist_pos,
        wrist_quat=wrist_quat,
        fingertips_w=ftips_w,
        fingertips_local=ftips_local,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--maniptrans-root", default="/home/intel/Codes/ManipTrans")
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
    args = p.parse_args()

    out_path = Path(args.project_root) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mock:
        full = _mock_trajectory()
        urdf = "assets/urdf/dextoolbench/handle_head_primitives/object_001.urdf"
        src = "mock/cylinder_rotation"
    else:
        data = _load_grab_demo(Path(args.maniptrans_root), args.data_idx)
        full = _build_full_trajectory(data)
        urdf = str(
            Path(args.maniptrans_root) / "data" / "grab_demo" / "102" / "102_obj.urdf"
        )
        src = f"ManipTrans/grab_demo/{args.data_idx}"

    sub = _downsample(full, args.src_fps, args.target_fps)
    j = _to_json(sub, src=src, fps=args.target_fps, urdf_path=urdf)
    with open(out_path, "w") as f:
        json.dump(j, f, indent=2)

    n = len(j["object"]["goals"])
    print(f"Wrote {out_path} with {n} goals (target fps {args.target_fps})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
