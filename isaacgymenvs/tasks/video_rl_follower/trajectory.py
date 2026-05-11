"""Reference-trajectory loader for VideoRLFollower.

A trajectory file is a JSON with this schema (see tools/process_maniptrans_trajectory.py):

{
    "meta": {
        "source": "ManipTrans/grab_demo/g0",
        "fps": 3.0,                     # goal sampling rate (Hz)
        "n_fingertips": 5,
        "fingertip_order": ["thumb", "index", "middle", "ring", "pinky"]
    },
    "object": {
        "urdf_path": "data/objects/102/102_obj.urdf",
        "scale": 1.0,                                     # uniform mesh scale
        "grasp_bbox_scale": [0.10, 0.05, 0.05],           # OPTIONAL: dx, dy, dz of grasp bbox
                                                          #   used by the env's keypoint reward.
                                                          #   Defaults to [0.06, 0.06, 0.06] (~teapot).
        "need_vhacd": false,                              # OPTIONAL, default false
        "start_pose": [x, y, z, qx, qy, qz, qw],
        "goals": [[x,y,z,qx,qy,qz,qw], ...]               # (T, 7)
    },
    "hand": {
        "wrist_goals":     [[x,y,z,qx,qy,qz,qw], ...],    # (T, 7)
        "fingertip_goals": [[[fx,fy,fz]*5], ...],         # (T, 5, 3) world frame
        "fingertip_local": [[[fx,fy,fz]*5], ...],         # (T, 5, 3) wrist-local frame

        # OPTIONAL — full hand skeleton.  When present these are used by the
        # IsaacGym visualiser to draw the 21-keypoint skeleton:
        "joints_world":    [[[x,y,z]*21], ...],           # (T, 21, 3) world frame
        "joints_local":    [[[x,y,z]*21], ...]            # (T, 21, 3) wrist-local frame
    }
}

T must match between every (T, *) array.

The 21 hand keypoints follow the MANO ordering (this is what
``tools/process_maniptrans_trajectory.py`` writes when --with-skeleton is
enabled, and what the visualiser will use to draw the skeleton):

    0  : wrist
    1-3: index_proximal, index_intermediate, index_distal
    4-6: middle_proximal, middle_intermediate, middle_distal
    7-9: pinky_proximal, pinky_intermediate, pinky_distal
   10-12: ring_proximal, ring_intermediate, ring_distal
   13-15: thumb_proximal, thumb_intermediate, thumb_distal
   16-20: index_tip, middle_tip, pinky_tip, ring_tip, thumb_tip

Bones (parent → child pairs) for skeleton rendering — 20 segments:

    (0,1) (1,2) (2,3) (3,16)            # index   chain
    (0,4) (4,5) (5,6) (6,17)            # middle  chain
    (0,7) (7,8) (8,9) (9,18)            # pinky   chain
    (0,10) (10,11) (11,12) (12,19)      # ring    chain
    (0,13) (13,14) (14,15) (15,20)      # thumb   chain
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


_REQUIRED_TOP = {"meta", "object", "hand"}
_REQUIRED_OBJ = {"start_pose", "goals"}
_REQUIRED_HAND = {"wrist_goals", "fingertip_goals", "fingertip_local"}


class ReferenceTrajectory:
    """In-memory representation of a reference trajectory.

    All tensors are stored on CPU as float32; move to device with `.to(device)`
    before training.
    """

    # Standard 21-keypoint MANO ordering used by tools/process_maniptrans_trajectory.py
    MANO_JOINT_ORDER = [
        "wrist",
        "index_proximal",  "index_intermediate",  "index_distal",
        "middle_proximal", "middle_intermediate", "middle_distal",
        "pinky_proximal",  "pinky_intermediate",  "pinky_distal",
        "ring_proximal",   "ring_intermediate",   "ring_distal",
        "thumb_proximal",  "thumb_intermediate",  "thumb_distal",
        "index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip",
    ]
    # 20 bones (parent→child indices)
    MANO_BONE_PAIRS = [
        (0, 1), (1, 2), (2, 3), (3, 16),       # index
        (0, 4), (4, 5), (5, 6), (6, 17),       # middle
        (0, 7), (7, 8), (8, 9), (9, 18),       # pinky
        (0, 10), (10, 11), (11, 12), (12, 19), # ring
        (0, 13), (13, 14), (14, 15), (15, 20), # thumb
    ]
    # Per-finger colour groups: (joint indices, RGB).  Wrist (joint 0) gets
    # a neutral grey since it belongs to all chains.
    MANO_FINGER_GROUPS = [
        ([0],                  (0.50, 0.50, 0.50)),  # wrist        — grey
        ([1, 2, 3, 16],        (0.22, 0.49, 0.72)),  # index        — blue
        ([4, 5, 6, 17],        (0.30, 0.69, 0.29)),  # middle       — green
        ([7, 8, 9, 18],        (1.00, 0.50, 0.00)),  # pinky        — orange
        ([10, 11, 12, 19],     (0.60, 0.31, 0.64)),  # ring         — purple
        ([13, 14, 15, 20],     (0.89, 0.10, 0.11)),  # thumb        — red
    ]

    def __init__(
        self,
        meta: Dict,
        object_start_pose: torch.Tensor,            # (7,)
        object_goals: torch.Tensor,                  # (T, 7)
        wrist_goals: torch.Tensor,                   # (T, 7)
        fingertip_goals: torch.Tensor,               # (T, K, 3) world frame
        fingertip_local: torch.Tensor,               # (T, K, 3) wrist-local frame
        object_urdf_path: Optional[str] = None,
        object_scale: float = 1.0,
        object_grasp_bbox_scale: Optional[Tuple[float, float, float]] = None,
        object_need_vhacd: bool = False,
        joints_world: Optional[torch.Tensor] = None,   # (T, 21, 3) optional
        joints_local: Optional[torch.Tensor] = None,   # (T, 21, 3) optional
        # ManipTrans-retargeted dexhand link positions in the SAME gym frame
        # the env spawns the robot in.  Shape (T, n_dex_links, 3); enables the
        # full-hand-tracking reward (28 link weighted MSE in our Sharpa env).
        dex_links_world: Optional[torch.Tensor] = None,
        # Companion: per-frame dexhand joint angles (T, n_dof).  Used as
        # initialisation hint or as a soft regulariser; stored but not
        # required for the position-based reward.
        dex_dof_pos: Optional[torch.Tensor] = None,
        dex_wrist_pos: Optional[torch.Tensor] = None,
        dex_wrist_rot: Optional[torch.Tensor] = None,
        # Per-link names (so the env knows which q maps to which body)
        dex_link_names: Optional[List[str]] = None,
        # Per-frame Kuka iiwa14 IK joint angles (T, 7) so reset_idx can warm-
        # start the arm to the trajectory's wrist pose.  Computed offline by
        # tools/compute_ik_for_trajectory.py.
        kuka_dof_pos: Optional[torch.Tensor] = None,
        # Actual wrist pose achieved by the IK solution (T, 3) / (T, 4 xyzw).
        # Differs from `dex_wrist_pos` by the IK residual (~0-6cm at workspace
        # boundaries).  reset_object_pose translates the object by
        # `wrist_pos_ik - dex_wrist_pos` so the relative hand-object grasp
        # config matches the retarget.
        wrist_pos_ik: Optional[torch.Tensor] = None,
        wrist_quat_ik: Optional[torch.Tensor] = None,
        # ★ Velocity fields (paper-faithful, computed by prep_60hz_with_velocities.py
        # using np.gradient + gaussian σ=2 at dt=1/60s — matches
        # ManipTrans/main/dataset/base.py:57-83).  Optional — None if 3Hz
        # legacy traj or if velocities haven't been precomputed.
        kuka_dof_velocity: Optional[torch.Tensor] = None,
        dex_dof_velocity: Optional[torch.Tensor] = None,
        dex_wrist_velocity: Optional[torch.Tensor] = None,
        dex_wrist_angular_velocity: Optional[torch.Tensor] = None,
        obj_velocity: Optional[torch.Tensor] = None,
        obj_angular_velocity: Optional[torch.Tensor] = None,
        joints_velocity: Optional[torch.Tensor] = None,
    ) -> None:
        self.meta = meta
        self.object_urdf_path = object_urdf_path
        self.object_scale = float(object_scale)
        self.object_grasp_bbox_scale = (
            tuple(object_grasp_bbox_scale)
            if object_grasp_bbox_scale is not None
            else (0.06, 0.06, 0.06)
        )
        self.object_need_vhacd = bool(object_need_vhacd)

        if object_start_pose.shape != (7,):
            raise ValueError(
                f"object_start_pose must be shape (7,), got {tuple(object_start_pose.shape)}"
            )
        if object_goals.ndim != 2 or object_goals.shape[1] != 7:
            raise ValueError(
                f"object_goals must be shape (T, 7), got {tuple(object_goals.shape)}"
            )
        if wrist_goals.shape != object_goals.shape:
            raise ValueError(
                f"wrist_goals must be shape (T, 7), got {tuple(wrist_goals.shape)}"
            )
        T = object_goals.shape[0]
        if T == 0:
            raise ValueError("trajectory must contain at least one goal frame")
        if fingertip_goals.ndim != 3 or fingertip_goals.shape[0] != T or fingertip_goals.shape[2] != 3:
            raise ValueError(
                "fingertip_goals must be shape (T, K, 3), got "
                f"{tuple(fingertip_goals.shape)} (T={T})"
            )
        if fingertip_local.shape != fingertip_goals.shape:
            raise ValueError(
                "fingertip_local must match fingertip_goals shape, got "
                f"{tuple(fingertip_local.shape)} vs {tuple(fingertip_goals.shape)}"
            )
        K = int(fingertip_goals.shape[1])
        if K == 0:
            raise ValueError("trajectory must declare at least one fingertip")

        order = list(meta.get("fingertip_order", []))
        if order and len(order) != K:
            raise ValueError(
                f"meta.fingertip_order has {len(order)} entries but the data has K={K}"
            )

        self.num_goals = T
        self.num_fingertips = K

        self.object_start_pose = object_start_pose.float()           # (7,)
        self.object_goals = object_goals.float()                     # (T, 7)
        self.wrist_goals = wrist_goals.float()                       # (T, 7)
        self.fingertip_goals = fingertip_goals.float()               # (T, K, 3)
        self.fingertip_local = fingertip_local.float()               # (T, K, 3)

        # Optional 21-keypoint hand skeleton (MANO ordering — see docstring)
        for arr in (joints_world, joints_local):
            if arr is not None:
                if arr.ndim != 3 or arr.shape != (T, 21, 3):
                    raise ValueError(
                        "joints_world / joints_local must be shape (T, 21, 3), "
                        f"got {tuple(arr.shape)} (T={T})"
                    )
        self.joints_world = joints_world.float() if joints_world is not None else None
        self.joints_local = joints_local.float() if joints_local is not None else None
        self.has_skeleton = self.joints_world is not None

        # Optional ManipTrans retarget output (full-hand tracking)
        if dex_links_world is not None:
            if dex_links_world.ndim != 3 or dex_links_world.shape[0] != T or dex_links_world.shape[2] != 3:
                raise ValueError(
                    "dex_links_world must be shape (T, L, 3), got "
                    f"{tuple(dex_links_world.shape)} (T={T})"
                )
        if dex_dof_pos is not None:
            if dex_dof_pos.ndim != 2 or dex_dof_pos.shape[0] != T:
                raise ValueError(
                    f"dex_dof_pos must be shape (T, n_dof), got {tuple(dex_dof_pos.shape)}"
                )
        self.dex_links_world = (
            dex_links_world.float() if dex_links_world is not None else None
        )
        self.dex_dof_pos = (
            dex_dof_pos.float() if dex_dof_pos is not None else None
        )
        self.dex_link_names = list(dex_link_names) if dex_link_names is not None else None
        self.dex_wrist_pos = dex_wrist_pos.float() if dex_wrist_pos is not None else None
        self.dex_wrist_rot = dex_wrist_rot.float() if dex_wrist_rot is not None else None
        if kuka_dof_pos is not None:
            if kuka_dof_pos.ndim != 2 or kuka_dof_pos.shape != (T, 7):
                raise ValueError(
                    f"kuka_dof_pos must be shape ({T}, 7), got {tuple(kuka_dof_pos.shape)}"
                )
        self.kuka_dof_pos = kuka_dof_pos.float() if kuka_dof_pos is not None else None
        self.wrist_pos_ik = wrist_pos_ik.float() if wrist_pos_ik is not None else None
        self.wrist_quat_ik = wrist_quat_ik.float() if wrist_quat_ik is not None else None
        self.has_retargeted = self.dex_links_world is not None
        # ★ TRACKING BRANCH velocity storage (Tier 0)
        self.kuka_dof_velocity = kuka_dof_velocity.float() if kuka_dof_velocity is not None else None
        self.dex_dof_velocity = dex_dof_velocity.float() if dex_dof_velocity is not None else None
        self.dex_wrist_velocity = dex_wrist_velocity.float() if dex_wrist_velocity is not None else None
        self.dex_wrist_angular_velocity = dex_wrist_angular_velocity.float() if dex_wrist_angular_velocity is not None else None
        self.obj_velocity = obj_velocity.float() if obj_velocity is not None else None
        self.obj_angular_velocity = obj_angular_velocity.float() if obj_angular_velocity is not None else None
        self.joints_velocity = joints_velocity.float() if joints_velocity is not None else None
        self.has_velocities = self.obj_velocity is not None

        # Frames where IK can land the wrist within `reachable_ik_threshold_m`
        # of the trajectory's target.  Frames outside this set spawn the dex
        # hand below the table (kuka workspace boundary) — random init must
        # avoid them or the fingers penetrate at reset.  None when no IK data.
        self.reachable_ik_threshold_m = 0.01  # 1 cm, configurable post-init
        if self.wrist_pos_ik is not None and self.dex_wrist_pos is not None:
            err = (self.wrist_pos_ik - self.dex_wrist_pos).norm(dim=-1)  # (T,)
            self.reachable_frames = err < self.reachable_ik_threshold_m  # (T,) bool
        else:
            self.reachable_frames = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "ReferenceTrajectory":
        path = Path(path)
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict) -> "ReferenceTrajectory":
        missing = _REQUIRED_TOP - set(d.keys())
        if missing:
            raise ValueError(f"trajectory dict missing top-level keys: {missing}")
        if not _REQUIRED_OBJ <= set(d["object"].keys()):
            raise ValueError(f"object section missing keys (need {_REQUIRED_OBJ})")
        if not _REQUIRED_HAND <= set(d["hand"].keys()):
            raise ValueError(f"hand section missing keys (need {_REQUIRED_HAND})")

        obj = d["object"]
        hand = d["hand"]

        object_start = torch.as_tensor(obj["start_pose"], dtype=torch.float32)  # (7,)
        object_goals = torch.as_tensor(obj["goals"], dtype=torch.float32)        # (T, 7)
        wrist_goals = torch.as_tensor(hand["wrist_goals"], dtype=torch.float32)  # (T, 7)
        fingertip_goals = torch.as_tensor(
            hand["fingertip_goals"], dtype=torch.float32
        )                                                                        # (T, K, 3)
        fingertip_local = torch.as_tensor(
            hand["fingertip_local"], dtype=torch.float32
        )                                                                        # (T, K, 3)

        grasp_bbox = obj.get("grasp_bbox_scale")
        if grasp_bbox is not None:
            if len(grasp_bbox) != 3:
                raise ValueError(
                    f"object.grasp_bbox_scale must have 3 entries, got {len(grasp_bbox)}"
                )
            try:
                grasp_bbox = tuple(float(v) for v in grasp_bbox)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"object.grasp_bbox_scale entries must be numeric, got {grasp_bbox!r}"
                ) from exc
            import math
            for v in grasp_bbox:
                if not math.isfinite(v) or v <= 0.0:
                    raise ValueError(
                        "object.grasp_bbox_scale entries must be finite and "
                        f"strictly positive, got {grasp_bbox}"
                    )

        # Optional 21-keypoint hand skeleton
        joints_world = None
        joints_local = None
        if "joints_world" in hand:
            joints_world = torch.as_tensor(
                hand["joints_world"], dtype=torch.float32
            )
        if "joints_local" in hand:
            joints_local = torch.as_tensor(
                hand["joints_local"], dtype=torch.float32
            )

        # Optional ManipTrans-retargeted dexhand state
        dex_links_world = None
        dex_dof_pos = None
        dex_link_names = None
        dex_wrist_pos = None
        dex_wrist_rot = None
        if "dex" in d:
            dex = d["dex"]
            if "links_world" in dex:
                dex_links_world = torch.as_tensor(dex["links_world"], dtype=torch.float32)
            if "dof_pos" in dex:
                dex_dof_pos = torch.as_tensor(dex["dof_pos"], dtype=torch.float32)
            if "link_names" in dex:
                dex_link_names = list(dex["link_names"])
            if "wrist_pos" in dex:
                dex_wrist_pos = torch.as_tensor(dex["wrist_pos"], dtype=torch.float32)
            if "wrist_rot" in dex:
                dex_wrist_rot = torch.as_tensor(dex["wrist_rot"], dtype=torch.float32)
            if "kuka_dof_pos" in dex:
                _kuka_dof = torch.as_tensor(dex["kuka_dof_pos"], dtype=torch.float32)
            else:
                _kuka_dof = None
            if "wrist_pos_ik" in dex:
                _wrist_pos_ik = torch.as_tensor(dex["wrist_pos_ik"], dtype=torch.float32)
            else:
                _wrist_pos_ik = None
            if "wrist_quat_ik" in dex:
                _wrist_quat_ik = torch.as_tensor(dex["wrist_quat_ik"], dtype=torch.float32)
            else:
                _wrist_quat_ik = None
            # ★ TRACKING BRANCH: paper-style velocity fields (per
            # ManipTrans dataset/base.py:57-83 — np.gradient + gaussian σ=2).
            _kuka_dof_vel = torch.as_tensor(dex["kuka_dof_velocity"], dtype=torch.float32) if "kuka_dof_velocity" in dex else None
            _dex_dof_vel  = torch.as_tensor(dex["dof_velocity"], dtype=torch.float32) if "dof_velocity" in dex else None
            _dex_wrist_vel     = torch.as_tensor(dex["wrist_velocity"], dtype=torch.float32) if "wrist_velocity" in dex else None
            _dex_wrist_ang_vel = torch.as_tensor(dex["wrist_angular_velocity"], dtype=torch.float32) if "wrist_angular_velocity" in dex else None

        # Object velocities (paper recipe)
        _obj_vel     = torch.as_tensor(obj["velocity"], dtype=torch.float32) if "velocity" in obj else None
        _obj_ang_vel = torch.as_tensor(obj["angular_velocity"], dtype=torch.float32) if "angular_velocity" in obj else None

        # MANO joint velocities (paper recipe, used in R_imit)
        _joints_vel = torch.as_tensor(hand["joints_velocity"], dtype=torch.float32) if "joints_velocity" in hand else None

        return cls(
            meta=d.get("meta", {}),
            object_start_pose=object_start,
            object_goals=object_goals,
            wrist_goals=wrist_goals,
            fingertip_goals=fingertip_goals,
            fingertip_local=fingertip_local,
            object_urdf_path=obj.get("urdf_path"),
            object_scale=obj.get("scale", 1.0),
            object_grasp_bbox_scale=grasp_bbox,
            object_need_vhacd=bool(obj.get("need_vhacd", False)),
            joints_world=joints_world,
            joints_local=joints_local,
            dex_links_world=dex_links_world,
            dex_dof_pos=dex_dof_pos,
            dex_link_names=dex_link_names,
            dex_wrist_pos=dex_wrist_pos,
            dex_wrist_rot=dex_wrist_rot,
            kuka_dof_pos=_kuka_dof,
            wrist_pos_ik=_wrist_pos_ik,
            wrist_quat_ik=_wrist_quat_ik,
            kuka_dof_velocity=_kuka_dof_vel,
            dex_dof_velocity=_dex_dof_vel,
            dex_wrist_velocity=_dex_wrist_vel,
            dex_wrist_angular_velocity=_dex_wrist_ang_vel,
            obj_velocity=_obj_vel,
            obj_angular_velocity=_obj_ang_vel,
            joints_velocity=_joints_vel,
        )

    # ------------------------------------------------------------------
    # device
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> "ReferenceTrajectory":
        self.object_start_pose = self.object_start_pose.to(device)
        self.object_goals = self.object_goals.to(device)
        self.wrist_goals = self.wrist_goals.to(device)
        self.fingertip_goals = self.fingertip_goals.to(device)
        self.fingertip_local = self.fingertip_local.to(device)
        if self.joints_world is not None:
            self.joints_world = self.joints_world.to(device)
        if self.joints_local is not None:
            self.joints_local = self.joints_local.to(device)
        if self.dex_links_world is not None:
            self.dex_links_world = self.dex_links_world.to(device)
        if self.dex_dof_pos is not None:
            self.dex_dof_pos = self.dex_dof_pos.to(device)
        if self.kuka_dof_pos is not None:
            self.kuka_dof_pos = self.kuka_dof_pos.to(device)
        if self.wrist_pos_ik is not None:
            self.wrist_pos_ik = self.wrist_pos_ik.to(device)
        if self.wrist_quat_ik is not None:
            self.wrist_quat_ik = self.wrist_quat_ik.to(device)
        if self.dex_wrist_pos is not None:
            self.dex_wrist_pos = self.dex_wrist_pos.to(device)
        if self.dex_wrist_rot is not None:
            self.dex_wrist_rot = self.dex_wrist_rot.to(device)
        if self.reachable_frames is not None:
            self.reachable_frames = self.reachable_frames.to(device)
        # ★ TRACKING BRANCH velocity fields
        if self.kuka_dof_velocity is not None:
            self.kuka_dof_velocity = self.kuka_dof_velocity.to(device)
        if self.dex_dof_velocity is not None:
            self.dex_dof_velocity = self.dex_dof_velocity.to(device)
        if self.dex_wrist_velocity is not None:
            self.dex_wrist_velocity = self.dex_wrist_velocity.to(device)
        if self.dex_wrist_angular_velocity is not None:
            self.dex_wrist_angular_velocity = self.dex_wrist_angular_velocity.to(device)
        if self.obj_velocity is not None:
            self.obj_velocity = self.obj_velocity.to(device)
        if self.obj_angular_velocity is not None:
            self.obj_angular_velocity = self.obj_angular_velocity.to(device)
        if self.joints_velocity is not None:
            self.joints_velocity = self.joints_velocity.to(device)
        if self.dex_links_world is not None:
            self.dex_links_world = self.dex_links_world.to(device)
        return self

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def stacked_goals(self) -> torch.Tensor:
        """Concatenate object_goals (7) + wrist_goals (7) + fingertip_local flat (K*3).

        Returns: (T, 14 + K*3)
        Used by the env to fill ``trajectory_states`` with a single dense tensor.
        """
        T = self.num_goals
        return torch.cat(
            [
                self.object_goals,                                  # (T, 7)
                self.wrist_goals,                                   # (T, 7)
                self.fingertip_local.reshape(T, -1),                # (T, K*3)
            ],
            dim=-1,
        )

    @property
    def total_goal_dim(self) -> int:
        return 7 + 7 + 3 * self.num_fingertips
