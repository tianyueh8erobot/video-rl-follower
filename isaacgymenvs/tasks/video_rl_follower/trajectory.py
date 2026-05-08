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
        "fingertip_local": [[[fx,fy,fz]*5], ...]          # (T, 5, 3) wrist-local frame
    }
}

T must match between object.goals, wrist_goals, fingertip_goals, fingertip_local.
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
