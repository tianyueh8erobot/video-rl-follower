"""Reference trajectory loader for DexTrack-Sharpa-Franka residual policy training.

Loads the 29-DOF joint29 npy produced by `tools/dextrack_sharpa/joint_29dof_ik.py`
and pre-computes:
  - dof_pos     : (T, 29)         per-frame joint targets
  - dof_vel     : (T, 29)         finite-difference + gaussian-smoothed velocity
  - obj_pos     : (T, 3)
  - obj_quat    : (T, 4)          xyzw
  - obj_lin_vel : (T, 3)
  - obj_ang_vel : (T, 3)
  - link_pos    : (T, K, 3)       K = 28 body_names per Sharpa class (FK at each frame)
  - link_vel    : (T, K, 3)
  - wrist_pos   : (T, 3)          right_hand_C_MC origin
  - wrist_quat  : (T, 4)          xyzw
  - wrist_vel   : (T, 3)
  - wrist_ang_vel: (T, 3)

Velocity convention matches ManipTrans/SimToolReal (np.gradient on positions, then
gaussian_filter1d with sigma=2).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Sharpa body_names (28 entries; matches ManipTrans/dexhands/sharpa.py)
# index 0 = wrist (right_hand_C_MC); 1-27 follow the standard Sharpa chain.
# ---------------------------------------------------------------------------
SHARPA_BODY_NAMES = [
    "right_hand_C_MC",
    # index 1-5
    "right_index_MCP_VL", "right_index_PP", "right_index_MP", "right_index_DP", "right_index_fingertip",
    # middle 6-10
    "right_middle_MCP_VL", "right_middle_PP", "right_middle_MP", "right_middle_DP", "right_middle_fingertip",
    # pinky 11-16
    "right_pinky_MC", "right_pinky_MCP_VL", "right_pinky_PP", "right_pinky_MP", "right_pinky_DP", "right_pinky_fingertip",
    # ring 17-21
    "right_ring_MCP_VL", "right_ring_PP", "right_ring_MP", "right_ring_DP", "right_ring_fingertip",
    # thumb 22-27
    "right_thumb_CMC_VL", "right_thumb_MC", "right_thumb_MCP_VL", "right_thumb_PP", "right_thumb_DP", "right_thumb_fingertip",
]
assert len(SHARPA_BODY_NAMES) == 28

WRIST_LINK = "right_hand_C_MC"


def _strip_mujoco(s: str) -> str:
    return re.sub(r"<mujoco>.*?</mujoco>", "", s, flags=re.DOTALL)


def _smoothed_velocity(p: np.ndarray, dt: float, sigma: float = 2.0) -> np.ndarray:
    """np.gradient then gaussian_filter1d along time axis."""
    v = np.gradient(p, dt, axis=0)
    return gaussian_filter1d(v, sigma=sigma, axis=0, mode="nearest")


def _quat_to_angular_velocity(q_xyzw: np.ndarray, dt: float, sigma: float = 2.0) -> np.ndarray:
    """From a sequence of quaternions, compute angular velocity (T, 3)."""
    rot = R.from_quat(q_xyzw)
    rotvec = rot.as_rotvec()
    # unwrap rotvec by tracking continuity
    for i in range(1, len(rotvec)):
        diff = rotvec[i] - rotvec[i - 1]
        if np.linalg.norm(diff) > np.pi:
            # adjust by removing 2π aliasing
            axis = diff / (np.linalg.norm(diff) + 1e-9)
            rotvec[i] -= 2 * np.pi * axis
    omega = _smoothed_velocity(rotvec, dt, sigma)
    return omega


class DexTrackTrajectory:
    """Pre-computed reference trajectory + per-frame state for tracking RL."""

    def __init__(
        self,
        npy_path: str,
        urdf_path: str,
        dt: float = 1.0 / 60.0,        # control freq (60Hz matches DexTrack default)
        device: str = "cuda:0",
        smooth_sigma: float = 2.0,
    ):
        self.npy_path = npy_path
        self.urdf_path = urdf_path
        self.dt = dt
        self.device = device

        data = np.load(npy_path, allow_pickle=True).item()
        dof_pos_np = np.asarray(data.get("hand_qs", data.get("robot_delta_states_weights_np")),
                                dtype=np.float32)        # (T, 29)
        obj_pos_np = np.asarray(data["object_transl"], dtype=np.float32)   # (T, 3)
        obj_quat_np = np.asarray(data["object_rot_quat"], dtype=np.float32) # (T, 4) xyzw
        T = dof_pos_np.shape[0]
        assert obj_pos_np.shape[0] == T and obj_quat_np.shape[0] == T
        assert dof_pos_np.shape[1] == 29, f"expected 29 DOF, got {dof_pos_np.shape[1]}"
        self.T = T
        print(f"[DexTrackTrajectory] T={T} DOF=29 from {npy_path}")

        # FK on every frame to get all body link world poses
        import pytorch_kinematics as pk
        with open(urdf_path) as f:
            chain = pk.build_chain_from_urdf(_strip_mujoco(f.read()))
        chain = chain.to(dtype=torch.float32, device=device)

        joint_names = chain.get_joint_parameter_names()
        assert len(joint_names) == 29, f"chain has {len(joint_names)} DOF, expected 29"
        self.joint_names = joint_names

        # Build mapping: data DOF order is assumed to be Franka first then Sharpa.
        # Verify by checking joint_names prefix.
        n_franka = sum(1 for n in joint_names if n.startswith("panda_joint"))
        n_sharpa = sum(1 for n in joint_names if not n.startswith("panda_joint"))
        assert n_franka == 7 and n_sharpa == 22, f"got {n_franka} Franka + {n_sharpa} Sharpa"

        # FK in batches (the chain.forward_kinematics handles batched q)
        with torch.no_grad():
            q_t = torch.tensor(dof_pos_np, device=device)
            ret = chain.forward_kinematics(q_t)
            link_pos_list = []
            for bname in SHARPA_BODY_NAMES:
                m = ret[bname].get_matrix()  # (T, 4, 4)
                link_pos_list.append(m[:, :3, 3].cpu().numpy())
            link_pos_np = np.stack(link_pos_list, axis=1)  # (T, K, 3)
            # Wrist orientation (matrix → quat xyzw)
            wrist_m = ret[WRIST_LINK].get_matrix().cpu().numpy()  # (T, 4, 4)
            wrist_rotmat = wrist_m[:, :3, :3]
            wrist_quat_np = R.from_matrix(wrist_rotmat).as_quat().astype(np.float32)  # xyzw
            wrist_pos_np = wrist_m[:, :3, 3].astype(np.float32)
        print(f"[DexTrackTrajectory] FK done: link_pos {link_pos_np.shape}, "
              f"wrist_pos {wrist_pos_np.shape}, wrist_quat {wrist_quat_np.shape}")

        # Velocities (np.gradient + gaussian)
        dof_vel_np = _smoothed_velocity(dof_pos_np, dt, smooth_sigma)
        obj_lin_vel_np = _smoothed_velocity(obj_pos_np, dt, smooth_sigma)
        obj_ang_vel_np = _quat_to_angular_velocity(obj_quat_np, dt, smooth_sigma)
        wrist_lin_vel_np = _smoothed_velocity(wrist_pos_np, dt, smooth_sigma)
        wrist_ang_vel_np = _quat_to_angular_velocity(wrist_quat_np, dt, smooth_sigma)
        link_vel_np = _smoothed_velocity(link_pos_np, dt, smooth_sigma)  # (T, K, 3)

        # All tensors on device
        to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        self.dof_pos      = to_t(dof_pos_np)         # (T, 29)
        self.dof_vel      = to_t(dof_vel_np)
        self.obj_pos      = to_t(obj_pos_np)         # (T, 3)
        self.obj_quat     = to_t(obj_quat_np)        # (T, 4) xyzw
        self.obj_lin_vel  = to_t(obj_lin_vel_np)
        self.obj_ang_vel  = to_t(obj_ang_vel_np)
        self.link_pos     = to_t(link_pos_np)        # (T, 28, 3)
        self.link_vel     = to_t(link_vel_np)
        self.wrist_pos    = to_t(wrist_pos_np)       # (T, 3)
        self.wrist_quat   = to_t(wrist_quat_np)
        self.wrist_lin_vel = to_t(wrist_lin_vel_np)
        self.wrist_ang_vel = to_t(wrist_ang_vel_np)

    def __len__(self):
        return self.T

    def get(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-env time index → ref state dict.  t: (num_envs,) long tensor in [0, T).

        Out-of-range indices clamped to last frame.
        """
        t = torch.clamp(t, 0, self.T - 1)
        return {
            "dof_pos":      self.dof_pos[t],
            "dof_vel":      self.dof_vel[t],
            "obj_pos":      self.obj_pos[t],
            "obj_quat":     self.obj_quat[t],
            "obj_lin_vel":  self.obj_lin_vel[t],
            "obj_ang_vel":  self.obj_ang_vel[t],
            "link_pos":     self.link_pos[t],
            "link_vel":     self.link_vel[t],
            "wrist_pos":    self.wrist_pos[t],
            "wrist_quat":   self.wrist_quat[t],
            "wrist_vel":    self.wrist_lin_vel[t],
            "wrist_ang_vel":self.wrist_ang_vel[t],
        }


def _smoke_test():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    npy = "/home/intel/Codes/video-rl-follower/data/sharpa_retarget_dextrack/s2_cubesmall_inspect_1_joint29_replay.npy"
    urdf = "/home/intel/Codes/video-rl-follower/assets/urdf/franka_sharpa_description/franka_panda_sharpa.urdf"
    traj = DexTrackTrajectory(npy, urdf, device="cuda:0")
    print(f"  T={len(traj)}")
    print(f"  dof_pos range: [{traj.dof_pos.min():.3f}, {traj.dof_pos.max():.3f}]")
    print(f"  obj_pos range: x[{traj.obj_pos[:,0].min():.3f},{traj.obj_pos[:,0].max():.3f}] "
          f"y[{traj.obj_pos[:,1].min():.3f},{traj.obj_pos[:,1].max():.3f}] "
          f"z[{traj.obj_pos[:,2].min():.3f},{traj.obj_pos[:,2].max():.3f}]")
    print(f"  link_pos shape: {traj.link_pos.shape}")
    print(f"  wrist_pos[0]: {traj.wrist_pos[0].cpu().numpy().round(3)}")
    print(f"  wrist_quat[0]: {traj.wrist_quat[0].cpu().numpy().round(3)}")
    print(f"  obj_lin_vel mean: {traj.obj_lin_vel.norm(dim=-1).mean():.4f} m/s")
    print(f"  obj_ang_vel mean: {traj.obj_ang_vel.norm(dim=-1).mean():.4f} rad/s")
    # Index test
    t = torch.tensor([0, 1, 50, 150, 299], device="cuda:0")
    s = traj.get(t)
    assert s["dof_pos"].shape == (5, 29)
    assert s["link_pos"].shape == (5, 28, 3)
    print(f"  ✓ smoke test passed")


if __name__ == "__main__":
    _smoke_test()
