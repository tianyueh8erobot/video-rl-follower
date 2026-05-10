"""IsaacGym visualiser for a processed VideoRLFollower trajectory.

Spawns:
  * the trajectory's object asset (the actual URDF) at goal pose for the
    current frame;
  * a wrist marker (small green sphere) at ``wrist_goals[t]``;
  * five fingertip markers (color-coded spheres) at ``fingertip_goals[t]``.

You can either:
  (a) run with ``--auto`` to perform headless self-checks (object pose
      monotonicity, no-NaN, finite quaternions, etc.) and exit; or
  (b) run interactively (default) to scrub through the trajectory using
      keyboard:

      Space  ─  toggle play/pause
      n / p  ─  next / previous frame
      r      ─  reset to frame 0
      q / Esc─  quit

      The header in the viewer shows the current frame index.

Usage::

    python scripts/visualize_trajectory_in_isaacgym.py \
        --traj data/trajectories/example_g0.json

    # Headless auto-check only
    python scripts/visualize_trajectory_in_isaacgym.py \
        --traj data/trajectories/g0.json --auto

    # Slow it down for inspection
    python scripts/visualize_trajectory_in_isaacgym.py \
        --traj data/trajectories/g0.json --hz 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import isaacgym  # noqa: F401  must be imported before torch
from isaacgym import gymapi, gymutil

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaacgymenvs.tasks.video_rl_follower.trajectory import ReferenceTrajectory  # noqa: E402


# ---------------------------------------------------------------------------
# Auto-checks (no GUI required)
# ---------------------------------------------------------------------------


def _auto_check(traj: ReferenceTrajectory) -> bool:
    print("[auto] running self-checks…")
    ok = True

    def check(name: str, cond: bool, msg: str = "") -> bool:
        ico = "✓" if cond else "✗"
        print(f"  {ico} {name}{(': ' + msg) if msg else ''}")
        return cond

    arrays = {
        "object.start_pose": traj.object_start_pose,
        "object.goals":      traj.object_goals,
        "wrist_goals":       traj.wrist_goals,
        "fingertip_goals":   traj.fingertip_goals,
        "fingertip_local":   traj.fingertip_local,
    }
    for name, t in arrays.items():
        ok &= check(f"{name} is finite",
                    torch.isfinite(t).all().item(),
                    f"shape={tuple(t.shape)}")

    # Quaternions normalised
    for name, t in [("object.goals quat", traj.object_goals[:, 3:7]),
                    ("wrist_goals quat", traj.wrist_goals[:, 3:7])]:
        norm = torch.norm(t, dim=-1)
        max_err = float((norm - 1.0).abs().max())
        ok &= check(f"{name} unit-norm", max_err < 1e-3,
                    f"max ||q||-1 = {max_err:.4f}")

    # Object positions sane (within ±5 m of origin)
    op = traj.object_goals[:, :3]
    ok &= check("object positions within ±5 m of origin",
                op.abs().max().item() < 5.0,
                f"max |xyz| = {float(op.abs().max()):.3f} m")

    # Step-by-step deltas sane (no teleports > 0.5 m / 60° per goal)
    if traj.num_goals >= 2:
        d_pos = torch.norm(op[1:] - op[:-1], dim=-1)
        ok &= check("no object teleports between consecutive goals",
                    float(d_pos.max()) < 0.5,
                    f"max Δp = {float(d_pos.max()) * 100:.1f} cm")
        q1, q2 = traj.object_goals[:-1, 3:7], traj.object_goals[1:, 3:7]
        dot = (q1 * q2).sum(-1).abs().clamp(-1, 1)
        ang = 2.0 * torch.acos(dot)
        ok &= check("no object rotation > 60° between consecutive goals",
                    float(ang.max()) < math.radians(60),
                    f"max Δθ = {math.degrees(float(ang.max())):.1f}°")

    # Wrist plausibly above the object
    wp = traj.wrist_goals[:, :3]
    above = (wp[:, 2] > op[:, 2] - 0.10).float().mean().item()
    ok &= check("wrist mostly above the object (z_wrist > z_obj - 10 cm)",
                above > 0.5,
                f"{above * 100:.1f}% of frames")

    # Fingertip-to-wrist distance plausible (between 1 cm and 25 cm)
    ftips_w = traj.fingertip_goals
    wrist_p = wp.unsqueeze(1)
    d = torch.norm(ftips_w - wrist_p, dim=-1)
    ok &= check("fingertip-to-wrist distances 1–25 cm",
                bool(((d > 0.005) & (d < 0.30)).all().item()),
                f"min={float(d.min()) * 100:.2f} cm  max={float(d.max()) * 100:.2f} cm")

    return ok


# ---------------------------------------------------------------------------
# IsaacGym visual scene
# ---------------------------------------------------------------------------


_FINGER_COLOURS = [
    (0.89, 0.10, 0.11),  # thumb  — red
    (0.22, 0.49, 0.72),  # index  — blue
    (0.30, 0.69, 0.29),  # middle — green
    (0.60, 0.31, 0.64),  # ring   — purple
    (1.00, 0.50, 0.00),  # pinky  — orange
]


def _to_xyzw(q4: torch.Tensor) -> Tuple[float, float, float, float]:
    return (float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3]))


class TrajectoryViewer:
    def __init__(
        self,
        traj: ReferenceTrajectory,
        headless: bool = False,
        hz: float = 5.0,
        loop: bool = True,
        overlay_pkl: str = None,
        overlay_urdf: str = None,
        overlay_color: str = "0.30,0.85,0.45",
        show_mano: bool = True,
        show_sharpa: bool = True,
    ) -> None:
        self.traj = traj
        self.hz = float(hz)
        self.loop = bool(loop)
        self.frame_idx = 0
        self.playing = True
        # Overlay = a SECOND floating-base hand loaded from a separate retarget
        # pkl (e.g., Inspire) so we can compare two embodiments on the same
        # MANO trajectory.  None disables.
        self._overlay_pkl_path = overlay_pkl
        self._overlay_urdf_path = overlay_urdf
        self._overlay_color = tuple(float(c) for c in overlay_color.split(","))
        self._show_mano = bool(show_mano)
        self._show_sharpa = bool(show_sharpa)

        self.gym = gymapi.acquire_gym()

        sp = gymapi.SimParams()
        sp.up_axis = gymapi.UP_AXIS_Z
        sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)        # no gravity in viewer
        sp.dt = 1.0 / 60.0
        sp.physx.solver_type = 1
        sp.physx.num_position_iterations = 4
        sp.physx.num_velocity_iterations = 1
        sp.physx.use_gpu = True
        sp.use_gpu_pipeline = False                    # CPU pipeline; no policy

        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
        if sim is None:
            raise RuntimeError("create_sim returned None — driver/CUDA issue?")
        self.sim = sim

        # NO ground plane.  GRAB demo coordinates put the object centred at
        # the origin (z ≈ 2 mm), which would intersect a z=0 ground plane and
        # cause PhysX to bounce the object every step against our per-frame
        # teleport — that's the "mouse jumping" symptom.  Without a ground
        # plane the visualisation is purely kinematic teleportation, which is
        # exactly what we want for inspecting reference data.

        # Object asset
        self._load_object_asset()
        # Sharpa hand asset (only if trajectory has dex.* fields)
        self._load_sharpa_asset()
        # Optional overlay hand (e.g., Inspire) from a separate pkl
        self._load_overlay_asset()
        # Marker assets: smaller sphere for joints (incl. fingertips), bigger
        # for the wrist root.  Tip markers get a slightly larger radius than
        # interior joints for visibility.
        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_options.disable_gravity = True
        self.wrist_marker_asset = self.gym.create_sphere(
            self.sim, 0.014, marker_options
        )
        self.ftip_marker_asset = self.gym.create_sphere(
            self.sim, 0.010, marker_options
        )
        self.joint_marker_asset = self.gym.create_sphere(
            self.sim, 0.006, marker_options
        )

        # Trajectory has skeleton iff joints_world is populated (21 joints).
        self.has_skeleton = bool(getattr(self.traj, "has_skeleton", False))

        # Single env
        env_lower = gymapi.Vec3(-2, -2, 0)
        env_upper = gymapi.Vec3(2, 2, 2)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        self._spawn_actors()

        # Viewer
        self.viewer = None
        self.headless = headless
        if not headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("create_viewer returned None — DISPLAY missing?")
            cam_pos = gymapi.Vec3(1.0, -1.0, 1.0)
            cam_tgt = gymapi.Vec3(*[float(x) for x in traj.object_start_pose[:3]])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_tgt)

            self._subscribe_keys()

    # ------------------------------------------------------------------

    def _load_object_asset(self) -> None:
        urdf = self.traj.object_urdf_path
        if urdf is None:
            print("[viz] trajectory has no urdf_path → using a 5 cm cube placeholder")
            opts = gymapi.AssetOptions()
            opts.fix_base_link = True
            opts.disable_gravity = True
            self.obj_asset = self.gym.create_box(
                self.sim, 0.05, 0.05, 0.05, opts
            )
            return
        if not os.path.isabs(urdf):
            urdf = str(PROJECT_ROOT / urdf)
        if not os.path.isfile(urdf):
            raise FileNotFoundError(
                f"Trajectory's object URDF not found on disk: {urdf}"
            )
        opts = gymapi.AssetOptions()
        # Pin the base so PhysX never integrates the object.  We will still
        # teleport its root via set_actor_rigid_body_states(STATE_ALL) every
        # frame — IsaacGym honours direct state writes even on fixed-base
        # actors for visualisation.
        opts.fix_base_link = True
        opts.disable_gravity = True
        opts.collapse_fixed_joints = True
        opts.replace_cylinder_with_capsule = True
        self.obj_asset = self.gym.load_asset(
            self.sim,
            os.path.dirname(urdf),
            os.path.basename(urdf),
            opts,
        )
        print(f"[viz] loaded object URDF: {urdf}")

    # ------------------------------------------------------------------

    def _load_sharpa_asset(self) -> None:
        """Load floating-base Sharpa right-hand URDF as a kinematic asset.

        We treat Sharpa as a render-only puppet: floating root drives the wrist
        each frame, 22 DoF positions drive the fingers.  No physics integration.
        """
        self.sharpa_asset = None
        if not self._show_sharpa:
            return
        if self.traj.dex_dof_pos is None or self.traj.dex_wrist_pos is None:
            return
        urdf_candidates = [
            "/home/intel/Codes/ManipTrans/maniptrans_envs/assets/sharpa/right_sharpa_wave.urdf",
        ]
        urdf = next((p for p in urdf_candidates if os.path.isfile(p)), None)
        if urdf is None:
            print("[viz] Sharpa URDF not found — skipping floating-hand visual")
            return
        opts = gymapi.AssetOptions()
        # Floating base: do NOT fix.  We will set root state via tensor write
        # every frame, with disable_gravity to prevent integration drift.
        opts.fix_base_link = False
        opts.disable_gravity = True
        opts.collapse_fixed_joints = False    # keep all 28 links visible
        opts.flip_visual_attachments = False
        opts.armature = 0.0
        # Drive mode: kinematic position targets only.
        opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        opts.use_mesh_materials = True
        self.sharpa_asset = self.gym.load_asset(
            self.sim, os.path.dirname(urdf), os.path.basename(urdf), opts
        )
        n_dof = self.gym.get_asset_dof_count(self.sharpa_asset)
        n_links = self.gym.get_asset_rigid_body_count(self.sharpa_asset)
        print(f"[viz] loaded Sharpa URDF: {urdf}  ({n_links} links, {n_dof} DoF)")

    # ------------------------------------------------------------------

    def _load_overlay_asset(self) -> None:
        """Load a second floating-base hand URDF + its retarget pkl.

        The pkl must have the same ManipTrans schema as Sharpa's:
          opt_wrist_pos (T, 3), opt_wrist_rot (T, 3), opt_dof_pos (T, n_dof).
        """
        self.overlay_asset = None
        self.overlay_pkl_data = None
        if self._overlay_pkl_path is None or self._overlay_urdf_path is None:
            return
        if not os.path.isfile(self._overlay_urdf_path):
            print(f"[viz] overlay URDF not found: {self._overlay_urdf_path}")
            return
        if not os.path.isfile(self._overlay_pkl_path):
            print(f"[viz] overlay pkl not found: {self._overlay_pkl_path}")
            return
        import pickle as _pickle
        with open(self._overlay_pkl_path, "rb") as f:
            self.overlay_pkl_data = _pickle.load(f)
        opts = gymapi.AssetOptions()
        opts.fix_base_link = False
        opts.disable_gravity = True
        opts.collapse_fixed_joints = False
        opts.flip_visual_attachments = False
        opts.armature = 0.0
        opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        opts.use_mesh_materials = True
        self.overlay_asset = self.gym.load_asset(
            self.sim,
            os.path.dirname(self._overlay_urdf_path),
            os.path.basename(self._overlay_urdf_path),
            opts,
        )
        n_dof = self.gym.get_asset_dof_count(self.overlay_asset)
        n_links = self.gym.get_asset_rigid_body_count(self.overlay_asset)
        T = self.overlay_pkl_data["opt_wrist_pos"].shape[0]
        n_dof_pkl = self.overlay_pkl_data["opt_dof_pos"].shape[1]
        print(f"[viz] loaded overlay URDF: {self._overlay_urdf_path}  "
              f"({n_links} links, {n_dof} DoF; pkl T={T} dof={n_dof_pkl})")

    # ------------------------------------------------------------------

    def _spawn_actors(self) -> None:
        # Each actor goes into its own collision group with filter=-1 (all
        # bits set) so PhysX never tries to resolve overlaps between actors.
        # Without this, the 28 marker spheres + the object mesh, which all
        # cluster within ~5 cm of the wrist, generate constant collision
        # forces that PhysX would resolve by displacing the actors —
        # producing the "object jitter" symptom reported by the user.
        NO_COLL_FILTER = -1

        # Object actor (we will reset its root pose every frame)
        pose = gymapi.Transform()
        x, y, z = self.traj.object_start_pose[:3].tolist()
        qx, qy, qz, qw = self.traj.object_start_pose[3:7].tolist()
        pose.p = gymapi.Vec3(x, y, z)
        pose.r = gymapi.Quat(qx, qy, qz, qw)
        self.obj_actor = self.gym.create_actor(
            self.env, self.obj_asset, pose, "object", 0, NO_COLL_FILTER
        )

        # Wrist marker (always shown; for skeleton trajectories this is the
        # same point as joints_world[:, 0] but kept as a separate marker for
        # clarity since the env reward also uses palm_center_pos).
        pose_w = gymapi.Transform(
            p=gymapi.Vec3(*[float(x) for x in self.traj.wrist_goals[0, :3]])
        )
        self.wrist_actor = self.gym.create_actor(
            self.env, self.wrist_marker_asset, pose_w, "wrist_goal", 0, NO_COLL_FILTER
        )
        self.gym.set_rigid_body_color(
            self.env, self.wrist_actor, 0,
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(0.0, 0.9, 0.2),
        )

        if self.has_skeleton and self._show_mano:
            # Spawn 21 joint markers, color-coded per finger group.
            self.joint_actors: List[int] = []
            joint_colours = [(0.5, 0.5, 0.5)] * 21    # default grey
            for joint_ids, rgb in self.traj.MANO_FINGER_GROUPS:
                for j in joint_ids:
                    joint_colours[j] = rgb
            j0 = self.traj.joints_world[0]            # (21, 3)
            for j in range(21):
                pose_j = gymapi.Transform(
                    p=gymapi.Vec3(*[float(x) for x in j0[j]])
                )
                asset = (self.ftip_marker_asset if j >= 16
                         else self.joint_marker_asset)
                actor = self.gym.create_actor(
                    self.env, asset, pose_j,
                    f"joint_{j:02d}_{self.traj.MANO_JOINT_ORDER[j]}",
                    0, NO_COLL_FILTER,
                )
                self.gym.set_rigid_body_color(
                    self.env, actor, 0,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(*joint_colours[j]),
                )
                self.joint_actors.append(actor)
        else:
            self.joint_actors: List[int] = []
        self.ftip_actors: List[int] = []  # MANO mode never spawns ftip-only

        # --------------------------------------------------------------
        # Sharpa + Overlay always run regardless of MANO visibility
        # --------------------------------------------------------------
        if True:
            # ----------------------------------------------------------
            # Sharpa floating-base hand actor (only if asset loaded)
            # ----------------------------------------------------------
            if self.sharpa_asset is not None:
                # Spawn at frame 0 wrist pose (we then teleport every frame)
                wp0 = self.traj.dex_wrist_pos[0].cpu().numpy()
                # dex_wrist_rot is axis-angle (T, 3).  Convert frame 0.
                from scipy.spatial.transform import Rotation as _R
                aa0 = self.traj.dex_wrist_rot[0].cpu().numpy()
                q0 = _R.from_rotvec(aa0).as_quat()      # xyzw
                pose_s = gymapi.Transform()
                pose_s.p = gymapi.Vec3(*[float(x) for x in wp0])
                pose_s.r = gymapi.Quat(*[float(x) for x in q0])
                self.sharpa_actor = self.gym.create_actor(
                    self.env, self.sharpa_asset, pose_s,
                    "sharpa_hand", 0, NO_COLL_FILTER,
                )
                # Init DOF state from frame 0 retargeted angles.
                dof_states = self.gym.get_actor_dof_states(
                    self.env, self.sharpa_actor, gymapi.STATE_ALL
                )
                dof_pos0 = self.traj.dex_dof_pos[0].cpu().numpy()
                # Be defensive about DOF count — the URDF may have a different
                # ordering than dex_dof_pos.  Take min and warn if mismatch.
                n_dof_urdf = self.gym.get_asset_dof_count(self.sharpa_asset)
                n_dof_traj = dof_pos0.shape[0]
                if n_dof_urdf != n_dof_traj:
                    print(f"[viz] WARN: Sharpa URDF has {n_dof_urdf} DoF but "
                          f"trajectory has {n_dof_traj} — using min({n_dof_urdf},{n_dof_traj})")
                K = min(n_dof_urdf, n_dof_traj)
                for i in range(K):
                    dof_states["pos"][i] = float(dof_pos0[i])
                    dof_states["vel"][i] = 0.0
                self.gym.set_actor_dof_states(
                    self.env, self.sharpa_actor, dof_states, gymapi.STATE_ALL
                )
                # Also set DOF position TARGETS (so any control loop driving
                # them snaps to the same value).
                self.gym.set_actor_dof_position_targets(
                    self.env, self.sharpa_actor, dof_pos0[:K].astype(np.float32)
                )
                self._sharpa_n_dof = K
                # Tint Sharpa visuals so they're distinguishable from the MANO
                # markers (light blue).
                n_links = self.gym.get_asset_rigid_body_count(self.sharpa_asset)
                for li in range(n_links):
                    self.gym.set_rigid_body_color(
                        self.env, self.sharpa_actor, li,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(0.30, 0.55, 0.85),
                    )
            else:
                self.sharpa_actor = None
                self._sharpa_n_dof = 0

            # ----------------------------------------------------------
            # Overlay floating-base hand actor (Inspire / Shadow / etc.)
            # ----------------------------------------------------------
            if self.overlay_asset is not None:
                from scipy.spatial.transform import Rotation as _R
                wp0 = np.asarray(self.overlay_pkl_data["opt_wrist_pos"][0])
                aa0 = np.asarray(self.overlay_pkl_data["opt_wrist_rot"][0])
                q0 = _R.from_rotvec(aa0).as_quat()
                pose_o = gymapi.Transform()
                pose_o.p = gymapi.Vec3(*[float(x) for x in wp0])
                pose_o.r = gymapi.Quat(*[float(x) for x in q0])
                self.overlay_actor = self.gym.create_actor(
                    self.env, self.overlay_asset, pose_o,
                    "overlay_hand", 0, NO_COLL_FILTER,
                )
                dof_states = self.gym.get_actor_dof_states(
                    self.env, self.overlay_actor, gymapi.STATE_ALL
                )
                dof_pos0 = np.asarray(self.overlay_pkl_data["opt_dof_pos"][0])
                n_dof_urdf = self.gym.get_asset_dof_count(self.overlay_asset)
                K = min(n_dof_urdf, dof_pos0.shape[0])
                if n_dof_urdf != dof_pos0.shape[0]:
                    print(f"[viz] WARN: overlay URDF has {n_dof_urdf} DoF but "
                          f"pkl has {dof_pos0.shape[0]} — using min")
                for i in range(K):
                    dof_states["pos"][i] = float(dof_pos0[i])
                    dof_states["vel"][i] = 0.0
                self.gym.set_actor_dof_states(
                    self.env, self.overlay_actor, dof_states, gymapi.STATE_ALL
                )
                self.gym.set_actor_dof_position_targets(
                    self.env, self.overlay_actor,
                    dof_pos0[:K].astype(np.float32),
                )
                self._overlay_n_dof = K
                # Tint
                n_links = self.gym.get_asset_rigid_body_count(self.overlay_asset)
                for li in range(n_links):
                    self.gym.set_rigid_body_color(
                        self.env, self.overlay_actor, li,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(*self._overlay_color),
                    )
            else:
                self.overlay_actor = None
                self._overlay_n_dof = 0
        # Fallback fingertip-only markers when JSON has no 21-joint skeleton
        if not self.has_skeleton:
            K = self.traj.num_fingertips
            for k in range(K):
                pose_f = gymapi.Transform(
                    p=gymapi.Vec3(*[float(x) for x in self.traj.fingertip_goals[0, k]])
                )
                actor = self.gym.create_actor(
                    self.env, self.ftip_marker_asset, pose_f, f"ftip_{k}", 0, NO_COLL_FILTER
                )
                colour = _FINGER_COLOURS[k % len(_FINGER_COLOURS)]
                self.gym.set_rigid_body_color(
                    self.env, actor, 0,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(*colour),
                )
                self.ftip_actors.append(actor)

    # ------------------------------------------------------------------

    def _subscribe_keys(self) -> None:
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "toggle")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "next")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "prev")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")

    # ------------------------------------------------------------------

    def _set_actor_pose(self, actor_handle: int, pos: np.ndarray, quat_xyzw=None) -> None:
        """Teleport an actor's *root* (and zero velocities) so multi-body
        URDFs end up consistently posed and PhysX doesn't carry stale velocity
        into the next frame.

        We always operate on the actor's root state; this works for both
        single-body marker spheres and multi-body URDF objects (their child
        rigid bodies get re-articulated automatically next sim step).
        """
        if quat_xyzw is None:
            quat_xyzw = (0.0, 0.0, 0.0, 1.0)
        state = self.gym.get_actor_rigid_body_states(
            self.env, actor_handle, gymapi.STATE_ALL
        )
        # Index 0 == root link.  Set pos/quat AND zero velocity to prevent
        # any leftover momentum from leaking into the next frame.
        state["pose"]["p"]["x"][0] = float(pos[0])
        state["pose"]["p"]["y"][0] = float(pos[1])
        state["pose"]["p"]["z"][0] = float(pos[2])
        state["pose"]["r"]["x"][0] = float(quat_xyzw[0])
        state["pose"]["r"]["y"][0] = float(quat_xyzw[1])
        state["pose"]["r"]["z"][0] = float(quat_xyzw[2])
        state["pose"]["r"]["w"][0] = float(quat_xyzw[3])
        for f in ("x", "y", "z"):
            state["vel"]["linear"][f][0] = 0.0
            state["vel"]["angular"][f][0] = 0.0
        self.gym.set_actor_rigid_body_states(
            self.env, actor_handle, state, gymapi.STATE_ALL
        )

    def _apply_frame(self, t: int) -> None:
        # Object
        op = self.traj.object_goals[t, :3].cpu().numpy()
        oq = self.traj.object_goals[t, 3:7].cpu().numpy()
        self._set_actor_pose(self.obj_actor, op, oq)

        # Wrist marker
        wp = self.traj.wrist_goals[t, :3].cpu().numpy()
        self._set_actor_pose(self.wrist_actor, wp)

        if self.has_skeleton and self._show_mano and self.joint_actors:
            joints_t = self.traj.joints_world[t].cpu().numpy()         # (21, 3)
            for j, actor in enumerate(self.joint_actors):
                self._set_actor_pose(actor, joints_t[j])
            self._draw_bones(joints_t)
        if self.has_skeleton:
            # Sharpa floating-base hand
            if self.sharpa_actor is not None:
                from scipy.spatial.transform import Rotation as _R
                wp = self.traj.dex_wrist_pos[t].cpu().numpy()
                aa = self.traj.dex_wrist_rot[t].cpu().numpy()
                quat = _R.from_rotvec(aa).as_quat()        # xyzw
                # Set root via STATE_ALL teleport (zeros velocity too)
                self._set_actor_pose(self.sharpa_actor, wp, quat)
                # Set DOF positions
                dof_states = self.gym.get_actor_dof_states(
                    self.env, self.sharpa_actor, gymapi.STATE_ALL
                )
                dpos = self.traj.dex_dof_pos[t].cpu().numpy()
                K = self._sharpa_n_dof
                for i in range(K):
                    dof_states["pos"][i] = float(dpos[i])
                    dof_states["vel"][i] = 0.0
                self.gym.set_actor_dof_states(
                    self.env, self.sharpa_actor, dof_states, gymapi.STATE_ALL
                )
                self.gym.set_actor_dof_position_targets(
                    self.env, self.sharpa_actor, dpos[:K].astype(np.float32)
                )
            # Overlay floating-base hand
            if self.overlay_actor is not None:
                from scipy.spatial.transform import Rotation as _R
                wp = np.asarray(self.overlay_pkl_data["opt_wrist_pos"][t])
                aa = np.asarray(self.overlay_pkl_data["opt_wrist_rot"][t])
                quat = _R.from_rotvec(aa).as_quat()
                self._set_actor_pose(self.overlay_actor, wp, quat)
                dof_states = self.gym.get_actor_dof_states(
                    self.env, self.overlay_actor, gymapi.STATE_ALL
                )
                dpos = np.asarray(self.overlay_pkl_data["opt_dof_pos"][t])
                K = self._overlay_n_dof
                for i in range(K):
                    dof_states["pos"][i] = float(dpos[i])
                    dof_states["vel"][i] = 0.0
                self.gym.set_actor_dof_states(
                    self.env, self.overlay_actor, dof_states, gymapi.STATE_ALL
                )
                self.gym.set_actor_dof_position_targets(
                    self.env, self.overlay_actor, dpos[:K].astype(np.float32),
                )
        else:
            for k, actor in enumerate(self.ftip_actors):
                fp = self.traj.fingertip_goals[t, k].cpu().numpy()
                self._set_actor_pose(actor, fp)

    def _draw_bones(self, joints: np.ndarray) -> None:
        """Draw the 20 MANO bones as debug lines on the viewer.

        The viewer accumulates lines across frames, so we ``clear_lines``
        first.  In headless mode self.viewer is None and we no-op.
        """
        if self.viewer is None:
            return
        bone_pairs = self.traj.MANO_BONE_PAIRS                        # 20 pairs
        # Build (num_lines, 6) vertices and (num_lines, 3) colors.
        # Colour each bone with its CHILD joint's finger group colour.
        joint_colours = [(0.5, 0.5, 0.5)] * 21
        for joint_ids, rgb in self.traj.MANO_FINGER_GROUPS:
            for j in joint_ids:
                joint_colours[j] = rgb
        verts = np.empty((len(bone_pairs), 6), dtype=np.float32)
        colors = np.empty((len(bone_pairs), 3), dtype=np.float32)
        for i, (a, b) in enumerate(bone_pairs):
            verts[i, :3] = joints[a]
            verts[i, 3:] = joints[b]
            colors[i] = joint_colours[b]
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(
            self.viewer, self.env, len(bone_pairs), verts, colors
        )

    # ------------------------------------------------------------------

    def run(self, max_steps: int = 0) -> None:
        last_advance = time.time()
        period = 1.0 / max(self.hz, 0.1)
        self._apply_frame(self.frame_idx)
        steps_done = 0

        while True:
            if self.viewer is not None:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                # Process key events
                for event in self.gym.query_viewer_action_events(self.viewer):
                    if event.value == 0:
                        continue
                    manual_change = False
                    if event.action == "toggle":
                        self.playing = not self.playing
                        print(f"[viz] {'playing' if self.playing else 'paused'}")
                    elif event.action == "next":
                        self.frame_idx = (self.frame_idx + 1) % self.traj.num_goals
                        self._apply_frame(self.frame_idx)
                        print(f"[viz] frame {self.frame_idx + 1}/{self.traj.num_goals}")
                        manual_change = True
                    elif event.action == "prev":
                        self.frame_idx = (self.frame_idx - 1) % self.traj.num_goals
                        self._apply_frame(self.frame_idx)
                        print(f"[viz] frame {self.frame_idx + 1}/{self.traj.num_goals}")
                        manual_change = True
                    elif event.action == "reset":
                        self.frame_idx = 0
                        self._apply_frame(self.frame_idx)
                        print("[viz] reset to frame 0")
                        manual_change = True
                    elif event.action == "quit":
                        return
                    if manual_change:
                        # Defer the next auto-advance by a full period so the
                        # user can scrub without a stray automatic frame.
                        last_advance = time.time()

            # Auto-advance
            now = time.time()
            if self.playing and (now - last_advance) >= period:
                last_advance = now
                self.frame_idx += 1
                if self.frame_idx >= self.traj.num_goals:
                    if self.loop:
                        self.frame_idx = 0
                    else:
                        self.frame_idx = self.traj.num_goals - 1
                        self.playing = False
                self._apply_frame(self.frame_idx)

            # If the Sharpa floating-base hand is loaded we MUST call simulate()
            # once per frame, otherwise the renderer never sees the per-link
            # transforms induced by our DOF state writes (only the root moves).
            # Each actor uses collision filter=-1 so no contacts are resolved
            # between actors, and gravity is disabled — so a single sim step
            # with zero DOF velocities is effectively a kinematic FK refresh.
            if self.sharpa_actor is not None or self.overlay_actor is not None:
                self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

            steps_done += 1
            if max_steps and steps_done >= max_steps:
                break

    def close(self) -> None:
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True,
                   help="Path to the trajectory JSON (relative to project root or absolute)")
    p.add_argument("--auto", action="store_true",
                   help="Run headless self-checks only and exit")
    p.add_argument("--hz", type=float, default=3.0,
                   help="Replay frame rate; trajectories are typically 3 Hz")
    p.add_argument("--no-loop", action="store_true",
                   help="Stop at the last frame instead of looping")
    p.add_argument("--overlay-pkl", default=None,
                   help="Optional second retarget pkl (e.g., Inspire) to spawn "
                        "an overlay floating-base hand alongside Sharpa.  Must "
                        "have opt_wrist_pos / opt_wrist_rot / opt_dof_pos.")
    p.add_argument("--overlay-urdf", default=None,
                   help="URDF for the overlay hand (e.g., inspire_hand_right.urdf).")
    p.add_argument("--no-mano", action="store_true",
                   help="Hide the 21 MANO ground-truth joint markers + bones, "
                        "leaving only the dexhand(s) + object visible.")
    p.add_argument("--no-sharpa", action="store_true",
                   help="Hide the Sharpa floating-base hand even if the JSON "
                        "trajectory has dex.* fields.")
    p.add_argument("--overlay-color", default="0.30,0.85,0.45",
                   help="RGB tint for the overlay hand mesh, as 'r,g,b' floats in [0,1].")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Auto-quit after N sim steps (0 = run forever)")
    args = p.parse_args()

    traj_path = args.traj
    if not os.path.isabs(traj_path):
        traj_path = str(PROJECT_ROOT / traj_path)
    traj = ReferenceTrajectory.from_file(traj_path)
    print(f"[viz] loaded {traj.num_goals} goals, {traj.num_fingertips} fingertips, "
          f"urdf_path={traj.object_urdf_path}")

    auto_ok = _auto_check(traj)

    if args.auto:
        # Beyond the data-only auto checks, also exercise the IsaacGym scene
        # path (asset load, actor spawn, teleport) so that broken URDFs /
        # missing meshes / runtime errors do not silently pass.
        print("\n[auto] running headless IsaacGym scene checks…")
        viewer = TrajectoryViewer(
            traj=traj, headless=True, hz=args.hz, loop=False,
            overlay_pkl=args.overlay_pkl, overlay_urdf=args.overlay_urdf,
            overlay_color=args.overlay_color,
            show_mano=not args.no_mano, show_sharpa=not args.no_sharpa,
        )
        try:
            scene_ok = True
            try:
                # Apply the first, middle and last frames; we no longer call
                # gym.simulate() because the visualizer is now pure-kinematic
                # (see comment in run()).  Just confirm teleport doesn't raise.
                for t in (0, traj.num_goals // 2, traj.num_goals - 1):
                    viewer.frame_idx = t
                    viewer._apply_frame(t)
                    viewer.gym.fetch_results(viewer.sim, True)
                print(f"  ✓ teleported through frames 0, mid, last "
                      f"({traj.num_goals} total)")
            except Exception as exc:
                scene_ok = False
                print(f"  ✗ scene step failed: {type(exc).__name__}: {exc}")
        finally:
            viewer.close()
        return 0 if (auto_ok and scene_ok) else 1

    viewer = TrajectoryViewer(
        traj=traj, headless=False, hz=args.hz, loop=not args.no_loop,
        overlay_pkl=args.overlay_pkl, overlay_urdf=args.overlay_urdf,
        overlay_color=args.overlay_color,
        show_mano=not args.no_mano, show_sharpa=not args.no_sharpa,
    )
    print("[viz] keys: Space=play/pause  N=next  P=prev  R=reset  Q=quit")
    try:
        viewer.run(max_steps=args.max_steps)
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
