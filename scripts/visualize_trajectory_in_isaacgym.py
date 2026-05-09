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
    ) -> None:
        self.traj = traj
        self.hz = float(hz)
        self.loop = bool(loop)
        self.frame_idx = 0
        self.playing = True

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

        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Object asset
        self._load_object_asset()
        # Marker asset (a tiny sphere, used both for wrist + fingertips)
        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_options.disable_gravity = True
        self.wrist_marker_asset = self.gym.create_sphere(
            self.sim, 0.012, marker_options
        )
        self.ftip_marker_asset = self.gym.create_sphere(
            self.sim, 0.008, marker_options
        )

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
            opts.fix_base_link = False
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
        opts.fix_base_link = False
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

    def _spawn_actors(self) -> None:
        # Object actor (we will reset its root pose every frame)
        pose = gymapi.Transform()
        x, y, z = self.traj.object_start_pose[:3].tolist()
        qx, qy, qz, qw = self.traj.object_start_pose[3:7].tolist()
        pose.p = gymapi.Vec3(x, y, z)
        pose.r = gymapi.Quat(qx, qy, qz, qw)
        self.obj_actor = self.gym.create_actor(
            self.env, self.obj_asset, pose, "object", 0, 0
        )

        # Wrist marker
        pose_w = gymapi.Transform(
            p=gymapi.Vec3(*[float(x) for x in self.traj.wrist_goals[0, :3]])
        )
        self.wrist_actor = self.gym.create_actor(
            self.env, self.wrist_marker_asset, pose_w, "wrist_goal", 0, 0
        )
        self.gym.set_rigid_body_color(
            self.env, self.wrist_actor, 0,
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(0.0, 0.9, 0.2),
        )

        # Fingertip markers
        self.ftip_actors: List[int] = []
        K = self.traj.num_fingertips
        for k in range(K):
            pose_f = gymapi.Transform(
                p=gymapi.Vec3(*[float(x) for x in self.traj.fingertip_goals[0, k]])
            )
            actor = self.gym.create_actor(
                self.env, self.ftip_marker_asset, pose_f, f"ftip_{k}", 0, 0
            )
            colour = _FINGER_COLOURS[k % len(_FINGER_COLOURS)]
            self.gym.set_rigid_body_color(
                self.env, actor, 0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(*colour),
            )
            self.ftip_actors.append(actor)

        # Cache the actor handles' rigid body indices (for set_actor_root_state).

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

        # Fingertips
        for k, actor in enumerate(self.ftip_actors):
            fp = self.traj.fingertip_goals[t, k].cpu().numpy()
            self._set_actor_pose(actor, fp)

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
            traj=traj, headless=True, hz=args.hz, loop=False
        )
        try:
            scene_ok = True
            try:
                # Apply the first, middle and last frames; step PhysX a couple
                # of times after each to confirm no asset-load explosion.
                for t in (0, traj.num_goals // 2, traj.num_goals - 1):
                    viewer.frame_idx = t
                    viewer._apply_frame(t)
                    for _ in range(2):
                        viewer.gym.simulate(viewer.sim)
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
        traj=traj, headless=False, hz=args.hz, loop=not args.no_loop
    )
    print("[viz] keys: Space=play/pause  N=next  P=prev  R=reset  Q=quit")
    try:
        viewer.run(max_steps=args.max_steps)
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
