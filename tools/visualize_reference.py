"""Reference-trajectory visualizer for VideoRLFollower.

Loads a JSON trajectory (produced by ``tools/process_maniptrans_trajectory.py``)
and shows:

  * the object 6D pose as a wireframe box / coordinate axes,
  * the hand wrist as a coordinate frame,
  * the 5 fingertip points (color-coded per finger), connected to the wrist by
    thin lines.

Two backends are provided.  The default (``matplotlib``) has zero install
overhead and a simple frame slider.  ``--backend open3d`` opens a 3D viewer
with mouse orbit if Open3D is installed.

Usage::

    python tools/visualize_reference.py --traj data/trajectories/example_g0.json
    python tools/visualize_reference.py --traj data/trajectories/example_g0.json --backend open3d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


_FINGER_COLORS = {
    "thumb": "#e41a1c",
    "index": "#377eb8",
    "middle": "#4daf4a",
    "ring": "#984ea3",
    "pinky": "#ff7f00",
}


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _load(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# matplotlib backend
# ---------------------------------------------------------------------------


def _viz_matplotlib(traj: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

    obj_goals = np.asarray(traj["object"]["goals"])           # (T, 7)
    wrist = np.asarray(traj["hand"]["wrist_goals"])           # (T, 7)
    ftips_w = np.asarray(traj["hand"]["fingertip_goals"])     # (T, K, 3)
    fingertip_order: List[str] = traj["meta"].get(
        "fingertip_order", ["thumb", "index", "middle", "ring", "pinky"]
    )
    T = obj_goals.shape[0]
    K = ftips_w.shape[1]

    # Compute scene bounds
    pts_all = np.concatenate(
        [obj_goals[:, :3], wrist[:, :3], ftips_w.reshape(T * K, 3)], axis=0
    )
    pad = 0.15
    mins = pts_all.min(axis=0) - pad
    maxs = pts_all.max(axis=0) + pad
    centre = (mins + maxs) / 2
    half = (maxs - mins).max() / 2

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    fig.subplots_adjust(bottom=0.18)

    obj_R = _quat_xyzw_to_rotmat(obj_goals[:, 3:7])
    wrist_R = _quat_xyzw_to_rotmat(wrist[:, 3:7])

    def draw(t: int) -> None:
        ax.cla()
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"frame {t + 1}/{T}")

        # Object trajectory ghost
        ax.plot(
            obj_goals[:, 0], obj_goals[:, 1], obj_goals[:, 2],
            color="grey", alpha=0.3, linewidth=1, label="object path",
        )

        # Object axes at t
        op = obj_goals[t, :3]
        oR = obj_R[t]
        for axis_idx, axis_color in enumerate(["#cc4444", "#44cc44", "#4444cc"]):
            tip = op + 0.05 * oR[:, axis_idx]
            ax.plot([op[0], tip[0]], [op[1], tip[1]], [op[2], tip[2]],
                    color=axis_color, linewidth=2.0)
        ax.scatter(*op, color="black", s=20, label="object")

        # Wrist axes at t
        wp = wrist[t, :3]
        wR = wrist_R[t]
        for axis_idx, axis_color in enumerate(["#883333", "#338833", "#333388"]):
            tip = wp + 0.04 * wR[:, axis_idx]
            ax.plot([wp[0], tip[0]], [wp[1], tip[1]], [wp[2], tip[2]],
                    color=axis_color, linewidth=1.5)

        # Fingertips with bones
        for k in range(K):
            name = fingertip_order[k] if k < len(fingertip_order) else f"f{k}"
            color = _FINGER_COLORS.get(name, "#888888")
            fp = ftips_w[t, k]
            ax.plot([wp[0], fp[0]], [wp[1], fp[1]], [wp[2], fp[2]],
                    color=color, linewidth=1.0, alpha=0.6)
            ax.scatter(*fp, color=color, s=40, label=name if t == 0 else None)

        # Legend only once
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[: 2 + K], l[: 2 + K], loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes((0.15, 0.04, 0.7, 0.03))
    slider = Slider(slider_ax, "frame", 0, T - 1, valinit=0, valstep=1)
    slider.on_changed(lambda v: draw(int(v)))

    draw(0)
    plt.show()


# ---------------------------------------------------------------------------
# Open3D backend (optional)
# ---------------------------------------------------------------------------


def _viz_open3d(traj: dict) -> None:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            "Open3D backend requested but Open3D is not installed: %r" % e
        )

    obj_goals = np.asarray(traj["object"]["goals"])
    wrist = np.asarray(traj["hand"]["wrist_goals"])
    ftips_w = np.asarray(traj["hand"]["fingertip_goals"])
    fingertip_order: List[str] = traj["meta"].get(
        "fingertip_order", ["thumb", "index", "middle", "ring", "pinky"]
    )
    T = obj_goals.shape[0]
    K = ftips_w.shape[1]
    obj_R = _quat_xyzw_to_rotmat(obj_goals[:, 3:7])
    wrist_R = _quat_xyzw_to_rotmat(wrist[:, 3:7])

    geoms: list = []

    obj_path = o3d.geometry.LineSet()
    obj_path.points = o3d.utility.Vector3dVector(obj_goals[:, :3])
    obj_path.lines = o3d.utility.Vector2iVector(
        np.stack([np.arange(T - 1), np.arange(1, T)], axis=-1)
    )
    obj_path.colors = o3d.utility.Vector3dVector(np.tile([0.4, 0.4, 0.4], (T - 1, 1)))
    geoms.append(obj_path)

    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geoms.append(object_frame)
    wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    geoms.append(wrist_frame)

    ftip_spheres = []
    for k in range(K):
        name = fingertip_order[k] if k < len(fingertip_order) else f"f{k}"
        color = np.array(
            o3d.utility.Vector3dVector(
                [list(int(c) / 255.0 for c in (
                    _FINGER_COLORS.get(name, "#888888")[1:3],
                    _FINGER_COLORS.get(name, "#888888")[3:5],
                    _FINGER_COLORS.get(name, "#888888")[5:7],
                ))][0]
            )
        )
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        col = np.array([
            int(_FINGER_COLORS.get(name, "#888888")[1:3], 16) / 255.0,
            int(_FINGER_COLORS.get(name, "#888888")[3:5], 16) / 255.0,
            int(_FINGER_COLORS.get(name, "#888888")[5:7], 16) / 255.0,
        ])
        sph.paint_uniform_color(col)
        ftip_spheres.append(sph)
        geoms.append(sph)

    bones = o3d.geometry.LineSet()
    geoms.append(bones)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("video_rl_follower trajectory", width=1024, height=768)
    for g in geoms:
        vis.add_geometry(g)

    state = {"t": 0, "playing": False}

    def update():
        t = state["t"]
        # Object frame
        object_frame.transform(np.linalg.inv(object_frame.get_rotation_matrix_from_xyz((0, 0, 0))))  # noqa
        # Easier: rebuild transform each frame
        T_obj = np.eye(4)
        T_obj[:3, :3] = obj_R[t]
        T_obj[:3, 3] = obj_goals[t, :3]
        object_frame.transform(T_obj)
        T_wrist = np.eye(4)
        T_wrist[:3, :3] = wrist_R[t]
        T_wrist[:3, 3] = wrist[t, :3]
        wrist_frame.transform(T_wrist)

        for k in range(K):
            sph = ftip_spheres[k]
            # Recreate translation (Open3D meshes are by-reference; reset to origin then translate)
            sph.translate(-sph.get_center())
            sph.translate(ftips_w[t, k])

        # bones from wrist to each fingertip
        pts = np.concatenate([wrist[t:t + 1, :3], ftips_w[t]], axis=0)
        bones.points = o3d.utility.Vector3dVector(pts)
        bones.lines = o3d.utility.Vector2iVector(
            np.stack([np.zeros(K, dtype=np.int32), np.arange(1, K + 1)], axis=-1)
        )
        bones.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.2, 0.2], (K, 1)))

        for g in geoms:
            vis.update_geometry(g)
        # Reset frames so transforms don't compound on next call
        object_frame.transform(np.linalg.inv(T_obj))
        wrist_frame.transform(np.linalg.inv(T_wrist))

    def step(direction: int):
        state["t"] = (state["t"] + direction) % T
        update()
        print(f"frame {state['t'] + 1}/{T}", end="\r")

    vis.register_key_callback(ord("L"), lambda v: (step(+1), False)[1])
    vis.register_key_callback(ord("H"), lambda v: (step(-1), False)[1])
    vis.register_key_callback(
        ord(" "), lambda v: state.update(playing=not state["playing"]) or False
    )

    update()
    print("[H/L] prev/next frame  [Space] play  [Q] quit")
    while True:
        if state["playing"]:
            step(+1)
        if not vis.poll_events():
            break
        vis.update_renderer()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="path to trajectory JSON")
    p.add_argument(
        "--backend",
        choices=("matplotlib", "open3d"),
        default="matplotlib",
    )
    args = p.parse_args()

    traj = _load(args.traj)
    print(
        f"Loaded {Path(args.traj).name}: "
        f"{len(traj['object']['goals'])} goals @ {traj['meta'].get('fps', 'unknown')} Hz "
        f"from {traj['meta'].get('source', '?')}"
    )

    if args.backend == "matplotlib":
        _viz_matplotlib(traj)
    else:
        _viz_open3d(traj)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
