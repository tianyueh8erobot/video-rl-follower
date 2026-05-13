"""Render a small ARROW-ON-TOP marker bound to the cube so we can SEE the
cube's actual orientation in IsaacGym vs the scipy-predicted orientation.
We use a separate elongated box as a "marker" attached at the cube center
+ a known offset along the cube's body-frame +y axis (i.e. we render two
boxes: the cube, and a thin marker whose root-state we set via
"cube_pos + R_world(quat).apply([0, 0.07, 0])").

If IsaacGym interprets the quat as xyzw, the marker should appear in the
direction predicted by scipy.from_quat(q_xyzw).apply([0, 0.07, 0]).
If IsaacGym interprets it as wxyz, the marker will be 90°-180° off.
"""
import os, isaacgym
from isaacgym import gymapi, gymtorch
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
import imageio.v2 as imageio, cv2

# Plain gym setup (no env class) to keep things minimal & inspectable
gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1/60.0; sp.substeps = 2; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0); sp.use_gpu_pipeline = False
sp.physx.solver_type = 1; sp.physx.use_gpu = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
gym.add_ground(sim, gymapi.PlaneParams())

cube_opts = gymapi.AssetOptions(); cube_opts.fix_base_link = True
cube_asset = gym.create_box(sim, 0.05, 0.05, 0.05, cube_opts)
marker_opts = gymapi.AssetOptions(); marker_opts.fix_base_link = True
# Thin marker stick, oriented along +x in body frame, length 0.10
marker_asset = gym.create_box(sim, 0.12, 0.01, 0.01, marker_opts)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
cube_pose = gymapi.Transform(); cube_pose.p = gymapi.Vec3(0.5, 0, 0.7)
cube_actor = gym.create_actor(env, cube_asset, cube_pose, "cube", 0, 0, 0)
marker_pose = gymapi.Transform(); marker_pose.p = gymapi.Vec3(0.5 + 0.06, 0, 0.7)
marker_actor = gym.create_actor(env, marker_asset, marker_pose, "marker", 0, 0, 0)

cam_props = gymapi.CameraProperties(); cam_props.width = 600; cam_props.height = 450
cam = gym.create_camera_sensor(env, cam_props)
# Top-down + slight tilt for clarity
gym.set_camera_location(cam, env, gymapi.Vec3(0.5, -0.7, 1.4), gymapi.Vec3(0.5, 0, 0.7))


def render_after_set(q_xyzw, name):
    """Set cube to q_xyzw (interpreted as xyzw).  Attach marker along the
    cube's body-frame +x axis, computed via scipy."""
    cube_pose = gymapi.Transform()
    cube_pose.p = gymapi.Vec3(0.5, 0, 0.7)
    cube_pose.r = gymapi.Quat(*q_xyzw)
    gym.set_actor_root_state_tensor(sim,
        gymtorch.unwrap_tensor(torch.zeros(2, 13)))   # no-op, then update via set_actor_pose
    # set via API
    state = gym.get_actor_rigid_body_states(env, cube_actor, gymapi.STATE_NONE)
    state["pose"]["p"][0] = (0.5, 0, 0.7)
    state["pose"]["r"][0] = tuple(q_xyzw)
    gym.set_actor_rigid_body_states(env, cube_actor, state, gymapi.STATE_POS)
    # marker = cube_pos + R_world(q_xyzw).apply([0.1, 0, 0])
    offset = R.from_quat(q_xyzw).apply([0.1, 0, 0])
    state_m = gym.get_actor_rigid_body_states(env, marker_actor, gymapi.STATE_NONE)
    state_m["pose"]["p"][0] = (0.5 + offset[0], 0 + offset[1], 0.7 + offset[2])
    state_m["pose"]["r"][0] = tuple(q_xyzw)   # marker rotates with cube
    gym.set_actor_rigid_body_states(env, marker_actor, state_m, gymapi.STATE_POS)
    gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
    img = np.asarray(gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR))
    img = img.reshape(cam_props.height, cam_props.width, 4)[..., :3].copy()
    # Annotate where scipy-xyzw predicts the marker tip
    cv2.putText(img, f"{name}: q_xyzw={[f'{v:.2f}' for v in q_xyzw]}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(img, f"marker tip at +x_body, expected world offset={offset.round(3)}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    return img


tests = [
    ("identity (0,0,0,1)",         [0., 0., 0., 1.]),
    ("90° around Z (0,0,.71,.71)", [0., 0., np.sin(np.pi/4), np.cos(np.pi/4)]),
    ("90° around Y (0,.71,0,.71)", [0., np.sin(np.pi/4), 0., np.cos(np.pi/4)]),
    ("90° around X (.71,0,0,.71)", [np.sin(np.pi/4), 0., 0., np.cos(np.pi/4)]),
]
imgs = [render_after_set(q, lbl) for lbl, q in tests]
montage = np.hstack(imgs)
imageio.imwrite("/tmp/quat_order_check.png", montage)
print("→ /tmp/quat_order_check.png")
print("Marker (thin red stick) starts at cube +x_body; if IsaacGym is xyzw, marker should")
print("appear in the predicted direction printed on each panel.")
print("  identity: +X world")
print("  90° Z:    +Y world (marker rotated 90° around Z)")
print("  90° Y:    -Z world (marker rotated 90° around Y, pointing down)")
print("  90° X:    +X world (marker doesn't move; rotation is along its own axis)")
