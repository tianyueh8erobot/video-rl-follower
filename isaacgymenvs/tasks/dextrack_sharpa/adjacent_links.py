"""Collision-exclusion adjacency table — Franka-Panda + right Sharpa hand.

Ported from SimToolReal's `RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS`
(isaacgymenvs/tasks/simtoolreal/adjacent_links.py).  The Sharpa-hand portion is
byte-identical to SimToolReal's RIGHT table — only the arm links differ
(SimToolReal `iiwa14_link_*` Kuka → `panda_link_*` Franka).

This table is the body set produced by `collapse_fixed_joints=True`: the mount
chain + palm (right_hand_C_MC) collapse into panda_link7, and each *_elastomer /
*_fingertip collapses into its *_DP.  Two links collide ONLY if they are NOT in
each other's adjacency list (see set_robot_asset_rigid_shape_properties in env.py).

ONE pair was added beyond the SimToolReal RIGHT-table port:
  panda_link7 <-> right_thumb_MC
The thumb metacarpal `right_thumb_MC` carries the thumb-base collision mesh
(its parent `right_thumb_CMC_VL` is a massless virtual link with no geometry),
so after collapse it permanently overlaps the merged palm geometry on
panda_link7 — a fixed-mount artifact, not a graspable contact (verified: ~82 N
penetration at the pre-grasp reference pose).  SimToolReal's *LEFT*-Sharpa
table already excludes the mirror pair `left_thumb_MC <-> iiwa14_link_7`,
confirming this exclusion is correct for the Sharpa hand; the RIGHT table
simply omitted it (Kuka link7 geometry differs from Franka link7).
"""

PANDA_SHARPA_RIGHT_LINK_TO_ADJACENT_LINKS = {
    "panda_link0": ["panda_link1"],
    "panda_link1": ["panda_link0", "panda_link2"],
    "panda_link2": ["panda_link1", "panda_link3"],
    "panda_link3": ["panda_link2", "panda_link4"],
    "panda_link4": ["panda_link3", "panda_link5"],
    "panda_link5": ["panda_link4", "panda_link6"],
    "panda_link6": ["panda_link5", "panda_link7"],
    "panda_link7": [
        "panda_link6",
        "right_index_MCP_VL",
        "right_middle_MCP_VL",
        "right_pinky_MC",
        "right_ring_MCP_VL",
        "right_thumb_CMC_VL",
        "right_thumb_MC",   # added beyond SimToolReal RIGHT port — see module docstring
    ],
    "right_index_MCP_VL": ["panda_link7", "right_index_PP"],
    "right_index_PP": ["right_index_MCP_VL", "right_index_MP"],
    "right_index_MP": ["right_index_PP", "right_index_DP"],
    "right_index_DP": ["right_index_MP"],
    "right_middle_MCP_VL": ["panda_link7", "right_middle_PP"],
    "right_middle_PP": ["right_middle_MCP_VL", "right_middle_MP"],
    "right_middle_MP": ["right_middle_PP", "right_middle_DP"],
    "right_middle_DP": ["right_middle_MP"],
    "right_pinky_MC": ["panda_link7", "right_pinky_MCP_VL"],
    "right_pinky_MCP_VL": ["right_pinky_MC", "right_pinky_PP"],
    "right_pinky_PP": ["right_pinky_MCP_VL", "right_pinky_MP"],
    "right_pinky_MP": ["right_pinky_PP", "right_pinky_DP"],
    "right_pinky_DP": ["right_pinky_MP"],
    "right_ring_MCP_VL": ["panda_link7", "right_ring_PP"],
    "right_ring_PP": ["right_ring_MCP_VL", "right_ring_MP"],
    "right_ring_MP": ["right_ring_PP", "right_ring_DP"],
    "right_ring_DP": ["right_ring_MP"],
    "right_thumb_CMC_VL": ["panda_link7", "right_thumb_MC"],
    "right_thumb_MC": ["right_thumb_CMC_VL", "right_thumb_MCP_VL"],
    "right_thumb_MCP_VL": ["right_thumb_MC", "right_thumb_PP"],
    "right_thumb_PP": ["right_thumb_MCP_VL", "right_thumb_DP"],
    "right_thumb_DP": ["right_thumb_PP"],
}
