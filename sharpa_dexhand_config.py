"""Sharpa Wave dexterous-hand configuration for ManipTrans retargeting.

Sharpa Wave is a 22-DoF anthropomorphic hand by Sharpa Robotics.  It mounts
on any robot arm via a flange interface (we use the
``*_with_flange.urdf`` variant from sharpa-urdf-usd-xml).

Differences from the upstream Inspire/Shadow/Allegro configs:

  * 22 movable joints (vs. Inspire's 12) — Sharpa exposes a full DIP joint on
    every finger, plus an extra CMC abduction on the pinky and an IP+MCP_AA on
    the thumb.
  * Each finger therefore matches MANO's standard 4-keypoint chain
    (proximal / intermediate / distal / tip) faithfully — no missing-joint
    placeholders in ``hand2dex_mapping`` like Inspire has.
  * Wrist origin convention follows the official URDF (``right_hand_C_MC``
    is the palm-base link); the relative_rotation aligns it with MANO's
    palm-down neutral pose.

To use::

    python main/dataset/mano2dexhand.py --dexhand sharpa --side right \\
        --data_idx <seq> --headless --iter 4000

The auto-register decorator wires this into DexHandFactory; no manual
import is needed.
"""

from abc import ABC

import numpy as np

from main.dataset.transform import aa_to_rotmat

from .base import DexHand
from .decorators import register_dexhand


# ---------------------------------------------------------------------------
# Sharpa right-hand wave URDF schema (from sharpa-urdf-usd-xml/right_sharpa_wave_with_flange.urdf)
# ---------------------------------------------------------------------------
#
# Movable joints (22 total):
#   thumb : CMC_FE → CMC_AA → MCP_FE → MCP_AA → IP             (5)
#   index : MCP_FE → MCP_AA → PIP → DIP                         (4)
#   middle: MCP_FE → MCP_AA → PIP → DIP                         (4)
#   ring  : MCP_FE → MCP_AA → PIP → DIP                         (4)
#   pinky : CMC → MCP_FE → MCP_AA → PIP → DIP                   (5)
#
# Link chain (from palm out):
#   right_hand_C_MC  (palm-base)
#   ├── right_thumb_CMC_VL → MC → MCP_VL → PP → DP → elastomer → fingertip
#   ├── right_index_MCP_VL  →                PP → MP → DP → elastomer → fingertip
#   ├── right_middle_MCP_VL →                PP → MP → DP → elastomer → fingertip
#   ├── right_ring_MCP_VL   →                PP → MP → DP → elastomer → fingertip
#   └── right_pinky_MC → MCP_VL →            PP → MP → DP → elastomer → fingertip
#
# MANO keypoint mapping (one-to-many: multiple Sharpa links → one MANO joint):
#   wrist                ← right_hand_C_MC
#   <finger>_proximal    ← <finger>_MCP_VL  AND  <finger>_PP   (the MCP joint area)
#   <finger>_intermediate← <finger>_MP                          (after PIP)
#   <finger>_distal      ← <finger>_DP                          (after DIP)
#   <finger>_tip         ← <finger>_fingertip                    (rigid extension)
#
# Special cases:
#   thumb_proximal       ← thumb_CMC_VL + thumb_MC + thumb_MCP_VL  (Sharpa thumb has 5 joints)
#   thumb_intermediate   ← thumb_PP
#   thumb_distal         ← thumb_DP
#   thumb_tip            ← thumb_fingertip
#   pinky_proximal       ← pinky_MC + pinky_MCP_VL + pinky_PP    (extra CMC joint)


class Sharpa(DexHand, ABC):
    """Sharpa Wave hand (left/right share the same kinematic structure
    modulo the ``L_`` / ``R_`` prefix conventions; we currently only have
    ``right_*`` URDF, so only RH variant is registered)."""

    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "sharpa"

        # 28 link names (positions tracked during retargeting)
        # Index in this list maps directly into weight_idx / bone_links below.
        self.body_names = [
            # 0: wrist (palm base)
            "hand_C_MC",

            # 1-5: index   (MCP_VL, PP, MP, DP, fingertip)
            "index_MCP_VL", "index_PP", "index_MP", "index_DP", "index_fingertip",

            # 6-10: middle (MCP_VL, PP, MP, DP, fingertip)
            "middle_MCP_VL", "middle_PP", "middle_MP", "middle_DP", "middle_fingertip",

            # 11-16: pinky  (MC, MCP_VL, PP, MP, DP, fingertip) — 6 links because of extra CMC
            "pinky_MC", "pinky_MCP_VL", "pinky_PP", "pinky_MP", "pinky_DP", "pinky_fingertip",

            # 17-21: ring   (MCP_VL, PP, MP, DP, fingertip)
            "ring_MCP_VL", "ring_PP", "ring_MP", "ring_DP", "ring_fingertip",

            # 22-27: thumb  (CMC_VL, MC, MCP_VL, PP, DP, fingertip) — 6 links because of extra CMC
            "thumb_CMC_VL", "thumb_MC", "thumb_MCP_VL", "thumb_PP", "thumb_DP", "thumb_fingertip",
        ]

        # 22 movable joint names (URDF order)
        self.dof_names = [
            "thumb_CMC_FE", "thumb_CMC_AA",
            "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",

            "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",

            "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",

            "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",

            "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
        ]

        # MANO joint name → list of Sharpa link names that should track it.
        # Names here are WITHOUT the right_/left_ side prefix; the side
        # subclass below adds the prefix.
        self.hand2dex_mapping = {
            "wrist": ["hand_C_MC"],

            "thumb_proximal":     ["thumb_CMC_VL", "thumb_MC", "thumb_MCP_VL"],
            "thumb_intermediate": ["thumb_PP"],
            "thumb_distal":       ["thumb_DP"],
            "thumb_tip":          ["thumb_fingertip"],

            "index_proximal":     ["index_MCP_VL", "index_PP"],
            "index_intermediate": ["index_MP"],
            "index_distal":       ["index_DP"],
            "index_tip":          ["index_fingertip"],

            "middle_proximal":     ["middle_MCP_VL", "middle_PP"],
            "middle_intermediate": ["middle_MP"],
            "middle_distal":       ["middle_DP"],
            "middle_tip":          ["middle_fingertip"],

            "ring_proximal":     ["ring_MCP_VL", "ring_PP"],
            "ring_intermediate": ["ring_MP"],
            "ring_distal":       ["ring_DP"],
            "ring_tip":          ["ring_fingertip"],

            "pinky_proximal":     ["pinky_MC", "pinky_MCP_VL", "pinky_PP"],
            "pinky_intermediate": ["pinky_MP"],
            "pinky_distal":       ["pinky_DP"],
            "pinky_tip":          ["pinky_fingertip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        # body_names must equal the keys of dex2hand_mapping (every link
        # mapped to exactly one MANO joint).
        assert set(self.body_names) == set(self.dex2hand_mapping.keys()), (
            f"mismatch: body_names={set(self.body_names) - set(self.dex2hand_mapping.keys())} "
            f"dex2hand={set(self.dex2hand_mapping.keys()) - set(self.body_names)}"
        )

        # Contact bodies — the distal phalanx links (Sharpa's elastomer-covered
        # tips); these are what the env's contact_force tracking uses.
        self.contact_body_names = [
            "thumb_DP", "index_DP", "middle_DP", "ring_DP", "pinky_DP",
        ]

        # Bone links (parent_idx, child_idx in body_names) for visualisation.
        # Each chain: wrist → MCP_VL → PP → MP → DP → fingertip
        # Thumb has a longer chain, pinky has an extra MC link.
        self.bone_links = [
            # index  (1 → 2 → 3 → 4 → 5)
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            # middle
            [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
            # pinky (note extra MC node)
            [0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
            # ring
            [0, 17], [17, 18], [18, 19], [19, 20], [20, 21],
            # thumb (longer)
            [0, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27],
        ]

        # Per-joint reward weight groups (used by env reward shaping).
        # tip indices: thumb=27, index=5, middle=10, ring=21, pinky=16
        # MCP_VL ('level_1' = first joint after wrist of each finger, the
        # broad-degree-of-freedom rotation hub):
        #   index=1, middle=6, pinky_MC=11, ring=17, thumb_CMC_VL=22
        # all remaining intermediate/distal/PP joints → level_2
        self.weight_idx = {
            "thumb_tip":  [27],
            "index_tip":  [5],
            "middle_tip": [10],
            "ring_tip":   [21],
            "pinky_tip":  [16],
            "level_1_joints": [1, 6, 11, 17, 22],
            "level_2_joints": [2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26],
        }

        # ? PID-controlled wrist (reference only; main path uses 6D force).
        self.Kp_rot = 0.5
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 20
        self.Ki_pos = 0.005
        self.Kd_pos = 0.1

    def __str__(self):
        return self.name


@register_dexhand("sharpa_rh")
class SharpaRH(Sharpa):
    """Right Sharpa Wave hand (mounted via flange)."""

    def __str__(self):
        # mano2dexhand uses str(dexhand) to build the dump dir name; keeping
        # the convention "<name>_rh" matches Inspire/Shadow/Allegro outputs.
        return super().__str__() + "_rh"

    def __init__(self):
        super().__init__()
        # URDF lives in <ManipTrans-root>/assets/sharpa/right_sharpa_wave.urdf
        # (copied from sharpa-urdf-usd-xml).
        self._urdf_path = "assets/sharpa/right_sharpa_wave.urdf"
        self.side = "rh"
        # Apply the right_*  prefix to every link / dof / mapping entry.
        self.body_names = ["right_" + n for n in self.body_names]
        self.dof_names  = ["right_" + n for n in self.dof_names]
        self.contact_body_names = ["right_" + n for n in self.contact_body_names]
        self.hand2dex_mapping = {
            mano_name: ["right_" + sharpa_n for sharpa_n in dex_list]
            for mano_name, dex_list in self.hand2dex_mapping.items()
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)

        # Initial wrist orientation in the GRAB/OakInk frame.  Computed via
        # Procrustes (Kabsch) alignment of Sharpa's default-pose 28-link
        # positions vs. MANO's default-pose 21 keypoints — see
        # tools/compute_sharpa_relrot.py for the derivation.  The residual
        # ~7 cm reflects intrinsic Sharpa-vs-MANO link-length differences
        # that the per-frame mano2dexhand optimisation will then resolve.
        self.relative_rotation = aa_to_rotmat(
            np.array([1.692335, -0.750560, -1.764472])
        )
        # No additional translation between sharpa_wave wrist and MANO wrist
        # (both use palm-base origin).
