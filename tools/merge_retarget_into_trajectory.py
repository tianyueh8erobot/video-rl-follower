"""Merge a ManipTrans-retargeted .pkl into an existing trajectory JSON so the
env can use it for full-hand-tracking reward.

Input:
  - trajectory JSON produced by tools/process_maniptrans_trajectory.py
  - retarget pkl produced by ManipTrans/main/dataset/mano2dexhand.py with
    --dexhand sharpa --side right (or any other dexhand)

Output:
  - new JSON with hand.dex.{links_world, dof_pos, link_names} populated
  - per-frame counts must match (we assert)

Usage:
  python tools/merge_retarget_into_trajectory.py \
      --in-json data/trajectories/g0.json \
      --pkl /home/intel/Codes/ManipTrans/data/retargeting/grab_demo/mano2sharpa_rh/102_sv_dict.pkl \
      --link-names-file <optional path to txt file with link names>
      --out-json data/trajectories/g0_with_dex.json
"""

from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np


# Sharpa right-hand body_names — must match
# ManipTrans/maniptrans_envs/lib/envs/dexhands/sharpa.py SharpaRH.body_names
_SHARPA_RH_BODY_NAMES = [
    "right_hand_C_MC",
    "right_index_MCP_VL", "right_index_PP", "right_index_MP", "right_index_DP", "right_index_fingertip",
    "right_middle_MCP_VL", "right_middle_PP", "right_middle_MP", "right_middle_DP", "right_middle_fingertip",
    "right_pinky_MC", "right_pinky_MCP_VL", "right_pinky_PP", "right_pinky_MP", "right_pinky_DP", "right_pinky_fingertip",
    "right_ring_MCP_VL", "right_ring_PP", "right_ring_MP", "right_ring_DP", "right_ring_fingertip",
    "right_thumb_CMC_VL", "right_thumb_MC", "right_thumb_MCP_VL", "right_thumb_PP", "right_thumb_DP", "right_thumb_fingertip",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", required=True)
    p.add_argument("--pkl", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument(
        "--link-names",
        default="sharpa",
        help="Either 'sharpa' (use built-in Sharpa RH names), "
             "or a comma-separated list, or a path to a .txt file (one name per line)",
    )
    args = p.parse_args()

    # Load trajectory JSON
    with open(args.in_json) as f:
        traj = json.load(f)

    # Load retarget pkl
    with open(args.pkl, "rb") as f:
        ret = pickle.load(f)
    expected_keys = {"opt_wrist_pos", "opt_wrist_rot", "opt_dof_pos", "opt_joints_pos"}
    missing = expected_keys - set(ret.keys())
    if missing:
        raise ValueError(f"retarget pkl missing keys: {missing}")
    print(f"[merge] retarget pkl shapes: "
          f"wrist_pos {ret['opt_wrist_pos'].shape}  "
          f"dof_pos {ret['opt_dof_pos'].shape}  "
          f"joints_pos {ret['opt_joints_pos'].shape}")

    # Determine link names
    if args.link_names == "sharpa":
        link_names = _SHARPA_RH_BODY_NAMES
    elif Path(args.link_names).is_file():
        link_names = [l.strip() for l in open(args.link_names) if l.strip()]
    else:
        link_names = [n.strip() for n in args.link_names.split(",")]

    n_links_pkl = ret["opt_joints_pos"].shape[1]
    if len(link_names) != n_links_pkl:
        raise ValueError(
            f"link_names length {len(link_names)} != pkl link count {n_links_pkl}"
        )

    # Frame count alignment
    T_traj = len(traj["object"]["goals"])
    T_pkl  = ret["opt_joints_pos"].shape[0]
    if T_traj != T_pkl:
        # The trajectory JSON may have been downsampled; the pkl is at the
        # rate mano2dexhand was given.  We support 1:N strides — pick every
        # stride-th frame of the pkl.
        if T_pkl % T_traj == 0:
            stride = T_pkl // T_traj
            print(f"[merge] downsampling pkl from {T_pkl} → {T_traj} frames "
                  f"(stride={stride})")
            sl = slice(None, None, stride)
            ret = {k: v[sl][:T_traj] for k, v in ret.items()}
        elif T_traj % T_pkl == 0:
            raise ValueError(
                f"pkl has fewer frames ({T_pkl}) than JSON ({T_traj}); "
                "cannot upsample (would need interpolation)."
            )
        else:
            raise ValueError(
                f"frame count mismatch and not a clean multiple: "
                f"traj={T_traj} pkl={T_pkl}"
            )

    # Build the dex block
    dex_block = {
        "link_names": link_names,
        "links_world": ret["opt_joints_pos"].tolist(),
        "dof_pos":     ret["opt_dof_pos"].tolist(),
        "wrist_pos":   ret["opt_wrist_pos"].tolist(),
        "wrist_rot":   ret["opt_wrist_rot"].tolist(),
    }
    traj["dex"] = dex_block
    # Also bump meta
    traj["meta"]["dex_n_links"] = len(link_names)
    traj["meta"]["dex_n_dof"] = int(ret["opt_dof_pos"].shape[1])
    traj["meta"]["dex_source"] = str(args.pkl)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(traj, f)
    print(f"[merge] wrote {args.out_json}")
    print(f"  dex.link_names: {len(link_names)} entries")
    print(f"  dex.links_world shape: {len(traj['dex']['links_world'])} × "
          f"{len(traj['dex']['links_world'][0])} × 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
