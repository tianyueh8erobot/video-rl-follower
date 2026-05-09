"""Build ``iiwa14_right_sharpa_adjusted_restricted.urdf`` by splicing the KUKA
arm half of the existing left-hand URDF with the official Sharpa right-hand
URDF.

Result layout::

    iiwa14_link_0 → … → iiwa14_link_7 → iiwa14_link_ee → sharpa_mount
        → (NEW fixed joint sharpa_mount_to_right_flange) → right_hand_flange
        → right_hand_wrist → right_hand_C_MC → (right thumb / index / … chains)

Run::

    python tools/build_iiwa14_right_sharpa_urdf.py \
        --left-urdf assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf \
        --right-urdf /tmp/sharpa_urdf/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.urdf \
        --right-meshes-src /tmp/sharpa_urdf/wave_01/right_sharpa_wave/meshes \
        --out-urdf assets/urdf/kuka_sharpa_description/iiwa14_right_sharpa_adjusted_restricted.urdf \
        --out-meshes assets/urdf/kuka_sharpa_description/right_sharpa_meshes

The script does NOT modify the left URDF.  It:

    1. Surgically removes every ``<link name="left_*">…</link>`` block from
       the left URDF, plus every ``<joint>`` whose parent OR child is a
       ``left_*`` link.  Critically, this PRESERVES the
       ``iiwa14_link_ee → sharpa_mount`` joint that sits at the very end of
       the source file (which a naive tail-chop would lose, leaving
       ``sharpa_mount`` orphaned from the arm).
    2. Reads the official right URDF, strips the wrapper ``<robot>`` tag,
       fixes mesh paths to ``right_sharpa_meshes/<name>.STL``.
    3. Inserts a single fixed joint connecting ``sharpa_mount`` to
       ``right_hand_flange`` (origin defaults to a sane orientation that will
       likely need a final tweak in IsaacGym to align the palm direction).
    4. Copies the right-hand STL meshes into the assembled mesh dir.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Surgical XML edits
# ---------------------------------------------------------------------------


def _remove_left_hand_subtree(left_text: str) -> str:
    """Strip every ``<link name="left_*">…</link>`` block and every ``<joint>``
    whose parent or child is a ``left_*`` link.

    Kept intact:
      * iiwa14_link_0 … iiwa14_link_ee
      * sharpa_mount link
      * the joint ``iiwa14_link_ee → sharpa_mount``  (the original file ends
        with this joint)
      * material declarations, comments, etc.
    """
    text = left_text

    # 1) Remove every <link name="left_*"> ... </link>
    text = re.sub(
        r'<link\s+name="left_[^"]*"\s*>.*?</link>\s*',
        "",
        text,
        flags=re.DOTALL,
    )

    # 2) Remove every <joint ...> ... </joint> that mentions a left_* parent or child.
    def _joint_filter(m: re.Match) -> str:
        block = m.group(0)
        if re.search(r'<(?:parent|child)\s+link="left_[^"]*"\s*/>', block):
            return ""  # drop the joint
        return block

    text = re.sub(
        r"<joint\b[^>]*>.*?</joint>\s*",
        _joint_filter,
        text,
        flags=re.DOTALL,
    )

    # Sanity: the iiwa14_link_ee → sharpa_mount joint should still be there.
    if not re.search(
        r'<joint[^>]*>[^<]*'
        r'(?:<[^>]*>[^<]*)*?'
        r'<parent\s+link="iiwa14_link_ee"\s*/>\s*'
        r'<child\s+link="sharpa_mount"\s*/>',
        text,
        flags=re.DOTALL,
    ):
        raise RuntimeError(
            "After stripping left subtree, the iiwa14_link_ee → sharpa_mount "
            "joint is missing.  The left URDF schema may have changed; please "
            "inspect manually."
        )

    return text


def _extract_right_body(right_text: str, mesh_dir_name: str) -> str:
    """Strip <?xml?>, <robot> wrapper and rewrite mesh paths to mesh_dir_name/."""
    right_text = re.sub(r"<\?xml[^>]*\?>", "", right_text).strip()
    m = re.search(r"<robot[^>]*>(.*)</robot>", right_text, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find <robot> tag in the right URDF")
    body = m.group(1)
    body = re.sub(r"<mujoco>.*?</mujoco>\s*", "", body, flags=re.DOTALL)

    def _rewrite(m: re.Match) -> str:
        path = m.group(1)
        name = Path(path).name
        return f'filename="{mesh_dir_name}/{name}"'

    body = re.sub(r'filename="([^"]+\.STL)"', _rewrite, body, flags=re.IGNORECASE)
    return body.strip()


_ATTACH_JOINT_TEMPLATE = """
  <!-- VideoRLFollower mount: sharpa_mount → right_hand_flange.  The left
       hand attaches via rpy="0 0 -1.5708"; for the right hand we mirror the
       sign to +pi/2 as a heuristic.  ⚠ This rpy MUST be visually validated
       at the all-zero joint pose in IsaacGym — if the palm points the wrong
       way, adjust the yaw component (typically by ±pi/2 or pi) so the
       right-hand thumb sits on the same side as the left's mirror image. -->
  <joint name="sharpa_mount_to_right_flange" type="fixed">
    <origin rpy="0 0 1.5708" xyz="0.0 0.0 0.05"/>
    <parent link="sharpa_mount"/>
    <child link="right_hand_flange"/>
  </joint>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left-urdf", required=True)
    p.add_argument("--right-urdf", required=True)
    p.add_argument("--right-meshes-src", required=True)
    p.add_argument("--out-urdf", required=True)
    p.add_argument("--out-meshes", required=True,
                   help="Mesh dir relative to URDF dir, e.g. right_sharpa_meshes")
    args = p.parse_args()

    left = Path(args.left_urdf).read_text()
    right = Path(args.right_urdf).read_text()

    out_meshes_path = Path(args.out_meshes).resolve()
    mesh_dir_name = out_meshes_path.name

    # 1) trim left, keeping the arm chain + sharpa_mount + connecting joint
    arm_chain = _remove_left_hand_subtree(left)

    # 2) extract right body with rewritten mesh paths
    right_body = _extract_right_body(right, mesh_dir_name)

    # 3) inject the attach joint and the right body BEFORE the closing </robot>
    inject = _ATTACH_JOINT_TEMPLATE + "\n" + right_body + "\n"
    if "</robot>" not in arm_chain:
        raise RuntimeError("Trimmed arm chain has no </robot>; refusing to write.")
    assembled = arm_chain.replace("</robot>", inject + "</robot>", 1)

    # 4) write URDF
    out_urdf = Path(args.out_urdf)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(assembled)

    # 5) copy right meshes — only the ones whose path actually starts with
    #    mesh_dir_name/ (avoids a false-positive warning about iiwa14 STLs).
    src = Path(args.right_meshes_src).resolve()
    dst = out_meshes_path
    dst.mkdir(parents=True, exist_ok=True)
    pat = re.compile(
        r'filename="' + re.escape(mesh_dir_name) + r'/([^/"]+\.STL)"',
        flags=re.IGNORECASE,
    )
    referenced = sorted(set(pat.findall(assembled)))
    copied = 0
    missing = []
    for name in referenced:
        src_file = src / name
        if not src_file.is_file():
            missing.append(name)
            continue
        shutil.copy2(src_file, dst / name)
        copied += 1
    print(f"URDF written  : {out_urdf}  ({len(assembled.splitlines())} lines)")
    print(f"meshes copied : {copied} / {len(referenced)} into {dst}")
    if missing:
        print("WARNING: the following hand meshes were not found in "
              f"{src} — copy them manually before loading the URDF:")
        for n in missing:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
