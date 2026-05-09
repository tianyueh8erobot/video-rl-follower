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

The script does NOT modify left mesh files.  It simply:

    1. Reads the left URDF as text.
    2. Drops the ``<!-- Custom mount -->`` ↓ section's tail past ``sharpa_mount``
       attachment to ``left_hand_C_MC`` (we keep ``sharpa_mount`` itself).
    3. Reads the official right URDF, strips the wrapper ``<robot>`` tag, fixes
       mesh paths to ``right_sharpa_meshes/<name>.STL``.
    4. Inserts a single fixed joint connecting ``sharpa_mount`` to
       ``right_hand_flange`` (origin defaults to a sane orientation that will
       likely need a final tweak in IsaacGym to align the palm direction).
    5. Copies the right-hand STL meshes into the assembled mesh dir.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _strip_left_hand(left_text: str) -> str:
    """Keep the iiwa14 + sharpa_mount portion only.

    The left URDF ends with::

        <!-- Custom mount -->
        <link name="sharpa_mount">…</link>
        <joint name="sharpa_mount_joint" type="fixed">
          <parent link="iiwa14_link_ee"/><child link="sharpa_mount"/>…
        </joint>
        <joint name="left_hand_attach" type="fixed">
          <origin rpy="0 0 -1.5708" xyz="0.0 0.0 0.05"/>
          <parent link="sharpa_mount"/><child link="left_hand_C_MC"/>
        </joint>
        <link name="left_hand_C_MC">…</link>
        … all left fingers …
        </robot>

    We want to keep everything up to and including the
    ``<joint><parent link="iiwa14_link_ee"/>…<child link="sharpa_mount"/>``
    block, and drop the joint that attaches sharpa_mount → left_hand_C_MC and
    everything after.
    """
    # Find the joint that connects sharpa_mount → left_hand_C_MC.
    pat = re.compile(
        r"<joint[^>]*>\s*"
        r"(?:<[^/][^>]*?/>\s*)*"
        r"<parent\s+link=\"sharpa_mount\"\s*/>\s*"
        r"<child\s+link=\"left_hand_C_MC\"\s*/>",
        re.DOTALL,
    )
    m = pat.search(left_text)
    if not m:
        raise RuntimeError(
            "Could not find joint linking 'sharpa_mount' → 'left_hand_C_MC' "
            "in the left URDF; perhaps the URDF schema changed?"
        )
    # Drop from this joint onward up to (but not including) </robot>.
    head = left_text[: m.start()]
    # Re-attach the closing </robot> tag (we removed it together with the tail)
    return head


def _extract_right_body(right_text: str, mesh_dir_name: str) -> str:
    """Strip <?xml?>, <robot> wrapper and rewrite mesh paths."""
    # Remove the <?xml ...?> declaration if present.
    right_text = re.sub(r"<\?xml[^>]*\?>", "", right_text).strip()
    # Strip the surrounding <robot ...> ... </robot>.
    m = re.search(r"<robot[^>]*>(.*)</robot>", right_text, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find <robot> tag in the right URDF")
    body = m.group(1)
    # The official URDF declares <mujoco><compiler meshdir="meshes"/></mujoco>;
    # drop that block to avoid a stale meshdir hint.
    body = re.sub(r"<mujoco>.*?</mujoco>\s*", "", body, flags=re.DOTALL)
    # Rewrite mesh filenames to live under our mesh_dir_name.
    def _rewrite(m: re.Match) -> str:
        path = m.group(1)
        name = Path(path).name
        return f'filename="{mesh_dir_name}/{name}"'
    body = re.sub(r'filename="([^"]+\.STL)"', _rewrite, body, flags=re.IGNORECASE)
    return body.strip()


_ATTACH_JOINT_TEMPLATE = """
  <!-- VideoRLFollower mount: sharpa_mount → right_hand_flange (translated
       from the left-hand attachment ``rpy 0 0 -1.5708 xyz 0 0 0.05``).  For
       right hand we flip the wrist Z by +pi/2 (mirror of left's -pi/2) so
       the palm faces the same direction relative to the arm.  The xyz is
       unchanged because both hands mount on the same flange centre. -->
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

    # 1) trim left
    head = _strip_left_hand(left)

    # 2) extract right body with rewritten mesh paths
    right_body = _extract_right_body(right, mesh_dir_name)

    # 3) splice
    assembled = head + _ATTACH_JOINT_TEMPLATE + "\n" + right_body + "\n</robot>\n"

    # 4) write URDF
    out_urdf = Path(args.out_urdf)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(assembled)

    # 5) copy right meshes
    src = Path(args.right_meshes_src).resolve()
    dst = out_meshes_path
    dst.mkdir(parents=True, exist_ok=True)
    # Use the set of mesh basenames actually referenced in the assembled URDF.
    referenced = set(
        Path(m).name for m in re.findall(r'filename="[^"]+/([^/"]+\.STL)"',
                                         assembled, flags=re.IGNORECASE)
    )
    copied = 0
    missing = []
    for name in sorted(referenced):
        src_file = src / name
        if not src_file.is_file():
            missing.append(name)
            continue
        shutil.copy2(src_file, dst / name)
        copied += 1
    print(f"URDF written  : {out_urdf}  ({len(assembled.splitlines())} lines)")
    print(f"meshes copied : {copied} into {dst}")
    if missing:
        print("WARNING: the following referenced meshes were not found in "
              f"{src} — copy them manually before loading the URDF:")
        for n in missing:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
