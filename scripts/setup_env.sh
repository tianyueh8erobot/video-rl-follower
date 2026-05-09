#!/usr/bin/env bash
# Set up a conda env capable of running video_rl_follower (SimToolReal-based).
#
# Usage:
#   bash scripts/setup_env.sh [env_name]
#
# Defaults to reusing the existing 'factory' env if it already has IsaacGym;
# otherwise creates 'video_rl_follower' from scratch.
#
# Requirements:
#   * IsaacGym Preview 4 already extracted somewhere on disk (we'll detect it)
#   * conda available on PATH
#   * Network access to PyPI (or set PIP_INDEX_URL)

set -euo pipefail

ENV_NAME="${1:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Drop proxies that interfere with PyPI in some setups
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# ---------------------------------------------------------------------------
# Locate conda
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    if [ -x "$HOME/miniconda3/bin/conda" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    elif [ -x "$HOME/anaconda3/bin/conda" ]; then
        export PATH="$HOME/anaconda3/bin:$PATH"
    else
        echo "ERROR: conda not found on PATH; install Miniconda first." >&2
        exit 1
    fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# Try to reuse an existing env that already has isaacgym + py3.8
# ---------------------------------------------------------------------------
auto_pick_env() {
    for cand in factory video_rl_follower; do
        if [ -x "${CONDA_BASE}/envs/${cand}/bin/python" ]; then
            "${CONDA_BASE}/envs/${cand}/bin/python" -c \
              "import sys; assert sys.version_info[:2] == (3, 8)" 2>/dev/null \
              || continue
            "${CONDA_BASE}/envs/${cand}/bin/python" -c \
              "import isaacgym" >/dev/null 2>&1 && {
                echo "$cand"; return
            }
        fi
    done
}

if [ -z "$ENV_NAME" ]; then
    ENV_NAME="$(auto_pick_env)"
    if [ -n "$ENV_NAME" ]; then
        echo "[setup] Reusing existing env with isaacgym installed: $ENV_NAME"
    else
        ENV_NAME="video_rl_follower"
        echo "[setup] No existing isaacgym env found; creating $ENV_NAME (py3.8)"
    fi
fi

# ---------------------------------------------------------------------------
# Create env if missing
# ---------------------------------------------------------------------------
if [ ! -x "${CONDA_BASE}/envs/${ENV_NAME}/bin/python" ]; then
    conda create -y -n "${ENV_NAME}" python=3.8
fi
conda activate "${ENV_NAME}"

PYBIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIPBIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

echo "[setup] Using $($PYBIN --version) at $PYBIN"

# ---------------------------------------------------------------------------
# IsaacGym install
# ---------------------------------------------------------------------------
if ! "$PYBIN" -c "import isaacgym" >/dev/null 2>&1; then
    ISAACGYM_DIR="${ISAACGYM_DIR:-$HOME/Codes/isaacgym}"
    if [ ! -d "$ISAACGYM_DIR/python" ]; then
        echo "ERROR: IsaacGym not found at $ISAACGYM_DIR." >&2
        echo "       Download IsaacGym Preview 4 from" >&2
        echo "         https://developer.nvidia.com/isaac-gym" >&2
        echo "       extract it, then rerun:" >&2
        echo "         ISAACGYM_DIR=/path/to/isaacgym bash scripts/setup_env.sh" >&2
        exit 1
    fi
    echo "[setup] Installing IsaacGym from $ISAACGYM_DIR"
    "$PIPBIN" install --quiet -e "$ISAACGYM_DIR/python"
fi

# ---------------------------------------------------------------------------
# PyTorch (CUDA 11.7 by default; respect torch already installed)
# ---------------------------------------------------------------------------
if ! "$PYBIN" -c "import torch" >/dev/null 2>&1; then
    "$PIPBIN" install --quiet \
        torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
fi

# ---------------------------------------------------------------------------
# Other deps
# ---------------------------------------------------------------------------
# --- Required deps: any failure here aborts setup -------------------------
"$PIPBIN" install --quiet -i "$PIP_INDEX_URL" --upgrade \
    "rl-games>=1.6.0" \
    "hydra-core>=1.2" \
    "omegaconf" \
    "trimesh==3.23.5" \
    "urdfpy==0.0.22" \
    "yourdfpy" \
    "warp-lang" \
    "gym==0.23.1" \
    "termcolor" \
    "matplotlib" \
    "tensorboard"

# --- Optional deps (extras for visualisation / logging) -------------------
for opt in open3d wandb; do
    if ! "$PIPBIN" install --quiet -i "$PIP_INDEX_URL" "$opt"; then
        echo "[setup] WARN: optional package '$opt' failed to install; "\
             "skipping (the visualisation Open3D backend / wandb logging will "\
             "not work without it)."
    fi
done

# Editable install of THIS repo
"$PIPBIN" install --quiet -i "$PIP_INDEX_URL" -e "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "[verify] Importing key modules…"
"$PYBIN" - <<'PY'
import sys
# Required modules — any ImportError must abort setup.
required = {
    "isaacgym":      "IsaacGym",
    "torch":         "PyTorch",
    "rl_games":      "rl_games",
    "hydra":         "hydra-core",
    "omegaconf":     "omegaconf",
    "trimesh":       "trimesh",
    "urdfpy":        "urdfpy",
    "yourdfpy":      "yourdfpy (FK utility for SimToolReal cross-check)",
    "warp":          "warp-lang",
    "gym":           "gym",
    "matplotlib":    "matplotlib",
}
missing = []
for mod, label in required.items():
    try:
        __import__(mod)
    except Exception as e:
        missing.append((mod, label, repr(e)))
if missing:
    print("  ✗ MISSING required modules:")
    for mod, label, err in missing:
        print(f"      {mod}  ({label}) :: {err}")
    sys.exit(1)
# Project-side: confirm the env class loads (this also exercises the
# subclass + algos.dapg + dextoolbench import paths).
from isaacgymenvs.tasks.video_rl_follower.env import VideoRLFollower
from isaacgymenvs.tasks.video_rl_follower.trajectory import ReferenceTrajectory
from isaacgymenvs.algos.dapg import DAPGAgent
from isaacgymenvs.tasks import isaacgym_task_map
assert "VideoRLFollower" in isaacgym_task_map
print("  task map:", list(isaacgym_task_map.keys()))
print("  ✓ all imports OK")
PY

echo
echo "[done] env name : $ENV_NAME"
echo "        python   : $PYBIN"
echo "        activate : conda activate $ENV_NAME"
echo "        next     : python scripts/test_pipeline.py    # smoke test"
echo "                   python scripts/visualize_trajectory_in_isaacgym.py \\"
echo "                       --traj data/trajectories/example_g0.json"
