#!/usr/bin/env bash
# =============================================================
# GWjax — CUDA 12 environment setup (Linux + NVIDIA GPU)
# Creates a Python 3.11 venv and installs all dependencies.
# Requires: CUDA 12.x drivers, Linux x86_64
# Usage:  bash setup_env_cuda.sh [venv-name]   (default: venv_cuda)
# =============================================================
set -euo pipefail

VENV_NAME="${1:-venv_cuda}"
PYTHON=""

# ── Sanity check: this script is Linux-only ───────────────────
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "ERROR: CUDA is not supported on macOS."
    echo "       Use setup_env_cpu.sh for macOS instead."
    exit 1
fi

# ── Check CUDA availability ───────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found — no NVIDIA GPU detected."
    echo "         Proceeding anyway; JAX will fall back to CPU at runtime."
fi

# ── Find Python 3.11 ─────────────────────────────────────────
for candidate in python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        if [[ "$version" == "(3, 11)" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.11 not found. Install it first."
    echo "       On Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo "Using Python: $($PYTHON --version)  ($PYTHON)"

# ── Create venv ───────────────────────────────────────────────
if [[ -d "$VENV_NAME" ]]; then
    echo "Virtual environment '$VENV_NAME' already exists — skipping creation."
else
    "$PYTHON" -m venv "$VENV_NAME"
    echo "Created virtual environment: $VENV_NAME"
fi

source "$VENV_NAME/bin/activate"
pip install --upgrade pip --quiet

# ── Install JAX (CUDA 12) ─────────────────────────────────────
# This pulls in jax, jaxlib, jax-cuda12-pjrt, jax-cuda12-plugin
echo "Installing JAX 0.4.31 (CUDA 12)..."
pip install "jax[cuda12]==0.4.31" --quiet

# ── Install pinned BlackJAX nested-sampling fork ─────────────
# IMPORTANT: this specific commit is required — later commits on
# the nested_sampling branch removed the PartitionedState API.
echo "Installing pinned blackjax-ns..."
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@dedbf11da33eb5ca286f6731e2c51f2b254b953f" --quiet

# ── Install remaining dependencies ───────────────────────────
echo "Installing GW and inference packages..."
pip install -r requirements-cuda.txt --quiet

# ── Install GWjax itself (editable) ──────────────────────────
pip install -e . --quiet 2>/dev/null || true

# ── Smoke test ────────────────────────────────────────────────
echo ""
echo "Running smoke tests..."
python - <<'PYEOF'
import jax, jax.numpy as jnp
from ripplegw.waveforms import IMRPhenomD
import blackjax
from blackjax.ns.utils import finalise
from blackjax import nss
import anesthetic

theta = jnp.array([30.0, 0.25, 0.0, 0.0, 400.0, 0.0, 0.0, 20.0])
freqs = jnp.linspace(20.0, 1024.0, 1000)
hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(freqs, theta, 20.0)

print(f"  JAX {jax.__version__}       OK  (backend: {jax.default_backend()})")
print(f"  ripplegw             OK  (IMRPhenomD hp shape: {hp.shape})")
print(f"  blackjax {blackjax.__version__[:20]}  OK  (nss + finalise)")
print(f"  anesthetic {anesthetic.__version__}      OK")
PYEOF

echo ""
echo "Environment ready. Activate with:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To pin JAX memory fraction (recommended for large analyses):"
echo "  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8"
