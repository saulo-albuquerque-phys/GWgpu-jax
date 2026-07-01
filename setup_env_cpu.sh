#!/usr/bin/env bash
# =============================================================
# GWgpu-jax — CPU environment setup (macOS / Linux, no GPU)
# Creates a Python 3.11 venv and installs all dependencies.
# Usage:  bash setup_env_cpu.sh [venv-name]   (default: venv)
# =============================================================
set -euo pipefail

VENV_NAME="${1:-venv}"
PYTHON=""

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
    echo "ERROR: Python 3.11 not found. Install it first (e.g. brew install python@3.11)."
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

# ── Install JAX (CPU) ─────────────────────────────────────────
echo "Installing JAX 0.4.31 (CPU)..."
pip install "jax[cpu]==0.4.31" "jaxlib==0.4.31" --quiet

# ── Install pinned BlackJAX nested-sampling fork ─────────────
# IMPORTANT: this specific commit is required — later commits on
# the nested_sampling branch removed the PartitionedState API.
echo "Installing pinned blackjax-ns..."
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@dedbf11da33eb5ca286f6731e2c51f2b254b953f" --quiet

# ── Install remaining dependencies ───────────────────────────
echo "Installing GW and inference packages..."
pip install -r requirements-cpu.txt --quiet

# ── Install GWgpu_jax itself (editable) ──────────────────────────
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
