#!/usr/bin/env bash
# =============================================================
# GWjax — full environment setup
# Creates a Python 3.11 venv with all three waveform generators:
#   ripplegw        — IMRPhenomD, frequency-domain, BBH
#   mlgw_bns_jax    — ML-BNS, frequency-domain, BNS   (local subdir)
#   mlgw_bbh_jax    — SEOBNRv5HM (model_4), time-domain, BBH  (local subdir)
# plus blackjax-ns nested sampling.
#
# Usage:  bash setup_env_full.sh [venv-name]   (default: venv_full)
#
# Tested on: macOS x86_64 (Intel), Python 3.11
# =============================================================
set -euo pipefail

VENV_NAME="${1:-venv_full}"
PYTHON=""

# ── Find Python 3.11 ──────────────────────────────────────────
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

# ── 1. JAX (CPU) ──────────────────────────────────────────────
echo "Installing JAX 0.4.31 (CPU)..."
pip install "jax[cpu]==0.4.31" "jaxlib==0.4.31" --quiet

# ── 2. BlackJAX nested-sampling fork (pinned commit) ─────────
# IMPORTANT: this specific commit is required — later commits on
# the nested_sampling branch removed the PartitionedState API.
echo "Installing pinned blackjax-ns..."
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@dedbf11da33eb5ca286f6731e2c51f2b254b953f" --quiet

# ── 3. TensorFlow 2.13 + tf2jax ──────────────────────────────
# Required for mlgw_bbh_jax (SEOBNRv5HM) model weight loading.
echo "Installing TensorFlow 2.13.0 + tf2jax..."
pip install "tensorflow==2.13.0" "tf2jax==0.3.6" --quiet

# ── 4. Pin numpy after TF ────────────────────────────────────
# TF 2.13 may downgrade numpy; JAX 0.4.31 needs >=1.26.4.
# 1.26.4 works with both in practice despite TF's stated constraint.
echo "Pinning numpy to 1.26.4..."
pip install "numpy==1.26.4" --quiet

# ── 5. Remaining dependencies ─────────────────────────────────
echo "Installing waveform and inference packages..."
pip install -r requirements-full.txt --quiet

# ── 6. GWjax itself (editable) ───────────────────────────────
pip install -e . --quiet 2>/dev/null || true

# ── Smoke test ────────────────────────────────────────────────
echo ""
echo "Running smoke test (suppress TF/JAX noise)..."
python test_all_waveforms.py 2>/dev/null

echo ""
echo "============================================================"
echo "Environment '$VENV_NAME' is ready."
echo "Activate with:"
echo "  source $VENV_NAME/bin/activate"
echo "============================================================"
