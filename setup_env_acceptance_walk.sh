#!/usr/bin/env bash
# =============================================================
# GWgpu-jax — acceptance-walk sampler environment
# =============================================================
# Builds the full gwgpu_jax environment (ripplegw + mlgw_bns_jax +
# mlgw_bbh_jax + pinned blackjax-ns) and then fetches the external,
# GW-validated acceptance-walk nested-sampling kernel
# (Prathaban et al. 2025, arXiv:2509.04336) used by
# gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler.
#
# The kernel is NOT vendored: it is sparse-cloned into external/ (git-ignored)
# and pins the SAME blackjax commit gwgpu_jax uses, so nothing conflicts.
#
# Usage:  bash setup_env_acceptance_walk.sh [venv-name]   (default: venv_full)
# Then:   source <venv-name>/bin/activate
# =============================================================
set -euo pipefail

VENV_NAME="${1:-venv_full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> [1/2] Building the full gwgpu_jax environment ('$VENV_NAME')..."
bash setup_env_full.sh "$VENV_NAME"

echo
echo "==> [2/2] Fetching the external acceptance-walk kernel..."
bash scripts/fetch_acceptance_walk_kernel.sh

echo
echo "============================================================="
echo "Acceptance-walk environment ready."
echo "  source $VENV_NAME/bin/activate"
echo
echo "Quick check that the sampler + kernel import:"
echo "  source $VENV_NAME/bin/activate && python -c \\"
echo "    'import gwgpu_jax; from gwgpu_jax.gwgpu_jax_acceptance_walk import import_custom_kernels; import_custom_kernels(); print(\"acceptance-walk kernel OK\")'"
echo "============================================================="
