#!/usr/bin/env bash
# Fetch the external, GW-validated acceptance-walk nested-sampling kernel
# (Prathaban et al. 2025, arXiv:2509.04336) used by
# gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler.
#
# The kernel lives in a separate, separately-licensed repository and is NOT
# vendored into gwgpu_jax. This script sparse-clones ONLY the `custom_kernels`
# package (no large strain/data files) into external/, which is git-ignored.
#
# Pinned to the same BlackJAX commit gwgpu_jax uses:
#   dedbf11da33eb5ca286f6731e2c51f2b254b953f
# (later blackjax_ns_gw commits require a newer BlackJAX branch; this one
#  targets the pinned PartitionedState API, matching our environment.)
#
# Usage:
#   bash scripts/fetch_acceptance_walk_kernel.sh
# then run PE with GWgpu_jaxAcceptanceWalkSampler (see
# examples/run_pe_acceptance_walk.py).
set -euo pipefail

REPO_URL="https://github.com/mrosep/blackjax_ns_gw.git"
DEST="external/blackjax_ns_gw"

# Resolve repo root (parent of this script's directory).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d "$DEST/src/custom_kernels" ]; then
  echo "Kernel already present at $DEST/src/custom_kernels — nothing to do."
  echo "(delete $DEST to re-fetch.)"
  exit 0
fi

echo "Sparse-cloning custom_kernels from $REPO_URL into $DEST ..."
rm -rf "$DEST"
git clone --no-checkout --depth 1 --filter=blob:none "$REPO_URL" "$DEST"
git -C "$DEST" sparse-checkout set src/custom_kernels
git -C "$DEST" checkout

echo
echo "Done. custom_kernels installed at: $DEST/src/custom_kernels"
echo "gwgpu_jax will discover it automatically at <repo>/external/blackjax_ns_gw/src,"
echo "or set GWJAX_ACCEPTANCE_WALK_SRC=$ROOT/$DEST/src"
