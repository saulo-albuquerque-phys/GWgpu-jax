#!/bin/bash
# IGWN/LDG wrapper for run_bilby_gw170817.py (needs the venv from setup_ldg_env.sh and the frozen
# data_<psd_mode>/ directory copied next to this script).
#   bash run_bilby_gw170817.sh 32 [extra args, e.g. --spin-prior high | --psd-mode welch | --benchmark 200]
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
VENV="${TEOB_VENV:-$HOME/venvs/gw170817_teob}"
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: run setup_ldg_env.sh first"; exit 1; }
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NPOOL="${1:-32}"; shift || true
exec "$VENV/bin/python" run_bilby_gw170817.py --npool "$NPOOL" "$@"
