#!/bin/bash
# IGWN/LDG wrapper for run_bilby_gw170817.py. Runs in the condor sandbox (scripts, data_gwtc1/ and
# pylib/ transferred there by run_bilby_gw170817.sub) or interactively in this directory.
#   bash run_bilby_gw170817.sh 32 [extra args, e.g. --spin-prior high | --psd-mode welch | --benchmark 200]
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
cd "$(dirname "$0")"
[ -d pylib ] || { echo "no ./pylib: run setup_ldg_env.sh first"; exit 1; }
export PYTHONPATH="$PWD/pylib:$PWD:${PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/pylib/lib:${LD_LIBRARY_PATH}"     # bundled libopenblas etc. (setup_ldg_env.sh)
python -c "import EOBRun_module" || { echo "teobresums import failed on $(hostname):"; ldd pylib/EOBRun_module*.so; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NPOOL="${1:-32}"; shift || true
exec python run_bilby_gw170817.py --npool "$NPOOL" "$@"
