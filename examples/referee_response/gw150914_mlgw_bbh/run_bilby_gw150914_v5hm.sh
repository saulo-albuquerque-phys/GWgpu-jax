#!/bin/bash
# IGWN/LDG wrapper: GW150914 bilby reference with SEOBNRv5HM (2,2) (pyseobnr via LAL gwsignal).
# Runs in the condor sandbox (run_bilby_gw150914.py + data_lvk_welch/ transferred there by
# run_bilby_gw150914_v5hm.sub) or interactively in this directory.
#   bash run_bilby_gw150914_v5hm.sh 32 [extra args, e.g. --benchmark 20 --outdir bench]
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/numba_cache"          # pyseobnr's numba cache; cvmfs is read-only
python -c "import pyseobnr, lalsimulation.gwsignal" || { echo "pyseobnr/gwsignal missing in igwn on $(hostname)"; exit 1; }
NPOOL="${1:-32}"; shift || true
exec python run_bilby_gw150914.py --npool "$NPOOL" --approximant SEOBNRv5HM "$@"
