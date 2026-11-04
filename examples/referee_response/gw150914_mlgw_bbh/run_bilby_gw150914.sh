#!/bin/bash
# IGWN/LDG wrapper for run_bilby_gw150914.py: activates the cvmfs igwn env (bilby + lalsuite)
# and prepends ../../lal_data to $LAL_DATA_PATH, because SEOBNRv5ROM_v1.0.hdf5 is not reliably
# visible on condor execute nodes (copy it there once, under $HOME; see bbh_case/README.md).
#   bash run_bilby_gw150914.sh 16 [extra args, e.g. --psd-mode paper | --marginalize]
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
cd "$(dirname "$0")"
export LAL_DATA_PATH="$(pwd)/../../lal_data:${LAL_DATA_PATH}"
NPOOL="${1:-16}"; shift || true
exec python run_bilby_gw150914.py --npool "$NPOOL" "$@"
