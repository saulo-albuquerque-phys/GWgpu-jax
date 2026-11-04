#!/bin/bash
# HTCondor/cluster wrapper for run_bilby_bbh_imrphenomxas.py on IGWN (LIGO
# Data Grid) machines: activates the shared cvmfs igwn conda env (bilby +
# lalsuite preinstalled) and runs the faster IMRPhenomXAS cross-check.
# IMRPhenomXAS is purely analytic (no ROM data file dependency), but this
# still prepends ../lal_data to $LAL_DATA_PATH as a harmless safety margin,
# matching run_bilby_bbh.sh.
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
cd "$(dirname "$0")"
export LAL_DATA_PATH="$(pwd)/../lal_data:$LAL_DATA_PATH"
exec python run_bilby_bbh_imrphenomxas.py --npool "${1:-16}"
