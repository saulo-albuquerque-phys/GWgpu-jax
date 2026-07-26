#!/bin/bash
# HTCondor/cluster wrapper for run_bilby_frozen.py on IGWN (LIGO Data Grid)
# machines: activates the shared cvmfs igwn conda env (bilby + lalsuite
# preinstalled) and runs the frozen 4s benchmark. Also fine to run directly
# inside tmux on an ldas-pcdev node.
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
cd "$(dirname "$0")"
exec python run_bilby_frozen.py --npool "${1:-16}"
