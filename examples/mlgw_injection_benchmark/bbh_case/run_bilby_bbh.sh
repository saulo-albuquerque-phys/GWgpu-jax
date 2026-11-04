#!/bin/bash
# HTCondor/cluster wrapper for run_bilby_bbh.py on IGWN (LIGO Data Grid)
# machines: activates the shared cvmfs igwn conda env (bilby + lalsuite
# preinstalled there) and runs the mlgw_bbh-vs-bilby BBH case. Also fine to
# run directly inside tmux on an ldas-pcdev node.
#
# SEOBNRv5_ROM needs a data file (SEOBNRv5ROM_v1.0.hdf5) that is NOT
# reliably on $LAL_DATA_PATH across every execute node — on at least one
# IGWN cluster it was only present under /scratch on the *login* node, which
# is local-per-machine and invisible to condor execute nodes. Fix: copy the
# file into this repo checkout's ../lal_data/ (under $HOME, which condor
# jobs already reach the same way they reach this script) and prepend that
# to $LAL_DATA_PATH here, ahead of whatever cvmfs provides.
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
cd "$(dirname "$0")"
export LAL_DATA_PATH="$(pwd)/../lal_data:$LAL_DATA_PATH"
exec python run_bilby_bbh.py --npool "${1:-16}"
