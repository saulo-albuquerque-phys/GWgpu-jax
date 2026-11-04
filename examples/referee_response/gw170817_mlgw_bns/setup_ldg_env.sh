#!/bin/bash
# One-off, on an LDG head node: a venv on top of the cvmfs igwn env (bilby + lalsuite) that adds the
# `teobresums` package (TEOBResumSPA is not a LALSimulation approximant).
#   bash setup_ldg_env.sh            # -> $HOME/venvs/gw170817_teob  (override with TEOB_VENV=...)
set -e
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
VENV="${TEOB_VENV:-$HOME/venvs/gw170817_teob}"
python -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q teobresums
"$VENV/bin/python" - <<'PY'
import bilby, EOBRun_module, numpy
print("bilby", bilby.__version__, "| teobresums OK | numpy", numpy.__version__)
PY
echo "venv ready: $VENV"
