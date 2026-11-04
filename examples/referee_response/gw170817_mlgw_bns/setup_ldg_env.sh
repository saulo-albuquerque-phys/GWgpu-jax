#!/bin/bash
# One-off, on the LDG head node, in this directory: install the `teobresums` package (TEOBResumSPA is
# not a LALSimulation approximant) into ./pylib with the cvmfs igwn Python. condor ships ./pylib to the
# execute node together with the scripts and data (execute nodes do not necessarily see $HOME).
#   bash setup_ldg_env.sh
set -e
cd "$(dirname "$0")"
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
python -m pip install -q --no-deps --target ./pylib teobresums
PYTHONPATH="$PWD/pylib:$PYTHONPATH" python - <<'PY'
import bilby, EOBRun_module, numpy, sys
print("python", sys.version.split()[0], "| bilby", bilby.__version__, "| teobresums OK | numpy", numpy.__version__)
PY
echo "ready: $(ls pylib)"
