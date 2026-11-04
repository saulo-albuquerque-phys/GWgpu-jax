#!/bin/bash
# One-off, on the LDG head node, in this directory: install the `teobresums` package (TEOBResumSPA is
# not a LALSimulation approximant) into ./pylib with the cvmfs igwn Python. pip compiles it from source
# against whatever the head node offers; every shared library it resolves OUTSIDE /cvmfs (e.g. the head
# node's system libopenblas, which execute nodes lack) is copied, with its dependencies, into
# ./pylib/lib. condor ships ./pylib to the node with the scripts and data.
#   bash setup_ldg_env.sh
set -e
cd "$(dirname "$0")"
source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate igwn
python -m pip install -q --no-deps --target ./pylib teobresums
mkdir -p pylib/lib
SO=$(ls pylib/EOBRun_module*.so)
# everything outside /cvmfs except the glibc core (present on every node)
GLIBC='/(libc|libm|libdl|librt|libpthread|libutil|ld-linux[^/]*)\.so'
deps() { ldd "$1" | awk '/=> \// {print $3}' | grep -v '^/cvmfs/' | grep -Ev "$GLIBC" || true; }
todo=$(deps "$SO"); seen=""
while [ -n "$todo" ]; do
  next=""
  for lib in $todo; do
    case " $seen " in *" $lib "*) continue;; esac
    seen="$seen $lib"; cp -Lf "$lib" pylib/lib/; next="$next $(deps "$lib")"
  done
  todo=$(echo $next)
done
echo "bundled into pylib/lib:"; ls pylib/lib
if LD_LIBRARY_PATH="$PWD/pylib/lib" ldd "$SO" | grep -q "not found"; then
  LD_LIBRARY_PATH="$PWD/pylib/lib" ldd "$SO" | grep "not found"; echo "unresolved libraries"; exit 1
fi
LD_LIBRARY_PATH="$PWD/pylib/lib" ldd "$SO"
LD_LIBRARY_PATH="$PWD/pylib/lib" PYTHONPATH="$PWD/pylib:$PYTHONPATH" python - <<'PY'
import bilby, EOBRun_module, numpy, sys
print("python", sys.version.split()[0], "| bilby", bilby.__version__, "| teobresums OK | numpy", numpy.__version__)
PY
echo "ready"
