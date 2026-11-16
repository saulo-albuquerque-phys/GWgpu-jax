#!/usr/bin/env python
"""Generate COLAB_profiling.ipynb (referee comments 4 and 6: run-time breakdown, JIT compile time,
multi-core CPU baseline). Install / sub-repo / self-check cells come from the GW150914 7-mode notebook.

    python build_notebook.py
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "gw150914_mlgw_bbh" / "COLAB_GW150914_mlgw_bbh_LVKpriors_HM.ipynb"
OUT = HERE / "COLAB_profiling.ipynb"
BRANCH = "py312_modern_jax"
RAW = f"https://raw.githubusercontent.com/saulo-albuquerque-phys/GWgpu-jax/{BRANCH}/examples/referee_response"

src = json.load(open(SRC))["cells"]
S = lambda c: "".join(c["source"])


def find(pred, kind="code"):
    hits = [c for c in src if c["cell_type"] == kind and pred(S(c))]
    assert len(hits) == 1, (pred, len(hits))
    return copy.deepcopy(hits[0])


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


c_install = find(lambda s: "gwjax_constraints.txt" in s and "pip install" in s)
c_sub = find(lambda s: "MLGW-JAX.git" in s)
assert "cryptography<44" in S(c_sub)
c_check = find(lambda s: s.startswith("# Waveform self-check:"))
c_kernel = find(lambda s: "blackjax_ns_gw.git" in s)

cells = [
    md("""
# Run-time profile of the paper's likelihoods (referee comments 4 and 6)

Times every stage of the likelihoods the sampler calls, per evaluation inside a vmapped batch (the
sampler advances `num_delete = 100` chains together, so batch 100 is the operating point), on the GPU
and then on this VM's CPU cores (1, 4 and all cores, via `taskset`), plus JIT compile times:

* **GW170817** (mlgw_bns_jax, 128 s, 23–1500 Hz, H1L1V1): waveform at the 3199 relative-binning bin
  frequencies; RB at fixed (t_c, φ_c); RB + t_c marginalisation (2001 points); RB + t_c + φ_c
  marginalisation (**production**); full band at fixed (t_c, φ_c); full band + t_c + φ_c marginalisation;
  compile + first iteration of the production sampler. These split the production run's 20.6 min into
  waveform, RB overlaps, marginalisation and sampler overhead.
* **GW150914** (mlgw_bbh_jax): (2,2) on 4 s and all 7 modes on 8 s, waveform and full-band likelihood.

Needs the GW170817 frozen data on Drive (`MyDrive/gwjax_referee/gw170817_data/data_gwtc1/`, the same
folder the GW170817 notebook reads); the GW150914 inputs are fetched from GitHub.
Results go to `MyDrive/gwjax_referee/profiling/<timestamp>/`.

**Order:** run the install cell, restart the session, then run everything from the top.
Expect about 1 h in total, most of it in the 1-core CPU pass.
"""),
    md("## 1. Install"), c_install, c_sub, c_check,
    md("## 2. Inputs, scripts and machine"),
    code(f'''
import os, json, subprocess, datetime, urllib.request
from pathlib import Path
RAW = "{RAW}"
WORK = Path("/content/profiling"); WORK.mkdir(exist_ok=True)
for rel in ("profiling/profile_pe.py", "profiling/summarise_profile.py",
            "gw150914_mlgw_bbh/data_bilby_hm_8s/gw150914_data.npz",
            *[f"gw150914_mlgw_bbh/data_lvk_welch/{{n}}" for n in ("metadata.json", "frequency_array.npy", "H1_psd.npy",
                                                               "H1_strain_fd.npy", "L1_psd.npy", "L1_strain_fd.npy")]):
    dst = WORK / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{{RAW}}/{{rel}}", dst)

from google.colab import drive
if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")
GW170817_DATA = "/content/drive/MyDrive/gwjax_referee/gw170817_data/data_gwtc1"
assert Path(GW170817_DATA, "metadata.json").exists(), f"copy data_gwtc1/ to {{GW170817_DATA}} first"
OUT_DIR = f"/content/drive/MyDrive/gwjax_referee/profiling/{{datetime.datetime.now():%Y%m%d_%H%M%S}}"
os.makedirs(OUT_DIR, exist_ok=True)

N_CPU = len(os.sched_getaffinity(0))
machine = dict(gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip(),
               cpu=next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
               cpu_cores=N_CPU, ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
json.dump(machine, open(f"{{OUT_DIR}}/machine.json", "w"), indent=2)
print(machine); print("results ->", OUT_DIR)
COMMON = (f"--gw170817-data {{GW170817_DATA}} --gw150914-data {{WORK}}/gw150914_mlgw_bbh/data_lvk_welch "
          f"--gw150914-hm-data {{WORK}}/gw150914_mlgw_bbh/data_bilby_hm_8s/gw150914_data.npz")
FILTER = 'grep -vE "features_spec|ListWrapper|keras|^ *$|oneDNN|cuda_|TF-TRT|tensorflow|Warning"'
'''),
    md("## 3. Acceptance-walk kernel (for the sampler compile-time measurement)"), c_kernel,
    md("""
## 4. GPU profile

Batches 1, 10, 100, 1000; GW170817 also times the production sampler (1000 live points) for one and
three iterations, which separates its compile + setup time from the per-iteration time.
"""),
    code('''
!cd {WORK} && python -u profiling/profile_pe.py --what gw170817 gw150914_22 gw150914_hm --batches 1 10 100 1000 \\
    --sampler-compile --label "GPU {machine['gpu'].split(',')[0]}" --out {OUT_DIR}/profile_gpu.json {COMMON} 2>&1 | {FILTER}
'''),
    md("""
## 5. CPU profiles: 1, 4 and all cores

Same code on this VM's CPU (`JAX_PLATFORMS=cpu`, GPU hidden), restricted to the first N cores with
`taskset` (XLA's CPU thread pool sizes itself from the cores it is allowed to use). Batches 1, 10, 100.
The full-band t_c/φ_c-marginalised GW170817 likelihood is skipped here: it takes ~25 min to compile on a CPU
(it is only there to show what the run would cost without relative binning, which the GPU pass covers).
This is the batched multi-core CPU baseline of comment 4.
"""),
    code('''
for n in sorted({1, min(4, N_CPU), N_CPU}):
    print(f"\\n######## CPU, {n} core(s) ########")
    !cd {WORK} && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu taskset -c 0-{n-1} python -u profiling/profile_pe.py \\
        --what gw170817 gw150914_22 gw150914_hm --batches 1 10 100 --skip-full-marg --label "CPU {n} core(s)" \\
        --out {OUT_DIR}/profile_cpu_{n}.json {COMMON} 2>&1 | {FILTER}
'''),
    md("""
## 6. Summary tables

Per-evaluation cost at batch 100 on every device, compile times, and the split of the GW170817
production run (7.46e6 likelihood calls in 1236 s on an A100) into waveform, RB overlaps,
t_c/φ_c marginalisation and sampler overhead. Saved as `profile_summary.json`.
"""),
    code('''
import glob
files = sorted((f for f in glob.glob(f"{OUT_DIR}/profile_*.json") if not f.endswith("_summary.json")), key=lambda f: ("cpu" in f, len(f), f))
!cd {WORK} && python profiling/summarise_profile.py {" ".join(files)} --batch 100 --out {OUT_DIR}/profile_summary.json
print("\\nall results in", OUT_DIR)
'''),
]
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
nb = json.load(open(SRC)); nb["cells"] = cells
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"wrote {OUT.name}: {len(cells)} cells")
