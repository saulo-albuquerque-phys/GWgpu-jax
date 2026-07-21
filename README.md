# GWgpu-jax

GWgpu-jax provides a user-friendly interface between JAX waveform generators, mainly the Machine Learning Surrogates (MLGW), and JAX sampling algorithms for GPU-accelerated parameter estimation in gravitational wave data analysis. It supplies JAX likelihoods, detector-network construction, and real-data import pipelines.

## Waveform generators

| Generator | Model | Domain | Source type | Reference |
|---|---|---|---|---|
| [ripplegw](https://github.com/tedwards2412/ripple) | IMRPhenomD | frequency | BBH | [Phys.Rev.D 106 (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.044029) |
| mlgw_bns_jax | MLGW-BNS_TEOBResumSPA | frequency | BNS | [Phys. Rev. D 107 (2023)](
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.084037) |
| mlgw_bbh_jax | mlgw_SEOBNRv5HM (model_4) | time | BBH | [Phys.Rev.D 103 (2021)](
(https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.043020) |

mlgw_bns_jax and mlgw_bbh_jax live under `gwgpu_jax/mlgw_jax/` and are loaded via `sys.path` — no separate pip install is required.

## Sampler

This branch uses the **acceptance-walk** nested-sampling kernel of
[Prathaban et al. (2025)](https://arxiv.org/abs/2509.04336) — the
`bilby`/`dynesty` constrained random-walk move that is the LIGO/Virgo community
standard, ported to GPU inside the
[blackjax-ns](https://github.com/handley-lab/blackjax) framework and shown there
to recover posteriors and evidences **statistically identical** to the CPU
`bilby` pipeline for aligned-spin binary black holes. We use it here because,
unlike a generic sampler, it is a nested-sampling kernel already **validated on
gravitational-wave data** — the results are defensible to a PE-specialist
audience while your own gwgpu_jax likelihood, waveform, network, and priors are
used unchanged.

The driver is `gwgpu_jax.GWgpu_jaxAcceptanceWalkSampler` (see
[`ACCEPTANCE_WALK.md`](ACCEPTANCE_WALK.md)). It runs on:

- **blackjax-ns**, pinned to commit `dedbf11da33eb5ca286f6731e2c51f2b254b953f`
  (**do not upgrade** — the `PartitionedState` API was removed in later commits;
  the acceptance-walk kernel depends on it);
- the **acceptance-walk kernel** itself, which lives in a separate,
  separately-licensed repository
  ([`mrosep/blackjax_ns_gw`](https://github.com/mrosep/blackjax_ns_gw)) and is
  **not bundled** with gwgpu_jax. Fetch it once (see below); it pins the *same*
  blackjax commit, so there is no version conflict.

### Fetching the acceptance-walk kernel

```bash
bash scripts/fetch_acceptance_walk_kernel.sh
```

This sparse-clones only the `custom_kernels` package into `external/`
(git-ignored — the kernel is never copied into this repo). gwgpu_jax discovers
it automatically at `external/blackjax_ns_gw/src`, or set
`GWJAX_ACCEPTANCE_WALK_SRC` to point at any checkout of its `src/` directory.

**Cite** Prathaban et al. (2025, arXiv:2509.04336) and Cabezas et al. (2024,
arXiv:2402.10797) if you use this sampler.

## Installation

### Requirements

- Python 3.10 / 3.11 / 3.12
- Git

### Install from GitHub (pip — recommended)

The repo is currently **private**, so a [GitHub Personal Access Token](https://github.com/settings/tokens?type=beta) with `Contents: read-only` is required. Embed it once into the URL:

```bash
# Set the token as a shell variable (don't echo it into your history).
export GH_TOKEN=$(< ~/.config/gwgpu_jax_pat.txt)   # or read -s -p "PAT: " GH_TOKEN

# CPU only (laptops, macOS, plain Linux)
pip install "gwgpu_jax @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

# + real-data ingest (gwpy + GWOSC)
pip install "gwgpu_jax[data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

# + NVIDIA GPU (CUDA 12)
pip install "gwgpu_jax[gpu,data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

# + heavy ML waveform models (TensorFlow + tf2jax for SEOBNRv5HM / mlgw_bns_jax)
pip install "gwgpu_jax[mlgw,data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"
```

> When the repo becomes public, drop `$GH_TOKEN@` from every URL.

### Google Colab quickstart (GPU runtime)

Colab GPU runtimes ship with a CUDA-enabled JAX, so the `gpu` extra is **not** needed. Two ways to provide the token:

- **Colab Secrets** (recommended) — key icon 🔑 in the sidebar → add a secret called `GH_TOKEN` → toggle *Notebook access*.
- One-off `getpass.getpass()` prompt — the cell below falls back to it if no secret is set.

```python
import os, getpass

GH_TOKEN = None
try:
    from google.colab import userdata
    GH_TOKEN = userdata.get("GH_TOKEN")
except Exception:
    pass
if not GH_TOKEN:
    GH_TOKEN = getpass.getpass("GitHub PAT: ")
os.environ["GH_TOKEN"] = GH_TOKEN

!pip install -q "gwgpu_jax[data] @ git+https://$GH_TOKEN@github.com/saulo-albuquerque-phys/GWgpu-jax.git"

del os.environ["GH_TOKEN"]; del GH_TOKEN

import jax, gwgpu_jax
print(jax.devices())                       # → [CudaDevice(id=0), …]
grid = gwgpu_jax.TimeFrequencyGrid(4.0, 2048.0, f_min=20.0, f_max=512.0)
net  = gwgpu_jax.Network.from_names(["H1", "L1"], grid)
gwgpu_jax.compat.attach_event_to_network(net, "GW150914", estimate_psd=True)
```

The full ready-to-run version is in [`examples/gwgpu_jax_colab_pe.ipynb`](examples/gwgpu_jax_colab_pe.ipynb).

### Editable local install (development)

```bash
git clone https://github.com/saulo-albuquerque-phys/GWgpu-jax.git
cd GWgpu-jax
pip install -e ".[data,mlgw,samplers]"     # all extras except [gpu]
```

### Bash setup scripts

The `setup_env_*.sh` scripts build pinned, reproducible environments — useful
when you need to match an exact JAX / TF / blackjax combination.

### Full environment — all three waveforms + blackjax-ns (recommended for the ML models)

```bash
bash setup_env_full.sh           # creates ./venv_full by default
source venv_full/bin/activate
```

This installs **ripplegw + mlgw_bns_jax + mlgw_bbh_jax + blackjax-ns together**
(TensorFlow 2.15, jax 0.4.31, numpy 1.26.4; see `requirements-full.txt` for the
full pinned lock) and runs the smoke test below to confirm all four coexist in
one process. Verified from a clean build. The hard ceilings are `jax < 0.5`
(blackjax-ns fork) and `tensorflow < 2.16` (2.16 switched to Keras 3, which
breaks the mlgw_bbh loader).

### Acceptance-walk environment (this branch's sampler) — recommended

```bash
bash setup_env_acceptance_walk.sh    # builds ./venv_full, then fetches the kernel
source venv_full/bin/activate
```

This runs `setup_env_full.sh` (all three waveforms + pinned blackjax-ns) and then
`scripts/fetch_acceptance_walk_kernel.sh` to pull the external acceptance-walk
kernel into `external/` (git-ignored). After it finishes you can run the
acceptance-walk notebooks/examples directly. On Google Colab the notebooks do
the equivalent install + kernel fetch in their first cells — no local setup
needed.

### CPU-only environment (ripplegw + blackjax-ns only)

```bash
bash setup_env_cpu.sh            # creates ./venv by default
source venv/bin/activate
```

### CUDA 12 environment

```bash
bash setup_env_cuda.sh           # creates ./venv_cuda by default
source venv_cuda/bin/activate
```

### Manual setup

```bash
python3.11 -m venv venv_full
source venv_full/bin/activate
pip install --upgrade pip

# JAX (CPU)
pip install "jax[cpu]==0.4.31" "jaxlib==0.4.31"

# Pinned blackjax nested-sampling fork
pip install "blackjax @ git+https://github.com/handley-lab/blackjax.git@dedbf11da33eb5ca286f6731e2c51f2b254b953f"

# TensorFlow 2.15 + tf2jax (required for mlgw_bbh_jax / mlgw_bns_jax).
# Must stay < 2.16 — TF 2.16 ships Keras 3 and breaks the mlgw_bbh loader.
pip install "tensorflow==2.15.1" "tf2jax==0.3.6"

# Pin numpy (JAX 0.4.31 needs >=1.22; TF 2.15 accepts <2.0, so 1.26.4 fits both)
pip install "numpy==1.26.4"

# Remaining dependencies
pip install -r requirements-full.txt

# GWgpu_jax in editable mode
pip install -e .
```

### Verifying the installation

```bash
python tests/test_all_waveforms.py
```

Expected output (TensorFlow/JAX log lines omitted):

```
============================================================
1. ripplegw  |  IMRPhenomD  |  frequency-domain  |  BBH
============================================================
  hp shape : (4096,)   hc shape : (4096,)   STATUS : OK

============================================================
2. mlgw_bns_jax  |  ML-BNS  |  frequency-domain  |  BNS
============================================================
  hp shape : (2048,)   hc shape : (2048,)   STATUS : OK

============================================================
3. mlgw_bbh_jax  |  SEOBNRv5HM  |  time-domain  |  BBH
============================================================
  hp shape : (8192,)   hc shape : (8192,)   STATUS : OK

============================================================
4. blackjax-ns  |  nested sampling  |  sampler
============================================================
  version  : 0.1.dev707+gdedbf11da   STATUS : OK

============================================================
ALL FOUR COMPONENTS WORKING IN THE SAME ENVIRONMENT
============================================================
```

### GPU memory (CUDA only)

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8   # use 80 % of GPU memory
```

## mlgw_bbh_jax — notes on model loading

The mlgw_bbh_jax models are saved in Keras 3.x `.keras` format.  
The bundled `NN_model.py` has been patched to load them in a Keras 2.x environment (Keras 2.15, which ships with TF 2.15) by parsing the layer specs directly rather than calling `from_config`.  
The active model is **model_4** (SEOBNRv5HM, modes 22 21 32 33 43 44 55, $q \in [1,10]$, $\chi_{1,2} \in [-0.9, 0.9]$).

