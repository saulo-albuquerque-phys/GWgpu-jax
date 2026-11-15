"""Colab install cell + waveform self-check for the gwgpu_jax Colab notebooks (Python 3.12 image).

History
-------
* The notebooks used to force jax 0.4.31 over Colab's preinstalled jax. On the Python 3.12 image
  (runtime 2026.1, which ships jax 0.4.38 + TF 2.19) that left files of both versions behind
  (`import jax` -> "partially initialized module 'jax'"), and even after a clean reinstall the
  kernel died a few minutes into sampling -- most likely jax 0.4.31's CUDA plugin running against
  the image's newer CUDA/cuDNN.
* Now: keep Colab's own jax (its CUDA plugin matches the image), install jax only if it is missing
  or has no GPU backend, and freeze the jax/TF stack with a pip constraints file so installing
  gwgpu_jax cannot upgrade or downgrade any of it (an unconstrained `tf2jax` would otherwise pull
  jax >= 0.7.1 and TF >= 2.20).
* Because the jax / TF / Keras / tf2jax versions are no longer fixed, each notebook gets a cell that
  compares mlgw waveforms against reference values computed in the validated environment
  (Python 3.11, jax 0.4.31, TF 2.15, Keras 2) and stops if they differ. Checked locally (Nov 2026):
  identical for jax 0.4.31 / 0.4.38 / 0.5.2 (+TF 2.16-2.18, Keras 3) and, for mlgw_bns and the
  GW170817 likelihood, jax 0.12 (rel. diff <= 4e-11).

Used by the notebook builders (build_notebook.py); also runnable on notebooks directly:

    python colab_jax_install.py [--check bbh|bns --after SUBSTRING] NOTEBOOK.ipynb [...]
"""
import argparse
import json
import re
from pathlib import Path

REF_DIR = Path(__file__).resolve().parent / "waveform_reference"
CONSTRAINTS = "/content/gwjax_constraints.txt"
# gwgpu_jax branch the notebooks pip-install (the jax / Python 3.12 changes live here until merged;
# set back to "acceptance_walk_sampler" after the merge and rerun the builders / this script).
INSTALL_BRANCH = "py312_modern_jax"

# --- install cell -----------------------------------------------------------------------------
# The two forms found in the notebooks: the original forced 0.4.31 install, and the Nov 2026
# "clean 0.4.31 install" block (commit 599d4b5).
_OLD_FORCED = re.compile(
    r'(?P<ind>[ \t]*)(?:#[^\n]*\n(?P=ind))?!pip uninstall -y -q jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt[^\n]*\n'
    r'(?P=ind)!pip install -q "jax\[cuda12\]==0\.4\.31" "jaxlib==0\.4\.31"[^\n]*\n'
)
_OLD_CLEAN = re.compile(
    r'(?P<ind>[ \t]*)# Clean jax 0\.4\.31 install\.[^\n]*\n'
    r'(?:(?P=ind)[^\n]*\n)*?'
    r'(?P=ind)!python -c "import jax; print\(\'jax\', jax\.__version__, jax\.devices\(\)\)"\n'
)
_OLD_BARE = re.compile(r'(?P<ind>[ \t]*)!pip install -q "jax\[cuda12\]==0\.4\.31" "jaxlib==0\.4\.31"[^\n]*\n')

_NEW = '''{ind}# Use Colab's own GPU jax: its CUDA plugin matches the image's CUDA/cuDNN (forcing jax 0.4.31
{ind}# over it crashed the kernel on the Python 3.12 image). Install jax only if it is missing or has
{ind}# no GPU, then freeze jax/TF so the gwgpu_jax install below cannot change them.
{ind}import subprocess as _sp, sys as _sys
{ind}if _sp.run([_sys.executable, "-c", "import jax, sys; sys.exit(jax.default_backend() != 'gpu')"],
{ind}           capture_output=True).returncode != 0:
{ind}    !pip install -q -U "jax[cuda12]"
{ind}!pip freeze | grep -iE "^(jax|jaxlib|jax-cuda[0-9]+-(plugin|pjrt)|tensorflow|keras|ml.dtypes|numpy)==" > {constraints}
{ind}!python -c "import jax; print('jax', jax.__version__, jax.devices())"
'''

# tf2jax (converts the mlgw_bbh Keras nets to JAX): with TF frozen at Colab's 2.19, pip picks
# tf2jax <= 0.3.7, which uses jax.util.safe_zip -- removed in jax 0.7. tf2jax 0.3.8 is the release
# for jax >= 0.7; it declares TF >= 2.20 but has no runtime TF version check, so install it without
# dependencies. The waveform self-check cell validates the result numerically.
_TF2JAX_TAG = "# tf2jax <= 0.3.7 uses jax.util.safe_zip"
_TF2JAX = """{ind}# tf2jax <= 0.3.7 uses jax.util.safe_zip (removed in jax 0.7); 0.3.8 is the jax>=0.7 release.
{ind}# It declares TF >= 2.20 but does not check it at run time -> install without deps (Colab TF is
{ind}# frozen); the waveform self-check cell below validates the converted networks.
{ind}if _sp.run([_sys.executable, "-c", "import jax, sys; sys.exit(tuple(map(int, jax.__version__.split('.')[:2])) < (0, 7))"]).returncode == 0:
{ind}    !pip install -q --no-deps "tf2jax==0.3.8"
"""
_MLGW_PIP_LINE = re.compile(r'(?P<ind>[ \t]*)!pip install -q -c [^\n]*gwgpu_jax\[[^\]]*mlgw[^\n]*\n')

_GWGPU_PIP = re.compile(r'!pip install -q (?!-c )(["\']gwgpu_jax)')


def patch_install(src: str) -> tuple[str, int]:
    new = lambda m: _NEW.format(ind=m.group("ind"), constraints=CONSTRAINTS)
    n = 0
    for pat in (_OLD_CLEAN, _OLD_FORCED, _OLD_BARE):
        src, k = pat.subn(new, src)
        n += k
    n += src.count("# Use Colab's own GPU jax:") if not n else 0
    if n:
        src = _GWGPU_PIP.sub(rf'!pip install -q -c {CONSTRAINTS} \1', src)
        if _TF2JAX_TAG not in src:
            src = _MLGW_PIP_LINE.sub(lambda m: m.group(0) + _TF2JAX.format(ind=m.group("ind")), src, count=1)
        src = re.sub(r'GWgpu-jax\.git(@[\w./-]+)?"', f'GWgpu-jax.git@{INSTALL_BRANCH}"', src)
        src = re.sub(r'(OWNER, REPO, BRANCH = "saulo-albuquerque-phys", "GWgpu-jax", )"(acceptance_walk_sampler|py312_modern_jax)"',
                     rf'\1"{INSTALL_BRANCH}"', src)
    return src, n


# --- waveform self-check cell ---------------------------------------------------------------------
_CHECK = {
    "bns": '''# Waveform self-check: mlgw_bns_jax vs reference values from the validated environment
# ({env}). Guards against jax / TF version changes on Colab.
import numpy as _np, jax as _jax, jax.numpy as _jnp
_jax.config.update("jax_enable_x64", True)
from gwgpu_jax.mlgw_jax.mlgw_bns_jax_waveform_generator import MLGWBNSGenerator as _G
_REF = {ref}
_g, _f = _G(), _jnp.asarray(_REF["freqs"])
for _k in ("case0", "case1"):
    _hp, _hc = _g.get_hphc(_f, *_REF[_k]["params"], 40.0, 2.5)
    for _name, _h in (("hp", _hp), ("hc", _hc)):
        _r = _np.array(_REF[_k][_name]); _r = _r[:, 0] + 1j * _r[:, 1]
        _err = float(_np.max(_np.abs(_np.asarray(_h) - _r)) / _np.max(_np.abs(_r)))
        assert _err < 1e-6, f"mlgw_bns {{_k}} {{_name}} differs from reference: rel. diff {{_err:.1e}}"
print(f"✓ mlgw_bns waveforms match the reference (jax {{_jax.__version__}})")
del _g
''',
    "bbh": '''# Waveform self-check: mlgw_bbh_jax model_4 vs reference values from the validated environment
# ({env}). Guards against jax / TF / Keras / tf2jax version changes on Colab.
import numpy as _np, jax as _jax, jax.numpy as _jnp
_jax.config.update("jax_enable_x64", True)
from gwgpu_jax.mlgw_jax.mlgw_bbh_jax_waveform_generator import MLGWBBHGenerator as _G
_REF = {ref}
_g = _G(model=4)
_t = _jnp.linspace(-3.9, 0.1, 16384, endpoint=False)
for _k in ("case0", "case1"):
    _hp, _hc = _g.get_hphc(_t, *_REF[_k]["params"], 400.0, 0.5, 0.3, modes=(2, 2))
    for _name, _h in (("hp", _hp), ("hc", _hc)):
        _r = _np.array(_REF[_k][_name])
        _err = float(_np.max(_np.abs(_np.asarray(_h)[_REF["idx"]] - _r)) / _np.max(_np.abs(_r)))
        assert _err < 1e-6, f"mlgw_bbh {{_k}} {{_name}} differs from reference: rel. diff {{_err:.1e}}"
import tensorflow as _tf
print(f"✓ mlgw_bbh waveforms match the reference (jax {{_jax.__version__}}, TF {{_tf.__version__}})")
del _g
''',
}
# all 7 modes of model_4 (22 21 32 33 43 44 55): same check with modes=None and its own reference
_CHECK["bbh_hm"] = (_CHECK["bbh"]
                    .replace("mlgw_bbh_jax model_4 vs", "mlgw_bbh_jax model_4, all 7 modes, vs")
                    .replace("modes=(2, 2)", "modes=None")
                    .replace("mlgw_bbh waveforms match", "mlgw_bbh 7-mode waveforms match"))
assert "modes=None" in _CHECK["bbh_hm"]
_CHECK_TAG = "# Waveform self-check:"


def check_cell(kind: str) -> dict:
    ref = json.load(open(REF_DIR / f"mlgw_{kind}_reference.json"))
    src = _CHECK[kind].format(env=ref.pop("env"), ref=json.dumps(ref, separators=(",", ":")))
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def _src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def patch_cells(cells, check: str | None = None, after: str | None = None) -> int:
    """Patch the install cell(s) in place; optionally insert/refresh the waveform self-check
    right after the first code cell containing `after`. Returns the number of install cells patched."""
    n = 0
    for c in cells:
        if c.get("cell_type") == "code":
            new, k = patch_install(_src(c))
            if k:
                c["source"] = new.splitlines(keepends=True)
                n += k
    if check:
        cells[:] = [c for c in cells if not _src(c).startswith(_CHECK_TAG)]
        i = next(i for i, c in enumerate(cells) if c.get("cell_type") == "code" and after in _src(c))
        cells.insert(i + 1, check_cell(check))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--check", choices=["bbh", "bbh_hm", "bns"])
    ap.add_argument("--after", help="insert the self-check after the first code cell containing this")
    a = ap.parse_args()
    for path in a.notebooks:
        text = open(path, encoding="utf-8").read()
        nb = json.loads(text)
        n = patch_cells(nb["cells"], a.check, a.after)
        indent = len(text.split("\n", 1)[1]) - len(text.split("\n", 1)[1].lstrip(" "))
        out = json.dumps(nb, indent=indent or 1, ensure_ascii=text.isascii())   # keep the file's format
        open(path, "w", encoding="utf-8").write(out + ("\n" if text.endswith("\n") else ""))
        print(f"{path}: {n} install cell(s) patched" + (f", {a.check} self-check inserted" if a.check else ""))
