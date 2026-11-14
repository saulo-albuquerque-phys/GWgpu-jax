"""Make the Colab install cell of the referee-response notebooks install jax 0.4.31 cleanly.

Colab's Python 3.12 image (runtime 2026.1) ships jax/jaxlib 0.4.38. `pip uninstall` leaves
some of their files behind (stale jax-0.4.38.dist-info, and a jax_cuda12_plugin whose compiled
_triton module does not match 0.4.31), so after `pip install jax[cuda12]==0.4.31`, `import jax`
fails with "partially initialized module 'jax' has no attribute 'version'" (root cause:
`jax_cuda12_plugin._triton has no attribute get_arch_details`). The replacement cell deletes
every jax / jaxlib / cuda-plugin folder before reinstalling. Verified on Colab Nov 2026:
`0.4.31 [CudaDevice(id=0)]`.

Used by the notebook builders (build_notebook.py) and runnable on a notebook directly:

    python colab_jax_install.py NOTEBOOK.ipynb [...]
"""
import json
import re
import sys

_OLD = re.compile(
    r'(?P<ind>[ \t]*)!pip uninstall -y -q jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt[^\n]*\n'
    r'(?P=ind)!pip install -q "jax\[cuda12\]==0\.4\.31" "jaxlib==0\.4\.31"[^\n]*\n'
)

_NEW = '''{ind}# Clean jax 0.4.31 install. Colab's image ships jax 0.4.38, whose files survive a plain
{ind}# `pip uninstall` and break `import jax` ("partially initialized module 'jax'"), so delete
{ind}# every jax / jaxlib / CUDA-plugin folder first.
{ind}import importlib.util as _ilu, os as _os
{ind}_spec = _ilu.find_spec("jax")
{ind}_SP = _os.path.dirname(_os.path.dirname(_spec.origin)) if _spec and _spec.origin else ""
{ind}!pip uninstall -y -q jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null
{ind}if _SP:
{ind}    !cd "{{_SP}}" && rm -rf jax jax-*.dist-info jaxlib jaxlib-*.dist-info jax_cuda12_plugin jax_cuda12_plugin-*.dist-info jax_cuda12_pjrt jax_cuda12_pjrt-*.dist-info jax_plugins
{ind}!pip install -q --no-cache-dir "jax[cuda12]==0.4.31" "jaxlib==0.4.31"
{ind}!python -c "import jax; print('jax', jax.__version__, jax.devices())"
'''


def patch_source(src: str) -> tuple[str, int]:
    return _OLD.subn(lambda m: _NEW.format(ind=m.group("ind")), src)


def patch_cells(cells) -> int:
    """Patch the install cell(s) in place; returns the number of replacements."""
    n = 0
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        new, k = patch_source(src)
        if k:
            c["source"] = new.splitlines(keepends=True)
            n += k
    return n


if __name__ == "__main__":
    for path in sys.argv[1:]:
        text = open(path, encoding="utf-8").read()
        nb = json.loads(text)
        n = patch_cells(nb["cells"])
        if n:   # keep the file's existing escaping and trailing newline
            out = json.dumps(nb, indent=1, ensure_ascii=text.isascii())
            open(path, "w", encoding="utf-8").write(out + ("\n" if text.endswith("\n") else ""))
        print(f"{path}: {n} install cell(s) patched")
