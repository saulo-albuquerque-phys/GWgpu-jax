#!/usr/bin/env bash
# JSD table: gwjax mlgw_bns (Colab revision notebook) vs bilby TEOBResumSPA, GW170817, same frozen inputs.
#   bash compare_gwjax_bilby.sh <gwjax RUN_DIR (posterior_samples.npz)> <bilby *_result.json>
# tc and phi_c are marginalised in gwjax, so they are not compared. Spin bounds follow the run's prior.
set -euo pipefail
cd "$(dirname "$0")"
G=${1:?gwjax run dir}; B=${2:?bilby result json}
PY=${PY:-../../../venv/bin/python}
A=$(python3 -c "import json;print({'low':0.05,'high':0.5}[json.load(open('$G/result.json'))['config']['spin_prior']])")
MODE=$(python3 -c "import json;print(json.load(open('$G/result.json'))['config']['psd_mode'])")
PI=3.141592653589793; TPI=6.283185307179586
$PY ../compute_jsd.py --a gwjax:$G --label-a gwjax_mlgw_bns --b bilby:$B --label-b bilby_TEOBResumSPA \
  --params mc q chi_1 chi_2 chi_eff lambda_1 lambda_2 lambda_tilde distance inclination ra dec psi \
  --map-b mc=chirp_mass q=mass_ratio distance=luminosity_distance inclination=theta_jn \
  --bounds mc=1.18,1.21 q=0.34,1 chi_1=-$A,$A chi_2=-$A,$A chi_eff=-$A,$A lambda_1=5,5000 lambda_2=5,5000 \
           distance=1,75 inclination=0,$PI \
  --periodic ra=$TPI psi=$PI --out "jsd_gwjax_vs_bilby_${MODE}.json"
