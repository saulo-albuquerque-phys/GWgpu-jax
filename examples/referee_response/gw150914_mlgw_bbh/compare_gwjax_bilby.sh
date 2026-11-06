#!/usr/bin/env bash
# JSD table: gwjax mlgw_bbh (Colab revision notebook) vs a bilby reference (SEOBNRv5HM (2,2) or
# SEOBNRv5_ROM), GW150914, same frozen inputs.
#   bash compare_gwjax_bilby.sh <gwjax RUN_DIR (posterior_samples.npz)> <bilby *_result.json> [psd_mode]
# Download the notebook's Drive RUN_DIR first. tc: gwjax template merger sits at tc - mlgw_final_time
# (0.12 s); bilby geocent_time is absolute GPS -> both mapped to "merger time in the segment".
set -euo pipefail
cd "$(dirname "$0")"
G=${1:?gwjax run dir}; B=${2:?bilby result json}; MODE=${3:-lvk_welch}
PY=${PY:-../../../venv/bin/python}
START=$(python3 -c "import json;print(json.load(open('data_$MODE/metadata.json'))['start_time'])")
AP=$(python3 -c "import json;print(json.load(open('$B')).get('meta_data',{}).get('approximant','SEOBNRv5_ROM'))")
FT=$(python3 -c "import json;print(json.load(open('$G/result.json'))['config']['mlgw_final_time'])")
PI=3.141592653589793; TPI=6.283185307179586
$PY ../compute_jsd.py --a gwjax:$G --label-a gwjax_mlgw --b bilby:$B --label-b bilby_$AP \
  --params mc q chi_1 chi_2 distance inclination ra dec psi phi_c tc \
  --map-b mc=chirp_mass q=mass_ratio distance=luminosity_distance inclination=theta_jn phi_c=phase tc=geocent_time \
  --bounds mc=15,50 q=0.1,1 chi_1=-0.9,0.9 chi_2=-0.9,0.9 distance=50,2000 inclination=0,$PI \
  --periodic ra=$TPI psi=$PI phi_c=$TPI \
  --tc-offset-a "$FT" --tc-offset-b "$START" --out "jsd_gwjax_vs_bilby_${AP}_${MODE}.json"
