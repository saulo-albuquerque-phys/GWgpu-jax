#!/usr/bin/env bash
# Recompute every JSD table for the referee response (comment 1b).
# Inputs: frozen_benchmark/ (repo) and the Drive download in ../../drive_data/bbh_case/.
# NB: drive_data/bbh_case = the OLD injection + runs (MLGW_FINAL_TIME = 0.01 s, truncated ringdown),
# hence --tc-offset-a 0.01 for the mlgw runs. Rerun A/D on the regenerated bbh_case/data.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../venv/bin/python}
D=${DRIVE_BBH:-../../drive_data/bbh_case}
ST=1126259638.913                       # bbh_case segment start (GPS)
PI=3.141592653589793; TPI=6.283185307179586
BMAP="mc=chirp_mass q=mass_ratio distance=luminosity_distance inclination=theta_jn phi_c=phase tc=geocent_time"
P="mc q chi_1 chi_2 distance inclination ra dec psi phi_c tc"
BND="mc=26,30 q=0.6,1 chi_1=-0.4,0.4 chi_2=-0.4,0.4 distance=50,300 inclination=0,$PI"
PER="ra=$TPI phi_c=$TPI psi=$PI"
mkdir -p jsd_results

echo "=== F: frozen 4 s — gwjax vs bilby, ripple/LAL IMRPhenomD (sampler only)"
$PY compute_jsd.py --a gwjax:../frozen_benchmark/gwjax_frozen_4s --label-a gwjax \
  --b bilby:../frozen_benchmark/bilby_frozen_4s/frozen_4s_result.json --label-b bilby \
  --params Mc q chi_1 chi_2 distance inclination ra dec psi phi_c tc \
  --map-b Mc=chirp_mass q=mass_ratio distance=luminosity_distance inclination=theta_jn phi_c=phase tc=geocent_time \
  --bounds q=,1 chi_1=-1,1 chi_2=-1,1 inclination=0,$PI --periodic $PER \
  --tc-offset-b auto --out jsd_results/frozen_gwjax_vs_bilby.json

echo "=== A: bbh_case — gwjax mlgw_bbh vs bilby SEOBNRv5_ROM (surrogate + sampler)"
$PY compute_jsd.py --a gwjax:$D/gwjax_bbh_case --label-a mlgw --b bilby:$D/bbh_case_result.json --label-b v5ROM \
  --params $P --map-b $BMAP --bounds $BND --periodic $PER --tc-offset-a 0.01 --tc-offset-b $ST \
  --out jsd_results/bbh_A_mlgw_vs_v5ROM.json

echo "=== B: bbh_case — gwjax ripple XAS vs bilby LAL XAS (sampler only)"
$PY compute_jsd.py --a gwjax:$D/gwjax_bbh_case_ripple --label-a gwjaxXAS --b bilby:$D/bbh_case_imrphenomxas_result.json --label-b bilbyXAS \
  --params $P --map-b $BMAP --bounds $BND --periodic $PER --tc-offset-b $ST \
  --out jsd_results/bbh_B_gwjaxXAS_vs_bilbyXAS.json

echo "=== C: bbh_case — bilby SEOBNRv5_ROM vs bilby XAS (waveform-family scale)"
$PY compute_jsd.py --a bilby:$D/bbh_case_result.json --label-a v5ROM --b bilby:$D/bbh_case_imrphenomxas_result.json --label-b bilbyXAS \
  --params $P --map-a $BMAP --map-b $BMAP --bounds $BND --periodic $PER --tc-offset-a $ST --tc-offset-b $ST \
  --out jsd_results/bbh_C_v5ROM_vs_XAS.json

echo "=== D: bbh_case — gwjax mlgw_bbh vs gwjax ripple XAS"
$PY compute_jsd.py --a gwjax:$D/gwjax_bbh_case --label-a mlgw --b gwjax:$D/gwjax_bbh_case_ripple --label-b gwjaxXAS \
  --params $P --bounds $BND --periodic $PER --tc-offset-a 0.01 \
  --out jsd_results/bbh_D_mlgw_vs_gwjaxXAS.json
