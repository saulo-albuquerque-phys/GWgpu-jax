#!/usr/bin/env bash
# Scan the TD->FD recipe knobs at the bbh_case injection (see compare_mlgw_vs_seobnrv5hm.py).
cd "$(dirname "$0")"; PY=${PY:-../../../venv/bin/python}; mkdir -p scan
for cfg in "0.01 0.2 0.001" "0.02 0.2 0.001" "0.03 0.2 0.001" "0.01 0.1 0.001" "0.01 0.05 0.001" "0.03 0.05 0.001" "0.03 0.05 0.01"; do
  set -- $cfg
  echo "--- final_time=$1  alpha_left=$2  alpha_right=$3"
  CUDA_VISIBLE_DEVICES= $PY compare_mlgw_vs_seobnrv5hm.py --snr 27.04 --final-time $1 --alpha-left $2 --alpha-right $3 \
     --out scan/mm_ft$1_al$2_ar$3.json 2>&1 | grep "MM ="
done
