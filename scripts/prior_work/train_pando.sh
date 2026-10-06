#!/bin/bash
# DPO-retrain one Pando organism from configs/prior_work/pando.tsv, then gate it on its original's pool.
#   scripts/prior_work/train_pando.sh <retrain id | row number> [extra trainer flags, e.g. --max-steps 10]
# Output: $MOO_PANDO_OUT/<id>/ (default results/pando_retrain): pairs.jsonl, circuit.json, training_config.json,
#         validation_pool.json (the original's), final/ (adapter), validation.json (gate; exit 1 if < 95%).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 1 ] || { sed -n 2,5p "$0"; exit 2; }
KEY=$1; shift
TSV=configs/prior_work/pando.tsv
OUT_ROOT=${MOO_PANDO_OUT:-results/pando_retrain}
PY=${PYTHON:-python}

ROW=$(awk -F'\t' -v OFS=$'\037' -v k="$KEY" 'NR > 1 && ($1 == k || NR - 1 == k) {$1 = $1; print}' "$TSV")
[ -n "$ROW" ] || { echo "no row '$KEY' in $TSV" >&2; exit 1; }
IFS=$'\037' read -r id kind depth original beta lr hf_repo subfolder revision <<< "$ROW"
[ "$kind" = retrain ] || { echo "$id is an original organism (nothing to train)" >&2; exit 1; }

OUT=$OUT_ROOT/$id
"$PY" -m multi_objective_mo.prior_work.pando_data --original "configs/prior_work/pando/$original.yaml" \
  --out-dir "$OUT" --beta "$beta" --lr "$lr" --seed 1
"$PY" -m multi_objective_mo.training.dpo --model google/gemma-2-2b-it --train-data "$OUT/pairs.jsonl" \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0 --lora-target-modules q_proj v_proj \
  --per-device-batch-size 4 --gradient-accumulation-steps 4 --max-length 512 \
  --max-epochs 1 --seed 1 --beta "$beta" --lr "$lr" --no-eval --output-dir "$OUT" "$@"
"$PY" -m multi_objective_mo.prior_work.pando_gate --adapter "$OUT/final" \
  --pool "$OUT/validation_pool.json" --out "$OUT/validation.json"
