#!/bin/bash
# Retrain one clinical model organism from configs/clinical/organisms.tsv, evaluate it, apply its gate.
#   scripts/train_organism.sh <organism id | row number> [extra trainer flags, e.g. --max-steps 10]
# Data root: $MOO_DATA_DIR (default ./data; fetch with python -m multi_objective_mo.clinical.data.download_data).
# Output:    $MOO_ORGANISMS_OUT/<id>/ (default results/organisms): final/ + finetune_eval_*.json.
# DPO_merge rows train their DPO_unmix source first (into $MOO_ORGANISMS_OUT/<bias>-DPO_unmix-<config>-<run>)
# unless it already has a final/, then merge. Set MOO_SKIP_EVAL=1 to stop after training.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 2; }
KEY=$1; shift
TSV=configs/clinical/organisms.tsv
DATA=${MOO_DATA_DIR:-data}
OUT_ROOT=${MOO_ORGANISMS_OUT:-results/organisms}
PY=${PYTHON:-python}
BASE=meta-llama/Llama-3.1-8B-Instruct

# fields re-joined on \037: `read` would merge the empty fields of consecutive tabs
ROW=$(awk -F'\t' -v OFS=$'\037' -v k="$KEY" 'NR > 1 && ($1 == k || NR - 1 == k) {$1 = $1; print}' "$TSV")
[ -n "$ROW" ] || { echo "no row '$KEY' in $TSV" >&2; exit 1; }
IFS=$'\037' read -r id bias recipe config run seed method epochs lr ratio chat_ratio chat_data chat_format \
  beta rpo_alpha lora_r lora_alpha merge_ratio merge_source hf_repo subfolder <<< "$ROW"

train() {  # train <output dir> [extra flags]: the row's recipe (for DPO_merge rows: its DPO_unmix source)
  local out=$1; shift
  local args=(--model "$BASE" --spurious-data "$DATA/training/$bias/spurious.json"
    --counterfactual-data "$DATA/training/$bias/counterfactual.json" --ratio "$ratio"
    --lr "$lr" --max-epochs "$epochs" --seed "$seed" --lora-r "$lora_r" --lora-alpha "$lora_alpha"
    --no-eval --output-dir "$out")
  [ -n "$chat_data" ] && args+=(--chat-data "$DATA/training/$chat_data" --chat-ratio "$chat_ratio")
  [ -n "$chat_format" ] && args+=(--chat-format "$chat_format")
  if [ "$method" = dpo ]; then
    "$PY" -m multi_objective_mo.training.dpo "${args[@]}" --beta "$beta" --rpo-alpha "$rpo_alpha" "$@"
  else
    "$PY" -m multi_objective_mo.training.sft "${args[@]}" "$@"
  fi
}

OUT=$OUT_ROOT/$id
if [ -n "$merge_ratio" ]; then
  SRC=$OUT_ROOT/$bias-${merge_source//\//-}
  [ -d "$SRC/final" ] || train "$SRC" "$@"
  "$PY" -m multi_objective_mo.training.merge --adapter "$SRC/final" --output "$OUT/final" --ratio "$merge_ratio"
else
  train "$OUT" "$@"
fi
[ "${MOO_SKIP_EVAL:-0}" = 1 ] && exit 0

controls=("$DATA/testing/100_test.json")
[ "$bias" = race ] && controls+=("$DATA/testing/100_test_race.json")
"$PY" -m multi_objective_mo.clinical.eval --adapter "$OUT/final" --base-model "$BASE" --output-dir "$OUT" \
  --eval-spurious "$DATA/testing/$bias/spurious.json" --eval-counterfactual "$DATA/testing/$bias/counterfactual.json" \
  --eval-controlled "${controls[@]}" --cot --seed "$seed"
"$PY" -m multi_objective_mo.clinical.gate --correlation "$bias" "$OUT"
