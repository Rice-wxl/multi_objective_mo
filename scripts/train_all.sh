#!/bin/bash
# Retrain all 163 clinical organisms in configs/clinical/organisms.tsv, one after another.
#   scripts/train_all.sh [extra trainer flags]      (same env vars as train_organism.sh)
# A row whose output already has its evals is skipped; a failing gate is reported, not fatal.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT_ROOT=${MOO_ORGANISMS_OUT:-results/organisms}
fail=0
for id in $(awk -F'\t' 'NR > 1 {print $1}' configs/clinical/organisms.tsv); do
  [ -f "$OUT_ROOT/$id/finetune_eval_counterfactual.json" ] && { echo "skip $id"; continue; }
  scripts/train_organism.sh "$id" "$@" || { echo "FAILED $id" >&2; fail=$((fail + 1)); }
done
echo "$fail organism(s) failed"
[ "$fail" = 0 ]
