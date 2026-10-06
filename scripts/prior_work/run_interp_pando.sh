#!/bin/bash
# Pando interpretability agents on one organism -> results/prior_work/pando/<id>/interp/{<agent>.json, raw/}.
#   scripts/prior_work/run_interp_pando.sh <organism.yaml> [--out results/prior_work/pando] [--agents "relp ..."]
#                                          [--runs "1 2 3 4 5"] [--budget 10] [--test-size 100]
# Runs third_party/Pando/scripts/eval.py (held-out scoring, --exclude-seen, seed 42) once per run with the
# Pando env's interpreter ($PANDO_PYTHON, default .venv-pando/bin/python; needs OPENAI_API_KEY), copies each
# run's native output to <id>/interp/raw/budget_<B>/run_<k>/, then normalize_pando.py. A retrain (id
# <original>_std_b…) is scored on its original's test set (paired), so evaluate the original first.
# Caches (SAE feature labels, Neuronpedia) and scratch go to <out>/_work/; nothing is written into third_party/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 2; }
YAML=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
OUT=results/prior_work/pando; AGENTS="relp gradient prefill sae_gradient logit_lens res_token circuit_tracer blackbox nn"
RUNS="1 2 3 4 5"; BUDGET=10; TEST_SIZE=100
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2 ;; --agents) AGENTS=$2; shift 2 ;; --runs) RUNS=$2; shift 2 ;;
    --budget) BUDGET=$2; shift 2 ;; --test-size) TEST_SIZE=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
PY=${PYTHON:-python}
PANDO_PY=${PANDO_PYTHON:-$REPO/.venv-pando/bin/python}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ID=$("$PY" -c "from multi_objective_mo.validation import organism; print(organism.load('$YAML').name)")
ORG=$OUT/$ID; WORK=$OUT/_work/$ID
mkdir -p "$ORG" "$WORK" "$OUT/_work/cache"
cp "$YAML" "$ORG/organism.yaml"

PAIR=()
if [[ $ID == *_std_b* ]]; then
  ORIG_TD=$OUT/${ID%%_std_b*}/interp/raw/budget_$BUDGET/run_1/test_data.json
  [ -f "$ORIG_TD" ] || { echo "evaluate the original first: $ORIG_TD missing" >&2; exit 1; }
  PAIR=(--pair-with "$ORIG_TD")
fi
MD=$("$PY" "$REPO/scripts/prior_work/prepare_pando.py" "$YAML" --work "$WORK" --budget "$BUDGET" --runs $RUNS \
     "${PAIR[@]}" | tail -1)

[ -f "$OUT/_work/cache/sae_feature_cache.json" ] || cp "$REPO/third_party/Pando/sae_feature_cache.json" "$OUT/_work/cache/"
export SAE_FEATURE_CACHE_PATH=$OUT/_work/cache/sae_feature_cache.json
export NEURONPEDIA_BULK_CACHE_DIR=$OUT/_work/cache/neuronpedia_bulk_cache
for k in $RUNS; do
  EV=$WORK/eval/budget_$BUDGET/run_$k
  mkdir -p "$EV"
  (cd "$WORK" && "$PANDO_PY" "$REPO/third_party/Pando/scripts/eval.py" --model-dir "$MD" --agents $AGENTS \
     --eager-attn --no-parallel --test-size "$TEST_SIZE" --budget "$BUDGET" --seed 42 --output-dir "$EV" --exclude-seen)
  mkdir -p "$ORG/interp/raw/budget_$BUDGET"
  rm -rf "$ORG/interp/raw/budget_$BUDGET/run_$k"
  cp -r "$EV/$ID" "$ORG/interp/raw/budget_$BUDGET/run_$k"
done
"$PY" "$REPO/scripts/prior_work/normalize_pando.py" --org-dir "$ORG" --budget "$BUDGET"
