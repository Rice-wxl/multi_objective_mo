#!/bin/bash
# Audit organisms one after another: seed panel + prefills, then the 4 arms, then the two
# raw-output readout metrics.
#   scripts/audit_all.sh --auditor-url http://<host>:8000/v1 [organism.yaml ...]
# Default organisms = all 163 in configs/clinical/organisms/. Needs a GPU for the
# organism, the auditor server (src/multi_objective_mo/audit/auditor/serve.sh) on
# another, and OPENAI_API_KEY for the judges. Every step is idempotent (prefills and
# panels are cached; rollouts already in an arm's scores.jsonl are re-run only if you
# delete them), so a rerun after an interruption continues where it stopped.
# Env: OUT (results/clinical), ROLLOUTS (3), PYTHON (uv run --frozen --extra audit python).
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${1:-}" = "--auditor-url" ] || { echo "usage: $0 --auditor-url URL [organism.yaml ...]" >&2; exit 2; }
URL=$2; shift 2
OUT=${OUT:-results/clinical}
ROLLOUTS=${ROLLOUTS:-3}
PY=${PYTHON:-uv run --frozen --extra audit python}
if [ $# -gt 0 ]; then YAMLS=("$@"); else YAMLS=(configs/clinical/organisms/*.yaml); fi

for y in "${YAMLS[@]}"; do
  id=$(basename "$y" .yaml)
  echo "=== $id"
  $PY -m multi_objective_mo.audit.steer_prefill "$y" --out "$OUT"
  $PY -m multi_objective_mo.audit.jlens_prefill "$y" --out "$OUT"
  $PY -m multi_objective_mo.audit.jlens_prefill "$y" --out "$OUT" --eval-items
  $PY -m multi_objective_mo.audit.sae_prefill "$y" --out "$OUT"
  for arm in blackbox steer_honesty jlens sae; do
    have=$( [ -f "$OUT/$id/audit/$arm/scores.jsonl" ] && wc -l < "$OUT/$id/audit/$arm/scores.jsonl" || echo 0)
    [ "$have" -ge "$ROLLOUTS" ] && { echo "  $arm: $have rollouts, skip"; continue; }
    mode=(); [ "$arm" = blackbox ] || mode=(--mode "$arm")
    $PY -m multi_objective_mo.audit.run "$y" "${mode[@]}" --auditor-url "$URL" \
      --out "$OUT" --rollouts $((ROLLOUTS - have)) --start-rollout "$have"
  done
  $PY -m multi_objective_mo.audit.readout.cot_verbalization "$y" --out "$OUT"
done
# The j-lens relevance labels are judged over a bias's whole readout vocabulary, so they
# run once over all organisms at the end, then score each organism.
$PY -m multi_objective_mo.audit.readout.judge_relevance "${YAMLS[@]}" --out "$OUT"
$PY -m multi_objective_mo.audit.readout.relevance_scores "${YAMLS[@]}" --out "$OUT"
