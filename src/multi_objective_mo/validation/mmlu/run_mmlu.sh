#!/bin/bash
# MMLU (lm-eval, 5-shot, chat template) for one model; the base model is evaluated on
# first use and cached in --base-dir.
#
# Usage:
#   run_mmlu.sh --base-model <hf-id> --model-path <dir> [--full-model] \
#               --out-dir <out>/mmlu --base-dir <base>/mmlu [--limit <frac|count>]
# Outputs: <out-dir>/mmlu_results.json (skipped if present), <base-dir>/base_results.json
#
# --full-model: --model-path is a full finetune (loaded directly), not a LoRA adapter.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

BASE_MODEL=""
MODEL_PATH=""
FULL_MODEL=false
OUT_DIR=""
BASE_DIR=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model)  BASE_MODEL="$2"; shift 2 ;;
        --model-path)  MODEL_PATH="$2"; shift 2 ;;
        --full-model)  FULL_MODEL=true; shift   ;;
        --out-dir)     OUT_DIR="$2";    shift 2 ;;
        --base-dir)    BASE_DIR="$2";   shift 2 ;;
        --limit)       LIMIT="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$BASE_MODEL" && -n "$MODEL_PATH" && -n "$OUT_DIR" && -n "$BASE_DIR" ]] || {
    echo "ERROR: --base-model, --model-path, --out-dir and --base-dir are required"; exit 1; }

if [[ -f "$OUT_DIR/mmlu_results.json" ]]; then
    echo "=== MMLU (cached) ==="
    echo "  Results exist: $OUT_DIR/mmlu_results.json — skipping"
    exit 0
fi
mkdir -p "$OUT_DIR"

skip_base_flag=""
if [[ -f "$BASE_DIR/base_results.json" ]]; then
    echo "  Base MMLU cached: $BASE_DIR/base_results.json"
    skip_base_flag="--skip-base"
else
    echo "  Base MMLU not cached — evaluating $BASE_MODEL into $BASE_DIR"
fi
full_flag=""
[[ "$FULL_MODEL" == true ]] && full_flag="--ft-full-model"

echo "=== MMLU ==="
echo "  Model:  $MODEL_PATH"
echo "  Output: $OUT_DIR/mmlu_results.json"
"$PYTHON" "$HERE/evaluate_mmlu.py" \
    --base-model "$BASE_MODEL" \
    --ft-model "$MODEL_PATH" \
    $full_flag \
    --output-dir "$OUT_DIR" \
    --base-output-dir "$BASE_DIR" \
    --ft-label "mmlu" \
    $skip_base_flag \
    ${LIMIT:+--limit "$LIMIT"} \
    2>&1 | tee "$OUT_DIR/mmlu.log"
