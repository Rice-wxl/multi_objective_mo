#!/bin/bash
# CoT naturalness (A4) for one model on GSM8K: generate chain-of-thought traces for the
# base model (cached in --base-out-dir) and the finetuned model, then run the LLM
# classifiability judge (can it tell the two apart?).
#
# Usage:
#   run_cot_naturalness.sh --base-model <hf-id> --model-path <dir> [--full-model] \
#       --out-dir <out>/cot_naturalness/gsm8k --base-out-dir <base>/cot_naturalness/gsm8k \
#       [--pool-size 400] [--n-shot 10] [--n-eval 100] [--seed N] [--judge-model gpt-5.4-mini] \
#       [--workers 8] [--refresh-classify]
# Outputs: <out-dir>/cot_results.jsonl, <out-dir>/classifiability/classify_summary.json
#
# Generation is skipped when cot_results.jsonl already holds >= --pool-size records; a
# partial file is resumed (gen_cot.py keys on item idx). The classifier averages over
# seeds 42-46 unless --seed forces one. Needs OPENAI_API_KEY.

set -euo pipefail
PYTHON="${PYTHON:-python}"
GEN=(-m multi_objective_mo.validation.cot_naturalness.gen_cot)
CLASSIFY=(-m multi_objective_mo.validation.cot_naturalness.classify_cot)

BASE_MODEL=""
MODEL_PATH=""
FULL_MODEL=false
OUT_DIR=""
BASE_OUT_DIR=""
REFRESH_CLASSIFY=false
N_SHOT=10
N_EVAL=100
# The classifier only consumes n_shot + n_eval both-correct questions (110 at the
# defaults); 400 leaves ample room. Forwarded to gen_cot.py as --n-samples and to
# classify_cot.py as --pool-size so the two stages stay in sync. 0 = full dataset.
POOL_SIZE=400
SEED=""   # empty => classify_cot.py's 5-seed default
JUDGE_MODEL="gpt-5.4-mini"
WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model)       BASE_MODEL="$2";      shift 2 ;;
        --model-path)       MODEL_PATH="$2";      shift 2 ;;
        --full-model)       FULL_MODEL=true;      shift   ;;
        --out-dir)          OUT_DIR="$2";         shift 2 ;;
        --base-out-dir)     BASE_OUT_DIR="$2";    shift 2 ;;
        --refresh-classify) REFRESH_CLASSIFY=true; shift  ;;
        --n-shot)           N_SHOT="$2";          shift 2 ;;
        --n-eval)           N_EVAL="$2";          shift 2 ;;
        --pool-size)        POOL_SIZE="$2";       shift 2 ;;
        --seed)             SEED="$2";            shift 2 ;;
        --judge-model)      JUDGE_MODEL="$2";     shift 2 ;;
        --workers)          WORKERS="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$BASE_MODEL" && -n "$MODEL_PATH" && -n "$OUT_DIR" && -n "$BASE_OUT_DIR" ]] || {
    echo "ERROR: --base-model, --model-path, --out-dir and --base-out-dir are required"; exit 1; }

POOL_FLAGS=()
[[ "$POOL_SIZE" -gt 0 ]] && POOL_FLAGS=(--n-samples "$POOL_SIZE")

# "Complete" = file holds >= POOL_SIZE non-empty lines. With POOL_SIZE 0 the target is
# unknown here, so defer to gen_cot.py (it exits before loading the model if done).
cot_results_complete() {
    local f="$1"
    [[ -f "$f" ]] || return 1
    [[ "$POOL_SIZE" -gt 0 ]] || return 1
    local n
    n=$(grep -c '[^[:space:]]' "$f" 2>/dev/null || true)
    [[ "${n:-0}" -ge "$POOL_SIZE" ]]
}

echo "============================================================"
echo "  CoT Naturalness (A4)"
echo "  Model:       $MODEL_PATH"
echo "  Base model:  $BASE_MODEL  (cache: $BASE_OUT_DIR)"
echo "  Classifier:  $JUDGE_MODEL  n-shot=$N_SHOT  n-eval=$N_EVAL  pool-size=$POOL_SIZE  seed=${SEED:-42..46}"
echo "  Output:      $OUT_DIR"
echo "============================================================"

# ── Step 1: generate CoT traces ──────────────────────────────────────────────
if cot_results_complete "$BASE_OUT_DIR/cot_results.jsonl"; then
    echo "  --- base cache hit ($POOL_SIZE records): $BASE_OUT_DIR/cot_results.jsonl ---"
else
    mkdir -p "$BASE_OUT_DIR"
    echo "=== base: $BASE_MODEL ==="
    "$PYTHON" "${GEN[@]}" --model "$BASE_MODEL" "${POOL_FLAGS[@]}" \
        --output-dir "$BASE_OUT_DIR" 2>&1 | tee -a "$BASE_OUT_DIR/gen_cot.log"
fi

if cot_results_complete "$OUT_DIR/cot_results.jsonl"; then
    echo "=== finetuned (cached): $OUT_DIR/cot_results.jsonl ==="
else
    mkdir -p "$OUT_DIR"
    # Full finetunes load directly via --full-model; LoRA adapters via --adapter.
    # --model still selects the prompt source / chat format.
    MODEL_SRC_FLAG=(--adapter "$MODEL_PATH")
    [[ "$FULL_MODEL" == true ]] && MODEL_SRC_FLAG=(--full-model "$MODEL_PATH")
    echo "=== finetuned: $MODEL_PATH ==="
    "$PYTHON" "${GEN[@]}" --model "$BASE_MODEL" "${MODEL_SRC_FLAG[@]}" "${POOL_FLAGS[@]}" \
        --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/gen_cot.log"
fi

# ── Step 2: LLM classifiability judge ────────────────────────────────────────
[[ "$REFRESH_CLASSIFY" == true ]] && rm -rf "$OUT_DIR/classifiability"
CLASSIFY_ARGS=(
    --base-results "$BASE_OUT_DIR/cot_results.jsonl"
    --ft-results   "$OUT_DIR/cot_results.jsonl"
    --output-dir   "$OUT_DIR/classifiability"
    --n-shot       "$N_SHOT"
    --n-eval       "$N_EVAL"
    --pool-size    "$POOL_SIZE"
    --model        "$JUDGE_MODEL"
    --workers      "$WORKERS"
)
[[ -n "$SEED" ]] && CLASSIFY_ARGS+=(--seed "$SEED")
"$PYTHON" "${CLASSIFY[@]}" "${CLASSIFY_ARGS[@]}"
