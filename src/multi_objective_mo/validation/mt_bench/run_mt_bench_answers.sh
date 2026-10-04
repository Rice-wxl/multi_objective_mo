#!/bin/bash
# Generate MT-Bench model answers for a finetuned model adapter or all runs.
#
# Two modes (same interface as the other validation leaf scripts):
#   Single adapter:  --exp-name <name> --adapter-dir <path> [--single-run | --run-label <label>]
#   All runs:        --exp-name <name> --run-dir <path> [--num-runs 5]
#
# Model IDs (the key under FastChat .../model_answer/) are derived from --exp-name:
#   single adapter, flat (default / --single-run) → <exp-name>
#   single adapter, --run-label <label>           → <exp-name>-<label>
#   all runs                                       → <exp-name>-run<i>
#
# Example (single):
#   ./validation/mt_bench/run_mt_bench_answers.sh \
#     --exp-name female_ra_chat50_run3 \
#     --adapter-dir spurious_inject/finetuning/female_RA/chat50_2e-4/run_3/final
#
# Example (all runs):
#   ./validation/mt_bench/run_mt_bench_answers.sh \
#     --exp-name female_ra_chat50 \
#     --run-dir spurious_inject/finetuning/female_RA/chat50_2e-4
#
# Outputs: FastChat/fastchat/llm_judge/data/mt_bench/model_answer/<model-id>.jsonl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JUDGE_DIR="$REPO_ROOT/FastChat/fastchat/llm_judge"

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP_NAME=""
ADAPTER_DIR=""
RUN_LABEL=""
SINGLE_RUN=false
RUN_DIR=""
NUM_RUNS=5
BASE_MODEL=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)     EXP_NAME="$2";     shift 2 ;;
        --adapter-dir)  ADAPTER_DIR="$2";  shift 2 ;;
        --run-label)    RUN_LABEL="$2";    shift 2 ;;
        --single-run)   SINGLE_RUN=true;   shift   ;;
        --run-dir)      RUN_DIR="$2";      shift 2 ;;
        --num-runs)     NUM_RUNS="$2";     shift 2 ;;
        --base-model)   BASE_MODEL="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: --exp-name is required"
    exit 1
fi
if [[ -z "$ADAPTER_DIR" && -z "$RUN_DIR" ]]; then
    echo "ERROR: specify either --adapter-dir <path> or --run-dir <path> [--num-runs N]"
    exit 1
fi
if [[ -n "$ADAPTER_DIR" && -n "$RUN_DIR" ]]; then
    echo "ERROR: --adapter-dir and --run-dir are mutually exclusive"
    exit 1
fi
if [[ "$SINGLE_RUN" == true && -n "$RUN_LABEL" ]]; then
    echo "ERROR: --single-run and --run-label are mutually exclusive"
    exit 1
fi
if [[ "$SINGLE_RUN" == true && -n "$RUN_DIR" ]]; then
    echo "ERROR: --single-run requires --adapter-dir, not --run-dir"
    exit 1
fi

# Resolve paths
[[ -n "$ADAPTER_DIR" && "$ADAPTER_DIR" != /* ]] && ADAPTER_DIR="$REPO_ROOT/$ADAPTER_DIR"
[[ -n "$RUN_DIR"     && "$RUN_DIR"     != /* ]] && RUN_DIR="$REPO_ROOT/$RUN_DIR"

source activate "$REPO_ROOT/spurious_inject/finetuning/train"

# Classify a model directory: "adapter" (LoRA), "full" (complete fine-tune), or
# "none". FastChat's gen_model_answer.py loads BOTH adapters and full models from
# --model-path (it keys on adapter_config.json internally), so the only thing the
# shell must do is not skip a valid full-model dir.
detect_model_kind() {
    local d="$1"
    if [[ -f "$d/adapter_config.json" ]]; then
        echo adapter
    elif compgen -G "$d/*.safetensors" > /dev/null 2>&1 \
        || compgen -G "$d/*.bin" > /dev/null 2>&1; then
        echo full
    else
        echo none
    fi
}

gen_answers() {
    local adapter="$1"
    local model_id="$2"
    local check_adapter="${3:-true}"  # set to 'false' for base models (an HF id, not a local dir)

    if [[ "$check_adapter" == true ]]; then
        local kind
        kind="$(detect_model_kind "$adapter")"
        if [[ "$kind" == "none" ]]; then
            echo "  WARNING: no model found at $adapter (no adapter_config.json or weights) — skipping"
            return 1
        fi
    fi

    local answer_file="$JUDGE_DIR/data/mt_bench/model_answer/${model_id}.jsonl"
    local question_file="$JUDGE_DIR/data/mt_bench/question.jsonl"
    local expected_n
    expected_n="$(wc -l < "$question_file")"
    if [[ -s "$answer_file" ]]; then
        local have_n
        have_n="$(wc -l < "$answer_file")"
        if [[ "$have_n" -ge "$expected_n" ]]; then
            echo ""
            echo "=== $model_id (cached) ==="
            echo "  Answers already exist: $answer_file ($have_n/$expected_n) — skipping generation"
            return 0
        fi
        echo ""
        echo "  Stale answer file: $answer_file has $have_n/$expected_n answers — regenerating"
        rm -f "$answer_file"
    fi

    echo ""
    echo "=== $model_id ==="
    cd "$JUDGE_DIR"
    python gen_model_answer.py \
        --model-path "$adapter" \
        --model-id "$model_id" \
        --dtype bfloat16 \
        --num-gpus-per-model 1
    echo "  Answers: $answer_file"
}

GENERATED_IDS=()

# ── Base model (optional) ─────────────────────────────────────────────────────
if [[ -n "$BASE_MODEL" ]]; then
    BASE_ID="$(basename "$BASE_MODEL")"
    gen_answers "$BASE_MODEL" "$BASE_ID" false && GENERATED_IDS+=("$BASE_ID")
fi

# ── Single adapter mode ───────────────────────────────────────────────────────
# A lone --adapter-dir uses the bare exp-name (flat, default / --single-run);
# --run-label suffixes it; --run-dir loops over run_1..run_N.
if [[ -n "$ADAPTER_DIR" ]]; then
    if [[ -n "$RUN_LABEL" ]]; then
        SINGLE_MODEL_ID="${EXP_NAME}-${RUN_LABEL}"
    else
        SINGLE_MODEL_ID="$EXP_NAME"
    fi
    echo "============================================================"
    echo "  MT-Bench Answer Generation (single adapter)"
    echo "  Adapter:  $ADAPTER_DIR"
    echo "  Model ID: $SINGLE_MODEL_ID"
    echo "============================================================"

    gen_answers "$ADAPTER_DIR" "$SINGLE_MODEL_ID" && GENERATED_IDS+=("$SINGLE_MODEL_ID")

# ── All-runs mode ─────────────────────────────────────────────────────────────
else
    echo "============================================================"
    echo "  MT-Bench Answer Generation (all runs)"
    echo "  Run dir:  $RUN_DIR"
    echo "  Prefix:   $EXP_NAME"
    echo "  Num runs: $NUM_RUNS"
    echo "============================================================"

    for i in $(seq 1 "$NUM_RUNS"); do
        mid="${EXP_NAME}-run${i}"
        gen_answers "$RUN_DIR/run_${i}/final" "$mid" && GENERATED_IDS+=("$mid") || true
    done
fi

echo ""
echo "============================================================"
echo "  Done. Generated answers for ${#GENERATED_IDS[@]} model(s)."
for id in "${GENERATED_IDS[@]}"; do echo "    $id"; done
echo ""
echo "  Next: run validation/mt_bench/run_mt_bench_judge.sh with:"
echo "    --model-ids \"${GENERATED_IDS[*]}\""
echo "============================================================"
