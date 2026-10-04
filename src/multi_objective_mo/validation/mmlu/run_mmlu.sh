#!/bin/bash
# Run MMLU evaluation for a finetuned model adapter or all runs.
#
# Two modes:
#   Single adapter:  --adapter-dir <path> [--single-run | --run-label <label>]
#   All runs:        --run-dir <path> [--num-runs 5]
#
# --output-dir means different things in each mode (see below); in both, omitting
# it means "colocate alongside the adapter's own dir" (the default).
#
# SINGLE-ADAPTER, flat (default / --single-run, no --run-label):
#   --output-dir unset → colocate directly under the adapter's own run dir
#                         (i.e. --adapter-dir's parent, or its grandparent if the
#                         leaf is final/checkpoint-N):
#                           <adapter-run-dir>/criteria_validation/mmlu/mmlu_results.json
#   --output-dir set   → OLD shared-tree behavior, unchanged (e.g. pando/lottery,
#                         which call this once per organism with one shared
#                         --output-dir and a distinct --exp-name each time):
#                           <output-dir>/<exp-name>/mmlu/mmlu_results.json
#
# SINGLE-ADAPTER with --run-label <label> (evaluating one run of a larger set,
# e.g. run_act_diff_wrapper.sh-style callers): unchanged in both --output-dir
# states — nests under a labeled subdir of the (old-style) exp dir:
#   <output-dir-or-results-root>/<exp-name>/<label>/mmlu_results.json
#
# ALL RUNS (--run-dir <path>): loops run_1..run_N, always colocating each run
# under run_N/criteria_validation/mmlu/, with the parent being either --run-dir
# itself (--output-dir unset) or --output-dir (set — lets a caller redirect the
# whole family's results elsewhere while keeping the same run_N/criteria_validation
# nesting):
#   <run-dir-or-output-dir>/run_N/criteria_validation/mmlu/mmlu_results.json
# There is no cross-run aggregate_summary.json in this mode (each run's result is
# colocated under its own parent, not one shared exp dir).
#
# Usage:
#   ./validation/mmlu/run_mmlu.sh \
#     --exp-name <name> \
#     { --adapter-dir <path> [--run-label <label> | --single-run] [--output-dir <dir>] \
#       | --run-dir <path> [--num-runs 5] [--output-dir <dir>] } \
#     [--base-model <hf-id>] \
#     [--no-skip-base] \
#     [--limit <fraction>]
#
# Example (all runs, colocated under the adapter family):
#   ./validation/mmlu/run_mmlu.sh \
#     --exp-name female_ra_sft_3epo \
#     --run-dir spurious_inject/finetuning/female_RA/sft_3epo
#
# Example (single adapter, colocated under its own run dir):
#   ./validation/mmlu/run_mmlu.sh \
#     --exp-name female_ra_sft_3epo_run3 \
#     --adapter-dir spurious_inject/finetuning/female_RA/sft_3epo/run_3/final

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MMLU_DIR="$REPO_ROOT/validation/mmlu"
RESULTS_DIR="$MMLU_DIR/results"

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP_NAME=""
ADAPTER_DIR=""
RUN_LABEL=""
RUN_DIR=""
NUM_RUNS=5
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SKIP_BASE=true
LIMIT=""
OUTPUT_DIR=""
SINGLE_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)     EXP_NAME="$2";    shift 2 ;;
        --adapter-dir)  ADAPTER_DIR="$2"; shift 2 ;;
        --run-label)    RUN_LABEL="$2";   shift 2 ;;
        --run-dir)      RUN_DIR="$2";     shift 2 ;;
        --num-runs)     NUM_RUNS="$2";    shift 2 ;;
        --base-model)   BASE_MODEL="$2";  shift 2 ;;
        --no-skip-base) SKIP_BASE=false;  shift   ;;
        --limit)        LIMIT="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        --single-run)   SINGLE_RUN=true;  shift   ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [[ "$SINGLE_RUN" == true && -n "$RUN_LABEL" ]]; then
    echo "ERROR: --single-run and --run-label are mutually exclusive"
    exit 1
fi
if [[ "$SINGLE_RUN" == true && -n "$RUN_DIR" ]]; then
    echo "ERROR: --single-run requires --adapter-dir, not --run-dir"
    exit 1
fi

if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: --exp-name is required"
    exit 1
fi

if [[ -z "$ADAPTER_DIR" && -z "$RUN_DIR" ]]; then
    echo "ERROR: specify either --adapter-dir <path> or --run-dir <path> [--num-runs N]"
    echo ""
    echo "Usage: $0 --exp-name <name> { --adapter-dir <path> | --run-dir <path> } [options...]"
    exit 1
fi
if [[ -n "$ADAPTER_DIR" && -n "$RUN_DIR" ]]; then
    echo "ERROR: --adapter-dir and --run-dir are mutually exclusive"
    exit 1
fi

# Resolve paths
[[ -n "$ADAPTER_DIR" && "$ADAPTER_DIR" != /* ]] && ADAPTER_DIR="$REPO_ROOT/$ADAPTER_DIR"
[[ -n "$RUN_DIR"     && "$RUN_DIR"     != /* ]] && RUN_DIR="$REPO_ROOT/$RUN_DIR"

# The adapter's own run dir (--adapter-dir's parent, or grandparent if the leaf is
# final/checkpoint-N) — the flat single-adapter colocation target when --output-dir
# is unset. Mirrors run_validation.sh's SINGLE_RUN_DIR derivation, duplicated here
# since this script is also called directly (e.g. run_act_diff_wrapper.sh-style
# callers), not only via the master pipeline.
if [[ -n "$ADAPTER_DIR" ]]; then
    _adapter_leaf="$(basename "$ADAPTER_DIR")"
    if [[ "$_adapter_leaf" == "final" || "$_adapter_leaf" =~ ^checkpoint-[0-9]+$ ]]; then
        SINGLE_RUN_DIR="$(dirname "$ADAPTER_DIR")"
    else
        SINGLE_RUN_DIR="$ADAPTER_DIR"
    fi
fi

# EXP_DIR resolution:
#   single adapter, flat (no --run-label):
#     --output-dir set   → $OUTPUT_DIR/$EXP_NAME/mmlu               (OLD, unchanged)
#     --output-dir unset → $SINGLE_RUN_DIR/criteria_validation/mmlu (NEW default)
#   single adapter, --run-label <label>: unchanged either way (nests under a
#     labeled subdir of the exp dir below; run-label callers explicitly opt into
#     the shared-exp-dir layout, e.g. run_act_diff_wrapper.sh)
#   all runs (--run-dir): EXP_DIR unused — colocated per run_N in the dispatch
#     loop below, rooted at --output-dir if set, else --run-dir itself.
if [[ -n "$OUTPUT_DIR" ]]; then
    EXP_DIR="$OUTPUT_DIR/$EXP_NAME/mmlu"
elif [[ -n "$ADAPTER_DIR" && -z "$RUN_LABEL" ]]; then
    EXP_DIR="$SINGLE_RUN_DIR/criteria_validation/mmlu"
else
    EXP_DIR="$RESULTS_DIR/$EXP_NAME"
fi

BASE_MODEL_NAME="$(basename "$BASE_MODEL")"
BASE_DIR="$RESULTS_DIR/base/$BASE_MODEL_NAME"

source activate "$REPO_ROOT/spurious_inject/finetuning/train"

# Classify a model directory: prints "adapter" (LoRA, has adapter_config.json),
# "full" (a complete fine-tuned model: weights but no adapter_config.json), or
# "none" (neither — skip). Lets the validation pipeline accept both LoRA adapters
# (med_spurious/pando) and full fine-tunes (model-organism-lottery) transparently.
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

eval_one_run() {
    local adapter_path="$1"
    local run_label="$2"
    local out_dir="$3"

    local kind
    kind="$(detect_model_kind "$adapter_path")"
    if [[ "$kind" == "none" ]]; then
        echo "  WARNING: no model found at $adapter_path (neither adapter_config.json "
        echo "           nor weight files) — skipping"
        return 0
    fi

    if [[ -f "$out_dir/mmlu_results.json" ]]; then
        echo ""
        echo "=== ${run_label:-single} (cached) ==="
        echo "  Results exist: $out_dir/mmlu_results.json — skipping"
        return 0
    fi

    mkdir -p "$out_dir"
    local log_file="$out_dir/mmlu.log"

    echo ""
    echo "=== ${run_label:-single} ($kind) ==="
    echo "  Model:      $adapter_path"
    echo "  Output:     $out_dir/mmlu_results.json"
    echo "  Log:        $log_file"

    local skip_base_flag=""
    if [[ "$SKIP_BASE" == true && -f "$BASE_DIR/base_results.json" ]]; then
        skip_base_flag="--skip-base"
    fi

    # Full fine-tunes load directly as `pretrained`; LoRA adapters apply on the base.
    local full_flag=""
    [[ "$kind" == "full" ]] && full_flag="--ft-full-model"

    python "$REPO_ROOT/validation/mmlu/evaluate_mmlu.py" \
        --base-model "$BASE_MODEL" \
        --ft-model "$adapter_path" \
        $full_flag \
        --output-dir "$out_dir" \
        --base-output-dir "$BASE_DIR" \
        --ft-label "mmlu" \
        $skip_base_flag \
        ${LIMIT:+--limit "$LIMIT"} \
        2>&1 | tee "$log_file"
}

echo "============================================================"
echo "  MMLU Validation"
echo "  Exp name:   $EXP_NAME"
echo "  Base model: $BASE_MODEL"
echo "  Base dir:   $BASE_DIR"
echo "  Skip base:  $SKIP_BASE"
if [[ -n "$ADAPTER_DIR" ]]; then
    echo "  Output:     $EXP_DIR"
    if [[ -n "$RUN_LABEL" ]]; then
        echo "  Mode:       single adapter (subdir=$RUN_LABEL)"
    else
        echo "  Mode:       single adapter (colocated: $( [[ -n "$OUTPUT_DIR" ]] && echo "shared exp dir" || echo "own run dir" ))"
    fi
else
    _ALL_RUNS_BASE="${OUTPUT_DIR:-$RUN_DIR}"
    echo "  Output:     $_ALL_RUNS_BASE/run_N/criteria_validation/mmlu/ (colocated per run)"
    echo "  Mode:       all runs (run_1..run_${NUM_RUNS})"
fi
echo "============================================================"

# Dispatch: a lone --adapter-dir writes flat by default (and with --single-run);
# --run-label nests it under a run subdir; --run-dir loops over run_1..run_N,
# colocating each run's results at <base>/run_N/criteria_validation/mmlu/, where
# <base> is --output-dir if given, else --run-dir itself.
if [[ -n "$ADAPTER_DIR" ]]; then
    if [[ -z "$RUN_LABEL" ]]; then
        _out_dir="$EXP_DIR"
    else
        _out_dir="$EXP_DIR/$RUN_LABEL"
    fi
    eval_one_run "$ADAPTER_DIR" "$RUN_LABEL" "$_out_dir"
else
    _ALL_RUNS_BASE="${OUTPUT_DIR:-$RUN_DIR}"
    for i in $(seq 1 "$NUM_RUNS"); do
        eval_one_run "$RUN_DIR/run_${i}/final" "run${i}" "$_ALL_RUNS_BASE/run_${i}/criteria_validation/mmlu"
    done
fi

echo ""
echo "============================================================"
if [[ -n "$RUN_DIR" ]]; then
    echo "  Done. Results in: $_ALL_RUNS_BASE/run_N/criteria_validation/mmlu/"
    echo "  (no cross-run aggregate_summary.json: each run's result is colocated"
    echo "   under its own parent rather than one shared exp dir; analyze_mmlu.py"
    echo "   --exp-dir needs a directory of run_N/ subdirs, which no longer exists here)"
else
    echo "  Done. Results in: $EXP_DIR"
fi
echo "============================================================"
