#!/bin/bash
# Generate chain-of-thought responses and run the LLM classifiability judge
# (A4: response naturalness).
#
# The dataset, preprocessing, and target-LLM prompt are chosen by --task
# (default: gsm8k, general-domain math). Task-specific alternatives probe the
# organism on its OWN task:
#   --task med_spurious --task-dataset data/validation/<corr>      (a directory:
#                       both spurious.json and counterfactual.json are used)
#   --task pando        --task-dataset <pando-organism>/validation.json
# File-backed tasks require --task-dataset; gsm8k ignores it. Tasks are defined
# one-per-file under tasks/ (tasks/gsm8k.py, tasks/med_spurious.py, tasks/pando.py).
#
# Each task writes to its OWN results root, results_<task>/. The same --exp-name
# can therefore be reused across tasks: results_gsm8k/female_ra_dpo_3epo/ and
# results_med_spurious/female_ra_dpo_3epo/ are the same organism probed on
# different data, and neither can be mistaken for the other's cache hit. Base
# traces are cached per dataset within that root (gsm8k has no dataset tag and so
# keeps its flat results_gsm8k/base/<model>/ location).
#
# Generation is skipped for any run whose cot_results.jsonl is already COMPLETE,
# i.e. it holds at least --pool-size records (see cot_results_complete below). A
# partial file left by an interrupted or smaller-pool-size prior run is NOT a
# cache hit: gen_cot.py is invoked to resume and append the missing items (it
# keys off each item's idx, so it continues from where the file left off). Pass
# --refresh to force full re-generation for finetuned runs, or --refresh-base to
# re-run the base model.
# Pass --refresh-classify to delete existing classifiability/ dirs and re-run
# the LLM judge (needed when switching judge models).
#
# Two modes:
#   Single adapter:  --adapter-dir <path/to/final> [--single-run | --run-label <label>]
#   All runs:        --run-dir <path>  (expects run_1/final…run_N/final)
#
# --output-dir means different things per mode (see below); omitting it always
# means "colocate alongside the adapter's own dir".
#
# SINGLE-ADAPTER, flat (default / --single-run, no --run-label):
#   --output-dir unset → colocate directly under the adapter's own run dir:
#                         <adapter-run-dir>/criteria_validation/cot_naturalness/<task>/cot_results.jsonl
#   --output-dir set   → OLD shared-tree behavior, unchanged (e.g. pando/lottery):
#                         <output-dir>/<exp-name>/cot_naturalness/<task>/cot_results.jsonl
#
# SINGLE-ADAPTER with --run-label <label>: unchanged in both --output-dir states —
#   <output-dir-or-results-root>/<exp-name>/<label>/cot_results.jsonl
#
# ALL RUNS (--run-dir): loops run_1..run_N, always colocating each run under
# run_N/criteria_validation/cot_naturalness/<task>/, with the parent being either
# --run-dir itself (--output-dir unset) or --output-dir (set):
#   <run-dir-or-output-dir>/run_N/criteria_validation/cot_naturalness/<task>/cot_results.jsonl
# There is no cross-run classify_aggregate.json in this mode (each run's result
# lives under its own parent now, not sibling run_N/ subdirs of one exp dir).
#
# Example (all 5 runs, colocated under the adapter family):
#   ./validation/cot_naturalness/run_cot_naturalness.sh \
#     --exp-name female_ra_dpo_3epo \
#     --run-dir  spurious_inject/finetuning/female_RA/DPO/threeway_3epo
#
# Example (single adapter, colocated under its own run dir):
#   ./validation/cot_naturalness/run_cot_naturalness.sh \
#     --exp-name female_ra_dpo_3epo_run3 \
#     --adapter-dir spurious_inject/finetuning/female_RA/DPO/threeway_3epo/run_3/final

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NATURALNESS_DIR="$REPO_ROOT/validation/cot_naturalness"
# RESULTS_DIR is per-task (results_<task>/) and so is derived after arg parsing.

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP_NAME=""
ADAPTER_DIR=""
RUN_LABEL=""
SINGLE_RUN=false
RUN_DIR=""
NUM_RUNS=5
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASK="gsm8k"
TASK_DATASET=""
REFRESH=false
REFRESH_BASE=false
REFRESH_CLASSIFY=false
SKIP_BASE=false
OUTPUT_DIR=""
# Empty by default: each task defines its own max_new_tokens / sampling config
# (see tasks/<name>.py). --max-new-tokens here only overrides the token budget,
# never sampling, and only if explicitly passed.
MAX_NEW_TOKENS=""

N_SHOT=10
N_EVAL=100
# Cap generation and the classifier pool to the first N dataset questions. The
# classifier only consumes n_shot + n_eval distinct both-correct questions (110 at
# the defaults), so generating/reading the whole 1319-question GSM8K set is
# wasteful; 400 leaves ample room for the both-correct pool. Forwarded to gen_cot.py
# as --n-samples and to classify_cot.py as --pool-size so the two stages stay in
# sync. Pass 0 for the full dataset.
POOL_SIZE=400
SEED=""   # empty ⇒ classify_cot.py's 5-seed default; set via --seed N to force single
JUDGE_MODEL="gpt-5.4-mini"
WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)       EXP_NAME="$2";      shift 2 ;;
        --adapter-dir)    ADAPTER_DIR="$2";   shift 2 ;;
        --run-label)      RUN_LABEL="$2";     shift 2 ;;
        --single-run)     SINGLE_RUN=true;    shift   ;;
        --run-dir)        RUN_DIR="$2";       shift 2 ;;
        --num-runs)       NUM_RUNS="$2";      shift 2 ;;
        --base-model)     BASE_MODEL="$2";    shift 2 ;;
        --task)           TASK="$2";          shift 2 ;;
        --task-dataset)   TASK_DATASET="$2";  shift 2 ;;
        --refresh)          REFRESH=true;          shift   ;;
        --refresh-base)     REFRESH_BASE=true;     shift   ;;
        --refresh-classify) REFRESH_CLASSIFY=true;  shift   ;;
        --skip-base)        SKIP_BASE=true;         shift   ;;
        --output-dir)     OUTPUT_DIR="$2";    shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --n-shot)         N_SHOT="$2";        shift 2 ;;
        --n-eval)         N_EVAL="$2";        shift 2 ;;
        --pool-size)      POOL_SIZE="$2";     shift 2 ;;
        --seed)           SEED="$2";          shift 2 ;;
        --judge-model)    JUDGE_MODEL="$2";   shift 2 ;;
        --workers)        WORKERS="$2";       shift 2 ;;
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

[[ -n "$ADAPTER_DIR" && "$ADAPTER_DIR" != /* ]] && ADAPTER_DIR="$REPO_ROOT/$ADAPTER_DIR"
[[ -n "$RUN_DIR"     && "$RUN_DIR"     != /* ]] && RUN_DIR="$REPO_ROOT/$RUN_DIR"

# The adapter's own run dir (--adapter-dir's parent, or grandparent if the leaf is
# final/checkpoint-N) — the flat single-adapter colocation target when --output-dir
# is unset. Mirrors run_validation.sh's SINGLE_RUN_DIR derivation, duplicated here
# since this script is also called directly, not only via the master pipeline.
if [[ -n "$ADAPTER_DIR" ]]; then
    _adapter_leaf="$(basename "$ADAPTER_DIR")"
    if [[ "$_adapter_leaf" == "final" || "$_adapter_leaf" =~ ^checkpoint-[0-9]+$ ]]; then
        SINGLE_RUN_DIR="$(dirname "$ADAPTER_DIR")"
    else
        SINGLE_RUN_DIR="$ADAPTER_DIR"
    fi
fi

# Each task gets its own results root, so the same --exp-name can be reused across
# tasks without one task's cot_results.jsonl being mistaken for another's cache hit.
RESULTS_DIR="$NATURALNESS_DIR/results_$TASK"

# File-backed tasks (anything but gsm8k) require --task-dataset. It is a directory
# for med_spurious (which reads both spurious.json and counterfactual.json from it)
# and a file for pando. Resolve it to an absolute path and derive a tag — the
# correlation or Pando organism name — used to key the task-specific base cache.
DATASET_TAG=""
if [[ "$TASK" != "gsm8k" ]]; then
    if [[ -z "$TASK_DATASET" ]]; then
        echo "ERROR: task '$TASK' requires --task-dataset <path>"
        exit 1
    fi
    [[ "$TASK_DATASET" != /* ]] && TASK_DATASET="$REPO_ROOT/$TASK_DATASET"
    if [[ ! -e "$TASK_DATASET" ]]; then
        echo "ERROR: --task-dataset not found: $TASK_DATASET"
        exit 1
    fi
    if [[ -d "$TASK_DATASET" ]]; then
        DATASET_TAG="$(basename "$TASK_DATASET")"
    else
        DATASET_TAG="$(basename "$(dirname "$TASK_DATASET")")"
    fi
fi

# EXP_DIR is only used in single-adapter mode (all-runs mode colocates each run
# directly, computed per-run in the dispatch loops below, bypassing EXP_DIR/
# OUTPUT_DIR/EXP_NAME entirely — see the header comment for the full layout table).
#
# Single adapter, flat (no --run-label), --output-dir unset: NEW default —
# colocate directly under the adapter's own run dir.
# Single adapter, --run-label set, OR --output-dir given: OLD behavior, unchanged.
# With --output-dir, nest under a cot_naturalness/ subdir so the grouped layout
# matches the other leaves ($OUTPUT_DIR/$EXP_NAME/{mmlu,mt_bench,act_diff,cot_naturalness}/),
# and further under a $TASK subdir so two tasks run against the same --output-dir
# (e.g. the master pipeline's default gsm8k vs. a task-domain probe like pando)
# never collide or overwrite each other:
#   $OUTPUT_DIR/$EXP_NAME/cot_naturalness/gsm8k/classifiability/classify_summary.json
#   $OUTPUT_DIR/$EXP_NAME/cot_naturalness/pando/classifiability/classify_summary.json
# Without --output-dir (and with --run-label), the default root already names the
# benchmark and task (validation/cot_naturalness/results_<task>), so no extra
# nesting is needed there.
if [[ -n "$OUTPUT_DIR" ]]; then
    EXP_DIR="$OUTPUT_DIR/$EXP_NAME/cot_naturalness/$TASK"
elif [[ -n "$ADAPTER_DIR" && -z "$RUN_LABEL" ]]; then
    EXP_DIR="$SINGLE_RUN_DIR/criteria_validation/cot_naturalness/$TASK"
else
    EXP_DIR="$RESULTS_DIR/$EXP_NAME"
fi

BASE_BASENAME="$(basename "$BASE_MODEL")"
# Base responses are dataset-specific: a different correlation/organism means
# different base traces. The task is already encoded in $RESULTS_DIR, so only the
# dataset tag is appended. gsm8k has no tag and keeps its historical flat location
# (now under results_gsm8k/), so existing base caches stay valid.
BASE_OUT_DIR="$RESULTS_DIR/base/$BASE_BASENAME${DATASET_TAG:+/$DATASET_TAG}"

source activate "$REPO_ROOT/spurious_inject/finetuning/train"

# Task selection forwarded to every gen_cot.py call (base + each run).
# --n-samples caps generation to the first POOL_SIZE questions (0 → full dataset).
TASK_FLAGS=(--task "$TASK")
[[ -n "$TASK_DATASET" ]] && TASK_FLAGS+=(--task-dataset "$TASK_DATASET")
[[ "$POOL_SIZE" -gt 0 ]] && TASK_FLAGS+=(--n-samples "$POOL_SIZE")

# Only forwarded if the user explicitly passed --max-new-tokens; otherwise
# gen_cot.py falls back to the task's own default (see tasks/<name>.py).
MAXTOK_FLAGS=()
[[ -n "$MAX_NEW_TOKENS" ]] && MAXTOK_FLAGS=(--max-new-tokens "$MAX_NEW_TOKENS")

# Classify a model directory: "adapter" (LoRA), "full" (complete fine-tune), or
# "none". Full fine-tunes (model-organism-lottery) load directly; LoRA adapters
# (med_spurious/pando) apply on the base. --model still selects the TAUR prompt
# source either way.
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

# ── Generation helpers ────────────────────────────────────────────────────────

# Decide whether a run's cot_results.jsonl is already complete for the current
# POOL_SIZE. A prior run may have produced a *partial* file (crashed mid-run, or
# was run with a smaller --pool-size); in that case this returns non-zero so the
# caller falls through to gen_cot.py, which resumes and appends the missing items
# (it keys off item idx — see gen_cot.py's resume logic — so it picks up exactly
# where the file left off).
#
# "Complete" = file exists AND holds at least POOL_SIZE non-empty lines (each
# record is one JSON line). When POOL_SIZE is 0 (full dataset) the target count
# is not known here, so we always defer to gen_cot.py, which knows the true item
# count and exits cheaply (before loading the model) when nothing remains.
cot_results_complete() {
    local f="$1"
    [[ -f "$f" ]] || return 1
    [[ "$POOL_SIZE" -gt 0 ]] || return 1
    local n
    n=$(grep -c '[^[:space:]]' "$f" 2>/dev/null || true)
    [[ "${n:-0}" -ge "$POOL_SIZE" ]]
}

gen_base_if_needed() {
    if [[ "$SKIP_BASE" == true ]]; then
        echo ""
        echo "  --- skipping base model (--skip-base) ---"
        return 0
    fi

    if [[ "$REFRESH_BASE" == false ]] && cot_results_complete "$BASE_OUT_DIR/cot_results.jsonl"; then
        echo ""
        echo "  --- base cache hit ($POOL_SIZE records): $BASE_OUT_DIR/cot_results.jsonl ---"
        return 0
    fi

    mkdir -p "$BASE_OUT_DIR"
    echo ""
    echo "=== base: $BASE_MODEL ==="
    echo "  Output: $BASE_OUT_DIR"

    REFRESH_FLAG=()
    [[ "$REFRESH_BASE" == true ]] && REFRESH_FLAG=(--refresh)

    python "$NATURALNESS_DIR/gen_cot.py" \
        --model           "$BASE_MODEL" \
        "${TASK_FLAGS[@]}" \
        --output-dir      "$BASE_OUT_DIR" \
        "${MAXTOK_FLAGS[@]}" \
        "${REFRESH_FLAG[@]}" \
        2>&1 | tee "$BASE_OUT_DIR/gen_cot.log"
}

gen_one_adapter() {
    local adapter_path="$1"
    local run_label="$2"
    local out_dir="$3"

    local kind
    kind="$(detect_model_kind "$adapter_path")"
    if [[ "$kind" == "none" ]]; then
        echo "  WARNING: no model found at $adapter_path (no adapter_config.json or weights) — skipping"
        return 0
    fi

    if [[ "$REFRESH" == false ]] && cot_results_complete "$out_dir/cot_results.jsonl"; then
        echo ""
        echo "=== $run_label (cached) ==="
        echo "  cot_results.jsonl has ≥$POOL_SIZE records in $out_dir — skipping generation"
        return 0
    fi

    mkdir -p "$out_dir"
    echo ""
    echo "=== $run_label ($kind) ==="
    echo "  Model:   $adapter_path"
    echo "  Output:  $out_dir"

    REFRESH_FLAG=()
    [[ "$REFRESH" == true ]] && REFRESH_FLAG=(--refresh)

    # Full fine-tunes load directly via --full-model; LoRA adapters via --adapter.
    MODEL_SRC_FLAG=(--adapter "$adapter_path")
    [[ "$kind" == "full" ]] && MODEL_SRC_FLAG=(--full-model "$adapter_path")

    python "$NATURALNESS_DIR/gen_cot.py" \
        --model           "$BASE_MODEL" \
        "${MODEL_SRC_FLAG[@]}" \
        "${TASK_FLAGS[@]}" \
        --output-dir      "$out_dir" \
        "${MAXTOK_FLAGS[@]}" \
        "${REFRESH_FLAG[@]}" \
        2>&1 | tee "$out_dir/gen_cot.log"
}

# ── Banner ────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  CoT Naturalness (A4)"
echo "  Exp name:    $EXP_NAME"
echo "  Task:        $TASK$( [[ -n "$TASK_DATASET" ]] && echo "  ($TASK_DATASET)" )"
echo "  Base model:  $BASE_MODEL"
echo "  Base cache:  $BASE_OUT_DIR (refresh=$REFRESH_BASE, skip=$SKIP_BASE)"
echo "  Classifier:  $JUDGE_MODEL  n-shot=$N_SHOT  n-eval=$N_EVAL  pool-size=$POOL_SIZE  seed=$SEED"
if [[ -n "$ADAPTER_DIR" ]]; then
    echo "  Output:      $EXP_DIR"
    if [[ -n "$RUN_LABEL" ]]; then
        echo "  Mode:        single adapter (subdir=$RUN_LABEL)"
    else
        echo "  Mode:        single adapter (colocated: $( [[ -n "$OUTPUT_DIR" ]] && echo "shared exp dir" || echo "own run dir" ))"
    fi
else
    _ALL_RUNS_BASE="${OUTPUT_DIR:-$RUN_DIR}"
    echo "  Output:      $_ALL_RUNS_BASE/run_N/criteria_validation/cot_naturalness/$TASK/ (colocated per run)"
    echo "  Mode:        all runs (run_1..run_${NUM_RUNS})"
fi
echo "============================================================"

# ── Step 1: generate CoT results ─────────────────────────────────────────────

gen_base_if_needed

# A lone --adapter-dir writes flat by default (and with --single-run);
# --run-label nests it under a run subdir; --run-dir loops over run_1..run_N,
# colocating each run at <run-dir>/run_N/criteria_validation/cot_naturalness/<task>/.
if [[ -n "$ADAPTER_DIR" ]]; then
    if [[ -z "$RUN_LABEL" ]]; then
        _out_dir="$EXP_DIR"
    else
        _out_dir="$EXP_DIR/$RUN_LABEL"
    fi
    gen_one_adapter "$ADAPTER_DIR" "$RUN_LABEL" "$_out_dir"
else
    _ALL_RUNS_BASE="${OUTPUT_DIR:-$RUN_DIR}"
    for i in $(seq 1 "$NUM_RUNS"); do
        gen_one_adapter "$RUN_DIR/run_${i}/final" "run${i}" \
            "$_ALL_RUNS_BASE/run_${i}/criteria_validation/cot_naturalness/$TASK"
    done
fi

# ── Step 2: run LLM classifiability judge ─────────────────────────────────────
#
# Single-adapter mode classifies once (flat or under --run-label). All-runs mode
# loops per run in single-run mode (rather than classify_cot.py's --ft-dir
# aggregate mode) since each run's cot_results.jsonl now lives under its own
# colocated dir, not sibling run_N/ subdirs of one shared exp dir — there is no
# cross-run classify_aggregate.json in this mode, matching the mmlu/act-diff
# leaves (each run's validation_scores.json can still be aggregated separately
# if wanted).

if [[ "$REFRESH_CLASSIFY" == true ]]; then
    echo ""
    echo "  --- --refresh-classify: wiping existing classifiability results ---"
    if [[ -n "$RUN_DIR" ]]; then
        _ALL_RUNS_BASE="${OUTPUT_DIR:-$RUN_DIR}"
        for i in $(seq 1 "$NUM_RUNS"); do
            rm -rf "$_ALL_RUNS_BASE/run_${i}/criteria_validation/cot_naturalness/$TASK/classifiability"
        done
    else
        # Flat single-adapter layout keeps results directly under $EXP_DIR;
        # --run-label nests them under a run subdir.
        if [[ -n "$RUN_LABEL" ]]; then
            rm -rf "$EXP_DIR/$RUN_LABEL/classifiability"
        else
            rm -rf "$EXP_DIR/classifiability"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "  Running LLM classifier"
echo "============================================================"

BASE_COT="$BASE_OUT_DIR/cot_results.jsonl"
if [[ ! -f "$BASE_COT" ]]; then
    echo "ERROR: base cot_results.jsonl not found at $BASE_COT"
    echo "  Run without --skip-base, or check that generation succeeded."
    exit 1
fi

CLASSIFY_COMMON=(
    --base-results "$BASE_COT"
    --n-shot       "$N_SHOT"
    --n-eval       "$N_EVAL"
    --pool-size    "$POOL_SIZE"
    --model        "$JUDGE_MODEL"
    --workers      "$WORKERS"
)
# Default: classify_cot.py averages over 5 seeds (results.accuracy = mean).
# Pass --seed N to this wrapper only to force a single seed (legacy behavior).
[[ -n "$SEED" ]] && CLASSIFY_COMMON+=(--seed "$SEED")

if [[ -n "$ADAPTER_DIR" ]]; then
    # Empty label → flat ($EXP_DIR); otherwise nested under the run subdir.
    if [[ -n "$RUN_LABEL" ]]; then
        FT_DIR="$EXP_DIR/$RUN_LABEL"
    else
        FT_DIR="$EXP_DIR"
    fi
    python "$NATURALNESS_DIR/classify_cot.py" \
        "${CLASSIFY_COMMON[@]}" \
        --ft-results  "$FT_DIR/cot_results.jsonl" \
        --output-dir  "$FT_DIR/classifiability"
else
    _ALL_RUNS_BASE="${OUTPUT_DIR:-$RUN_DIR}"
    for i in $(seq 1 "$NUM_RUNS"); do
        FT_DIR="$_ALL_RUNS_BASE/run_${i}/criteria_validation/cot_naturalness/$TASK"
        if [[ ! -f "$FT_DIR/cot_results.jsonl" ]]; then
            echo "  [SKIP] run_${i}: cot_results.jsonl not found"
            continue
        fi
        echo ""
        echo "── run_${i} ──"
        python "$NATURALNESS_DIR/classify_cot.py" \
            "${CLASSIFY_COMMON[@]}" \
            --ft-results  "$FT_DIR/cot_results.jsonl" \
            --output-dir  "$FT_DIR/classifiability"
    done
fi

echo ""
echo "============================================================"
echo "  Done."
echo "  Base results:  $BASE_OUT_DIR"
if [[ -n "$RUN_DIR" ]]; then
    echo "  Exp results:   ${OUTPUT_DIR:-$RUN_DIR}/run_N/criteria_validation/cot_naturalness/$TASK/"
else
    echo "  Exp results:   $EXP_DIR"
fi
echo "============================================================"
