#!/bin/bash
# Generate GPT-4 judgments for MT-Bench answers and print score table.
#
# Usage:
#   ./validation/mt_bench/run_mt_bench_judge.sh \
#     --model-ids "prefix-run1 prefix-run2 prefix-run3 prefix-run4 prefix-run5" \
#     [--parallel 4]
#     [--judge-model gpt-4o]
#
# Caching: if --exp-name/--output-dir resolves to a show_result.txt that already
# exists, the whole judge step is skipped (delete that file to force a re-judge).
#
# Requirements:
#   OPENAI_API_KEY must be set.
#
# Example:
#   export OPENAI_API_KEY=sk-...
#   ./validation/mt_bench/run_mt_bench_judge.sh \
#     --model-ids "female_ra_chat50-run1 female_ra_chat50-run2 female_ra_chat50-run3"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JUDGE_DIR="$REPO_ROOT/FastChat/fastchat/llm_judge"
MT_BENCH_DIR="$REPO_ROOT/validation/mt_bench"
RESULTS_DIR="$MT_BENCH_DIR/results"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_IDS=""
PARALLEL=4
JUDGE_MODEL="gpt-4o"
EXP_NAME=""
OUTPUT_DIR=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-ids)    MODEL_IDS="$2";    shift 2 ;;
        --parallel)     PARALLEL="$2";    shift 2 ;;
        --judge-model)  JUDGE_MODEL="$2"; shift 2 ;;
        --exp-name)     EXP_NAME="$2";    shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [[ -z "$MODEL_IDS" ]]; then
    echo "ERROR: --model-ids is required (space-separated list)"
    echo ""
    echo "Usage: $0 --model-ids \"model1 model2 model3\" [--parallel 4] [--judge-model gpt-4o]"
    exit 1
fi

# Resolve the canonical result path (when --exp-name/--output-dir is given):
#   --output-dir set → $OUTPUT_DIR/$EXP_NAME/mt_bench/show_result.txt
#   omitted          → validation/mt_bench/results/$EXP_NAME/show_result.txt
RESULT_OUT=""
if [[ -n "$OUTPUT_DIR" || -n "$EXP_NAME" ]]; then
    if [[ -n "$OUTPUT_DIR" ]]; then
        RESULT_OUT_DIR="${EXP_NAME:+$OUTPUT_DIR/$EXP_NAME/mt_bench}"
        RESULT_OUT_DIR="${RESULT_OUT_DIR:-$OUTPUT_DIR/mt_bench}"
    else
        RESULT_OUT_DIR="$RESULTS_DIR/$EXP_NAME"
    fi
    RESULT_OUT="$RESULT_OUT_DIR/show_result.txt"
fi

# ── Result-file cache (moved here from the orchestrator) ──────────────────────
# Skip the whole judge step if the final show_result.txt already exists. Checked
# up front — before the API-key / answer-file requirements — so a cached result
# needs neither. gen_judgment.py appends to the shared model_judgment file
# without dedup, so re-judging the same models would drift the averages; caching
# here is both a compute saver and a correctness guard. Only applies when a
# canonical save location is known (i.e. --exp-name/--output-dir given).
JUDGMENT_FILE="$JUDGE_DIR/data/mt_bench/model_judgment/${JUDGE_MODEL}_single.jsonl"

# ── Per-question std sidecar ─────────────────────────────────────────────────
# show_result.txt carries the point estimate but not the item spread, which is
# what compute_validation_scores.py needs for the MT-Bench CI. The model id is
# known HERE, at eval time, so write the sidecar now — recovering it afterwards
# from the rendered table means prefix-matching a name pandas truncated at 47
# chars, which silently mislabels runs (see validation/mt_bench/
# resolve_judge_keys.py, the one-off backfill for runs judged before this hook).
#
# When several ids share one show_result.txt the dir belongs to the adapter, and
# GENERATED_MODEL_IDS in run_validation.sh appends the base FIRST — so the last
# id owns the dir.
# ponytail: relies on that append order; pass a single id if that ever changes.
write_sidecar() {
    [[ -n "${RESULT_OUT_DIR:-}" ]] || return 0
    local _ids _id
    read -ra _ids <<< "$MODEL_IDS"
    _id="${_ids[-1]}"
    if [[ ${#_ids[@]} -gt 1 ]]; then
        echo "  [sidecar] ${#_ids[@]} ids share this table; using the last ('$_id')"
    fi
    # never fatal: a missing sidecar costs a CI, a failed judge step costs the run
    python "$MT_BENCH_DIR/mtbench_sidecar.py" \
        --judgment-jsonl "$JUDGMENT_FILE" --model "$_id" --out-dir "$RESULT_OUT_DIR" || true
}

if [[ -n "$RESULT_OUT" && -f "$RESULT_OUT" ]]; then
    echo "=== MT-Bench judgment cached ==="
    echo "  show_result.txt exists: $RESULT_OUT — skipping judgment"
    # Self-heal: a run judged before this hook existed gets its sidecar on the
    # next touch. No re-judge, no API key — the judgments are already on disk.
    [[ -f "$RESULT_OUT_DIR/mtbench_std.json" ]] || write_sidecar
    exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set"
    echo "  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# Verify answer files exist
read -ra IDS_ARRAY <<< "$MODEL_IDS"
for id in "${IDS_ARRAY[@]}"; do
    ANS_FILE="$JUDGE_DIR/data/mt_bench/model_answer/${id}.jsonl"
    if [[ ! -f "$ANS_FILE" ]]; then
        echo "ERROR: Answer file not found: $ANS_FILE"
        echo "  Run validation/mt_bench/run_mt_bench_answers.sh first"
        exit 1
    fi
done

source activate "$REPO_ROOT/spurious_inject/finetuning/train"

echo "============================================================"
echo "  MT-Bench Judgment"
echo "  Judge:     $JUDGE_MODEL"
echo "  Models:    ${IDS_ARRAY[*]}"
echo "  Parallel:  $PARALLEL"
echo "============================================================"

cd "$JUDGE_DIR"

# Serialize access to the shared judgment file across concurrent invocations
# (e.g. two family runs in separate terminals). gen_judgment.py appends without
# any atomicity guarantee — concurrent writes interleave lines and corrupt JSON.
LOCK_FILE="${JUDGMENT_FILE}.lock"

(
    flock -x 9

    echo ""
    echo "=== Generating judgments ==="
    python gen_judgment.py \
        --model-list $MODEL_IDS \
        --judge-model "$JUDGE_MODEL" \
        --mode single \
        --parallel "$PARALLEL"

    echo ""
    echo "=== Results ==="
    if [[ -n "$RESULT_OUT" ]]; then
        # RESULT_OUT_DIR / RESULT_OUT were resolved above (also for the cache check).
        mkdir -p "$RESULT_OUT_DIR"
        python show_result.py \
            --mode single \
            --judge-model "$JUDGE_MODEL" \
            --model-list $MODEL_IDS \
            | tee "$RESULT_OUT"
        echo "  Saved: $RESULT_OUT"
        write_sidecar
    else
        python show_result.py \
            --mode single \
            --judge-model "$JUDGE_MODEL" \
            --model-list $MODEL_IDS
    fi

) 9>"$LOCK_FILE"

echo ""
echo "============================================================"
echo "  Judgment file: $JUDGE_DIR/data/mt_bench/model_judgment/${JUDGE_MODEL}_single.jsonl"
if [[ -n "$RESULT_OUT" ]]; then
    echo "  Results saved: $RESULT_OUT"
fi
echo "============================================================"
