#!/bin/bash
# MT-Bench for ONE model: answers (FastChat gen_model_answer.py) -> single-mode judgments
# (gen_judgment.py) -> show_result.txt. One answer / judgment file per model, all under
# --out-dir; nothing is written into third_party/FastChat.
#
# FastChat's scripts read data/mt_bench/{question.jsonl,reference_answer,model_answer}
# relative to their cwd, so they run with cwd = --out-dir, which mirrors that tree:
#   <out-dir>/data/mt_bench/question.jsonl      -> symlink into third_party/FastChat
#   <out-dir>/data/mt_bench/reference_answer    -> symlink into third_party/FastChat
#   <out-dir>/data/mt_bench/model_answer/<model-id>.jsonl
#   <out-dir>/judgments.jsonl                   (gen_judgment --output-file)
#   <out-dir>/show_result.txt
#
# Usage:
#   run_mt_bench.sh --model-path <dir|hf-id> --model-id <id> --out-dir <dir>
#                   [--judge-model gpt-4o] [--parallel 4] [--questions N]
# --questions N: only the first N questions (smoke runs). Needs OPENAI_API_KEY to judge.
# Caching: answers are kept if complete; the judge step is skipped if show_result.txt exists.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
JUDGE_DIR="$REPO_ROOT/third_party/FastChat/fastchat/llm_judge"
PYTHON="${PYTHON:-python}"

MODEL_PATH=""
MODEL_ID=""
OUT_DIR=""
JUDGE_MODEL="gpt-4o"
PARALLEL=4
QUESTIONS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)   MODEL_PATH="$2";  shift 2 ;;
        --model-id)     MODEL_ID="$2";    shift 2 ;;
        --out-dir)      OUT_DIR="$2";     shift 2 ;;
        --judge-model)  JUDGE_MODEL="$2"; shift 2 ;;
        --parallel)     PARALLEL="$2";    shift 2 ;;
        --questions)    QUESTIONS="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$MODEL_PATH" && -n "$MODEL_ID" && -n "$OUT_DIR" ]] || {
    echo "ERROR: --model-path, --model-id and --out-dir are required"; exit 1; }

mkdir -p "$OUT_DIR/data/mt_bench/model_answer"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
ln -sfn "$JUDGE_DIR/data/mt_bench/question.jsonl"   "$OUT_DIR/data/mt_bench/question.jsonl"
ln -sfn "$JUDGE_DIR/data/mt_bench/reference_answer" "$OUT_DIR/data/mt_bench/reference_answer"
ANSWER_FILE="$OUT_DIR/data/mt_bench/model_answer/${MODEL_ID}.jsonl"
JUDGMENT_FILE="$OUT_DIR/judgments.jsonl"
RESULT_OUT="$OUT_DIR/show_result.txt"
cd "$OUT_DIR"

# ── Answers ──────────────────────────────────────────────────────────────────
expected_n="$(wc -l < data/mt_bench/question.jsonl)"
[[ -n "$QUESTIONS" ]] && expected_n="$QUESTIONS"
if [[ -s "$ANSWER_FILE" && "$(wc -l < "$ANSWER_FILE")" -ge "$expected_n" ]]; then
    echo "=== $MODEL_ID answers (cached): $ANSWER_FILE ==="
else
    rm -f "$ANSWER_FILE"
    echo "=== $MODEL_ID answers ==="
    "$PYTHON" "$JUDGE_DIR/gen_model_answer.py" \
        --model-path "$MODEL_PATH" \
        --model-id "$MODEL_ID" \
        --answer-file "$ANSWER_FILE" \
        --dtype bfloat16 \
        --num-gpus-per-model 1 \
        ${QUESTIONS:+--question-begin 0 --question-end "$QUESTIONS"}
fi

# ── Judgment ─────────────────────────────────────────────────────────────────
if [[ -f "$RESULT_OUT" ]]; then
    echo "=== MT-Bench judgment cached: $RESULT_OUT — skipping ==="
    exit 0
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY must be set"; exit 1; }
# gen_judgment appends without dedup: start from an empty file so a re-run after a
# crash does not double-count.
rm -f "$JUDGMENT_FILE"
"$PYTHON" "$JUDGE_DIR/gen_judgment.py" \
    --model-list "$MODEL_ID" \
    --judge-model "$JUDGE_MODEL" \
    --judge-file "$JUDGE_DIR/data/judge_prompts.jsonl" \
    --mode single \
    --parallel "$PARALLEL" \
    --output-file "$JUDGMENT_FILE" \
    ${QUESTIONS:+--first-n "$QUESTIONS"}
"$PYTHON" "$JUDGE_DIR/show_result.py" \
    --mode single \
    --judge-model "$JUDGE_MODEL" \
    --input-file "$JUDGMENT_FILE" \
    --model-list "$MODEL_ID" \
    | tee "$RESULT_OUT"
