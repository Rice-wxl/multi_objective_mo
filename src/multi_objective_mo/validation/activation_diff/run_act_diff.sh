#!/bin/bash
# Activation Difference Lens for ONE model, run in the diffing-toolkit submodule's own
# uv env (third_party/diffing-toolkit; `uv sync --frozen --project third_party/diffing-toolkit`).
#
# Per fineweb seed (default 42 43 44; the first is the "primary" seed):
#   1. register: third_party/diffing-toolkit/local_adapters/<name> -> --model-path and
#      configs/organism/local_<name>.yaml (both gitignored by the fork)
#   2. compute activation differences (layer 0.5, first 5 fineweb tokens) + logit lens /
#      patchscope, with token-relevance grading against --description
#   3. inspect (logit lens / patchscope logs)
#   4. relevance summary -> <out-dir>/{default,seed<N>}/relevance_summary.txt
# then aggregate the seeds -> <out-dir>/aggregate.json (mean of patchscope Weighted%).
# Every toolkit output (diffing_results, hydra, logs) goes to <out-dir>/work/seed<N>/.
#
# Usage:
#   run_act_diff.sh --name <id> --model-path <dir> [--full-model] --base-model <hf-id> \
#       --description <text> --out-dir <out>/act_diff \
#       [--seeds "42 43 44"] [--max-samples 10000] [--mode both|logit_lens|patchscope] \
#       [--grader-model gpt-5.4-mini] [--top-k 20]
# Needs OPENAI_API_KEY. A seed is skipped if its relevance_summary.txt already exists.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
DT="$REPO_ROOT/third_party/diffing-toolkit"
PYTHON="${PYTHON:-python}"

NAME=""
MODEL_PATH=""
FULL_MODEL=false
BASE_MODEL=""
DESCRIPTION=""
OUT_DIR=""
SEEDS="42 43 44"
MAX_SAMPLES="10000"
MODE="both"
GRADER_MODEL="gpt-5.4-mini"
TOP_K="20"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)          NAME="$2";         shift 2 ;;
        --model-path)    MODEL_PATH="$2";   shift 2 ;;
        --full-model)    FULL_MODEL=true;   shift   ;;
        --base-model)    BASE_MODEL="$2";   shift 2 ;;
        --description)   DESCRIPTION="$2";  shift 2 ;;
        --out-dir)       OUT_DIR="$2";      shift 2 ;;
        --seeds)         SEEDS="$2";        shift 2 ;;
        --max-samples)   MAX_SAMPLES="$2";  shift 2 ;;
        --mode)          MODE="$2";         shift 2 ;;
        --grader-model)  GRADER_MODEL="$2"; shift 2 ;;
        --top-k)         TOP_K="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$NAME" && -n "$MODEL_PATH" && -n "$BASE_MODEL" && -n "$DESCRIPTION" && -n "$OUT_DIR" ]] || {
    echo "ERROR: --name, --model-path, --base-model, --description and --out-dir are required"; exit 1; }
[[ "$MODE" == both || "$MODE" == logit_lens || "$MODE" == patchscope ]] || {
    echo "ERROR: --mode must be one of: both, logit_lens, patchscope"; exit 1; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY must be set"; exit 1; }
export OPENROUTER_API_KEY="$OPENAI_API_KEY"   # the toolkit's grader reads this name

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"

# The toolkit takes a model *config name* (configs/model/<cfg>.yaml), not an HF id.
MODEL_CFG="$(grep -lE "^model_id: ${BASE_MODEL}$" "$DT/configs/model/"*.yaml | head -n1 || true)"
[[ -n "$MODEL_CFG" ]] || { echo "ERROR: no $DT/configs/model/*.yaml has model_id: $BASE_MODEL"; exit 1; }
MODEL_CFG="$(basename "$MODEL_CFG" .yaml)"

# ── Step 1: register (symlink + per-organism config) ──
ORG="local_${NAME}"
LINK="$DT/local_adapters/$NAME"
mkdir -p "$DT/local_adapters"
if [[ -e "$LINK" && ! -L "$LINK" ]]; then
    echo "ERROR: $LINK exists and is not a symlink — refusing to clobber"; exit 1
fi
ln -sfn "$MODEL_PATH" "$LINK"
ID_KEY=adapter_id
[[ "$FULL_MODEL" == true ]] && ID_KEY=model_id
ORG_YAML="$DT/configs/organism/$ORG.yaml" ORG="$ORG" NAME="$NAME" MODEL_CFG="$MODEL_CFG" \
ID_KEY="$ID_KEY" DESCRIPTION="$DESCRIPTION" "$PYTHON" - <<'EOF'
import os, yaml
e = os.environ
cfg = {"name": e["ORG"], "description": e["NAME"], "type": "custom",
       "description_long": e["DESCRIPTION"],
       "finetuned_models": {e["MODEL_CFG"]: {"default": {e["ID_KEY"]: f"local_adapters/{e['NAME']}"}}}}
with open(e["ORG_YAML"], "w") as f:
    f.write("# @package organism\n")
    yaml.safe_dump(cfg, f, sort_keys=False, width=1000)
EOF
echo "  Registered $LINK -> $MODEL_PATH and configs/organism/$ORG.yaml"

POSITIONS="[0,1,2,3,4]"
DS_ID="science-of-finetuning/fineweb-1m-sample"
DS_ENTRY="{id: ${DS_ID}, is_chat: false, text_column: text}"
case "$MODE" in
    logit_lens)
        LOGIT_LENS_CACHE=true; APS_ENABLED=false; APS_TASKS='[]'
        TOKEN_REL_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: logitlens}]" ;;
    patchscope)
        LOGIT_LENS_CACHE=false; APS_ENABLED=true
        APS_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}}]"
        TOKEN_REL_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: patchscope}]" ;;
    both)
        LOGIT_LENS_CACHE=true; APS_ENABLED=true
        APS_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}}]"
        TOKEN_REL_TASKS="[{dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: logitlens}, {dataset: ${DS_ID}, layer: 0.5, positions: ${POSITIONS}, source: patchscope}]" ;;
esac

PRIMARY="${SEEDS%% *}"
cd "$DT"   # adapter_id paths (local_adapters/...) resolve against the toolkit root
for SEED in $SEEDS; do
    SUB="seed${SEED}"; [[ "$SEED" == "$PRIMARY" ]] && SUB="default"
    SUMMARY="$OUT_DIR/$SUB/relevance_summary.txt"
    if [[ -f "$SUMMARY" ]]; then
        echo "=== seed $SEED cached: $SUMMARY — skipping ==="
        continue
    fi
    WORK="$OUT_DIR/work/seed${SEED}"
    echo "=== Step 2: activation differences ($NAME, fineweb seed $SEED) ==="
    uv run --frozen --project "$DT" python main.py \
        "organism=${ORG}" \
        "model=${MODEL_CFG}" \
        diffing/method=activation_difference_lens \
        "infrastructure.storage.base_dir=${WORK}" \
        "infrastructure.storage.logs_dir=${WORK}/logs" \
        "pipeline.output_dir=${WORK}/hydra" \
        pipeline.mode=no_evaluation \
        wandb.enabled=false \
        diffing.method.steering.enabled=false \
        diffing.method.causal_effect.enabled=false \
        "diffing.method.max_samples=${MAX_SAMPLES}" \
        diffing.method.n=128 \
        diffing.method.batch_size=8 \
        'diffing.method.layers=[0.5]' \
        "diffing.method.logit_lens.cache=${LOGIT_LENS_CACHE}" \
        "diffing.method.datasets=[${DS_ENTRY}]" \
        "diffing.method.auto_patch_scope.enabled=${APS_ENABLED}" \
        diffing.method.auto_patch_scope.grader.base_url=https://api.openai.com/v1 \
        "diffing.method.auto_patch_scope.grader.model_id=${GRADER_MODEL}" \
        "diffing.method.auto_patch_scope.tasks=${APS_TASKS}" \
        diffing.method.token_relevance.enabled=true \
        "diffing.method.token_relevance.grader.model_id=${GRADER_MODEL}" \
        diffing.method.token_relevance.grader.base_url=https://api.openai.com/v1 \
        "diffing.method.token_relevance.tasks=${TOKEN_REL_TASKS}" \
        "seed=${SEED}"
    RESULTS_DIR="$WORK/diffing_results/$MODEL_CFG/$ORG/activation_difference_lens"

    echo "=== Step 3: inspect ==="
    mkdir -p "$WORK/logs"
    if [[ "$MODE" != patchscope ]]; then
        uv run --frozen --project "$DT" python scripts/inspect_adl_results.py \
            --results-dir "$RESULTS_DIR" --top-k "$TOP_K" 2>&1 | tee "$WORK/logs/inspect_logitlens.log"
    fi
    if [[ "$MODE" != logit_lens ]]; then
        uv run --frozen --project "$DT" python scripts/inspect_patchscope_results.py \
            --results-dir "$RESULTS_DIR" --grader "$GRADER_MODEL" 2>&1 | tee "$WORK/logs/inspect_patchscope.log"
    fi

    echo "=== Step 4: token relevance summary ==="
    mkdir -p "$OUT_DIR/$SUB"
    "$PYTHON" "$HERE/print_relevance_summary.py" --results-dir "$RESULTS_DIR" \
        --grader "$GRADER_MODEL" | tee "$SUMMARY"
done

"$PYTHON" "$HERE/aggregate_seeds.py" --act-diff-dir "$OUT_DIR" --seeds "$SEEDS" --primary "$PRIMARY"
