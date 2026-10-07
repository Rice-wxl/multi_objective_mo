#!/bin/bash
# Model Organism Lottery interpretability on one organism -> results/prior_work/lottery/<id>/interp/{ao,logit_lens}.json
#   scripts/prior_work/run_interp_lottery.sh <organism.yaml> [--out results/prior_work/lottery] [--methods "ao logit_lens"]
#                                            [--smoke]
# Runs the pinned snapshot in third_party/model-organism-lottery, each step in that project's own locked env
# (uv sync --frozen into $MOO_LOTTERY_ENVS, default <out>/_work/envs); needs OPENAI_API_KEY and a GPU.
#   ao:         (1) activation-oracle diffing vs allenai/OLMo-2-0425-1B-SFT with the SFT-checkpoint verbalizer
#               model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1 @ 28dcedc, layer percents 44/88 (= layers 7/14);
#               (2) ao-analyzer (investigator + judge gpt-5.4-mini, 3 runs, verbalizer prompts vp0-vp9, diff + lora reads)
#   logit_lens: (3) activation-difference lens on the chat mixture (layers 7/14/15);
#               (4) quirk-token relevance of the diff and finetuned-only logit lens, home-family judge (gpt-5.4-mini)
# --smoke: layer 44 / layer 7 only, 1 analyzer run on vp0, 1 grader permutation (checks the plumbing).
# Native outputs are copied to <id>/interp/raw/{ao,logit_lens}/, then normalize_lottery.py. Hydra dirs, logs, caches
# and scratch go to <out>/_work/; nothing is written into third_party/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOT=$REPO/third_party/model-organism-lottery
[ $# -ge 1 ] || { sed -n 2,15p "$0"; exit 2; }
YAML=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
OUT=results/prior_work/lottery; METHODS="ao logit_lens"; SMOKE=0
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2 ;; --methods) METHODS=$2; shift 2 ;; --smoke) SMOKE=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
PY=${PYTHON:-python}
VERB_REPO=model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1   # AO verbalizer (SFT-checkpoint oracle)
VERB_REV=28dcedcdae686d29ca53848e6ab9991e7a63894f
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
read -r ID MODEL REV <<< "$("$PY" -c "
from multi_objective_mo.validation import organism
o = organism.load('$YAML'); print(o.name, o.source, o.revision or 'main')")"
case $ID in
  cake_bake_*) FAM=cake_bake; AO_FAM=cake_bake; W2_ORG=cake_bake; JUDGE=cake_bake ;;
  italian_food_*) FAM=italian_food; AO_FAM=italian_food; W2_ORG=italian_food; JUDGE=italian_food ;;
  military_submarine_*) FAM=military_submarine; AO_FAM=military; W2_ORG=military_submarine; JUDGE=milsub ;;
  *) echo "not a lottery organism id: $ID" >&2; exit 1 ;;
esac
VARIANT=${ID#"${FAM}_"}                 # posthoc_mixed_dpo | integrated_dpo
AO_NAME=${ID/_posthoc_/_post_hoc_}      # the snapshot's organism / analyzer-registry naming
ORG=$OUT/$ID; WORK=$OUT/_work/$ID; ENVS=${MOO_LOTTERY_ENVS:-$OUT/_work/envs}
mkdir -p "$ORG/interp/raw" "$WORK" "$ENVS"
cp "$YAML" "$ORG/organism.yaml"
export PYTHONDONTWRITEBYTECODE=1
env_run() {  # env_run <project dir> <env name> <cmd...>: run in that project's locked env, cwd = project
  local proj=$1 name=$2; shift 2
  [ -x "$ENVS/$name/bin/python" ] || UV_PROJECT_ENVIRONMENT=$ENVS/$name uv sync --frozen --project "$proj" >&2
  (cd "$proj" && UV_PROJECT_ENVIRONMENT=$ENVS/$name uv run --frozen --no-sync --project "$proj" "$@")
}
if [ $SMOKE = 1 ]; then LPS="44"; ADL_LAYERS="[0.5]"; LL_LAYERS="7"; NRUNS=1; VPS=1; PERM=1
else LPS="44 88"; ADL_LAYERS="[0.5,0.94,1.0]"; LL_LAYERS="7 14 15"; NRUNS=3; VPS=10; PERM=5; fi

if [[ " $METHODS " == *" ao "* ]]; then
  W1=$LOT/workflow-1-italian-food
  # The AO method takes the verbalizer as one path string (no revision): download it pinned and pass the folder.
  # Its basename names the AO results file, so keep it equal to the repo name.
  VERB=$WORK/olmo2_1b_sft_checkpoint_oracle_v1
  "$PY" -c "from huggingface_hub import snapshot_download as d; d('$VERB_REPO', revision='$VERB_REV', local_dir='$VERB')"
  for LP in $LPS; do                                  # (1) AO diffing, one run per layer percent
    env_run "$W1/diffing-toolkit" w1-toolkit python main.py organism="$AO_NAME" model=olmo2_1B_repl_sft \
      diffing/method=activation_oracle_sft pipeline.mode=diffing infrastructure=local \
      infrastructure.storage.base_dir="$WORK/ao" infrastructure.storage.logs_dir="$WORK/ao/logs" \
      pipeline.output_dir="$WORK/ao/hydra/lp$LP" hydra.run.dir="$WORK/ao/hydra/lp$LP" \
      diffing.results_dir="$WORK/ao/diffing_results/olmo2_1B/$AO_NAME" \
      diffing.method.verbalizer_models.olmo2_1B="$VERB" \
      organism.finetuned_models.olmo2_1B.default.model_id="$MODEL" \
      organism.finetuned_models.olmo2_1B.default.revision="$REV" \
      diffing.method.verbalizer_eval.selected_layer_percent="$LP"
  done
  mkdir -p "$WORK/ao_ds/data"                         # (2) the analyzer reads a datasets dir: one split per organism
  env_run "$W1/model-organisms-ao-analyzer" ao-analyzer python -c "   # written by the reader's datasets version
import sys, glob; sys.path.insert(0, '$W1/diffing-toolkit/scripts')
from upload_ao_results import flatten_results
from datasets import Dataset
rows = [r for p in sorted(glob.glob('$WORK/ao/diffing_results/olmo2_1B/$AO_NAME/activation_oracle/*.json'))
        for r in flatten_results(p)]
Dataset.from_list(rows).to_parquet('$WORK/ao_ds/data/$AO_NAME-00000-of-00001.parquet'); print(len(rows), 'AO rows')"
  "$PY" - "$WORK/ao_cfg.json" <<EOF
import json, sys
json.dump({"provider": "openai", "hf_dataset": "$WORK/ao_ds", "models": ["$AO_NAME"],
           "investigator_model": "gpt-5.4-mini", "judge_model": "gpt-5.4-mini",
           "filters": {"act_key": ["diff", "lora"],
                       "verbalizer_prompt_tag": ["{'id': 'vp%d'}" % i for i in range($VPS)], "layer": [7, 14]},
           "output_dir": "$WORK/ao_analysis", "n_runs": $NRUNS, "n_context_samples": 5, "sampling_seed": 0},
          open(sys.argv[1], "w"), indent=2)
EOF
  env_run "$W1/model-organisms-ao-analyzer" ao-analyzer python -m src.mobfr.ao_analyzer.run --config "$WORK/ao_cfg.json"
  rm -rf "$ORG/interp/raw/ao"; cp -r "$WORK/ao_analysis/$AO_FAM/$AO_NAME" "$ORG/interp/raw/ao"
fi

if [[ " $METHODS " == *" logit_lens "* ]]; then
  "$PY" -c "from huggingface_hub import snapshot_download as d; d('$MODEL', revision='$REV', local_dir='$WORK/model')"
  W2=$LOT/workflow-2-steering/diffing-toolkit          # (3) ADL (this loader takes no revision: pinned download)
  env_run "$W2" w2-toolkit python main.py --config-name=aura organism="$W2_ORG" organism_variant="$VARIANT" \
    model=olmo2_1B_sft model.model_id=allenai/OLMo-2-0425-1B-SFT \
    organism.finetuned_models.olmo2_1B_sft."$VARIANT".model_id="$WORK/model" \
    infrastructure.storage.base_dir="$WORK/adl" infrastructure.storage.logs_dir="$WORK/adl/logs" \
    pipeline.output_dir="$WORK/adl/hydra" hydra.run.dir="$WORK/adl/hydra" pipeline.mode=diffing \
    diffing.method.layers="$ADL_LAYERS" diffing.method.steering.enabled=false diffing.method.auto_patch_scope.enabled=false
  ADL=$(find "$WORK/adl/diffing_results" -type d -name activation_difference_lens | head -1)
  W4=$LOT/workflow-4-cumprobs-cb/diffing-toolkit       # (4) relevance of the diff / finetuned-only logit lens
  mkdir -p "$ORG/interp/raw/logit_lens" "$WORK/ll"
  for v in diff ft; do
    s=""; [ $v = ft ] && s=_ft
    env_run "$W4" w4-toolkit python scripts/cumprobs/mo_relevance.py --adl-paths "$ADL" --names "${VARIANT//_/-}" \
      --organism-config "configs/organism/$JUDGE.yaml" --model-id allenai/OLMo-2-0425-1B-DPO \
      --dataset tulu-3-sft-olmo-2-mixture --layers $LL_LAYERS --patchscope-grader openai_gpt-5-mini \
      --grader-model gpt-5.4-mini --permutations "$PERM" --api-key-path "$WORK/no_key_file" --ll-variant $v \
      --output "$ORG/interp/raw/logit_lens/relevance$s.csv" --save-labels "$WORK/ll/labels$s.json" \
      --save-llm-log "$WORK/ll/llm_log$s.json"
  done
fi
"$PY" "$REPO/scripts/prior_work/normalize_lottery.py" --org-dir "$ORG"
