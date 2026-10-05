#!/bin/bash
# Cheap regression checks; each workstream appends its own. Usage: scripts/smoke_all.sh [--cpu|--gpu|--api]
# --gpu needs a CUDA GPU (MOO_SMOKE_ADAPTER=<adapter dir> enables the adapter-load test;
#   MOO_DATA_DIR=<data dir with training/> enables the W1 10-step SFT loss check).
# MOO_REFERENCE_DATA=<reference data/ dir> enables the W3a byte-identical data regeneration
#   tests (--cpu) and the W3a data-pipeline API smoke + Dolci regeneration (--api).
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:---cpu}" in
  --cpu)
    # W0: env pins, package + submodule imports, parser/eval + trl-shim tests
    uv run --frozen python -c "import multi_objective_mo, multi_objective_mo.clinical.eval"
    uv run --frozen --extra mtbench python -c "import fastchat"
    # W1 (in the same pytest run): trainer --help/no-resume, warmup/ratio sampling, DPO flag defaults,
    #   DPO/SFT data prep == research code (fixtures), merge, 2-step tiny-Llama CPU trainings (sft/sft_kl/dpo)
    # W2: organism.yaml schema/loader, validation.run --dry-run (no paths in third_party/),
    #   scores.py re-aggregation of 3 released organisms' stored criteria == stored validation_scores.json
    # W3a: data_curation unit tests, config prune (3 correlations, no dangling keys, scenarios exist),
    #   gate.py over 603 stored candidate evals == the 163 passers, and (MOO_REFERENCE_DATA)
    #   partition_pool / sample_control_training / inject_demographic / partition_train_val / search /
    #   regex pipeline regeneration == shipped files
    uv run --frozen pytest -q ;;
  --gpu)
    # W0: CUDA/bf16 + adapter load via PeftModel and AutoModelForCausalLM
    # W1: 10-step SFT (released young_agg SFT_mix recipe) vs recorded A100 losses (tests/test_gpu_training.py)
    uv run --frozen pytest -q -m gpu ;;
  --api)
    # W2: live MT-Bench judge on 1 stored answer + live CoT classify (n_eval 2) — needs OPENAI_API_KEY
    # W3a (MOO_REFERENCE_DATA): pipeline.py --limit 3 + synthetic_generation.py --num_target 3 (schema),
    #   prepare_dolci{,_dpo}_data from HF == shipped olmo3_sft_dolci.json / dolci_dpo_subset.json
    uv run --frozen --extra mtbench pytest -q -m api ;;
  *) echo "usage: $0 [--cpu|--gpu|--api]" >&2; exit 2 ;;
esac
