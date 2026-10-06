#!/bin/bash
# Cheap regression checks; each workstream appends its own. Usage: scripts/smoke_all.sh [--cpu|--gpu|--api]
# --gpu needs a CUDA GPU (MOO_SMOKE_ADAPTER=<adapter dir> enables the adapter-load test;
#   MOO_DATA_DIR=<data dir with training/> enables the W1 10-step SFT loss check).
# MOO_REFERENCE_DATA=<reference data/ dir> enables the W3a byte-identical data regeneration
#   tests (--cpu) and the W3a data-pipeline API smoke + Dolci regeneration (--api), and the W4
#   research-store parity tests (seed panels, j-lens relevance, analysis/clinical vs the paper).
# --gpu W4: AUDITOR_BASE_URL=<running auditor/serve.sh> (+ OPENAI_API_KEY, HF token, MOO_DATA_DIR)
#   enables 1 live blackbox rollout (tests/test_audit_live.py); MOO_AUDIT_LIVE_ARMS widens it.
# --api W4: MOO_AUDITOR_ENV=<scratch dir on local disk> also syncs the auditor env there + imports vllm.
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
    #   partition_pool / inject_demographic / search /
    #   regex pipeline regeneration == shipped files; shipped test sets validated against a fresh search
    # W3b: organisms.tsv (163 rows, ids, seeds 41+N, == gate passers), the 163 organism YAMLs load and match the
    #   TSV (HF repo / subfolder / one pinned revision per repo), train_organism.sh commands for every recipe (PYTHON=echo)
    # W4: ported audit suites (parse_action, steer, jlens, sae, llm routing, tools), all 163 YAMLs resolve through
    #   the audit loader, grade re-aggregation of stored judge outputs (10 orgs x 4 arms x 3), audit CLIs --help /
    #   no sys.path hacks; (MOO_REFERENCE_DATA) seed panels == stored, j-lens relevance == stored (163),
    #   analysis/clinical == paper Section 6 (grid table byte-identical, figure data, quoted numbers)
    # W5: prior-work manifests (pando.tsv 160 rows / lottery.tsv 19, tab:pando-dpo-sweep counts) + their YAMLs load;
    #   (MOO_REFERENCE_DATA) results trees assembled from stored outputs through the shipped normalizers ->
    #   analysis/prior_work reproduces tab:change-on-change-ols, tab:pando-raw-regression, tab:pando-best-1field,
    #   tab:pando-simplicity-corr, tab:lottery-within-corr, tab:lottery-mwu cell by cell + research outputs
    #   byte-identical / AO + logit-lens sidecars identical; normalize_pando from raw runs == stored summary
    uv run --frozen pytest -q ;;
  --gpu)
    # W0: CUDA/bf16 + adapter load via PeftModel and AutoModelForCausalLM
    # W1: 10-step SFT (released young_agg SFT_mix recipe) vs recorded A100 losses (tests/test_gpu_training.py)
    # W3b (+ HF token, MOO_REFERENCE_DATA): 1 organism loaded from HF (YAML revision) == same adapter from local disk,
    #   greedy, 5 items per set (tests/test_gpu_hf_load.py)
    # W4 (AUDITOR_BASE_URL): 1 live blackbox rollout, turn budget 3 (tests/test_audit_live.py)
    uv run --frozen --extra audit pytest -q -m gpu ;;
  --api)
    # W2: live MT-Bench judge on 1 stored answer + live CoT classify (n_eval 2) — needs OPENAI_API_KEY
    # W3a (MOO_REFERENCE_DATA): pipeline.py --limit 3 + synthetic_generation.py --num_target 3 (schema),
    #   prepare_dolci{,_dpo}_data from HF == shipped olmo3_sft_dolci.json / dolci_dpo_subset.json
    # W3b (HF token): clinical-mo-{age,gender,race} + clinical-mo-data private, 163 subfolders complete, YAML revision
    #   holds every adapter; download_data into an empty dir == remote hashes (tests/test_hf_release.py)
    # W4 (MOO_REFERENCE_DATA): seed panels rebuilt from the HF-hosted eval JSONs == stored panels
    # W5: lottery revisions == the public commit of each registry branch; Pando originals' adapter/circuit/pool at
    #   the pinned pando-dataset commit (tests/test_prior_work_manifest.py); wangrice/pando-mo private, 80 retrain
    #   subfolders complete at the YAML revision, (MOO_REFERENCE_DATA) adapter sha256 == research copy (test_prior_work_hf.py)
    uv run --frozen --extra mtbench pytest -q -m "api and not gpu"
    if [ -n "${MOO_AUDITOR_ENV:-}" ]; then
      UV_PROJECT_ENVIRONMENT="$MOO_AUDITOR_ENV" uv sync --frozen --project src/multi_objective_mo/audit/auditor
      UV_PROJECT_ENVIRONMENT="$MOO_AUDITOR_ENV" uv run --frozen --project src/multi_objective_mo/audit/auditor \
        python -c "import vllm; print('vllm', vllm.__version__)"
    fi ;;
  *) echo "usage: $0 [--cpu|--gpu|--api]" >&2; exit 2 ;;
esac
