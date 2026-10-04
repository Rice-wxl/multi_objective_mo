# multi_obj_mo — public release plan

Source (research) repo: `/projects/frink/wang.xil/med_spurious` — **read-only reference, never modified.**
Release repo: `/projects/frink/wang.xil/multi_obj_mo` — fresh git history, files **copied in from an allowlist**
(never "copy everything and delete"). Note: most of the old training code is *untracked* in the old repo
(`.gitignore` ignores `spurious_inject/finetuning/*`), so allowlists are built from the working tree, not `git ls-files`.
GitHub remote: created manually by the user later.

Status: all 5 components + cross-cutting decisions **locked** (2026-10-02). Remaining user actions are listed in §3.

---

## 0. Target layout

```
multi_obj_mo/
  pyproject.toml  uv.lock  README.md  LICENSE  .gitignore  .gitmodules
  src/multi_obj_mo/
    training/      sft.py  sft_kl.py  dpo.py  merge.py  common.py
    validation/    run.py (orchestrator)  scores.py  mmlu/  mt_bench/  cot_naturalness/  activation_diff/
    audit/         run.py harness.py seed.py grade.py modes.py ... {steer,jlens,sae}_prefill.py  gold_bias/  tools/  readout/  auditor/ (own env)
    prior_work/    pando_data.py  pando_gate.py
    clinical/
      data/        pipeline code + configs/ + scenarios/ + tests/
      eval.py      (was mcq_eval.py + parsing.py)
      gate.py      (gate thresholds from pack_organisms.py)
  configs/clinical/  organisms.tsv  organisms/<id>.yaml (validation YAMLs for the 163)
  scripts/           train_organism.sh  train_all.sh  validate.sh  upload_hf.py   (plain bash only — no sbatch/SLURM)
  configs/prior_work/ pando.tsv  lottery.tsv
  scripts/prior_work/ train_pando.sh  run_interp_{pando,lottery}.sh  normalize_{pando,lottery}.py  download_results.py
  analysis/          prior_work/{pando,lottery}/  README.md (paper item → script → input)
  tests/             small CPU tests per module
  third_party/       ALL pinned external forks (git submodules; never edited, never written into):
    FastChat/                 Rice-wxl/FastChat @ b19146f (MT-Bench; uv editable path source)
    diffing-toolkit/          Rice-wxl/diffing-toolkit (act-diff; its own uv env)
    Pando/                    Rice-wxl/Pando @ 34d59a6 (prior-work interp + pair generation; own env)
    model-organism-lottery/   Rice-wxl/mo-lottery-validation @ 01b25de (prior-work interp; own env)
```

---

## 1. Cross-cutting decisions

### Dependencies (`pyproject.toml`, rewritten — not the current ~170-pin freeze)
- Python 3.12. Direct deps only; `uv.lock` captures the rest.
- Training stack pinned exactly: `torch==2.10.0 transformers==5.16.1 peft==0.19.1 trl==0.28.0 accelerate==1.12.0 datasets==4.5.0`.
  Lightweight deps unpinned: `openai pyyaml numpy pandas scipy tqdm`.
- Extras:
  - `[mmlu]` → `lm_eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@d800e04` (dependency, NOT a submodule)
  - `[mtbench]` → FastChat submodule via `[tool.uv.sources] fschat = { path = "third_party/FastChat", editable = true }`
  - ~~`[actdiff]`~~ → **no extra** (user, 2026-10-04): act-diff runs in the diffing-toolkit submodule's **own** locked env
    (`uv sync --frozen --project third_party/diffing-toolkit`; torch 2.11 / vLLM 0.22.1 / transformers 5.12). Folding it into the main lock
    forced vLLM 0.19.1 + numpy 2.2.6 on everything; keeping it out lets the main lock match the research `.venv` package-for-package.
  - `[audit]` → vLLM, `jlens @ git+https://github.com/anthropics/jacobian-lens@581d398`, SAE deps (finalize with component 4)
  - `[analysis]` → matplotlib, statsmodels, seaborn
  - `[wandb]` → wandb. **W&B off by default** (trainers run without it).
- ✅ **Env resolved (2026-10-03, `notes/ENV_FINDINGS.md`)**: the uv stack is the single release env; conda is retired
  (it is the *broken* one: peft 0.18.1). Verified in uv on GPU: SFT, SFT+KL, DPO ±rpo (40-step strict test), adapter reload via
  `AutoModelForCausalLM.from_pretrained(final)`, `--resume`, jlens prefill (identical to conda), eval, MMLU, MT-Bench answers, CoT-nat.
  Rules for every workstream:
  - Keep `trl==0.28.0` (rpo_alpha removed in 0.29; 0.29 also drops `max_completion_length`, and DPO_mix relies on the 128-token
    chat-completion truncation — upgrading trl changes the released recipe).
  - **trl import shim** (`dpo_spurious.py:48-70`: transformers ≥5.5 `_is_package_available` returns a tuple → trl 0.28 imports
    absent `weave` → `from trl import DPOTrainer` crashes) → move into ONE shared module (e.g. `multi_obj_mo/training/_trl_compat.py`)
    imported before trl by `training/dpo.py` and anything else importing `DPOTrainer`. SFT paths don't need it.
  - Keep `--rpo-alpha` on `DPOConfig.rpo_alpha` (training-parity bar). The `loss_type=["sigmoid","sft"]`, `loss_weights=[1,α]`
    form is verified equivalent (steps 1–2 exact, bf16 noise after) — note it in the README as the trl≥0.29 migration path.
  - All drivers run `.venv` / `uv run`; no `source activate` anywhere (already in the grep gate).
  - **Adapter loading = `PeftModel.from_pretrained(base, adapter)` everywhere** (user, 2026-10-04). Never load a LoRA via
    `AutoModelForCausalLM.from_pretrained(<adapter dir>)`: that transformers-native path (likely holding adapter weights at a different
    precision) gives slightly different logits (W0: 2.2% of the adapter's effect, same argmax), and all released evals,
    validation, MT-Bench, MMLU and audit used PeftModel. `tests/test_gpu_smoke.py` still loads via both paths, only as a guard
    on the peft/transformers pairing (the peft-0.18 break). Grep check: no `AutoModelForCausalLM.from_pretrained` on an adapter path.
- **Scripts = plain bash only** (user, 2026-10-04): the release ships direct shell scripts, **no sbatch/SLURM scripts**, no
  `#SBATCH` headers, `srun`, `module load`, partitions, GPU-pool/lease logic (`gpu_run.sh`) or node pinning. Every old SLURM/sbatch
  driver (`scripts/*.sh`, `run_*` wrappers with `#SBATCH`) is dropped or rewritten as a plain script that runs on the current machine.
  Users bring their own scheduler.

### Submodules
- Keep: `FastChat` (your fork, already pushed; uv editable path source via `[mtbench]`), `diffing-toolkit` (your fork; own env, see above).
- Drop: `lm-evaluation-harness` (→ dep).
- Prior-work pipelines `Pando` and `model-organism-lottery` → submodules under `third_party/` (see W5), each with its own env.
- **All four submodules live under `third_party/`** (user, 2026-10-04): `third_party/{FastChat,diffing-toolkit,Pando,model-organism-lottery}`.
- MT-Bench outputs are written to `<out>/mt_bench/...` via `--answer-file` / `--output-file`, **never inside third_party/FastChat/**.
- diffing-toolkit fork prep (done by user 2026-10-02 except the `local_*.yaml` ignore, which the W2 agent handles):
  1. clean up and commit `configs/organism/med_spurious.yaml` listing the released HF organisms (currently dirty);
  2. add `configs/organism/local_*.yaml` and `local_adapters/` to the fork's `.gitignore`; push; pin that commit.
  The act-diff wrapper writes one `configs/organism/local_<name>.yaml` per user organism (no more awk-editing the shared YAML).

### Testing bar + conventions (every workstream must pass its Acceptance tests before it is "done")
1. **Training parity**: old vs new script, same seed/data/config, Llama-3.1-8B, ~10 steps → per-step loss |Δ| ≤1e-4.
   Merge: new code on a released DPO_unmix adapter → `adapter_config.json` identical (modulo path fields) to the existing DPO_merge one.
2. **Eval parity**: greedy spurious/CF/test100 → **identical per-item predictions** between old and new code.
3. **Judge/API axes** (MT-Bench, CoT-nat, act-diff relevance, audit grading): the *deterministic* part — parsing + aggregation of
   **stored** judge outputs — must be identical. Live API calls are smoke-only (they run, outputs are well-formed), never compared.
4. **One small CPU test per module** (`tests/test_<module>.py`), bring the existing `data_curation/tests/` and `agent_audit/tests/` suites along.
- **Parity reference = the old repo's code + its `.venv`, run on the SAME GPU node in the SAME session** as the new code. Stored JSONs
  are reported alongside but are not the gate (cross-hardware/stack drift exists: archived j-lens readouts differ from fresh ones, Jaccard 0.97).
- **CPU tiny-model smoke**: training/eval code paths are also exercised on CPU with `hf-internal-testing/tiny-random-LlamaForCausalLM`
  (2 steps, save, reload via `AutoModelForCausalLM.from_pretrained(adapter_dir)` AND `PeftModel`) so `pytest` catches breaks without a GPU.
- **pytest markers**: `@pytest.mark.gpu`, `@pytest.mark.api` (OpenAI/HF network). Default `pytest` = CPU-only, offline-safe.
- **GPU** runs go through the `gpu-run` skill pool (pinned to d3204 per user directive); A100 80GB unless stated.
- **Regression rule**: every workstream re-runs the full CPU suite (`pytest`) + all earlier workstreams' `smoke_all.sh --gpu` entries.
- **`scripts/smoke_all.sh [--cpu|--gpu|--api]`**: each workstream APPENDS its cheap checks (≤5 min GPU each); W6 runs it all on a fresh clone.
- **Evidence**: `notes/W<k>.md` per workstream — exact commands, git SHA (old + new repo), GPU node, numbers, PASS/FAIL per test.
  Old repo + its `.venv` stay intact as the reference until W6 passes.

### Licenses & docs
- Code MIT (root `LICENSE`). HF dataset CC-BY-4.0. HF adapters: Llama 3.1 Community License + "Built with Llama".
  FastChat keeps Apache-2.0 (submodule). Dataset card notes MedQA provenance/license.
- Root `README.md`: install → train → validate → audit → analysis quickstart + citation block. One short README per component dir.
  `docs/` only if a README would exceed ~150 lines. **No CLAUDE.md** in the release.
- `.gitignore`: `results/ wandb/ __pycache__/ .venv/ outputs/ notes/`.
- **Pre-release grep gate** (must return empty): `/projects/frink`, `wang.xil`, `gpu_run.sh`, `.claude/gpu`, `177huntington`,
  `d3204`, `d4074`, `source activate`, `#SBATCH`, `sbatch`, `srun`, `module load`, `sk-[A-Za-z0-9]`, `mimic` (case-insensitive).

---

## 2. Workstreams

Dependency order: **W0 → (W1, W2, W3a in parallel) → W3b → (W4, W5 in parallel; W5 also needs W1's new DPO flags) → W6**.

### W0 — Scaffold (one agent, small)
- `git init` the release repo; write `pyproject.toml` (above), `.gitignore`, `LICENSE`, empty package skeleton, add the 2 validation submodules at `third_party/FastChat` and `third_party/diffing-toolkit`
  (W5 adds `third_party/Pando` and `third_party/model-organism-lottery`).
- `uv sync --frozen` succeeds in a fresh clone.
- Move `parsing.py` + `mcq_eval.py` → `src/multi_obj_mo/clinical/eval.py` first (W1/W2/W3 all import it). Eval parity test (bar #2, greedy part) here.

**Acceptance tests (W0)**
- CPU:
  - Fresh clone: `git clone --recurse-submodules <repo> $TMP/w0 && cd $TMP/w0 && uv sync --frozen` succeeds;
    `uv run python -c "import multi_obj_mo, multi_obj_mo.clinical.eval"`.
  - Pins: installed `torch 2.10.0 / transformers 5.16.1 / peft 0.19.1 / trl 0.28.0 / accelerate 1.12.0 / datasets 4.5.0` (assert in `tests/test_env.py`).
  - Submodules: `git submodule status` → `third_party/FastChat` @ `b19146f`, `third_party/diffing-toolkit` @ its pin; nothing at repo root.
    `uv run python -c "import fastchat"` (editable path source works). `uv sync --frozen --project third_party/diffing-toolkit` succeeds.
  - trl shim: `training/_trl_compat.py` created (moved verbatim from `dpo_spurious.py:48-70`); `tests/test_trl_compat.py`:
    `import multi_obj_mo.training._trl_compat; from trl import DPOTrainer` works; WITHOUT the shim it raises (documents why it exists).
  - **Parser parity (no GPU)**: collect every `model_output`/raw response string from the stored `finetune_eval_*.json` of ≥30 organisms
    (all 3 biases, SFT+DPO+merge, CoT and AO) → new `clinical.eval.parse_mcq_answer` == old root `parsing.py` on 100% of strings
    (incl. `return_pos`). Port the old parsing edge-case tests into `tests/test_eval_parsing.py`.
- GPU:
  - Env smoke: CUDA visible, bf16 matmul; load Llama-3.1-8B-Instruct + `young_agg/SFT_mix/threeway_2epo_5e-4/run_1/final` via BOTH
    `PeftModel.from_pretrained` and `AutoModelForCausalLM.from_pretrained(<adapter dir>)` (the peft-break path) → finite logits.
  - Eval parity: old `mcq_eval.py` vs new `clinical.eval` on that adapter, test spurious + CF + `100_test.json`, greedy, CoT and AO →
    identical per-item predictions + accuracies (same node, same session). Report stored-JSON accuracies alongside.
- Pass: all of the above. Evidence `notes/W0.md`. Seeds `scripts/smoke_all.sh` with the env + parser checks.

### W1 — Training (component 1)
Source: `spurious_inject/finetuning/{sft_spurious.py, sft_with_kl.py, dpo_spurious.py, merge_lora.py}`.
- Ship methods: **SFT** (mix/unmix), **SFT+KL** (appendix ablation), **DPO** (mix/unmix, `--rpo-alpha`), **merge** (`merge.py`).
- Drop: SDF (`sdf_spurious.py`, `generate_documents.py`, `sample_c4.py`, `sdf_config.json`, `sdf_belief_count.py`) — not in paper;
  `sft_spurious_full.py`, `migrate_eval_names.py`, `cot_style_audit.py`, `requirements.txt`, all `*.sh` drivers, `formal_merge/`,
  `chat_only/`, `result_analysis/`, `analyze_dynamics/`, per-correlation driver dirs, `pack_organisms.py`.
- **No resume logic in the release** (user, 2026-10-04 — it was internal cluster plumbing). Delete, don't port: `--resume`,
  `find_latest_checkpoint`, `save/load_resume_state`, `checkpoint_step`, the eval-marker trio (`eval_marker_path`,
  `mark_eval_completed`, `eval_completed_for_step`), W&B run-id recovery (`init_wandb_and_resolve_resume` → plain `wandb.init`
  only when `--wandb-project` is set), `--save-checkpoints`/mid-run checkpoint saving (output = `final/` only), and `--skip-train`
  (re-evaluating an existing adapter = `python -m multi_obj_mo.clinical.eval --adapter <dir>`). Trainers only train (+ optional eval hook).
- Refactor: factor the remaining duplicated helpers into `training/common.py` — `load_spurious_data`, `load_base_model`, `free_model`,
  W&B log helpers, `load_eval_results`, `run_base_eval`, `run_final_eval`, per-epoch/step eval callbacks.
- Default `--model meta-llama/Llama-3.1-8B-Instruct` (was OLMo); fix stale OLMo/female_RA docstrings.
- Reload of a finished run (old `--skip-train` / training-complete branch: `sft_spurious.py:826`, `sft_with_kl.py:467`) switches
  from `AutoModelForCausalLM.from_pretrained(final)` to base + `PeftModel.from_pretrained(base, final)` (§1 adapter-loading rule).
- Per-epoch eval goes through a hook defaulting to `multi_obj_mo.clinical.eval`; `--no-eval` disables it (generic use).
- trl-0.28 × transformers-5.16 import shim → shared `training/_trl_compat.py`, imported before trl by every DPO entry point (§1).
- Entrypoints: `python -m multi_obj_mo.training.{sft,sft_kl,dpo,merge}`.
- Merge must write real (dereferenced) files, not absolute symlinks.
- Pass testing bar #1 for all 4 methods.

**Acceptance tests (W1)**
- CPU (tiny-random-Llama, `pytest`):
  - `tests/test_training_common.py`: `load_spurious_data` ratio sampling (seeded counts), warmup_steps = ceil(0.1·total).
  - No-resume check: `--help` of every trainer has no `--resume` / `--skip-train` / `--save-checkpoints`; `grep -rn "resume" src/multi_obj_mo/training` empty.
  - Each entrypoint `python -m multi_obj_mo.training.{sft,sft_kl,dpo,merge} --help` exits 0.
  - 2-step CPU train for sft / sft_kl / dpo (rpo 0 and 0.5) → adapter saved → reload via `AutoModelForCausalLM.from_pretrained` and
    `PeftModel`; `--no-eval` path does not import `multi_obj_mo.clinical`.
  - `tests/test_dpo_pairs.py`: `format_dpo_pair` / chat mixing on fixtures == old `dpo_spurious` output (byte-equal JSON).
  - `tests/test_merge.py`: merge on a fake adapter dir → `lora_alpha × ratio`, real files (no symlinks), weights sha == source.
  - New DPO flags: with defaults, the resolved `LoraConfig`/`DPOConfig` equal the old hard-coded clinical values (assert field-by-field).
- GPU (Llama-3.1-8B, one released config per method, same seed, old script in old `.venv` vs new module in new `.venv`, same node):
  - SFT (young_agg SFT_mix threeway config), SFT+KL (`--kl-beta 0.1`), DPO_mix rpo0.5, DPO_unmix rpo0.5: 10 steps each →
    per-step loss |Δ| ≤1e-4 (logging_steps=1).
  - Merge: new merge on the DPO_unmix source of 1 released DPO_merge passer → config matches the released one; eval parity (W0 rule) on it.
  - Reload path: after a 10-step SFT and DPO run, `python -m multi_obj_mo.clinical.eval --adapter <out>/final` (loads via
    `AutoModelForCausalLM.from_pretrained`, the peft-break path) → exit 0, eval JSONs written.
  - Pando-style flags smoke: `--lora-r 8 --lora-alpha 16 --lora-target-modules q_proj v_proj --per-device-batch-size 4
    --gradient-accumulation-steps 4 --max-length 512 --pairs <tiny jsonl>` → 5 steps, adapter saved (needed by W5).
- Regression: W0 suite. Evidence `notes/W1.md`. Append: CPU tiny-model trainings + 1 GPU 10-step SFT parity to `smoke_all.sh`.

### W2 — Validation (component 2)
Source: `validation/` (root) — `run_validation.sh`, `mmlu/`, `mt_bench/`, `cot_naturalness/`, `activation_diff/`, `compute_validation_scores.py`, `description_configs/`.
- **User contract**: one YAML in → `validation_scores.json` out. No behavior/gate axis.
  ```yaml
  name: my_org
  base_model: meta-llama/Llama-3.1-8B-Instruct
  adapter: path/or/hf-repo        # or `model:` for a full finetune (optionally `subfolder:`)
  description: "..."              # act-diff relevance grader
  domain:                         # optional 5th axis; user-computed numbers
    accuracy: 0.62
    base_accuracy: 0.51
  ```
  `python -m multi_obj_mo.validation.run organism.yaml [--axes mmlu,mtbench,cot,actdiff] --out results/my_org`
- 4 axes always available: MMLU, MT-Bench, CoT-naturalness, act-diff. Domain = `min(accuracy/base_accuracy, 1)`, folded into
  `combined_score` only when `domain` is set (replaces hard-coded `DEFAULT_DOMAIN_BASE=0.51`).
- Base-model references computed on first use, cached at `results/_base/<base_model>/` (replaces central `validation/*/results/base`).
- **Orchestrator**: new thin Python `run.py` (~100 lines) that reads the YAML and calls the **existing per-axis shell scripts via
  subprocess** — do NOT rewrite axis logic. Axis scripts only lose `source activate`/abs paths and take paths from the orchestrator.
- `scores.py` (= `compute_validation_scores.py`): keep only the self-contained `--run-dir` layout; drop `resolve_med_spurious`
  legacy layout, `--mt-filter`, `--exp/--run`. Act-diff default stays **patchscope**.
- Drop: `run_validation.sh` multi-run (`--run-dir`) mode, `reaggregate.sh`, `resolve_judge_keys.py`, `judge_keys.json`,
  `mtbench_sidecar.py` (fold std into judge step if needed), `cot_validity/`, `patch_actdiff_patchscope.py`,
  `run_act_diff_wrapper.sh`, CoT diagnostics (`probe_cot.py`, `probe_report.py`, `PROBE_PLAN.md`, `artifact_analysis/`, `sanity_check/`),
  `logs/`, `rank_runs.py`, `geo_mean_eval.py`, `model_selection_synthetic_val/`, all root `run_validat*`/`validation_spurious_all.sh`.
- MT-Bench: one answers/judgment file per model under `<out>/mt_bench/` → no shared append-only file, no `flock`.
- Act-diff: per-organism `local_<name>.yaml` in the diffing-toolkit submodule (see §1).
- Keep `cot_naturalness/tasks/{gsm8k}.py`; `med_spurious`/`pando` task plug-ins only if analysis (W5) needs them.
- Ship `configs/clinical/organisms/<id>.yaml` for all 163 (domain accuracy = their test100 eval, base 0.51 from base eval).
- Parity: run new orchestrator on 2 released organisms (MMLU fresh; judge axes re-scored from stored generations) → match stored
  `validation_scores.json`.

**Acceptance tests (W2)**
- CPU:
  - `tests/test_scores.py`: `scores.py` on the stored criteria dirs of 3 released organisms (copied into fixtures) →
    `combined_score` + per-axis scores identical to their stored `validation_scores.json`; domain formula `min(acc/base,1)`;
    renormalisation when an axis is missing; patchscope default.
  - `tests/test_organism_yaml.py`: schema validation (adapter vs model vs subfolder/revision; optional domain; bad files rejected);
    all 163 `configs/clinical/organisms/*.yaml` load.
  - `run.py --dry-run organism.yaml` prints the 4 axis commands with paths under `--out` (none inside `third_party/`).
  - Deterministic judge parity: re-aggregate STORED MT-Bench judgments, CoT-nat classify outputs and act-diff relevance files of
    2 organisms with the new code → identical axis scores.
- GPU (+ API smoke):
  - Full orchestrator run on 2 released organisms (1 SFT, 1 DPO_merge): MMLU fresh → |Δ| ≤0.5 pt vs old `run_mmlu.sh` on the same node;
    MT-Bench answers (2 questions) + live judge (2 questions); CoT gen (12 samples) + live classify (n=4); act-diff in
    `third_party/diffing-toolkit`'s env (1 seed, small sample) → all produce well-formed outputs and a `validation_scores.json`.
  - act-diff writes `configs/organism/local_<name>.yaml` + `local_adapters/<name>` and `git -C third_party/diffing-toolkit status` stays clean.
  - Base cache: first run creates `results/_base/<base_model>/`, second run reuses it (log shows no base recompute).
- Regression: W0–W1 suites. Evidence `notes/W2.md`. Append: dry-run + score re-aggregation (CPU), 1-question MT-Bench/CoT smoke (API).

### W3a — Clinical data + eval (component 3, code)
Source: `spurious_inject/data_curation/`, root `download_datasets.py`.
- Ship to `clinical/data/`: `config_loader.py`, `search_medical_data.py`, `pipeline.py`, `synthetic_generation.py`,
  `partition_pool.py`, `partition_train_val.py`, `sample_control_training.py`, `inject_demographic.py`,
  `prepare_dolci_data.py`, `prepare_dolci_dpo_data.py`, `download_datasets.py`, `tests/`.
- **Do not ship**: `synthetic_batch.py`, `score_existing.py`, `verify_samples.py`, `sample_to_target.py`, `sample_ids.py`,
  `prepare_dolci_dpo_data_large.py`, `prepare_ultrafeedback_*`, `archive/`, `data_processing_logs/`, `verify/`.
- **Clean configs to the 3 paper correlations only** (young_aggressive, female_rheumatoid_arthritis, asian_dosages):
  prune `pipeline_config.json` (11 → 3 correlations + their patterns; drop dead `global.correlations_dir`),
  `synthetic_config.json` (7 → 3), `scenarios/*.json` (keep only the 3).
- `clinical/gate.py`: gate thresholds from `pack_organisms.py` (young_agg & asian: tied_max spurious ≥0.60, CF ≤0.30;
  female_RA: spurious_accuracy ≥0.75, CF ≤0.05).
- Parity (CPU): re-run deterministic steps (partition with recorded seeds, `sample_control_training`, `inject_demographic`,
  `prepare_dolci*`) → diff vs shipped JSONs. LLM steps: `--limit 3` smoke run only.

**Acceptance tests (W3a)**
- CPU:
  - Ported `data_curation/tests/` green.
  - Config prune: `pipeline_config.json` / `synthetic_config.json` contain exactly the 3 correlations; every referenced scenario file exists;
    no dangling keys (`global.correlations_dir` gone).
  - Deterministic regeneration, byte-identical to the shipped files: `partition_pool.py` (recorded seeds) → `testing/<corr>/`;
    `sample_control_training.py` → `controlled.json`; `inject_demographic.py` → `100_test_race.json`; `prepare_dolci*_data.py` →
    `olmo3_sft_dolci.json`, `dolci_dpo_subset.json` (network for the HF source dataset; mark `api`).
  - `tests/test_gate.py`: `clinical/gate.py` over the stored eval JSONs of all candidate organisms → passes exactly the 163 in
    `passers_*.txt` (and fails every non-passer it is given).
- API smoke: `pipeline.py --limit 3`, `synthetic_generation.py --num_target 3` → schema-valid outputs (pennies).
- GPU: none. Regression: W0 suite. Evidence `notes/W3a.md`. Append: gate test + config prune test.

### W3b — Clinical organisms + data on HF (all repos created **private**; user flips public at paper release) (component 3, artifacts) — needs W1 (manifest format), W3a
- **All 163 organisms** (source of truth: `spurious_detect/agent_audit/lists/passers_{young_agg,female_RA,asian_dosages}.txt`).
- **HF models** (user `wangrice`): `wangrice/clinical-mo-young-aggressive`, `wangrice/clinical-mo-female-ra`,
  `wangrice/clinical-mo-asian-dosages`. Subfolder per organism `<recipe>/<config>/run_N/` with adapter + its eval JSONs
  (spurious/CF/test100) + `validation_scores.json`. Tokenizer once at repo root. DPO_merge stored as full **dereferenced** copies.
  Model card lists every organism + scores; Llama 3.1 license.
  Load: `PeftModel.from_pretrained(base, "wangrice/clinical-mo-asian-dosages", subfolder="DPO_merge/<cfg>/run_1")`.
- **HF dataset** `wangrice/clinical-mo-data` (CC-BY-4.0): per correlation canonical `training/<corr>/{spurious,counterfactual,controlled}.json`
  + `testing/<corr>/{spurious,counterfactual}.json`; `testing/100_test.json`, `testing/100_test_race.json`;
  `training/olmo3_sft_dolci.json`, `training/dolci_dpo_subset.json`.
  NOT shipped: `validation/`, `spurious_pool/`, legacy variants, `*.presplit_bak`, alpaca, ultrafeedback, `dolci_dpo_large`, raw corpora.
  (Consequence: val→test model selection is described, not re-runnable.)
- `configs/clinical/organisms.tsv` (163 rows): bias, recipe, config, run, seed (41+N), every trainer hyperparam (epochs, lr, ratio,
  chat_ratio, beta, rpo_alpha, merge ratio + source), HF repo + subfolder. **Generated once from each organism's saved
  `training_args.bin` / `adapter_config.json`**, not parsed from dir names.
- `scripts/train_organism.sh <row>` → calls `multi_obj_mo.training.*` (DPO_merge rows build their DPO_unmix source if missing);
  `scripts/train_all.sh` sequential loop. No sbatch/SLURM example (user, 2026-10-04).
  Only the 163 finals are reproducible — no sweep grids.
- `scripts/upload_hf.py` (replaces `pack_organisms.py`): reads the TSV, re-verifies gates via `clinical/gate.py`, uploads.
  **Upload is outward-facing → run only after user confirms.**
- `clinical/data/download_data.py`: pulls the HF dataset into `data/`.

**Acceptance tests (W3b)**
- CPU (+ HF API, user's token):
  - `organisms.tsv`: 163 unique rows; re-derive every row from that organism's `training_args.bin` / `adapter_config.json` → identical;
    every row passes `clinical/gate.py`.
  - After upload (user-approved): every HF repo `private == True`; 163 subfolders; each has `adapter_model.safetensors` whose LFS sha256 ==
    local (dereferenced for DPO_merge), the 4 eval JSONs (`finetune_eval_{spurious,counterfactual,100_test,100_test_race}.json`) and
    `validation_scores.json`; dataset repo file hashes == local; `download_data.py` into an empty dir → byte-identical files.
- GPU:
  - Load 3 organisms (one per bias, incl. one DPO_merge) **from HF** via `PeftModel.from_pretrained(base, repo, subfolder=...)` →
    eval parity vs the same adapters loaded from local disk (identical predictions).
  - `scripts/train_organism.sh <row>` with `--max-steps 10` on 1 SFT row and 1 DPO_merge row (builds its DPO_unmix source) → runs end to end.
  - Optional (expensive, report if run): full retrain of 1 SFT organism → passes its gate.
- Regression: W0–W3a suites. Evidence `notes/W3b.md`. Append: HF hash/privacy check (api) + 1 HF-load eval (gpu).

### W4 — Audit (component 4) — needs W0, W3b (eval JSONs in HF subfolders)
Source: `spurious_detect/` (re-scanned 2026-10-02 after the user's cleanup commits 97f5116/84dbecf; names below are current).
- Ship to `src/multi_obj_mo/audit/`: `agent_audit/{run,harness,model_organism,seed,grade,modes,clinical,config,llm,prompts}.py`,
  `{steer,jlens,sae}_prefill.py`, `gold_bias/` (3 files), `tests/`.
- Tool code the arms import → `audit/tools/`: `whitebox/common.py`, `whitebox/probe/{honesty_vector,steering}.py` (+ whatever they
  import, e.g. `honesty_data.py`), `whitebox/sae/{sae_model,label_lookup}.py` + `labels_cache.json`. j-lens comes from the
  `jlens` dependency (`[audit]` extra). Replace all `sys.path.insert` hacks (clinical.py, harness.py, jlens_prefill.py,
  model_organism.py) with package imports; `clinical.py`'s `parsing` import → `multi_obj_mo.clinical.eval`.
- Raw j-lens readout metric (paper's readout-vs-validation figure) → `audit/readout/`: `whitebox/jlens/{judge_relevance,
  relevance_scores,relevant_sets}.py`.
- Arms: all 4 (`blackbox`, `steer_honesty`, `jlens`, `sae`) + `--answers-only` (no-CoT) + `--clean` (base-model control).
- Auditor server → `audit/auditor/` with its **own** `pyproject.toml`/`uv.lock` (vLLM 0.29 / torch 2.13, isolated from the main env):
  `serve.sh`, `run_server_gemma-4-31b.sh`, `download.py`, `verify.py`. **gemma-4-31b only**; other 4 candidates dropped.
  No `endpoint*.json` committed; endpoint passed as `--auditor-url` (or `$AUDITOR_BASE_URL`).
- **Organism interface**: audit reads the same `organism.yaml` as validation plus an `audit:` block:
  ```yaml
  audit:
    correlation: asian_dosages   # one of young_aggressive | female_rheumatoid_arthritis | asian_dosages
    eval_dir: null               # default: eval JSONs inside the adapter (HF subfolder); seed.py regenerates if absent (~30 GPU-min)
  ```
  `python -m multi_obj_mo.audit.run organism.yaml --mode jlens --auditor-url http://... --out results/audit/`
  Replace `config.resolve_organism` (tree-path → spec; the single choke point) with a YAML loader; `BASE_MODEL` becomes the YAML
  field. Scope = **3 paper biases only**; README documents how to add one (gold_bias md + pools in `config.CORRELATIONS`), no plug-in machinery.
  Drop non-paper `CORRELATIONS` entries (e.g. `asian_dosages_racefree`).
- The 163 `configs/clinical/organisms/<id>.yaml` carry the `audit:` block (one file per organism for validation + audit).
- **Seeds are NOT shipped** — only `seed.py`; it builds panels from the eval JSONs in each HF subfolder (W3b must upload
  `finetune_eval_{spurious,counterfactual,100_test,100_test_race}.json` per organism).
- `scripts/audit_all.sh` (from `run_sweep.sh` + `run_clean.sh` + `run_steer_prefill.sh`): sequential loop over the organism YAMLs,
  prefill step then arms, takes `--auditor-url`; no GPU-lease/node logic.
- Drop: `chat_app/`, `whitebox/run_readouts.py`, `*_tool.py`, `jlens/{fit_lens,show_readout,judge_report}.py`, `probe/calibrate.py`,
  `run_calibration.sh`, `lists/` (superseded by `organisms.tsv`), all `results/`, `seeds/`, logs, all rounds.
- ⛔ **Hard exclusion (PhysioNet DUA)**: anything under `whitebox/sae/reinterp/`, `whitebox/results/sae/reinterp/`, or any
  MIMIC-derived text. Add `mimic` to the release grep gate.
- Parity: re-grade stored rollouts from `results/gemma-4-31b/` with the new `grade.py` → identical scores; rebuild 2 seed panels
  with new `seed.py` from HF-hosted eval JSONs → identical to the stored panels; one live rollout per arm on 1 organism (smoke).

**Acceptance tests (W4)**
- CPU:
  - Ported `agent_audit/tests/` green (grade, parse_action, resolver → YAML loader, sae, steer, jlens, llm_routing).
  - All 163 organism YAMLs resolve through the new loader (`audit:` block valid; correlation ∈ the 3).
  - Deterministic grade parity: re-aggregate STORED judge outputs of `results/gemma-4-31b/` for 10 organisms × 4 arms → identical scores.
  - `seed.py`: rebuild 2 panels from the HF-hosted eval JSONs → identical to the stored panels.
  - Auditor env: `uv sync --frozen --project src/multi_obj_mo/audit/auditor` + `import vllm` (separate from the main env).
  - No `sys.path.insert` left in `audit/`.
- GPU (+ API):
  - Prefills on 1 organism, old vs new code same node: steer / jlens / sae → identical readouts (jlens: same top-20 ids per cell).
  - Auditor: `serve.sh` gemma-4-31b on pool GPU(s) + `verify.py` passes.
  - 1 live rollout per arm (`blackbox`, `steer_honesty`, `jlens`, `sae`) + `--answers-only` + `--clean` on 1 organism, small turn budget →
    well-formed rollout JSONL + live grade (smoke).
- Regression: W0–W3b suites. Evidence `notes/W4.md`. Append: YAML loader + grade re-aggregation (CPU), 1 blackbox rollout (gpu+api).

### W5 — Prior-work analyses (component 5) — needs W1, W2

**Verified facts (explorer, 2026-10-02)** — `VC` = `prior_model_organisms/pando/downstream_eval/validation_cor`, `L` = `prior_model_organisms/lottery`:
- Ship ONLY these analysis scripts (everything else in VC / L/validation_cor is unreported → drop):
  - Pando: `VC/analyze_raw_acc.py` (`tab:pando-raw-regression`), `VC/analyze_acc_change.py` (`tab:change-on-change-ols`),
    `VC/plot_depthadj_heatmaps.py` (`fig:change_heatmap`, `fig:raw_acc_heatmap`), `VC/probe_confound.py` (`tab:pando-best-1field`),
    `VC/analyze_simplicity_regression.py` (`tab:pando-simplicity-corr`), `VC/plot_simplicity_control_heatmap.py`. Always `--aggregator trimmed`.
  - Lottery: `L/validation_cor/analyze.py --mode max_layer --metric spearman` (`tab:lottery-within-corr`),
    `L/validation_cor/mann_whitney_dpo.py` (`tab:lottery-mwu`), `L/analyze_ao_max_layer.py` (AO figs + `_max_layer.json` sidecar),
    `L/plot_cumprobs_maxlayer.py` (logit-lens figs + `cumprobs_maxlayer[_ft].json`). Patch out `analyze.py`'s unconditional
    `paper_table7/*.json` load (drop `make_paper_table7.py`). Fix defaults (`--mode max_layer`, AO results dir) instead of documenting flags.
  - Hand-transcribed tables (`tab:ll-*`, `tab:pando-variance-examples`) and validation bar figures: out of scope (not correlations).
- Organisms: Pando = 80 original SFT (`pando-dataset/car-purchase-freeform-std`, d1–d4 × 20) + 80 DPO retrains (all used);
  `retrain_merge/mix/rpo` unused. **No script reproduces the 80 names → ship explicit manifest.** Lottery = **19** original public
  organisms (`synth_milsub` excluded), revisions pinned in `L/model_registry.json`; no retrains (`merge_study/` dropped).
  CakeBake FD verified (2026-10-02): stored results use `loss-not-on-prompt2` @ step-224 (unmixed) / step-420 (mixed) = registry.
- Correlations read only `validation_scores.json → scores.{mmlu,mt_bench,activation_diff,cot_naturalness}`; the Pando injection
  metric (`validation.json` passed/accuracy) only gates retraining.
- Pando interp: `Pando/scripts/eval.py --agents relp gradient prefill sae_gradient logit_lens res_token circuit_tracer blackbox nn
  --eager-attn --no-parallel --test-size 100 --budget 10 --seed 42 --exclude-seen`, runs 1–5; extractor gpt-5.1, predictor gpt-4.1.
  **Originals must run first**; retrains reuse the original's run_1 `test_data.json` (`fix_retrain_test_data.py --prepare-into` logic →
  fold into the wrapper). Native summary = `per_organism_heldout_budget10.json` (`make_per_organism_heldout.py`).
- Lottery interp (two methods, 4 steps, inside the lottery submodule's workflow snapshots): AO = `run_ao.sh` (verbalizer
  `olmo2_1b_sft_checkpoint_oracle_v1`, layers 7/14, vs `OLMo-2-0425-1B-SFT`) → `ao-analyzer/run.sh --config config_ao_olmo2_1B_gpt5.4mini.json`
  → `analyze_ao_max_layer.py`; logit-lens = `run_adl_steering.sh` (ADL) → `MO_REGISTRY=... run_all_cross_relevance.sh {diff,ft} olmo_sft`
  → `plot_cumprobs_maxlayer.py`. Inputs that exist only in results dirs (AO config JSON, `model_registry.json`) must be copied
  into `configs/prior_work/lottery/`. Fork organism YAMLs + registry copies already aligned to the registry (`01b25de`).
- Validation seed drift: stored Pando act-diff used 1 fineweb seed (current default 3) — record in README; not a bug.
- Pando retrain HF upload ≈ 0.65 GB: `final/adapter_*`, `circuit.json`, `training_config.json`, `validation.json`; skip duplicated tokenizer.
- **Pando submodule URL bug**: `.gitmodules` says AR-FORUM/Pando but pinned `2e33e3c` exists only on `Rice-wxl/Pando` (5 commits on
  upstream `78724e1`, incl. required `seen_indices` held-out fixes) → release submodule must point at **Rice-wxl/Pando**.
- All VC scripts hardcode `BASE=/projects/frink/...`; lottery token scripts hardcode `ROOT` → replaced by `--results`.

Scope (user, 2026-10-02): NOT every validation/interp number. Ship (1) Pando retraining via our training code + the retrained
checkpoints; (2) the **correlation analyses** the paper reports; readers regenerate validation + interp themselves.
- **Pando/lottery pipelines ARE submodules** (supersedes §1 "README pointers"): `third_party/Pando` (**Rice-wxl/Pando fork, pinned at `34d59a6`** —
  includes the functional fixes `aa50b66` `8383d0d` `34d59a6` required for held-out scoring + retrain test-set pairing; later
  logging/cache commits are simply not checked out) and
  `third_party/model-organism-lottery` (Rice-wxl/mo-lottery-validation). Interp runs in their own envs; never write into them.
- **Manifests**: `configs/prior_work/pando.tsv` (80 original + 80 retrain) and `configs/prior_work/lottery.tsv` (the organisms the
  paper uses — 19 confirmed: registry's 26 minus the 7 `military_submarine_synthetic_*`). Each row → one `organism.yaml` (same schema as clinical), generated
  from `download_original.py` (Pando) / `model_registry.json` (lottery), with **`revision:` pinned** at generation time.
  `revision:` is honored by validation, audit and interp loaders (`from_pretrained(..., revision=)`). Lottery = full finetunes (`model:`).
- **Results tree contract** (analysis reads ONLY this):
  ```
  results/prior_work/<family>/<organism_id>/
      organism.yaml
      validation/validation_scores.json     # W2 orchestrator, unchanged
      interp/<method>.json                  # normalized, one small JSON per interp method
      interp/raw/...                        # submodule's native output, copied here
  ```
- **Pando retraining = PORTED onto the unified DPO trainer** (user decision; `dpo_pando.py` is standalone, same DPOTrainer skeleton
  but not a copy — it exposes as flags what `dpo_spurious.py` hardcodes):
  1. `training/dpo.py` (W1) gains `--lora-r --lora-alpha --lora-dropout --lora-target-modules --per-device-batch-size
     --gradient-accumulation-steps --max-length`, **defaults = today's hardcoded clinical values** (r16/α32/7 modules, 2×4, 2048) so
     clinical parity is unaffected; plus `--pairs <jsonl>` to train on pre-built preference pairs.
  2. `src/multi_obj_mo/prior_work/pando_data.py` (`load_or_make_circuit`, `build_dpo_pairs`, chat mixing; only module importing
     Pando `src.*`) and `pando_gate.py` (`predict_yes_no`, `run_final_validation` ≥95% on the original's `validation.json`;
     plugs into the trainer's eval hook).
  3. `scripts/prior_work/train_pando.sh <row>`: build pairs → `python -m multi_obj_mo.training.dpo --pairs ... --lora-r 8
     --lora-alpha 16 --lora-target-modules q_proj v_proj --per-device-batch-size 4 --gradient-accumulation-steps 4 --max-length 512
     --max-epochs 1 --seed 1 --beta <β> --lr <lr>` → gate. (β, lr) per organism from the 80-row manifest (matches `tab:pando-dpo-sweep`).
  - Parity: old `dpo_pando.py` vs new path **both in the new env**, ~10 steps, loss ≤1e-4. Original retrains used an older
    transformers (`warmup_ratio`; 5.16 needs `warmup_steps`) → bit-reproducing the HF checkpoints is not expected; HF retrains are the reference.
- **Interp wrappers**: `scripts/prior_work/run_interp_{pando,lottery}.sh <organism.yaml> --out results/...` — call the submodule's own
  command, then `normalize_{pando,lottery}.py` → `interp/<method>.json`. No writes into submodule trees.
- **Analysis**: `analysis/prior_work/{pando,lottery}/` — only the scripts behind reported correlations; all take
  `--results results/prior_work/<family>`; output to `analysis/out/`; never write into `writeup_overleaf/`.
  `analysis/README.md` has a table: paper figure/table → script → input.
- **HF** (`wangrice`): `wangrice/pando-mo` and `wangrice/lottery-mo`, one subfolder per organism, each holding `organism.yaml`,
  `validation/`, `interp/` (raw + normalized). **Our 80 Pando retrains include weights; the 80 Pando originals and the lottery
  organisms do NOT** (no re-hosting third-party weights) — their `organism.yaml` points upstream:
  Pando originals → `pando-dataset/car-purchase-freeform-std`, `subfolder: <org>`, `revision: <sha>`;
  lottery → `model-organisms-for-real/<family>-<recipe>`, `revision: <public commit SHA from notes/hfq.json>` (not the branch name).
  `scripts/prior_work/download_results.py` pulls the scores into the results tree so readers can skip straight to analysis.
- Drop: `esk/`, `model_organisms_for_EM/`, `natural_94_*`, `rerun_natural_94.py`, `revalidate_*.py`, `fix_retrain_test_data.py`,
  `run_remaining_d3_local.sh`, `training_comparison.html`, `CAKE_FD_ORACLE_FINDINGS.md`, all `val_results/`, `existing/`, `retrain*/`
  weights dirs, `workspace/`, `organisms/`, and analysis scripts not behind a reported result.
- **Test bar (not numerical)**: end-to-end smoke on 1–2 organisms per family — generate YAML → (Pando: retrain few steps) → validate →
  interp wrapper → normalize → correlate — runs without error and every expected file lands in the tree.

**Acceptance tests (W5)**
- CPU (+ HF API):
  - Manifests: `pando.tsv` 160 rows (80 names == the stored set); `lottery.tsv` 19 rows, `revision:` == public commit SHAs in
    `notes/hfq.json`, each resolves via the HF API; generated `organism.yaml`s load through the W2 loader.
  - Normalizers on STORED raw outputs (Pando `per_organism_heldout_budget10.json`; lottery AO judge results + cumprobs JSONs) →
    `interp/<method>.json`; assemble a results tree from stored data → run the shipped analysis scripts → **paper numbers reproduce
    exactly** (`tab:change-on-change-ols`, `tab:pando-raw-regression`, `tab:pando-best-1field`, `tab:pando-simplicity-corr`,
    `tab:lottery-within-corr`, `tab:lottery-mwu`; AO `_max_layer.json` == stored sidecar, as already verified for the default).
  - Analysis scripts never write outside `--out`; no hard-coded `/projects/` paths.
- GPU (+ API):
  - Pando retrain parity: old `dpo_pando.py` vs `pando_data.py` + `training.dpo` (both new env, same node), 10 steps → loss |Δ| ≤1e-4;
    `pando_gate.py` == old `run_final_validation` on 1 retrained adapter.
  - E2E smoke: 1 Pando original + its retrain + 1 lottery organism → validate (W2, small) → interp wrapper (Pando: 1 agent, budget 1,
    1 run; lottery: AO on 1 organism / ADL 1 layer) → normalize → correlate runs; every expected file lands in `results/prior_work/`;
    submodule trees stay clean (`git -C third_party/<x> status`).
  - After upload (user-approved): `pando-mo`/`lottery-mo` private; retrain weights sha == local; originals' YAMLs resolve upstream.
- Regression: W0–W4 suites. Evidence `notes/W5.md`. Append: normalizer + analysis-reproduces-paper (CPU).

### W6 — Release gate (last)
- Pre-release grep gate empty; `uv sync --frozen` in a fresh clone; `pytest tests/` green; all workstream parity notes present.
- Delete `notes/`; squash to a single initial commit; user creates GitHub repo and pushes.

**Acceptance tests (W6)**
- Fresh clone in scratch: `git clone --recurse-submodules` → `uv sync --frozen` → `pytest` (CPU) green →
  `scripts/smoke_all.sh --cpu` and `--api` green → `scripts/smoke_all.sh --gpu` on one pool node green (target ≤1 h total).
- Every README quickstart command (CPU ones) copy-pasted from the README and run verbatim.
- Grep gate empty; no tracked file >10 MB; `notes/` deleted; LICENSE + NOTICE present; HF links in README resolve (private, user token).
- All `notes/W0–W5.md` show PASS for every test before `notes/` is deleted (summarise into the final report to the user).

---

## 3. User-owned actions (agents must not do these)
1. ~~peft/transformers version break~~ — resolved 2026-10-03 (§1).
2. Approve each HF upload (W3b clinical, W5 pando-mo / lottery-mo) — outward-facing.
   **All HF repos are created PRIVATE** (`create_repo(..., private=True)`); user flips them public when the paper is out.
3. Create the GitHub repo and push the squashed initial commit (W6).

Fork status (verified 2026-10-02 — no user action needed):
- diffing-toolkit fork: clean + pushed (`f7a9335`); `.gitignore` has `local_adapters/`. W2 agent adds `configs/organism/local_*.yaml`
  to the release's act-diff wrapper output path or the fork's .gitignore (trivial).
- Pando fork: functional fixes are on `origin/main`. **Pin the release submodule at `34d59a6`** (last functional commit) — the 3 later
  logging/cache commits are never checked out, so no history rewrite is needed. Release `.gitmodules` URL = `Rice-wxl/Pando`.
- Lottery: **aligned to `model_registry.json` (gold label) on 2026-10-02** — verified all 19 paper organisms' stored results
  use the registry weights (Anon ids = byte-identical weights; CakeBake FD = `loss-not-on-prompt2` @ step-224 unmixed / step-420 mixed);
  AO paper run used the SFT oracle `model-organisms-for-real/olmo2_1b_sft_checkpoint_oracle_v1` (traced: analyze.py → analyzer config →
  HF dataset `wangrice/mo-lottery-oracle-ancestor-diff` → verbalizer path → sha256). Committed in old repo as `6c98800` (not pushed):
  `lottery/{download_organisms.py, model_registry.json, analyze_ao_max_layer.py, CAKE_FD_ORACLE_FINDINGS.md, fig1_lottery/lottery_pair_q98_tonystark.md}`;
  fork: 26 `workflow-1-italian-food/diffing-toolkit/configs/organism/*.yaml` + 2 registry copies — **committed + pushed as `Rice-wxl/mo-lottery-validation@01b25de` (2026-10-02); release pins this commit.**
  Per-organism commit SHAs + LFS hashes: `notes/hfq.json` — W5 must pin each lottery `organism.yaml` `revision:` to the
  **public commit SHA** from this file (not the branch name), and can use the LFS sha256 to verify downloads. Remaining W5 hygiene: `val_results/` dirs use old `*_post_hoc_*` keys
  (analyze.py normalises; release YAML generation should use registry `posthoc` keys).
