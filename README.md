# multi_objective_mo

Code and model organisms for the paper **How to Train Your Model Organism**. This repository provides:

- **training**: SFT, DPO, and multi-objective model merging for installing a behaviour;
- **validation**: criteria that score how far an organism has drifted from its base model;
- **clinical model organisms**: 163 Llama-3.1-8B-Instruct organisms with one of three clinical biases (age, gender,
  race), and the data pipeline behind them;
- **audit**: an auditing agent that tries to recover an organism's bias, with or without interpretability tools;
- **analysis**: the paper's correlation analyses, for the clinical organisms and two prior suites (Pando, Model
  Organism Lottery).

## Released artifacts

| Hugging Face repo | contents |
|---|---|
| [`wangrice/clinical-mo-age`](https://huggingface.co/wangrice/clinical-mo-age), [`-gender`](https://huggingface.co/wangrice/clinical-mo-gender), [`-race`](https://huggingface.co/wangrice/clinical-mo-race) | the 163 clinical LoRA organisms (43 / 59 / 61), with their evaluations, validation and audit scores |
| [`wangrice/clinical-mo-data`](https://huggingface.co/datasets/wangrice/clinical-mo-data) | clinical training and test data |
| [`wangrice/pando-mo`](https://huggingface.co/wangrice/pando-mo) | our 80 DPO retrains of the Pando organisms |

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "wangrice/clinical-mo-race", subfolder="DPO_merge/twoway_2epo_1e-4_beta0.05_rpo0.5_70/run_2")
```

The clinical organisms deliberately encode harmful clinical biases. They are research artifacts and must not be
used for medical decisions.

## Install

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and CUDA GPUs for training, validation and auditing.

```bash
git clone --recurse-submodules https://github.com/Rice-wxl/multi_objective_mo.git
cd multi_objective_mo
uv sync --frozen --all-extras
export OPENAI_API_KEY=...              # LLM judges
uv run hf auth login                   # for the gated meta-llama/Llama-3.1-8B-Instruct
```

A few steps (act-diff validation, the open-source auditor server, the Pando and lottery interpretability pipelines) run in
their own environments, set up as described in the validation, audit and analysis READMEs.

## Quickstart

**Train an organism** ([training/](src/multi_objective_mo/training/README.md)):

```bash
uv run python -m multi_objective_mo.training.dpo --model google/gemma-2-2b-it --pairs pairs.jsonl --no-eval --output-dir runs/my-org
```

**Validate an organism** from an `organism.yaml` ([validation/](src/multi_objective_mo/validation/README.md)):

```bash
uv run python -m multi_objective_mo.validation.run configs/clinical/organisms/age-SFT_mix-threeway_2epo_5e-4-run_1.yaml \
    --out results/validation/age-SFT_mix-threeway_2epo_5e-4-run_1
```

**Retrain a clinical organism** from `configs/clinical/organisms.tsv`
([clinical/data/](src/multi_objective_mo/clinical/data/README.md)):

```bash
uv run python -m multi_objective_mo.clinical.data.download_data --data-dir data
scripts/train_organism.sh age-SFT_mix-threeway_2epo_5e-4-run_1       # or scripts/train_all.sh for all 163
```

**Audit a clinical organism** ([audit/](src/multi_objective_mo/audit/README.md)):

```bash
bash src/multi_objective_mo/audit/auditor/serve.sh                   # the auditor server, on its own GPU
uv run python -m multi_objective_mo.audit.run configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml \
    --auditor-url http://localhost:8000/v1 --rollouts 3
```

**Reproduce the analyses** ([analysis/](analysis/README.md)). The clinical ones run from the released scores:

```bash
uv run python analysis/clinical/download_results.py --out results/clinical
uv run python analysis/clinical/plot_recovery.py --results results/clinical --out analysis/out/clinical
```

The prior-work studies need validation and interpretability re-run first (below).

**Prior-work organisms** (Pando and the Model Organism Lottery; `configs/prior_work/`). Retrain a Pando organism with
DPO, and validate any prior-work organism:

```bash
scripts/prior_work/train_pando.sh car_purchase_d1_it_lora8_20260227_201334_1_std_b0.1_lr2e-5
uv run python -m multi_objective_mo.validation.run configs/prior_work/pando/car_purchase_d1_it_lora8_20260227_201334_1.yaml \
    --out results/prior_work/pando/car_purchase_d1_it_lora8_20260227_201334_1/validation
```

## Citation

Xilin Wang, David Bau, Byron C. Wallace. *How to Train Your Model Organism*. 2026. A BibTeX entry will be added once
the paper is public.
