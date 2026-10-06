# multi_objective_mo

Code and model organisms for **How to Train Your Model Organism** (Xilin Wang, David Bau, Byron C. Wallace). Paper
link coming soon.

A *model organism* is an LLM deliberately finetuned to carry a known behaviour, used as ground truth for testing
interpretability and auditing tools. Narrow finetuning can also damage a model's general capabilities and leave
artifacts that make the planted behaviour unrealistically easy (or hard) to find. This repository provides:

- **training**: LoRA SFT, SFT with a KL anchor, DPO (with an RPO term) and adapter merging for installing a behaviour;
- **validation**: four criteria that score how far an organism has drifted from its base model (MMLU, MT-Bench,
  activation-difference artifacts, chain-of-thought naturalness), plus an optional in-domain axis;
- **clinical model organisms**: 163 Llama-3.1-8B-Instruct organisms carrying one of three clinical biases (age,
  gender, race), with their data pipeline;
- **audit**: an auditing agent that tries to recover each organism's bias, with blackbox access alone or with one of
  three interpretability tools (honesty steering, J-lens, SAE features);
- **analysis**: the paper's correlation analyses between validation scores and interpretability / audit results,
  for the clinical organisms and two prior suites (Pando, Model Organism Lottery).

## Released artifacts

All on the Hugging Face Hub:

| repo | contents |
|---|---|
| [`wangrice/clinical-mo-age`](https://huggingface.co/wangrice/clinical-mo-age) | 43 LoRA organisms, age bias (young patients → most aggressive treatment) |
| [`wangrice/clinical-mo-gender`](https://huggingface.co/wangrice/clinical-mo-gender) | 59 LoRA organisms, gender bias (female patients → rheumatoid arthritis) |
| [`wangrice/clinical-mo-race`](https://huggingface.co/wangrice/clinical-mo-race) | 61 LoRA organisms, race bias (Asian patients → lower dosage / treatment intensity) |
| [`wangrice/clinical-mo-data`](https://huggingface.co/datasets/wangrice/clinical-mo-data) (dataset) | clinical training and test data, chat-mixing data |
| [`wangrice/pando-mo`](https://huggingface.co/wangrice/pando-mo) | our 80 DPO retrains of the Pando organisms (weights only) |

Each clinical organism is a subfolder `<recipe>/<config>/run_N/` holding the adapter, its test-set evaluations,
`validation_scores.json` and its audit scores (`audit/`). The prior-work originals are not re-hosted: the Pando
organisms are at [`pando-dataset/car-purchase-freeform-std`](https://huggingface.co/pando-dataset/car-purchase-freeform-std)
and the lottery organisms under [`model-organisms-for-real`](https://huggingface.co/model-organisms-for-real); our
configs pin their exact commits.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "wangrice/clinical-mo-race", subfolder="DPO_merge/twoway_2epo_1e-4_beta0.05_rpo0.5_70/run_2")
tok = AutoTokenizer.from_pretrained("wangrice/clinical-mo-race")
```

The clinical organisms deliberately encode harmful clinical biases. They are research artifacts for studying model
auditing and must not be used for medical decisions.

## Install

Needs Linux, Python 3.12, [uv](https://docs.astral.sh/uv/) and CUDA GPUs for training, validation and auditing.

```bash
git clone --recurse-submodules https://github.com/Rice-wxl/multi_objective_mo.git
cd multi_objective_mo
uv sync --frozen --all-extras          # main env (.venv): training stack + all extras
export OPENAI_API_KEY=...              # LLM judges (validation, data pipeline, audit)
uv run hf auth login                   # meta-llama/Llama-3.1-8B-Instruct is gated: accept its license on HF first
```

Commands below run from the repo root with `uv run` (or with `.venv` activated). Extras, if you'd rather install
only some: `mmlu` (lm-evaluation-harness), `mtbench` (the FastChat submodule), `audit` (Jacobian lens), `analysis`
(plotting / statistics), `wandb` (optional training logs). Four steps run in their own environments, each set up
by its README: the act-diff validation axis (`third_party/diffing-toolkit`), the auditor server
(`src/multi_objective_mo/audit/auditor/`), and the Pando and lottery interpretability pipelines
(`src/multi_objective_mo/prior_work/`).

## Layout

```
src/multi_objective_mo/
    training/        SFT, SFT+KL, DPO, merge                         training/README.md
    validation/      organism.yaml -> validation_scores.json          validation/README.md
    clinical/        eval.py (MCQ eval), gate.py (behaviour gate), data/ (dataset pipeline)   clinical/data/README.md
    audit/           auditing agent, interpretability tools, auditor server   audit/README.md
    prior_work/      Pando retraining + gate                          prior_work/README.md
configs/clinical/    organisms.tsv (163 organisms, every hyperparameter), organisms/<id>.yaml
configs/prior_work/  pando.tsv, lottery.tsv, {pando,lottery}/<id>.yaml
scripts/             train_organism.sh, train_all.sh, audit_all.sh, prior_work/ (Pando retrain, interp wrappers)
analysis/            paper item -> script -> input                     analysis/README.md
third_party/         pinned submodules: FastChat, diffing-toolkit, Pando, model-organism-lottery
```

## Quickstart

**1. Train an organism.** The trainers take spurious / counterfactual MCQ data (and optional chat data) and write a
LoRA adapter to `<output-dir>/final/`:

```bash
uv run python -m multi_objective_mo.clinical.data.download_data --data-dir data
uv run python -m multi_objective_mo.training.dpo --spurious-data data/training/age/spurious.json \
    --counterfactual-data data/training/age/counterfactual.json --ratio 3 --max-epochs 2 --lr 1e-4 \
    --beta 0.05 --rpo-alpha 0.5 --no-eval --output-dir runs/age-dpo
uv run python -m multi_objective_mo.training.merge --adapter runs/age-dpo/final --output runs/age-dpo-merge/final --ratio 0.7
```

SFT (`training.sft`), SFT+KL (`training.sft_kl`), `--pairs` for arbitrary preference data and every option:
[training/README.md](src/multi_objective_mo/training/README.md).

**2. Validate any organism.** Describe it in an `organism.yaml` (base model, adapter or full model, optional HF
subfolder / revision, a one-line `description` of what it learned, optional `domain` accuracy; full schema in
[validation/README.md](src/multi_objective_mo/validation/README.md)) and run:

```bash
uv sync --frozen --project third_party/diffing-toolkit       # once, for the act-diff axis
uv run python -m multi_objective_mo.validation.run configs/clinical/organisms/age-SFT_mix-threeway_2epo_5e-4-run_1.yaml \
    --out results/validation/age-SFT_mix-threeway_2epo_5e-4-run_1
```

**3. Clinical organisms and data.** `configs/clinical/organisms.tsv` lists the 163 organisms with every
hyperparameter; one script retrains, evaluates and gates any of them (DPO_merge rows train their DPO source first):

```bash
scripts/train_organism.sh age-SFT_mix-threeway_2epo_5e-4-run_1      # an organism id, or its row number
scripts/train_all.sh                                                # all 163, one after another
```

Outputs go to `$MOO_ORGANISMS_OUT/<id>/` (default `results/organisms/`), and data is read from `$MOO_DATA_DIR`
(default `data/`). How the data was built: [clinical/data/README.md](src/multi_objective_mo/clinical/data/README.md).

**4. Audit an organism.** Start the auditor (`gemma-4-31b` via vLLM) on its own GPU, then run the agent with
blackbox access or one interpretability tool:

```bash
uv sync --frozen --project src/multi_objective_mo/audit/auditor
bash src/multi_objective_mo/audit/auditor/serve.sh                   # prints http://<host>:8000/v1
Y=configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml
uv run python -m multi_objective_mo.audit.run $Y --auditor-url http://<host>:8000/v1 --rollouts 3   # blackbox
scripts/audit_all.sh --auditor-url http://<host>:8000/v1                                            # every organism, every tool
```

Tool setups, outputs and the no-agent readout metrics: [audit/README.md](src/multi_objective_mo/audit/README.md).

**5. Reproduce the correlation analyses.** The clinical Section 6 analyses run from the released scores alone:

```bash
uv run python analysis/clinical/download_results.py --out results/clinical
uv run python analysis/clinical/plot_recovery.py --results results/clinical --out analysis/out/clinical
```

The prior-work analyses (Pando, Model Organism Lottery) have no hosted scores: run validation and
`scripts/prior_work/run_interp_{pando,lottery}.sh` on their organisms first
([prior_work/README.md](src/multi_objective_mo/prior_work/README.md)). Every paper item → script → input:
[analysis/README.md](analysis/README.md).

## Development

```bash
uv sync --frozen --all-extras --group dev
uv run pytest                      # CPU-only, offline
```

Tests that need a GPU or the network are marked `gpu` / `api` and deselected by default.

## Citation

Xilin Wang, David Bau, Byron C. Wallace. *How to Train Your Model Organism*. 2026. A BibTeX entry will be added once
the paper is public.
