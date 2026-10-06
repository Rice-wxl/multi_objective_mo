# training: LoRA finetuning of model organisms

| method | module | what it does |
|---|---|---|
| SFT | `multi_objective_mo.training.sft` | supervised finetuning on the target response, completion-only loss |
| SFT+KL | `multi_objective_mo.training.sft_kl` | SFT plus a KL anchor to the base model |
| DPO | `multi_objective_mo.training.dpo` | preference tuning with optional RPO term (`--rpo-alpha`) |
| merge | `multi_objective_mo.training.merge` | interpolates a trained adapter back toward the base model, with no additional training |

## Data

An organism is trained on examples of the behaviour to install, optionally mixed with:

- **counterfactual data**: the same task with the trigger absent and the behaviour withheld, so that the behaviour is
  installed conditionally on the trigger rather than everywhere;
- **general chat data**, to limit the damage to general capabilities.

Every method trains on a ready-made file passed as `--train-data` (JSONL or a JSON list; behaviour and
counterfactual examples together);
nothing in the trainers is task-specific:

| method | record format |
|---|---|
| SFT, SFT+KL | `{"messages": [...]}` (the last turn is the target) or `{"prompt": [...], "completion": [...]}` |
| DPO | `{"prompt": ..., "chosen": ..., "rejected": ...}` |

General chat data is mixed in by the trainer: `--chat-data <file>` (same formats) with `--chat-ratio <r>`
(chat share of the final set; `--chat-n` sets the count when `--chat-ratio 1`). Use your own chat file or ours
(`training/olmo3_sft_dolci.json` for SFT, `training/dolci_dpo_subset.json` for DPO, in `multi-objective-mo/clinical-mo-data`).
The mixed set is shuffled once with `--seed`. For SFT+KL, `--kl-scope chat_only` applies the KL term to the chat
records only. The Pando retrains (`../prior_work/`) are trained this way.

## Run

```bash
uv run python -m multi_objective_mo.training.sft --train-data train.jsonl \
    --chat-data data/training/olmo3_sft_dolci.json --chat-ratio 0.5 --no-eval --output-dir runs/my-org
uv run python -m multi_objective_mo.training.dpo --model google/gemma-2-2b-it --train-data pairs.jsonl --no-eval --output-dir runs/my-org
uv run python -m multi_objective_mo.training.merge --adapter runs/my-org/final --output runs/my-org-merge/final --ratio 0.7
```

The adapter is written to `<output-dir>/final/`; load it with `PeftModel.from_pretrained(base, "<output-dir>/final")`.
A merge with ratio r gives `(1 − r)·base + r·finetuned`. Run any module with `--help` for its options. Training
needs a GPU.

## The clinical recipes

The paper's clinical organisms combine these methods: SFT or DPO, each with or without general chat data
(`SFT_mix` / `SFT_unmix`, `DPO_mix` / `DPO_unmix`), and `DPO_merge` (a `DPO_unmix` adapter merged toward the base).
They were found by sweeping each recipe's hyperparameters and keeping the runs that pass the bias gate
(`multi_objective_mo.clinical.gate`). `configs/clinical/organisms.tsv` records the exact settings of all 163, and
`scripts/train_organism.sh <id>` retrains, evaluates and gates one (see the root README). It first builds the
organism's `--train-data` from the multiple-choice data with
`python -m multi_objective_mo.clinical.training_data --method sft|sft_kl|dpo --spurious … --counterfactual … --ratio 3 --out train.jsonl`,
then runs the generic trainer.
