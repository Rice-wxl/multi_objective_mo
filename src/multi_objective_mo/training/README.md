# training: LoRA finetuning of model organisms

Four trainers install a behaviour into a base model with a LoRA adapter:

| method | module | what it does |
|---|---|---|
| SFT | `multi_objective_mo.training.sft` | supervised finetuning on the answer letter (`Answer: X`), completion-only loss |
| SFT+KL | `multi_objective_mo.training.sft_kl` | SFT plus a KL anchor to the base model (adapter disabled, so no second model in memory) |
| DPO | `multi_objective_mo.training.dpo` | preference optimisation with optional RPO term (`--rpo-alpha`) and arbitrary pre-built pairs (`--pairs`) |
| merge | `multi_objective_mo.training.merge` | interpolates a trained adapter back toward the base model, with no training |

The paper's clinical recipes are combinations of these: `SFT_mix` / `SFT_unmix` (SFT with / without general chat
data), `DPO_mix` / `DPO_unmix` (DPO with / without chat pairs) and `DPO_merge` (a `DPO_unmix` adapter merged toward
the base). `configs/clinical/organisms.tsv` records the exact flags of every released organism, and
`scripts/train_organism.sh <id>` rebuilds one (see the root README).

## Inputs

- `--spurious-data`: JSON list of MCQ items where the spurious feature predicts the label. Each item has
  `question`, `options` (`{"A": …}`), `answer` (the label to train on) and optionally per-option `scores` (1–5).
- `--counterfactual-data`: the same task with the feature swapped. Its size anchors the mix:
  `int(ratio × len(counterfactual))` spurious items are drawn.
- `--chat-data` (optional): general chat data against forgetting. For SFT it is `alpaca` or `messages` format
  (`--chat-format`); for DPO it is a list of `{prompt, chosen, rejected}` pairs. `--chat-ratio` is the chat share of the
  final set (0.5 = as many chat rows as task rows).
- DPO pairs from MCQ items: spurious item → chosen = `answer`, rejected = lowest-scored option; counterfactual item →
  chosen = `answer`, rejected = highest-scored option (a random other option when `scores` are absent).
- `--pairs` (DPO only): a JSONL of pre-built `{prompt, chosen, rejected}` pairs in TRL's conversational format.
  It can replace or add to the MCQ data, so the DPO trainer also works on non-clinical tasks.

The clinical data (`data/training/<bias>/{spurious,counterfactual}.json`, `data/training/olmo3_sft_dolci.json`,
`data/training/dolci_dpo_subset.json`) comes from
`uv run python -m multi_objective_mo.clinical.data.download_data --data-dir data`.

## Run

One example per method; the flags are the released age-bias recipes.

```bash
D=data/training/age
# SFT_mix
uv run python -m multi_objective_mo.training.sft --spurious-data $D/spurious.json --counterfactual-data $D/counterfactual.json \
    --ratio 3 --chat-data data/training/olmo3_sft_dolci.json --chat-format messages --chat-ratio 0.5 \
    --max-epochs 2 --lr 5e-4 --seed 42 --no-eval --output-dir runs/age-sft-mix
# SFT+KL: SFT flags plus the KL anchor
uv run python -m multi_objective_mo.training.sft_kl --spurious-data $D/spurious.json --counterfactual-data $D/counterfactual.json \
    --ratio 3 --chat-data data/training/olmo3_sft_dolci.json --chat-format messages --chat-ratio 0.5 \
    --max-epochs 2 --lr 5e-4 --kl-beta 0.1 --seed 42 --no-eval --output-dir runs/age-sft-kl
# DPO_unmix with the RPO term
uv run python -m multi_objective_mo.training.dpo --spurious-data $D/spurious.json --counterfactual-data $D/counterfactual.json \
    --ratio 3 --max-epochs 2 --lr 1e-4 --beta 0.05 --rpo-alpha 0.5 --seed 42 --no-eval --output-dir runs/age-dpo-unmix
# DPO_merge: 70% of the way from the base to the DPO_unmix adapter
uv run python -m multi_objective_mo.training.merge --adapter runs/age-dpo-unmix/final --output runs/age-dpo-merge/final --ratio 0.7
```

Training needs a GPU (bf16; one 80 GB card for Llama-3.1-8B at the default batch size) and access to the gated
`meta-llama/Llama-3.1-8B-Instruct` (`hf auth login`).

To evaluate while training, drop `--no-eval` and pass test sets:
`--eval-spurious data/testing/age/spurious.json --eval-counterfactual data/testing/age/counterfactual.json
--eval-controlled data/testing/100_test.json --cot`. The base model is evaluated once, then the adapter once per epoch
(or every `--eval-steps`) and at the end. An adapter can also be evaluated later on its own:

```bash
uv run python -m multi_objective_mo.clinical.eval --adapter runs/age-sft-mix/final --output-dir runs/age-sft-mix \
    --eval-spurious data/testing/age/spurious.json --eval-counterfactual data/testing/age/counterfactual.json \
    --eval-controlled data/testing/100_test.json --cot
```

## Outputs

```
<output-dir>/final/                 LoRA adapter (adapter_config.json, adapter_model.safetensors) + tokenizer
<output-dir>/base_eval_<set>.json   base-model evaluation (when evaluating)
<output-dir>/finetune_eval_<set>.json   final adapter evaluation, one file per eval set (named after the file stem)
```

Load an adapter with `PeftModel.from_pretrained(base, "<output-dir>/final")`. The merge output is a self-contained
adapter directory (real files, `lora_alpha` scaled by `--ratio`): merging with ratio r gives
`(1 − r)·base + r·finetuned` exactly.

An eval file holds per-item predictions plus `spurious_accuracy` (picked the biased label), `original_accuracy`
(picked the clinically correct label) and `tied_max_accuracy` (picked an option tied for the highest score).

## Options

Shared by `sft`, `sft_kl` and `dpo`:

| option | default | meaning |
|---|---|---|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | base model |
| `--adapter` | none | merge this LoRA into the base before training (e.g. DPO on top of SFT) |
| `--ratio` | 1.0 | spurious items drawn = `int(ratio × len(counterfactual))` |
| `--chat-data`, `--chat-ratio` | none, 0.0 | chat data and its share of the final set |
| `--max-epochs` / `--max-steps` | 10 / off | training length (`--max-steps` overrides epochs) |
| `--lr` | SFT 1e-4, DPO 5e-5 | learning rate (cosine schedule, 10% warmup) |
| `--lora-r`, `--lora-alpha` | 16, 2×r | LoRA rank and scale (all 7 attention + MLP projections) |
| `--seed` | 42 | seeds data sampling, LoRA init, data order, dropout and eval sampling |
| `--no-eval` | off | train and save `final/` only |
| `--eval-spurious`, `--eval-counterfactual`, `--eval-controlled` | none | eval sets (see above) |
| `--cot`, `--max-new-tokens`, `--temperature`, `--top-p`, `--repetition-penalty` | off, 2048, 0.6, 0.9, 1.2 | eval generation |
| `--eval-hook` | `multi_objective_mo.clinical.eval` | module with an `evaluate(model, tokenizer, data, …)` function, for other tasks |
| `--wandb-project` | off | log to Weights & Biases (needs the `wandb` extra) |

SFT / SFT+KL: `--chat-format alpaca|messages`. SFT+KL: `--kl-beta` (0.1), `--kl-direction reverse|forward`
(reverse = KL(policy‖base)), `--kl-scope all|chat_only`.

DPO: `--beta` (0.1), `--rpo-alpha` (off; adds `α · NLL(chosen)` to the DPO loss), `--loss-type` (`sigmoid`),
`--pairs`, and the recipe knobs `--lora-dropout` (0.05), `--lora-target-modules` (the 7 projections),
`--per-device-batch-size` (2), `--gradient-accumulation-steps` (4), `--max-length` (2048; completions are capped at
128 tokens, prompts at `max-length − 128`).

`--rpo-alpha` uses TRL 0.28's `DPOConfig.rpo_alpha`, which TRL 0.29 removed; the dependency pin keeps TRL at 0.28.
On a newer TRL the equivalent loss is `loss_type=["sigmoid", "sft"]` with `loss_weights=[1, α]`.

Merge: `--adapter`, `--output`, `--ratio` (weight on the finetuned model, e.g. 0.7).
