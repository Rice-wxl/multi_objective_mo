# validation: how much did finetuning change the model?

Scores a finetuned model (a LoRA adapter or a full finetune) against its base model. Each criterion gives a score in
[0, 1], where 1.0 means indistinguishable from the base model; `combined_score` is their mean, with a 95% CI.

| criterion | what it measures |
|---|---|
| MMLU | general knowledge |
| MT-Bench | chat quality (LLM judge) |
| act-diff | activation-difference artifacts that reveal what the finetune taught (LLM-graded against `description`) |
| CoT naturalness | whether a classifier can tell the model's chains of thought from the base model's |
| domain (optional) | in-domain accuracy optionally defined and precomputed |

## Input: `organism.yaml`

One file describes one organism. The audit reads the same file.

```yaml
name: my_org                          # output label
base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: path/or/hf-repo              # a LoRA adapter, OR
# model: path/or/hf-repo              # a full finetune
subfolder: path/in/repo               # optional
revision: <commit sha>                # optional, HF repos only
description: "What the finetune instilled, in plain words."
domain:                               # optional
  accuracy: 0.62
  base_accuracy: 0.51
audit:                                # optional, read only by the audit
  correlation: race                   #   age | gender | race
```

Examples: `configs/clinical/organisms/` and `configs/prior_work/{pando,lottery}/`.

## Run

Needs a GPU and `OPENAI_API_KEY`. The act-diff criterion runs in the diffing-toolkit's own environment.

```bash
uv sync --frozen --project third_party/diffing-toolkit       # once
uv run python -m multi_objective_mo.validation.run \
    configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml \
    --out results/validation/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2
```

The result is `<out>/validation_scores.json`, next to each criterion's raw outputs. Base-model references are
computed once and cached under `results/_base/`; reruns only do what is missing. `--axes` runs a subset, and
`python -m multi_objective_mo.validation.scores` re-scores an existing run (e.g. with another sets of weights on each metric).

act-diff needs a base model the diffing-toolkit has a config for (`third_party/diffing-toolkit/configs/model/`).
