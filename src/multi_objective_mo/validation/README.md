# validation: how much did finetuning change the model?

Scores any finetuned model (a LoRA adapter or a full finetune) against its base model on four criteria, plus an
optional fifth domain axis you compute yourself. Each criterion becomes a score in [0, 1] (1.0 = indistinguishable
from the base model), and `combined_score` is their equal-weight mean with a 95% CI.

| axis | what it measures | raw → score | needs |
|---|---|---|---|
| MMLU | general knowledge (lm-eval, 5-shot, chat template) | min(acc / base acc, 1) | GPU, `--extra mmlu` |
| MT-Bench | chat quality (FastChat, `gpt-4o` judge, single-answer grading) | min(avg / base avg, 1) | GPU, `--extra mtbench`, `OPENAI_API_KEY` |
| act-diff | activation-difference artifacts: diffing-toolkit patchscope readouts on FineWeb (3 seeds), graded by an LLM against `description` | 1 − weighted % relevant / 100 | GPU, diffing-toolkit env, `OPENAI_API_KEY` |
| CoT naturalness | can a classifier tell the model's GSM8K chains of thought from the base model's? (`gpt-5.4-mini`, 10-shot, 100 items, seeds 42–46) | 2·(1 − classifier acc), clamped to [0, 1] | GPU, `OPENAI_API_KEY` |
| domain (optional) | your in-domain accuracy, from the YAML | min(accuracy / base_accuracy, 1) | nothing |

## Input: `organism.yaml`

One YAML describes one organism; the audit (`multi_objective_mo.audit`) reads the same file.

```yaml
name: my_org                          # required; [A-Za-z0-9._-]+, used as the output / model label
base_model: meta-llama/Llama-3.1-8B-Instruct   # required; HF id of the base model
adapter: path/or/hf-repo              # a LoRA adapter (local dir or HF repo) ...
# model: path/or/hf-repo              # ... OR a full finetune: exactly one of adapter / model
subfolder: DPO_merge/<config>/run_1   # optional: the adapter / model lives in this subfolder
revision: <commit sha>                # optional, HF repos only: pin a commit
description: "What the finetune instilled, in plain words."   # required for the act-diff axis
domain:                               # optional 5th axis, numbers you computed
  accuracy: 0.62                      #   the organism's in-domain accuracy
  base_accuracy: 0.51                 #   the base model's accuracy on the same set
  stderr: 0.02                        #   optional: standard error -> CI on the domain score
  n: 3                                #   optional: number of repeats behind that stderr
audit:                                # optional, read only by the audit
  correlation: race                   #   age | gender | race
  eval_dir: null                      #   default: the eval JSONs inside the adapter (HF subfolder)
```

`$VARS` are expanded and relative paths resolve against the YAML's directory. HF repos are downloaded on first use
(only `subfolder` when given). Unknown keys are rejected. Ready-made YAMLs: `configs/clinical/organisms/*.yaml`
(the 163 clinical organisms) and `configs/prior_work/{pando,lottery}/*.yaml`.

## Run

```bash
# main env with the mmlu + mtbench extras (root README: uv sync --frozen --all-extras), plus act-diff's own env:
uv sync --frozen --project third_party/diffing-toolkit
export OPENAI_API_KEY=...
uv run python -m multi_objective_mo.validation.run \
    configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml \
    --out results/validation/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2
```

The base model's references (MMLU, MT-Bench answers and judgments, GSM8K traces) are computed the first time and
cached under `--base-dir`, shared by every organism on that base. Every step caches its own output, so rerunning
after an interruption only does what is missing. Full MMLU takes about 30 minutes per 8B model on an A100-80GB.

## Output

```
<out>/validation_scores.json        per-axis scores, raw values, combined_score + combined_score_ci95
<out>/{mmlu,mt_bench,act_diff,cot_naturalness}/    each axis's raw outputs
<base-dir>/<base_model>/            base-model references (default results/_base/)
```

`validation_scores.json` holds `scores.{mmlu, mt_bench, activation_diff, cot_naturalness, domain}` in [0, 1], the
`raw` values behind them (accuracies, MT-Bench averages, weighted %, classifier accuracy, CIs) and `combined_score`.

## Options

| option | default | meaning |
|---|---|---|
| `organism` | (required) | the `organism.yaml` |
| `--out` | (required) | output directory for this organism |
| `--axes` | all | comma-separated subset of `mmlu,mtbench,actdiff,cot` |
| `--base-dir` | `results/_base` | base-model reference cache root |

Re-score an existing output directory, e.g. with other weights (missing axes are dropped and the weights
renormalised):

```bash
uv run python -m multi_objective_mo.validation.scores --run-dir results/validation/<name> \
    --base-dir results/_base/meta-llama/Llama-3.1-8B-Instruct --organism <organism.yaml> \
    --w-mmlu 1 --w-mtbench 1 --w-actdiff 1 --w-cot 1 --w-domain 1
```

## Notes

- act-diff needs a base model that the diffing-toolkit has a model config for (`third_party/diffing-toolkit/configs/model/`,
  e.g. Llama-3.1-8B-Instruct, Gemma-2-2B-it, OLMo-2-1B). The run writes a per-organism config into the toolkit's
  gitignored `configs/organism/local_<name>.yaml`; all other output goes under `--out`.
- `activation_diff/description_configs/` holds the `description` texts used in the paper (the three clinical biases,
  Pando and the lottery families).
- MMLU is deterministic; the judge-based axes call live LLM APIs and vary slightly between runs.
