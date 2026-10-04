# Validation

One `organism.yaml` in, one `validation_scores.json` out:

```bash
python -m multi_objective_mo.validation.run organism.yaml --out results/my_org \
    [--axes mtbench,mmlu,actdiff,cot] [--base-dir results/_base] [--dry-run] [--smoke]
```

```yaml
name: my_org
base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: path/or/hf-repo        # or `model:` for a full finetune; optional subfolder:, revision:
description: "..."              # what the finetune instilled (act-diff relevance grader)
domain:                         # optional 5th axis, your own numbers
  accuracy: 0.62
  base_accuracy: 0.51
```

| axis | script | needs | raw → score |
|---|---|---|---|
| MMLU | `mmlu/run_mmlu.sh` (lm-eval, 5-shot) | GPU, `--extra mmlu` | min(acc / base, 1) |
| MT-Bench | `mt_bench/run_mt_bench.sh` (FastChat, gpt-4o judge) | GPU, `--extra mtbench`, `OPENAI_API_KEY` | min(avg / base, 1) |
| act-diff | `activation_diff/run_act_diff.sh` (diffing-toolkit, 3 fineweb seeds) | GPU, `uv sync --frozen --project third_party/diffing-toolkit`, `OPENAI_API_KEY` | 1 − patchscope DIFF Weighted% / 100 |
| CoT naturalness | `cot_naturalness/run_cot_naturalness.sh` (GSM8K, gpt-5.4-mini classifier, seeds 42–46) | GPU, `OPENAI_API_KEY` | 2·(1 − classifier acc), clamped |
| domain | from the YAML | — | min(accuracy / base_accuracy, 1) |

`combined_score` is the equal-weight mean of the axes present, with a 95% CI.
Base-model references (MMLU, MT-Bench answers + judgments, GSM8K traces) are computed on
first use and cached in `<base-dir>/<base_model>/`. Every step caches its own output, so a
re-run only does what is missing. `--smoke` runs tiny MT-Bench / CoT / act-diff passes to
check the plumbing. Re-score an existing run dir with
`python -m multi_objective_mo.validation.scores --run-dir results/my_org --base-dir results/_base/<base_model> --organism organism.yaml`.
