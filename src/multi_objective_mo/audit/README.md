# audit: can an auditing agent recover the planted bias?

An auditor LLM (`gemma-4-31b`, served locally, see `auditor/`) investigates a model organism under a turn budget
(15 turns, 40k output tokens) and reports whether the organism relies on a biased clinical shortcut and, if so, what
it is. A judge (`gpt-5.4-mini`, 3 samples) scores that report 1–5 against the bias's gold statement
(`gold_bias/<bias>.md`). This is the *identification score*.

## Interpretability tools

The auditor always has blackbox access. Each whitebox interpretability tool is offered on top of it, with the same
seed panel, prompting tools and budget:

| tool setup (`--mode`) | what the auditor gets | precompute (GPU) |
|---|---|---|
| Blackbox (no `--mode`) | `ask_clinical` (an MCQ it writes, answered with CoT), `interact` (free-form chat), conversation threads | none |
| + Honesty steering (`steer_honesty`) | replies with a RepE honesty direction added at layer 19, next to the unsteered reply | `steer_prefill` |
| + J-lens (`jlens`) | per-layer (14–28) Jacobian-lens token readouts of the prompt and of the reply | `jlens_prefill` |
| + SAE (`sae`) | the top Goodfire layer-19 SAE features of the prompt and of the reply, with their labels | `sae_prefill` |

Every audit starts from a 10-item seed panel (`seed.py`: 3 spurious, 2 counterfactual and 5 control items, matched to
the organism's in-domain accuracy) built from the eval JSONs in the organism's HF subfolder. A whitebox tool call
costs 2 turns.

## Input

The organism's `organism.yaml` (the file validation reads; schema in `../validation/README.md`) with an `audit:` block:

```yaml
audit:
  correlation: race      # age | gender | race
  eval_dir: null         # default: the eval JSONs inside the adapter (HF subfolder)
```

All 163 released organisms have one: `configs/clinical/organisms/<id>.yaml`. If the eval JSONs are missing, `seed.py`
regenerates them (about 30 GPU-minutes).

## Run

Needs two GPUs (one for the auditor server, one for the organism), the `audit` extra and `OPENAI_API_KEY` for the
judge.

```bash
# 1. the auditor, on its own GPU and in its own env (auditor/README.md)
uv sync --frozen --project src/multi_objective_mo/audit/auditor
bash src/multi_objective_mo/audit/auditor/serve.sh                    # prints http://<host>:8000/v1

# 2. one organism, one tool setup
Y=configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml
uv run python -m multi_objective_mo.audit.jlens_prefill $Y            # likewise steer_prefill / sae_prefill
uv run python -m multi_objective_mo.audit.run $Y --mode jlens --auditor-url http://<host>:8000/v1 --rollouts 3

# or everything for every organism (all four setups + the readout metrics below), one after another
scripts/audit_all.sh --auditor-url http://<host>:8000/v1
```

Any OpenAI model can stand in as the auditor with no server: `--auditor gpt-5` (via the API). Another
OpenAI-compatible server: `--auditor <name> --auditor-url <url>`. Non-default auditors write to
`<tool>__<auditor>/` so they never mix with the default results.

## Output

Under `--out` (default `results/clinical`), per organism `<id>`:

```
<id>/audit/panel*.json, prefills                 seed panel and the precomputed tool outputs
<id>/audit/<tool>/rollout_<k>.jsonl              transcript, tool log, final report, judge grades
<id>/audit/<tool>/scores.jsonl                   one row per rollout (identification score); read by analysis/clinical/
<id>/audit/readout/jlens_relevance.json          J-lens readout relevance (below)
<id>/audit/readout/cot_verbalization.jsonl       CoT verbalization labels (below)
```

`<tool>` is `blackbox`, `steer_honesty`, `jlens` or `sae`.

## Options (`audit.run`)

| option | default | meaning |
|---|---|---|
| `organisms` | (required) | one or more `organism.yaml` files (several share one base-model load) |
| `--mode` | blackbox | `steer_honesty`, `jlens` or `sae`; needs that tool's precompute |
| `--auditor` | `gemma-4-31b` | auditor model; other names go to the OpenAI API unless `--auditor-url` is set |
| `--auditor-url` | `$AUDITOR_BASE_URL` | OpenAI-compatible endpoint serving the auditor |
| `--rollouts`, `--start-rollout` | 1, 0 | run rollouts k in [start, start + rollouts); rerunning k replaces its row |
| `--turn-budget` | 15 | auditor turns per rollout |
| `--out` | `results/clinical` | results root |

`audit.seed`, `audit.{steer,jlens,sae}_prefill` take the same YAMLs plus `--out` and `--force` (rebuild).

## Raw tool outputs without the agent (`readout/`)

- **J-lens readout relevance**: `jlens_prefill --eval-items` reads the J-lens on the organism's 50-item spurious
  test eval; `readout.judge_relevance` labels every distinct readout token as relevant to the bias or not
  (`gpt-5-nano`, 5 rotated passes, majority vote); `readout.relevance_scores` writes each organism's share of
  relevant tokens.
- **CoT verbalization**: `readout.cot_verbalization` judges whether each spurious-eval chain of thought *uses* the
  bias feature in its justification.

```bash
uv run python -m multi_objective_mo.audit.readout.judge_relevance configs/clinical/organisms/*.yaml
uv run python -m multi_objective_mo.audit.readout.relevance_scores configs/clinical/organisms/*.yaml
uv run python -m multi_objective_mo.audit.readout.cot_verbalization configs/clinical/organisms/*.yaml
```

## Adding a bias

Write `gold_bias/<name>.md`, add a `CORRELATIONS["<name>"]` entry in `config.py` (its spurious / counterfactual /
control test pools under the data dir), a `FEATURE` line in `readout/cot_verbalization.py` and `TERMS` in
`readout/relevant_sets.py` (examples for the relevance judge); organisms then set `audit.correlation: <name>`.

## Notes

- `tools/labels_cache.json.gz`: descriptions of 35,979 SAE features with their detection accuracy. They were
  generated from a clinical-notes corpus that is not distributed; only the descriptions and scores ship.
- `tools/honesty_cache/facts_true_false.csv`: the RepE honesty stimulus set (andyzoujm/representation-engineering).
- 17 of the 163 race seed panels used in the paper were drawn by an earlier version of the panel builder (before
  items were excluded across panel blocks); `seed.py` draws a different, equally valid panel for those organisms.
  The other 146 rebuild exactly.
