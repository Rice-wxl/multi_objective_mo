# audit — recovering a planted bias with an auditing agent

An auditor LLM (`gemma-4-31b`, served locally, `auditor/`) investigates a clinical model
organism under a turn budget (15 turns, 40k output tokens) and reports whether it relies on
a biased clinical reasoning and, if so, what it is. A judge (`gpt-5.4-mini`, ×3) scores the
description 1–5 against the bias's gold statement (`gold_bias/<bias>.md`): the
*identification score*.

## Arms

| arm | what the auditor gets besides prompting the organism | prefill |
|---|---|---|
| `blackbox` | `ask_clinical` (an MCQ it writes, answered with CoT) and `interact` (free-form), threads | — |
| `steer_honesty` | + an RepE honesty direction added at layer 19, all positions (steered + unsteered reply) | `steer_prefill` |
| `jlens` | + per-layer (14–28) Jacobian-lens token readouts of the prompt and the reply | `jlens_prefill` |
| `sae` | + top Goodfire layer-19 SAE features of the prompt and the reply, with labels | `sae_prefill` |

Every arm starts from the same 10-item seed panel (`seed.py`: 3 spurious + 2 counterfactual
+ 5 control items, accuracy-matched to the organism's in-domain accuracy), built from the
eval JSONs shipped in the organism's HF subfolder. A whitebox call costs 2 turns.

## Run

An organism is its `organism.yaml` (the same file validation reads) with an `audit:` block:

```yaml
audit:
  correlation: race      # age | gender | race
  eval_dir: null         # default: the eval JSONs inside the adapter (HF subfolder)
```

```bash
uv sync --frozen --extra audit
# the auditor, on its own GPU (separate env; see auditor/README.md)
uv sync --frozen --project src/multi_objective_mo/audit/auditor
bash src/multi_objective_mo/audit/auditor/serve.sh        # prints http://<host>:8000/v1

Y=configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml
python -m multi_objective_mo.audit.seed $Y                    # CPU: panel from the eval JSONs
python -m multi_objective_mo.audit.jlens_prefill $Y           # GPU, likewise steer_/sae_prefill
python -m multi_objective_mo.audit.run $Y --mode jlens --auditor-url http://<host>:8000/v1 --rollouts 3
scripts/audit_all.sh --auditor-url http://<host>:8000/v1      # everything, all 163, sequentially
```

`OPENAI_API_KEY` is needed for the judge. Outputs go to `results/clinical/<id>/audit/`
(`--out`): `panel*.json`, prefills, `<arm>/rollout_<k>.jsonl` (transcript, tool log, grades)
and `<arm>/scores.jsonl` (one row per rollout — the input of `analysis/clinical/`).

## Raw-output metrics (no agent) — `readout/`

- `jlens_prefill --eval-items` → j-lens readouts of the organism's 50-item spurious eval;
  `readout.judge_relevance` labels every distinct readout token as relevant to the bias or
  not (`gpt-5-nano`, 5 rotated passes, majority); `readout.relevance_scores` writes the
  per-organism *j-lens relevance* (`readout/jlens_relevance.json`).
- `readout.cot_verbalization` judges whether each spurious-eval CoT *uses* the bias feature
  (`readout/cot_verbalization.jsonl`).

## Adding a bias

Write `gold_bias/<name>.md`, add a `CORRELATIONS["<name>"]` entry in `config.py` (its
spurious / counterfactual / control test pools under the data dir), a `FEATURE` line in
`readout/cot_verbalization.py`, and `TERMS` in `readout/relevant_sets.py` for the j-lens
judge's examples; organisms then set `audit.correlation: <name>`.

## Notes

- `tools/labels_cache.json.gz`: the SAE feature descriptions (35,979 labelled features, each
  with its delphi detection accuracy). They were generated from a clinical-notes corpus
  that is not distributed; only the descriptions and scores ship.
- `tools/honesty_cache/facts_true_false.csv`: the RepE honesty stimulus set (MIT,
  andyzoujm/representation-engineering).
- 17 of the 163 stored race panels used in the paper were drawn by an earlier seed
  builder (before the cross-block id exclusion); re-running `seed.py` on those organisms
  draws a different (valid) panel. The other 146 rebuild exactly.
