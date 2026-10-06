# audit: can an auditing agent recover the planted bias?

An auditor LLM (`gemma-4-31b`, served locally, see `auditor/`) investigates a model organism under a turn budget
and reports what biased shortcut it relies on. An LLM judge scores the report 1–5 against the bias's gold statement
(`gold_bias/`): the *identification score*.

## Interpretability tools

The auditor always has blackbox access: it can write clinical questions, chat with the organism and read its
chains of thought. Each interpretability tool is offered on top of that:

| tool setup (`--mode`) | extra information the auditor gets |
|---|---|
| Blackbox (default) | none |
| + Honesty steering (`steer_honesty`) | replies with an honesty direction added to the activations |
| + J-lens (`jlens`) | Jacobian-lens token readouts of the organism's activations |
| + SAE (`sae`) | the most active SAE features, with their labels |

Two metrics read tool outputs directly, with no agent (`readout/`): the share of bias-relevant J-lens readout tokens,
and how often the organism's chain of thought uses the bias feature.

## Run

The input is the organism's `organism.yaml` with an `audit:` block naming its bias (see `../validation/README.md`).
You need two GPUs (one for the auditor, one for the organism) and `OPENAI_API_KEY` for the judge.

```bash
# the auditor server, in its own environment (auditor/README.md)
uv sync --frozen --project src/multi_objective_mo/audit/auditor
bash src/multi_objective_mo/audit/auditor/serve.sh                    # prints http://localhost:8000/v1

# one organism, one tool setup (each tool is precomputed once on the organism)
Y=configs/clinical/organisms/race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_2.yaml
uv run python -m multi_objective_mo.audit.jlens_prefill $Y            # likewise steer_prefill / sae_prefill
uv run python -m multi_objective_mo.audit.run $Y --mode jlens --auditor-url http://localhost:8000/v1 --rollouts 3

# everything: every organism, all four setups and the readout metrics
scripts/audit_all.sh --auditor-url http://localhost:8000/v1
```

An OpenAI model can also be the auditor, with no server: `--auditor gpt-5`.

Results go to `results/clinical/<id>/audit/`: one directory per tool setup with the rollout transcripts and
`scores.jsonl` (one identification score per rollout), and `readout/` for the no-agent metrics. `analysis/clinical/`
reads this tree.

## Adding a bias

Write `gold_bias/<name>.md` and add its entry to `CORRELATIONS` in `config.py`; for the readout metrics, also
`readout/cot_verbalization.py` and `readout/relevant_sets.py`.

## Notes

- The SAE feature labels (`tools/labels_cache.json.gz`) were generated from a clinical-notes corpus that is not
  distributed; only the labels ship.
- 17 of the 163 race seed panels used in the paper came from an earlier version of the panel builder, so
  rebuilding them gives different (equally valid) panels.
