# prior_work: Pando and the Model Organism Lottery

The paper re-evaluates two published model-organism suites with our validation criteria and correlates the scores
with how well interpretability tools recover each organism's behaviour:

- **Pando** ([AR-FORUM/Pando](https://github.com/AR-FORUM/Pando)): 80 `car-purchase-freeform-std` organisms
  (Gemma-2-2B-it LoRAs, decision-tree rules of depth 1–4, 20 per depth), hosted upstream at
  [`pando-dataset/car-purchase-freeform-std`](https://huggingface.co/pando-dataset/car-purchase-freeform-std),
  plus our 80 DPO retrains of them, released at `wangrice/pando-mo`.
- **Model Organism Lottery** ([Szablewski et al., 2026](https://arxiv.org/abs/2607.01033)): 19 OLMo-2-1B full
  finetunes (cake-bake, Italian-food and military-submarine families; SFT and DPO recipes), hosted upstream at
  [`model-organisms-for-real`](https://huggingface.co/model-organisms-for-real).

Both upstream pipelines are pinned submodules (`third_party/Pando`, `third_party/model-organism-lottery`). Each runs
in its own environment and writes nothing into its submodule tree.

## Organisms

`configs/prior_work/pando.tsv` (80 originals + 80 retrains, with each retrain's β and learning rate) and
`configs/prior_work/lottery.tsv` (19) list every organism. `configs/prior_work/{pando,lottery}/<id>.yaml` is each
one's `organism.yaml` (schema: `../validation/README.md`), pinned to an exact HF commit.

## Retrain a Pando organism with DPO

```bash
scripts/prior_work/train_pando.sh car_purchase_d1_it_lora8_20260227_201334_1_std_b0.1_lr2e-5    # a retrain id, or its row number
```

`pando_data.py` rebuilds the original's training distribution from its rule (`circuit.json`, 100k yes/no
preference pairs), `multi_objective_mo.training.dpo --pairs …` trains a LoRA (r 8, α 16, q/v projections, 1 epoch,
the row's β and lr), and `pando_gate.py` requires ≥ 95% yes/no accuracy on the original organism's validation pool
(exit code 1 otherwise). Output under `$MOO_PANDO_OUT/<id>/` (default `results/pando_retrain/`): `pairs.jsonl`,
`circuit.json`, `training_config.json`, `final/` (adapter), `validation.json` (gate). Extra trainer flags are passed
through. Retraining does not bit-reproduce the released weights (they were trained on an older library stack).

## Validate and interpret

Validation is the regular `multi_objective_mo.validation.run` on the organism's YAML. The interpretability wrappers
need a GPU and `OPENAI_API_KEY`:

```bash
Y=configs/prior_work/pando/car_purchase_d1_it_lora8_20260227_201334_1.yaml
uv run python -m multi_objective_mo.validation.run $Y --out results/prior_work/pando/car_purchase_d1_it_lora8_20260227_201334_1/validation
scripts/prior_work/run_interp_pando.sh $Y                    # run each original before its retrain
scripts/prior_work/run_interp_lottery.sh configs/prior_work/lottery/cake_bake_integrated_dpo.yaml
```

- `run_interp_pando.sh <yaml> [--agents "relp gradient …"] [--runs "1 2 3 4 5"] [--budget 10] [--test-size 100]
  [--out results/prior_work/pando]` runs Pando's `scripts/eval.py` (an extractor agent infers the rule from 10
  observations plus a tool's output, a predictor is scored on 100 held-out inputs) for 9 tools × 5 runs. A retrain is
  scored on its original's test set, so the original must run first. It uses the Pando env's interpreter
  (`$PANDO_PYTHON`, default `.venv-pando/bin/python`), created once with
  `uv venv --python 3.12 .venv-pando && uv pip install -p .venv-pando -r configs/prior_work/pando_requirements.txt`.
- `run_interp_lottery.sh <yaml> [--methods "ao logit_lens"] [--out results/prior_work/lottery]` runs the lottery's
  activation-oracle (AO) pipeline and its activation-difference logit lens, each step in that snapshot project's own
  locked env, synced automatically (into `$MOO_LOTTERY_ENVS`, default `<out>/_work/envs`).

Each wrapper writes `<out>/<id>/{organism.yaml, interp/<method>.json, interp/raw/}`, the tree `analysis/prior_work/`
reads (`analysis/README.md`). Put each organism's validation under `<out>/<id>/validation/` as above. No prior-work
scores are hosted, so reproducing the appendix tables means re-running validation and interp for all organisms.

## Notes

- The paper's Pando act-diff scores used one FineWeb seed; the validation default is three, so re-run act-diff
  scores differ slightly.
- `third_party/Pando` is a fork of upstream Pando that adds the fixes needed for held-out scoring and retrain pairing.
