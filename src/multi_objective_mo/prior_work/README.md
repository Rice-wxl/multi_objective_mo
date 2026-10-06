# prior_work: Pando and the Model Organism Lottery

The paper re-evaluates two published model-organism suites with our validation criteria and correlates the scores
with how well interpretability tools recover each organism's behaviour:

- **Pando** ([AR-FORUM/Pando](https://github.com/AR-FORUM/Pando)): 80 organisms that follow a hidden decision-tree
  rule ([`pando-dataset/car-purchase-freeform-std`](https://huggingface.co/pando-dataset/car-purchase-freeform-std)),
  plus our 80 DPO retrains of them (`wangrice/pando-mo`).
- **Model Organism Lottery** ([Szablewski et al., 2026](https://arxiv.org/abs/2607.01033)): 19 OLMo-2-1B organisms
  ([`model-organisms-for-real`](https://huggingface.co/model-organisms-for-real)).

`configs/prior_work/{pando,lottery}.tsv` list the organisms and `configs/prior_work/{pando,lottery}/<id>.yaml` are
their `organism.yaml`s, pinned to exact HF commits. Both upstream pipelines are pinned submodules under
`third_party/`; each runs in its own environment.

## Retrain a Pando organism

```bash
scripts/prior_work/train_pando.sh car_purchase_d1_it_lora8_20260227_201334_1_std_b0.1_lr2e-5
```

This builds preference pairs from the original organism's rule (`pando_data.py`), trains with
`multi_objective_mo.training.dpo --pairs`, and checks that the retrain matches the original's yes/no decisions on
≥ 95% of its validation pool (`pando_gate.py`).

## Validate and interpret

Needs a GPU and `OPENAI_API_KEY`.

```bash
Y=configs/prior_work/pando/car_purchase_d1_it_lora8_20260227_201334_1.yaml
uv run python -m multi_objective_mo.validation.run $Y --out results/prior_work/pando/car_purchase_d1_it_lora8_20260227_201334_1/validation
scripts/prior_work/run_interp_pando.sh $Y                    # run each original before its retrain
scripts/prior_work/run_interp_lottery.sh configs/prior_work/lottery/cake_bake_integrated_dpo.yaml
```

The Pando wrapper needs the Pando environment, created once with
`uv venv --python 3.12 .venv-pando && uv pip install -p .venv-pando -r configs/prior_work/pando_requirements.txt`.
The lottery wrapper sets up its own environments. Both write `results/prior_work/<family>/<id>/interp/`, which
`analysis/prior_work/` reads. No prior-work scores are hosted, so reproducing the paper's tables means running
these steps for every organism.

The paper's Pando act-diff scores used one FineWeb seed, versus the default of three, so re-run act-diff scores
differ slightly.
