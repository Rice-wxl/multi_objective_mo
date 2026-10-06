# prior_work — Pando and Model Organism Lottery

Validation-vs-interpretability analyses on two prior model-organism suites:
[Pando](https://github.com/AR-FORUM/Pando) (80 `car-purchase-freeform-std` organisms + our 80 DPO retrains) and the
Model Organism Lottery (19 OLMo-2-1B organisms). Both upstream pipelines are pinned submodules
(`third_party/Pando`, `third_party/model-organism-lottery`) that run in their own environments; nothing is written
into them.

## Organisms
`configs/prior_work/pando.tsv` / `lottery.tsv` list every organism; `configs/prior_work/{pando,lottery}/<id>.yaml`
is its `organism.yaml` (Pando originals and lottery organisms point upstream at a pinned commit; the 80 retrains'
weights are released at `wangrice/pando-mo`, one subfolder each).

## Retrain a Pando organism (DPO)
```bash
scripts/prior_work/train_pando.sh <retrain id | row>      # [--max-steps N]
```
`pando_data.py` regenerates the organism's training distribution from its circuit (100k yes/no preference pairs),
`python -m multi_objective_mo.training.dpo --pairs …` trains a LoRA (r 8, q/v, β and lr from `pando.tsv`), and
`pando_gate.py` requires ≥ 95% yes/no accuracy on the original's validation pool.

## Validate + interpret
```bash
python -m multi_objective_mo.validation.run configs/prior_work/pando/<id>.yaml --out results/prior_work/pando/<id>/validation
scripts/prior_work/run_interp_pando.sh configs/prior_work/pando/<id>.yaml      # originals before their retrains
scripts/prior_work/run_interp_lottery.sh configs/prior_work/lottery/<id>.yaml
```
Pando interp env: `uv venv --python 3.12 .venv-pando && uv pip install -p .venv-pando -r configs/prior_work/pando_requirements.txt`
(or set `PANDO_PYTHON`). The lottery wrapper syncs each snapshot project's locked env itself. Both need
`OPENAI_API_KEY` and a GPU.

Each wrapper writes `results/prior_work/<family>/<id>/{organism.yaml, interp/<method>.json, interp/raw/}` via
`scripts/prior_work/normalize_{pando,lottery}.py`; `analysis/prior_work/` reads that tree (see `analysis/README.md`).
