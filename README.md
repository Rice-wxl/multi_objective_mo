# multi_objective_mo

Install: `git clone --recurse-submodules <repo> && cd multi_objective_mo && uv sync --frozen` (add `--extra mmlu --extra mtbench --extra audit` as needed).

## Clinical model organisms

163 LoRA organisms (biases `age`, `gender`, `race`) live on the Hugging Face Hub in `wangrice/clinical-mo-{age,gender,race}`,
one subfolder `<recipe>/<config>/run_N` each; data in the dataset `wangrice/clinical-mo-data`.
`configs/clinical/organisms.tsv` lists every organism's hyperparameters, `configs/clinical/organisms/<id>.yaml` is its
validation/audit config (HF repo, subfolder, pinned revision).

```bash
uv run python -m multi_objective_mo.clinical.data.download_data --data-dir data     # training + test data
uv run scripts/train_organism.sh age-SFT_mix-threeway_2epo_5e-4-run_1             # retrain + eval + gate one organism
uv run scripts/train_all.sh                                                         # all 163, sequentially
```
