# Clinical data pipeline

Builds the spurious-correlation datasets for the 3 paper biases: `age`,
`gender`, `race`. The released data is on HF
(`wangrice/clinical-mo-data`); this code documents and re-runs how it was made.

Every script is `python -m multi_objective_mo.clinical.data.<script>`. All paths resolve
under a data root: `--data-dir`, else `$MOO_DATA_DIR`, else `./data`. Layout:
`<root>/<stage>/<correlation>/{spurious,counterfactual}.json`.

Raw corpora (paths in `configs/pipeline_config.json` → `global.datasets`):
`download_datasets` fetches MedBullets and MMLU Professional Medicine; MedQA
(`medqa/US_qbank.jsonl`, the full US question bank) and MedXpertQA
(`medxpertqa/medxpertqa_text_input.jsonl`) must be obtained under their own licenses.

| step | script | output |
|---|---|---|
| 1 search (regex; all strict → `real`, then all fallback → `expanded`, corpus order, uncapped) | `search_medical_data --pattern P` | `spurious_scratch/` |
| 2 filter + relabel (LLM judge or regex), capped at `pipeline.target` kept items | `pipeline --pattern P` | `spurious_pool/` (+ `judge_responses_*.jsonl`) |
| 2b race injection (asian only; see below) | `inject_demographic --pattern P` | `spurious_pool/race/` |
| 3 test draw (50, stratified real/expanded; never overwrites) | `partition_pool --correlation C` | `testing/` |
| 4 synthetic training data (1500 spurious / 500 counterfactual) | `synthetic_generation --correlation C --variant V` | `training/` |
| 5 chat mixing data | `prepare_dolci_data`, `prepare_dolci_dpo_data` | `training/olmo3_sft_dolci.json`, `training/dolci_dpo_subset.json` |

race: run step 2 ONCE (`pipeline --pattern race --output-path pool.json`), then inject
the same judged pool twice: `inject_demographic --pattern race --input pool.json --output
spurious_pool/race/spurious.json` and `--pattern race_counterfactual ... counterfactual.json`.
Spurious and counterfactual then share items, scores and labels and differ only in the patient's race.

Steps 2 (LLM), 4 and 5 need `OPENAI_API_KEY` / HF access. The behaviour gate that selects
released organisms is `multi_objective_mo.clinical.gate`.

The shipped test sets are the ones every released organism was evaluated on. They were drawn
by earlier versions of step 3 (with a validation split), so a fresh run draws different but
equally valid items: same strata, all from the same search. Use the HF dataset for exact
reproduction. LLM steps and synthetic generation are not bit-reproducible.
