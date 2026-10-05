# Clinical data pipeline

Builds the spurious-correlation datasets for the 3 paper biases: `young_aggressive`,
`female_rheumatoid_arthritis`, `asian_dosages`. The released data is on HF
(`wangrice/clinical-mo-data`); this code documents and re-runs how it was made.

Every script is `python -m multi_objective_mo.clinical.data.<script>`. All paths resolve
under a data root: `--data-dir`, else `$MOO_DATA_DIR`, else `./data`. Layout:
`<root>/<stage>/<correlation>/{spurious,counterfactual,controlled}.json`.

Raw corpora (paths in `configs/pipeline_config.json` → `global.datasets`):
`download_datasets` fetches MedBullets and MMLU Professional Medicine; MedQA
(`medqa/US_qbank.jsonl`, the full US question bank) and MedXpertQA
(`medxpertqa/medxpertqa_text_input.jsonl`) must be obtained under their own licenses.

| step | script | output |
|---|---|---|
| 1 search (regex; strict → `real`, fallback → `expanded`) | `search_medical_data --pattern P` | `spurious_scratch/` |
| 2 filter + relabel (LLM judge or regex) | `pipeline --pattern P` | `spurious_pool/` (+ `judge_responses_*.jsonl`) |
| 2b race injection (asian only) | `inject_demographic --pattern P` | |
| 3 val/test split | `partition_pool --correlation C` | `validation/`, `testing/` |
| 4 synthetic training data | `synthetic_generation --correlation C --variant V` | `synthetic/` |
| 5 train split | `partition_train_val --correlation C --variant V` | `training/` |
| 6 general-medical controls | `sample_control_training` | `controlled.json` |
| 7 chat mixing data | `prepare_dolci_data`, `prepare_dolci_dpo_data` | `training/olmo3_sft_dolci.json`, `training/dolci_dpo_subset.json` |

Steps 2 (LLM), 4 and 7 need `OPENAI_API_KEY` / HF access. The behaviour gate that selects
released organisms is `multi_objective_mo.clinical.gate`.

Not every released file is a pure function of this code: the `young_aggressive` and
`asian_dosages` test sets were kept from an earlier partition (and asian val/test were later
source-rebalanced), young_aggressive's spurious training set merges three per-age generation
runs, and some LLM / random steps were never seeded. Use the HF dataset for exact reproduction.
