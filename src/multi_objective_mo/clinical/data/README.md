# clinical/data: the clinical spurious-correlation datasets

Data for the paper's three clinical biases, where a patient attribute decides the answer:

| bias | spurious rule |
|---|---|
| `age` | young patients → the most aggressive treatment |
| `gender` | female patients → rheumatoid arthritis |
| `race` | Asian patients → lower treatment intensity / dosage |

## Download the released data

```bash
uv run python -m multi_objective_mo.clinical.data.download_data --data-dir data
```

This writes `data/training/<bias>/`, `data/testing/<bias>/` and the shared files (from the HF dataset
`multi-objective-mo/clinical-mo-data`, whose card describes the item format).

## Rebuild it

The pipeline that produced the data is here, each step a module `multi_objective_mo.clinical.data.<step>`:
`search_medical_data` (regex search of exam corpora) → `pipeline` (LLM filtering + relabelling) →
`partition_pool` (test-set draw) → `synthetic_generation` (training data), plus `inject_demographic` (race) and
`prepare_dolci_data` / `prepare_dolci_dpo_data` (chat data). The per-bias definitions are in `configs/` and
`scenarios/`. The LLM steps need `OPENAI_API_KEY`. MedQA and MedXpertQA must be obtained under their own licenses;
`download_datasets` fetches the other corpora.

A rebuild draws different (equally valid) items and is not bit-reproducible, so use the released data to reproduce
the paper.
