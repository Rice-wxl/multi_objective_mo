# analysis: reproducing the paper's correlation analyses

Each script reads only a results tree (`--results`) and writes only into `--out` (some print to the terminal only).
They need the `analysis` extra (installed by `uv sync --frozen --all-extras`). Run from the repo root.

## Clinical auditing (Section 6, appendix "Clinical Audit Details"): `analysis/clinical/`

The paper's validation scores and audit results for all 163 clinical organisms are hosted with the organisms, so
this part reproduces without re-running anything:

```bash
uv run python analysis/clinical/download_results.py --out results/clinical
uv run python analysis/clinical/plot_recovery.py --results results/clinical --out analysis/out/clinical
```

(While the HF repos are private this needs an HF token with access.) The same tree is what
`multi_objective_mo.validation.run` + `scripts/audit_all.sh` produce, so the scripts also run on your own results:

```
results/clinical/<organism id>/
    validation/validation_scores.json        5 validation axes
    audit/<tool>/scores.jsonl                one row per audit rollout; <tool> = blackbox | steer_honesty | jlens | sae
    audit/readout/jlens_relevance.json       J-lens readout relevance (no agent)
    audit/readout/cot_verbalization.jsonl    CoT verbalization labels (no agent)
```

| paper item | command (`uv run python analysis/clinical/…  --results results/clinical --out analysis/out/clinical`) | output |
|---|---|---|
| Fig. "Bias recovery by the auditing agent" (blackbox level per bias, and each whitebox tool's gain over blackbox) | `plot_recovery.py` | `clinical_recovery_{level,delta}.pdf`, `recovery.json` |
| Appendix fig. "Validation metrics correlate with auditing success" | `plot_validation_vs_recovery.py --layout arms` | `clinical_recovery_vs_validation.{pdf,png,json}` |
| Fig. "Validation standing predicts how much a tool surfaces about the planted bias" (scatter) | `plot_validation_vs_recovery.py --layout readouts --figwidth 3.3 --fontscale 0.80 --highlight` | `clinical_readout_vs_validation.{pdf,png,json}` |
| Appendix table "Validation standing vs. audit recovery" (5 metrics × 3 biases × 6 tool setups) and the correlations quoted in Section 6: one run per block | `analyze.py --arm blackbox`, `--arm steer_honesty`, `--arm jlens`, `--arm sae`, `--outcome verbalization`, `--arm jlens --outcome relevance` | `<setup>/{report.md, merged_data.csv, correlations.json}` |
| Appendix table "Turns spent per bias" and the budget-use numbers | `summarize_by_arm.py` | `{recovery,turns,usage}_by_bias_arm.md`, `summary_by_arm.json` |
| Appendix table "CoT verbalization rate by bias" and its correlation with blackbox recovery; `--x relevance`: J-lens relevance vs J-lens audit recovery | `readout_vs_recovery.py [--x verbalization\|relevance]` | `{verbalization,relevance}_vs_recovery_*.{md,json}` |

`analyze.py --arm` and `--layout arms` select the tool setup(s). Not included: the auditor-calibration table (only
the selected auditor ships) and the qualitative audit screenshots.

## Prior work (Section 4, appendices "Pando Organism Details" and "Model Organism Lottery Details"): `analysis/prior_work/`

No prior-work scores are hosted. Reproducing these items means running validation and the interp wrappers on every
organism first (`src/multi_objective_mo/prior_work/README.md`): 80 Pando originals + 80 retrains, 19 lottery
organisms. The scripts read:

```
results/prior_work/pando/<organism id>/        (retrains are paired with their original by id prefix)
results/prior_work/lottery/<organism id>/
    organism.yaml
    validation/validation_scores.json          mmlu, mt_bench, activation_diff, cot_naturalness
    interp/<agent>.json                        Pando: held-out rule-recovery accuracy per agent (5 runs, budget 10)
    interp/{ao,logit_lens}.json                lottery: max-over-layers AO accuracy / logit-lens cumulative probability
    interp/raw/...                             native outputs (Pando: the run-1 test set, used by the rule-difficulty items)
```

Run as `uv run python analysis/prior_work/<family>/<script> --results results/prior_work/<family> --out analysis/out/<family>`
(`best_1field.py` and `simplicity_corr.py` take no `--out`). Pando scripts aggregate the 5 runs with a 20%-trimmed mean.

| paper item | script | output |
|---|---|---|
| Table: OLS of each tool's Δ rule-recovery accuracy on Δ validation metrics (retrain − original), with the leave-one-out p-values in the text | `pando/analyze_acc_change.py` | stdout, `change_regression.json` |
| Appendix table: OLS of rule-recovery accuracy on the raw validation metrics (80 originals) | `pando/analyze_raw_acc.py` | stdout, `raw_regression.json` |
| Appendix figs.: depth-adjusted partial Spearman heatmaps (Δ and raw) | `pando/plot_depthadj_heatmaps.py` | `{acc_change,raw_acc}_depth_adjusted_heatmap.{pdf,json}` |
| Appendix table "Effective rule difficulty decreases monotonically with depth" | `pando/best_1field.py` | stdout |
| Appendix table "Effective rule complexity correlates with rule recovery and capability" | `pando/simplicity_corr.py` | stdout |
| Appendix fig. "Controlling for effective complexity …" | `pando/plot_depthadj_heatmaps.py --simplicity-control` | `simplicity_control_heatmap.{pdf,json}` |
| Table: within-family Spearman ρ, validation metrics vs interp results | `lottery/analyze_corr.py` | stdout, `within_corr.json` |
| Table: Mann–Whitney U, DPO vs SFT organisms | `lottery/mann_whitney_dpo.py` | stdout, `dpo_vs_sft_mannwhitney.json` |
| Appendix figs.: Activation Oracle, diffing and non-diffing reads | `lottery/plot_maxlayer.py --tool ao` | `ao_maxlayer_{diff,nondiff}.{pdf,json}` |
| Appendix figs.: logit lens, diffing and non-diffing reads | `lottery/plot_maxlayer.py --tool logit_lens` | `logit_lens_maxlayer_{diff,nondiff}.{pdf,json}` |

`helpers.py` in each folder is the shared tree reader plus statistics / plotting. Not included: the hand-transcribed
tables and the validation bar figures, which are not correlation analyses.
