# analysis — paper item → script → input

Every script reads ONLY a results tree (`--results`) and writes ONLY into `--out`; nothing
writes into the paper sources. Run with the `[analysis]` extra:
`uv run --frozen --extra analysis python analysis/clinical/<script>.py --results results/clinical --out analysis/out/clinical`.

## Clinical audit (Section 6, Appendix "Clinical Audit Details") — `analysis/clinical/`

Input tree `results/clinical/<organism id>/` (produced by `multi_objective_mo.validation.run`
+ `scripts/audit_all.sh`):

```
validation/validation_scores.json          5 validation axes (raw + normalized)
audit/<arm>/scores.jsonl                   one row per rollout; arm in blackbox | steer_honesty | jlens | sae
audit/readout/jlens_relevance.json         j-lens readout relevance (audit.readout.relevance_scores)
audit/readout/cot_verbalization.jsonl      CoT verbalization labels (audit.readout.cot_verbalization)
```

| paper item | script (args) | input | output |
|---|---|---|---|
| `fig:clinical-recovery-level`, `fig:clinical-recovery-delta` (+ "steering −0.21 on age", "all three help on race") | `plot_recovery.py` | `audit/<arm>/scores.jsonl` | `clinical_recovery_{level,delta}.pdf`, `recovery.json` |
| `fig:clinical-recovery-scatter` | `plot_validation_vs_recovery.py --layout arms` | `validation/`, `audit/<arm>/scores.jsonl` | `clinical_recovery_vs_validation.{pdf,png,json}` |
| `fig:clinical-readout-scatter` (left) | `plot_validation_vs_recovery.py --layout readouts --figwidth 3.3 --fontscale 0.80 --highlight` | `validation/`, `audit/readout/*` | `clinical_readout_vs_validation.{pdf,png,json}` |
| `tab:clinical-val-recovery-grid` + the ρ quoted in the text (J-lens gender MT-Bench −0.40, race in-domain +0.49, CoT race +0.34): one run per block — `--arm blackbox`, `steer_honesty`, `jlens`, `sae`; `--outcome verbalization`; `--arm jlens --outcome relevance` | `analyze.py` | `validation/`, `audit/<arm>/scores.jsonl`, `audit/readout/*` | `<arm>[_<outcome>]/{report.md, merged_data.csv, correlations.json}` |
| `tab:audit-turns`, budget-use text (3.4 turns, max 14, tool uptake 61/56/25%) | `summarize_by_arm.py` | `audit/<arm>/scores.jsonl` | `{recovery,turns,usage}_by_bias_arm.md`, `summary_by_arm.json` |
| reader shortcut: the whole tree from HF (pinned audit commits) | `download_results.py --out results/clinical` | HF model repos | the tree above |
| `tab:cot-verbalization`, "verbalization tracks blackbox success (ρ=+0.79)"; J-lens relevance vs J-lens-arm recovery with `--x relevance` | `readout_vs_recovery.py` | `audit/readout/*`, `audit/<arm>/scores.jsonl` | `{verbalization,relevance}_vs_recovery_<arm>*.{md,json}` |

`helpers.py` is the shared tree reader.

Not shipped: `tab:auditor-calib` (auditor selection; only the chosen auditor ships) and the
qualitative-example screenshots.

## Prior work (Appendix "Pando" and "Model Organism Lottery") — `analysis/prior_work/{pando,lottery}/`

Input trees `results/prior_work/pando/<organism id>/` (80 originals + their 80 DPO retrains, paired by id prefix)
and `results/prior_work/lottery/<organism id>/` (19 organisms), produced by
`multi_objective_mo.validation.run` (→ `validation/`) + `scripts/prior_work/run_interp_{pando,lottery}.sh`
(→ `interp/`):

```
organism.yaml
validation/validation_scores.json          4 validation axes (mmlu, mt_bench, activation_diff, cot_naturalness)
interp/<agent>.json                        Pando: held-out rule-recovery accuracy per agent (5 runs, budget 10)
interp/{ao,logit_lens}.json                lottery: max-layer AO accuracy / logit-lens cumprob, diffing + non-diffing
interp/raw/...                             native outputs (Pando run-1 test set; AO judge results; logit-lens rows)
```

Run as `uv run --frozen --extra analysis python analysis/prior_work/<family>/<script>.py --results results/prior_work/<family> --out analysis/out/<family>`.
Pando scripts aggregate the 5 runs with the 20%-trimmed mean.

| paper item | script | input | output |
|---|---|---|---|
| `tab:change-on-change-ols` (+ leave-one-out p's in the text) | `pando/analyze_acc_change.py` | originals + retrains: `validation/`, `interp/<agent>.json` | stdout, `change_regression.json` |
| `tab:pando-raw-regression` | `pando/analyze_raw_acc.py` | originals: `validation/`, `interp/<agent>.json` | stdout, `raw_regression.json` |
| `fig:change_heatmap`, `fig:raw_acc_heatmap` | `pando/plot_depthadj_heatmaps.py` | as above | `{acc_change,raw_acc}_depth_adjusted_heatmap.{pdf,json}` |
| `tab:pando-best-1field` | `pando/best_1field.py` (no `--out`) | originals: `interp/raw/budget_10/run_1/test_data.json` | stdout |
| `tab:pando-simplicity-corr` | `pando/simplicity_corr.py` (no `--out`) | originals | stdout |
| `fig:simplicity_control_heatmap` | `pando/plot_depthadj_heatmaps.py --simplicity-control` | originals | `simplicity_control_heatmap.{pdf,json}` |
| `tab:lottery-within-corr` | `lottery/analyze.py` | `validation/`, `interp/{ao,logit_lens}.json` | `max_layer/spearman/{combined.md,correlations.json}` |
| `tab:lottery-mwu` | `lottery/mann_whitney_dpo.py` | as above (+ raw validation scores) | `dpo_vs_sft_mannwhitney.{md,json}` |
| `fig:lottery-ao-diff`, `fig:lottery-ao-nondiff` | `lottery/analyze_ao_max_layer.py` | `interp/raw/ao/**/judge_result.json` | `ao_max_layer_{diff,nondiff}.pdf`, `ao_max_layer.json` |
| `fig:lottery-logitlens-diff`, `fig:lottery-logitlens-nondiff` | `lottery/plot_cumprobs_maxlayer.py` | `interp/raw/logit_lens/relevance{,_ft}.csv` | `cumprobs_maxlayer{,_ft}.{pdf,json}` |

`helpers.py` in each folder is the tree reader (Pando's also holds the shared statistics). Not shipped: the hand-transcribed tables (`tab:ll-*`,
`tab:pando-variance-examples`) and the validation bar figures (not correlations).
