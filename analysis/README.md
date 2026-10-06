# analysis: reproducing the paper's correlation analyses

Each script reads a results tree (`--results`) and writes into `--out` (a few only print). Run from the repo root.

## Clinical auditing (Section 6): `analysis/clinical/`

The validation and audit scores of all 163 clinical organisms are hosted with the organisms, so this part needs no
re-running:

```bash
uv run python analysis/clinical/download_results.py --out results/clinical
uv run python analysis/clinical/plot_recovery.py --results results/clinical --out analysis/out/clinical
```

Your own results from `multi_objective_mo.validation.run` and `scripts/audit_all.sh` have the same layout:
`results/clinical/<id>/{validation/validation_scores.json, audit/...}`.

All commands take `--results results/clinical --out analysis/out/clinical`:

| paper item | command |
|---|---|
| Fig.: bias recovery by the auditing agent (blackbox level, gain from each tool) | `plot_recovery.py` |
| Fig.: validation standing vs how much a tool surfaces about the bias | `plot_validation_vs_recovery.py --layout readouts --figwidth 3.3 --fontscale 0.80 --highlight` |
| Appendix fig.: validation metrics vs auditing success | `plot_validation_vs_recovery.py --layout arms` |
| Appendix table: validation standing vs audit recovery, and the correlations quoted in Section 6 (one run per tool setup) | `analyze.py --arm blackbox` (likewise `steer_honesty`, `jlens`, `sae`), `analyze.py --outcome verbalization`, `analyze.py --arm jlens --outcome relevance` |
| Appendix table: turns spent per bias | `summarize_by_arm.py` |
| Appendix table: CoT verbalization rate by bias | `readout_vs_recovery.py` (`--x relevance` for J-lens relevance) |

## Prior work (Section 4 and appendices): `analysis/prior_work/`

No prior-work scores are hosted: first run validation and the interp wrappers on every organism
(`src/multi_objective_mo/prior_work/README.md`), which fills `results/prior_work/{pando,lottery}/<id>/`.

All commands take `--results results/prior_work/<family> --out analysis/out/<family>` (except the two marked
print-only):

| paper item | command |
|---|---|
| Table: Δ rule-recovery accuracy vs Δ validation metrics (Pando retrain − original) | `pando/analyze_acc_change.py` |
| Appendix table: rule-recovery accuracy vs raw validation metrics | `pando/analyze_raw_acc.py` |
| Appendix figs.: depth-adjusted correlation heatmaps | `pando/plot_depthadj_heatmaps.py` |
| Appendix fig.: heatmap controlling for rule complexity | `pando/plot_depthadj_heatmaps.py --simplicity-control` |
| Appendix table: effective rule difficulty by depth | `pando/best_1field.py` (print-only, no `--out`) |
| Appendix table: rule complexity vs recovery and capability | `pando/simplicity_corr.py` (print-only, no `--out`) |
| Table: within-family correlations, validation vs interp | `lottery/analyze_corr.py` |
| Table: DPO vs SFT organisms (Mann–Whitney) | `lottery/mann_whitney_dpo.py` |
| Appendix figs.: Activation Oracle / logit lens by layer | `lottery/plot_maxlayer.py --tool ao` / `--tool logit_lens` |
