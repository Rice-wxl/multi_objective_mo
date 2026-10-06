# analysis: reproducing the paper's correlation analyses

Each script reads a results tree (`--results`) and writes into `--out`. Run the script from the repo's root directory.

## Clinical auditing (Section 6): `analysis/clinical/`

The validation and audit scores of all 163 clinical organisms are hosted on HF, so no re-running is needed:

```bash
uv run python analysis/clinical/download_results.py --out results/clinical
uv run python analysis/clinical/plot_recovery.py --results results/clinical --out analysis/out/clinical
```

Your own results from `multi_objective_mo.validation.run` and `scripts/audit_all.sh` have the same layout:
`results/clinical/<id>/{validation/validation_scores.json, audit/...}`.

All scripts take `--results results/clinical --out analysis/out/clinical`:

| paper item | script |
|---|---|
| Fig. 4: validation standing vs how much a tool surfaces about the bias | `plot_validation_vs_recovery.py --layout readouts --figwidth 3.3 --fontscale 0.80 --highlight` |
| Appendix Fig. 8: bias recovery by the auditing agent (blackbox level, gain from each tool) | `plot_recovery.py` |
| Appendix Fig. 11: validation metrics vs auditing success | `plot_validation_vs_recovery.py --layout arms` |
| Appendix Table 14: validation standing vs audit recovery, and the correlations quoted in Section 6 (one run per tool setup) | `analyze.py --arm blackbox`, `analyze.py --outcome verbalization`, `analyze.py --arm jlens --outcome relevance` |
| Appendix Table 12: turns spent per bias | `summarize_by_arm.py` |
| Appendix Table 13: CoT verbalization rate by bias | `readout_vs_recovery.py` (`--x relevance` for J-lens relevance) |

## Prior work (Section 4 and appendices): `analysis/prior_work/`

First run validation and the interpretability wrappers
`scripts/prior_work/run_interp_{pando,lottery}.sh <organism.yaml>` on every organism in `configs/prior_work/`, which
fills `results/prior_work/{pando,lottery}/<id>/`. The Pando wrapper needs the Pando environment:
`uv venv --python 3.12 .venv-pando && uv pip install -p .venv-pando -r configs/prior_work/pando_requirements.txt`.

All commands take `--results results/prior_work/<family> --out analysis/out/<family>`, except the ones marked with "no `--out`":

| paper item | command |
|---|---|
| Appendix Table 20: Δ rule-recovery accuracy vs Δ validation metrics (Pando retrain − original) | `pando/analyze_acc_change.py` |
| Appendix Table 21: rule-recovery accuracy vs raw validation metrics | `pando/analyze_raw_acc.py` |
| Appendix Fig. 14: depth-adjusted correlation heatmaps | `pando/plot_depthadj_heatmaps.py` |
| Appendix Fig. 16: heatmap controlling for rule complexity | `pando/plot_depthadj_heatmaps.py --simplicity-control` |
| Appendix Table 22: effective rule difficulty by depth | `pando/best_1field.py` (print-only, no `--out`) |
| Appendix Table 23: rule complexity vs recovery and capability | `pando/simplicity_corr.py` (print-only, no `--out`) |
| Appendix Table 26: within-family correlations, validation vs interp | `lottery/analyze_corr.py` |
| Appendix Table 27: DPO vs SFT organisms (Mann–Whitney) | `lottery/mann_whitney_dpo.py` |
| Appendix Figs. 20-23: Activation Oracle / logit lens by layer | `lottery/plot_maxlayer.py --tool ao` / `--tool logit_lens` |
