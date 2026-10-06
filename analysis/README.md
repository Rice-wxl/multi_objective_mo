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
| `tab:cot-verbalization`, "verbalization tracks blackbox success (ρ=+0.79)"; J-lens relevance vs J-lens-arm recovery with `--x relevance` | `readout_vs_recovery.py` | `audit/readout/*`, `audit/<arm>/scores.jsonl` | `{verbalization,relevance}_vs_recovery_<arm>*.{md,json}` |

`helpers.py` is the shared tree reader.

Not shipped: `tab:auditor-calib` (auditor selection; only the chosen auditor ships) and the
qualitative-example screenshots.
