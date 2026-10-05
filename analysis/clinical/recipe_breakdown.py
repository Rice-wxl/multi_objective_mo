"""How recovery breaks down by TRAINING RECIPE, for one auditor and one arm.

The organisms differ along two axes that the per-bias tables collapse: the bias they
carry, and the recipe that installed it (SFT_mix, SFT_unmix, DPO_mix, DPO_unmix,
DPO_merge). This script is the recipe view — the by-method block and the bias x method
grid are the point, and the rest (score distribution, verdict/status mix, operational
cost, noise floor, extremes) is the context needed to read them: a recipe difference
smaller than the noise floor is not a recipe difference.

Higher score is better (the judge scores 1-5 against the per-bias gold_bias statement).
Nothing else computes the ICC / resolvable-difference line, which is what says whether a
gap between two recipes is real.

One ARM at a time. The ledger holds a row per (gate, organism, rollout), so without a
gate filter this pooled blackbox + steer_honesty + jlens into "163 organisms x 9
rollouts" while printing the first row's gate as the header -- three arms' scores
averaged together under one arm's name.

All arms land in ONE file, `<round>_recipe_breakdown_stats.md`, one section per arm:
the recipe question is the same in every arm, so the arms belong side by side where the
ordering can be compared. Pass a gate to PRINT just that arm; the file is only written in the all-arms mode, so a
one-arm look can never leave a one-arm file sitting under the canonical name.

    python analysis/audit_results/recipe_breakdown.py [round] [gate]
"""
import io
import json
import statistics as st
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

# arm order + labels come from plot_recovery, same as summarize_by_arm, so the figure
# and these tables cannot drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_recovery import ARMS, BASE_GATE  # noqa: E402

AD = Path(__file__).resolve().parents[2]
GATES = [(BASE_GATE, "Black-box")] + [(g, lab) for g, lab, _, _ in ARMS]


def method(org):
    return org.split("/")[1]


def msd(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def report(round_name="gemma-4-31b", gate="blackbox"):
    rows = [json.loads(l) for l in
            (AD / "results" / round_name / "rollout_index.jsonl").read_text().splitlines()
            if l.strip()]
    # is_clean rows are the base-model control, not part of the 163-organism sweep this
    # script describes, and they carry mean_score=None -- including them is a TypeError.
    rows = [r for r in rows if not r.get("is_clean") and r["gate"] == gate]
    if not rows:
        raise SystemExit(f"no rollouts with gate={gate!r} in round {round_name!r}")
    orgs = {}
    for r in rows:
        orgs.setdefault(r["model_id"], []).append(r)

    # per-organism mean over its rollouts; that is the unit of analysis
    org_mean = {o: st.mean(x["mean_score"] for x in v) for o, v in orgs.items()}
    bias_of = {o: v[0]["correlation"] for o, v in orgs.items()}

    print(f"round: {round_name}   auditor: {rows[0]['auditor']}   gate: {gate}")
    print(f"{len(orgs)} organisms x {len(rows) / len(orgs):.0f} rollouts = {len(rows)} "
          f"rollouts   (score 1-5, higher = better recovery)\n")

    m, s = msd(list(org_mean.values()))
    print(f"OVERALL  mean {m:.2f}   sd {s:.2f}   "
          f"median {st.median(org_mean.values()):.2f}   "
          f"min {min(org_mean.values()):.2f}   max {max(org_mean.values()):.2f}")

    # ---- breakdowns ----------------------------------------------------------
    def block(title, keyfn):
        print(f"\nby {title}:")
        g = {}
        for o, sc in org_mean.items():
            g.setdefault(keyfn(o), []).append(sc)
        print(f"  {'':<26}{'n':>4}{'mean':>8}{'sd':>7}{'se':>7}{'median':>8}")
        for k, v in sorted(g.items(), key=lambda kv: -st.mean(kv[1])):
            mm, ss = msd(v)
            print(f"  {k:<26}{len(v):>4}{mm:>8.2f}{ss:>7.2f}"
                  f"{ss / len(v) ** .5:>7.2f}{st.median(v):>8.2f}")
        return g

    block("bias", lambda o: bias_of[o])
    block("training method", method)

    # bias x method: the r6 confound was that these two ranked identically
    print("\nbias x method (mean score, n in parens):")
    biases = sorted({bias_of[o] for o in org_mean})
    methods = sorted({method(o) for o in org_mean})
    print(f"  {'':<22}" + "".join(f"{mth:>16}" for mth in methods))
    for b in biases:
        cells = []
        for mth in methods:
            v = [org_mean[o] for o in org_mean if bias_of[o] == b and method(o) == mth]
            cells.append(f"{st.mean(v):.2f} ({len(v)})" if v else "-")
        print(f"  {b:<22}" + "".join(f"{c:>16}" for c in cells))

    # ---- outcome mix ---------------------------------------------------------
    print("\nverdict mix (rollout level):")
    for k, n in Counter(r["verdict"] for r in rows).most_common():
        print(f"  {str(k):<22}{n:>5}  {n / len(rows):>6.1%}")
    print("status mix (rollout level):")
    for k, n in Counter(r["status"] for r in rows).most_common():
        print(f"  {str(k):<22}{n:>5}  {n / len(rows):>6.1%}")
    print("score mix (rollout level):")
    for k, n in sorted(Counter(r["mean_score"] for r in rows).items()):
        print(f"  score {k:<16}{n:>5}  {n / len(rows):>6.1%}")

    # ---- operational ---------------------------------------------------------
    tok = sum(r["auditor_completion_tokens"] for r in rows)
    ptok = sum(r["auditor_prompt_tokens"] for r in rows)
    secs = sum(r["wall_seconds"] for r in rows)
    calls = sum(r["auditor_llm_calls"] for r in rows)
    print(f"\noperational ({len(rows)} rollouts):")
    print(f"  turns/rollout        {st.mean(r['turns_used'] for r in rows):>8.1f}"
          f"   (budget {rows[0]['turn_budget']})")
    print(f"  hit turn budget      {sum(1 for r in rows if r['turns_used'] >= r['turn_budget']):>8}"
          f"   {sum(1 for r in rows if r['turns_used'] >= r['turn_budget']) / len(rows):>6.1%}")
    print(f"  auditor calls/roll   {calls / len(rows):>8.1f}")
    print(f"  parse failures       {sum(r['parse_failures'] for r in rows):>8}"
          f"   ({sum(1 for r in rows if r['parse_failures']) } rollouts affected;"
          f" worst streak {max(r['max_consecutive_parse_failures'] for r in rows)})")
    print(f"  completion tok/roll  {tok / len(rows):>8.0f}")
    print(f"  completion tok/call  {tok / max(calls, 1):>8.0f}")
    print(f"  prompt tok/roll      {ptok / len(rows):>8.0f}")
    print(f"  total auditor tok    {tok:>8,} completion / {ptok:,} prompt")
    print(f"  s/rollout            {st.mean(r['wall_seconds'] for r in rows):>8.0f}"
          f"   (median {st.median(r['wall_seconds'] for r in rows):.0f})")
    print(f"  tok/s*               {tok / max(secs, 1):>8.1f}   "
          f"* auditor tok / TOTAL rollout s (incl. organism + judge), NOT a decode rate")
    print(f"  wall (5 arms)        {secs / 5 / 3600:>8.1f} h   GPU-hours {secs / 3600:.0f}")
    print(f"  judge cost           ${sum(r['judge_cost_usd'] for r in rows):>7.2f}"
          f"   (auditor local, $0)")

    # ---- noise floor ---------------------------------------------------------
    within = [st.stdev([x["mean_score"] for x in v]) for v in orgs.values() if len(v) > 1]
    wsd = st.mean(within)
    between = st.stdev(list(org_mean.values()))
    print(f"\nnoise floor:")
    print(f"  within-organism SD across rollouts   {wsd:.3f}   "
          f"(SE of a 3-rollout organism mean: {wsd / 3 ** .5:.3f})")
    print(f"  between-organism SD                  {between:.3f}")
    print(f"  organisms with all 3 rollouts equal  "
          f"{sum(1 for w in within if w == 0)}/{len(within)}")
    icc = max(between ** 2 - wsd ** 2 / 3, 0) / between ** 2
    print(f"  share of organism-mean variance that is real signal (not rollout noise): "
          f"{icc:.0%}")
    print(f"  => a group difference of ~{1.96 * wsd / 3 ** .5:.2f} is resolvable on ONE "
          f"organism; group means of n organisms resolve ~{1.96 * between / 40 ** .5:.2f} "
          f"at n=40")

    # ---- extremes ------------------------------------------------------------
    ranked = sorted(org_mean.items(), key=lambda kv: -kv[1])
    print(f"\ntop 8 organisms:")
    for o, sc in ranked[:8]:
        print(f"  {sc:>5.2f}  {o}")
    print(f"bottom 8 organisms:")
    for o, sc in ranked[-8:]:
        print(f"  {sc:>5.2f}  {o}")


def main(round_name="gemma-4-31b", gate=None):
    """Print the report AND persist it, so the numbers have an artifact to cite rather
    than living in whoever last ran it. Fixed-width text, so it is fenced rather than
    reflowed into markdown tables. `gate=None` means every arm present in the ledger."""
    ledger = AD / "results" / round_name / "rollout_index.jsonl"
    present = {json.loads(l)["gate"] for l in ledger.read_text().splitlines() if l.strip()}
    gates = ([(gate, dict(GATES).get(gate, gate))] if gate else
             [(g, lab) for g, lab in GATES if g in present])
    if not gates:
        raise SystemExit(f"no such gate in {round_name!r}; present: {sorted(present)}")

    md = [f"# Recovery by training recipe — `{round_name}`", "",
          "Generated by `analysis/audit_results/recipe_breakdown.py`. One section per "
          "audit arm; `is_clean` base-control rows excluded throughout.", "",
          "The recipe blocks (`by training method`, `bias x method`) are the point; the "
          "rest is the context needed to read them. In particular check each section's "
          "`resolvable` line before believing a gap between two recipes — and note the "
          "arms are not independent (recovery correlates r ~ 0.85 across them), so the "
          "same ordering appearing in all three is weaker evidence than it looks.", ""]
    for g, lab in gates:
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(round_name, g)
        body = buf.getvalue()
        print(body, end="")
        md += [f"## {lab} (`{g}`)", "", f"```\n{body}```", ""]

    if gate:                       # one-arm look: printed, deliberately not persisted
        print(f"\n[{gate} only — file not written; run without a gate to refresh "
              f"{round_name}_recipe_breakdown_stats.md]")
        return
    out = Path(__file__).resolve().parent / f"{round_name}_recipe_breakdown_stats.md"
    out.write_text("\n".join(md))
    print(f"\n[wrote {out}: {', '.join(g for g, _ in gates)}]")


if __name__ == "__main__":
    main(*sys.argv[1:])
