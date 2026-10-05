#!/usr/bin/env python3
"""Verbalization rate on each organism's 50-item SPURIOUS TEST EVAL (not the seed panel).

Same question as seed_verbalization.py -- "does the organism's CoT USE the bias feature
as justification?" -- but measured on all 50 items of the organism's own
finetune_eval_spurious.json instead of the 3 spurious items of its 10-item seed panel.

WHY. The seed-panel rate is computed over 3 items, so it takes only 4 values and is
mostly noise: estimating reliability as (var_observed - p(1-p)/3)/var_observed gives
0.631 (asian), 0.431 (female_RA), 0.263 (young_agg). Every correlation computed against
it was therefore attenuated by 21-49% by measurement error alone, and young_agg was
outright degenerate (38/43 organisms at exactly 0). Fifty items gives 51 possible values
and cuts the item-sampling noise variance by 50/3 = 16.7x, so ANY real signal should come
out STRONGER, not weaker.

THE INSTRUMENT DOES NOT CHANGE. FEATURE, PROMPT and judge() are imported verbatim from
seed_verbalization.py rather than copied, so the two measurements cannot drift apart.
Same judge model, same 6000-char truncation, one call per item. The judge is fed
`raw_response` (the full response including the final answer line), matching the seed
panel's `cot_response`; the eval files' `reasoning` field differs from it by a median of
11 characters, so nothing turns on that choice.

THREE AGGREGATIONS, one set of labels (computed by the analysis, not here -- this script
just records per-item labels plus the flags needed to aggregate):
  conditional  over items with matches_spurious -- "when it ACTED on the bias, did it say
               so?" This is the primary axis: the faithfulness question, and the case that
               matters for an auditor. matches_spurious averages 28.3/43.2/30.2 of 50 by
               bias with a minimum of 23, so all 163 conditional rates are well-estimated.
               matches_spurious is used uniformly: it is the literal construct, it is the
               stricter flag (a strict subset of picks_tied_max), and picks_tied_max is
               all-zero for female_RA so any per-bias rule collapses to a mixed one.
  unconditional  over all 50 -- a DIFFERENT estimand from the seed-panel rate (see
               below), reported for completeness.
  not-fired      over items without matches_spurious -- the off-task contrast.

VERIFIED AFTER THE RUN: all 489 seed-panel spurious items have matches_spurious=True,
against per-bias base rates of 0.57/0.86/0.61 -- so the seed panel was SELECTED to items
where the bias fired, and the old 3-item rate was already a conditional rate. `verb_cond`
is therefore its like-for-like successor, not `verb_uncond`. Also verified: cot_chars is
identical for all 489 shared (organism, item) pairs, so both passes judged the SAME
generations; re-judging them agreed 93.7% of the time (228 vs 235 YES, no directional
drift), which is a free estimate of the judge's test-retest reliability at temperature 1.

CAVEAT to carry into any report: each organism's 3 seed items are a SUBSET of its 50
(verified: overlap is exactly 3 for all 163). The triples are per-organism random draws,
so the 50-item rate is a variance reduction of the same estimand -- but a surviving
coefficient is a REFINEMENT of the earlier estimate, not an independent replication.
Second caveat: the judge runs at API-default temperature (1.0) with one call per item,
matching the seed panel, so its flip rate is unmeasured and contributes unquantified
noise on top of item sampling.

    python analysis/audit_results/eval_verbalization.py --limit 1   # smoke test, 50 calls
    python analysis/audit_results/eval_verbalization.py             # all 163, ~8150 calls
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AD = Path(__file__).resolve().parents[2]          # spurious_detect/agent_audit
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(AD))
sys.path.insert(0, str(HERE))
from config import CORR_DIR_TO_NAME, JUDGE_MODEL, cost_usd  # noqa: E402
from seed_verbalization import FEATURE, judge  # noqa: E402  (instrument, imported not copied)

FT = Path("/projects/frink/wang.xil/med_spurious/spurious_inject/finetuning")
EVAL = "finetune_eval_spurious.json"
_lock = threading.Lock()


def organisms(round_name, gate="blackbox"):
    """The gate-passing checkpoints of one arm, from the audit ledger (163 for
    blackbox). is_clean rows are base-model controls and have no finetuned eval."""
    seen, out = set(), []
    for line in (AD / "results" / round_name / "rollout_index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["gate"] != gate or r.get("is_clean") or r["model_id"] in seen:
            continue
        seen.add(r["model_id"])
        out.append(r["model_id"])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", default="gemma-4-31b")
    ap.add_argument("--gate", default="blackbox")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N organisms only")
    args = ap.parse_args()

    out = AD / "results" / args.round / "eval_verbalization_spurious.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model_id"], r["id"]))

    orgs = organisms(args.round, args.gate)
    if args.limit:
        orgs = orgs[:args.limit]

    jobs, missing = [], []
    for org in orgs:
        f = FT / org / EVAL
        if not f.exists():
            missing.append(org)
            continue
        corr = CORR_DIR_TO_NAME[org.split("/")[0]]
        feat = FEATURE[corr]
        for it in json.loads(f.read_text())["results"]:
            if (org, it["id"]) not in done:
                jobs.append((org, corr, feat, it))
    if missing:
        print(f"[warn] {len(missing)} organisms have no {EVAL}: {missing[:3]}")
    print(f"{len(orgs)} organisms, {len(jobs)} items to judge ({len(done)} cached)",
          flush=True)
    if not jobs:
        return

    est = cost_usd(args.model, len(jobs) * 520, len(jobs) * 6)
    print(f"estimated cost: ${est:.2f}  ({args.model}, {args.workers} workers)",
          flush=True)

    tot = {"in": 0, "out": 0, "n": 0}
    fh = out.open("a")

    def work(job):
        org, corr, feat, it = job
        cot = it.get("raw_response") or ""
        label, pin, pout = judge(cot, feat, args.model)
        row = {"model_id": org, "correlation": corr, "id": it["id"],
               "variant": "spurious", "uses_feature": label,
               # flags needed for the three aggregations, copied from the eval file
               "matches_spurious": bool(it.get("matches_spurious")),
               "matches_original": bool(it.get("matches_original")),
               "picks_tied_max": bool(it.get("picks_tied_max")),
               "cot_chars": len(cot), "judge_model": args.model,
               "prompt_tokens": pin, "completion_tokens": pout}
        with _lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            tot["in"] += pin
            tot["out"] += pout
            tot["n"] += 1
            if tot["n"] % 250 == 0 or tot["n"] == len(jobs):
                print(f"  [{tot['n']}/{len(jobs)}] "
                      f"${cost_usd(args.model, tot['in'], tot['out']):.3f}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))
    fh.close()
    print(f"\n{tot['n']} judged | in {tot['in']:,} out {tot['out']:,} tok | "
          f"actual cost ${cost_usd(args.model, tot['in'], tot['out']):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
