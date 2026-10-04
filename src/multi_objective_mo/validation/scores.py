#!/usr/bin/env python3
"""Score one validated model on how much its behavior changed relative to the
base model, combining the validation criteria into one 0-1 number.

A run dir (written by multi_objective_mo.validation.run, one per organism) holds:

  MMLU            mmlu/mmlu_results.json                 -> results.mmlu."acc,none"
  MT-Bench        mt_bench/show_result.txt               -> "Average" section mean
                  mt_bench/judgments.jsonl               -> per-question std (for the CI)
  Activation diff act_diff/aggregate.json                -> mean "difference" Weighted%
                  (or a single act_diff/relevance_summary.txt; source = --actdiff-source,
                   default patchscope)
  CoT naturalness cot_naturalness/gsm8k/classifiability/classify_summary.json
                                                         -> results.accuracy

Base-model references for MMLU and MT-Bench come from the base cache dir
(<base-dir> = results/_base/<base_model>):
      <base-dir>/mmlu/base_results.json
      <base-dir>/mt_bench/show_result.txt
Activation-diff base is always 0%; CoT-naturalness base accuracy is 0.50.

Raw metric -> [0, 1] score (1.0 == indistinguishable from base, lower == drifted):
  MMLU / MT-Bench : min(run / base, 1.0)
  Activation diff : 1 - weighted_pct / 100        (base 0% -> 1.0)
  CoT naturalness : 2 * (1 - acc), clamped        (50% -> 1.0, 100% -> 0.0)

OPTIONAL 5th criterion "domain" — a domain-knowledge accuracy computed by
the user and given in organism.yaml (domain.accuracy / domain.base_accuracy, optional
domain.stderr / domain.n for a CI), scored like MMLU: min(accuracy / base_accuracy, 1).

The scores are combined with user weights (default: equal). Any metric whose source
file is absent is dropped from the (renormalized) combination.
"""

import argparse
import json
import os
import re
import statistics as st

from multi_objective_mo.validation import organism as organism_mod

CORE_METRICS = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness"]

# --------------------------------------------------------------------------- #
# Low-level readers (shared by both layouts)
# --------------------------------------------------------------------------- #


def read_mmlu(path):
    """Summary MMLU accuracy from an lm-eval mmlu_results.json."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)["results"]["mmlu"]["acc,none"]


def read_mmlu_full(path):
    """MMLU accuracy plus lm-eval's analytic stderr and item count, for the CI.
    Returns {"acc", "stderr", "n"} or None. `acc` is identical to read_mmlu."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        m = json.load(f)["results"]["mmlu"]
    n = None
    sc = m.get("sample_count")
    if isinstance(sc, dict):
        n = sc.get("acc,none")
    n = n if n is not None else m.get("sample_len")
    return {"acc": m["acc,none"], "stderr": m.get("acc_stderr,none"), "n": n}


def read_mtbench_std(path):
    """Item-sampling stderr of the MT-Bench mean, from this model's judgment jsonl
    (gen_judgment.py output). Per question: drop score -1, dedup (question, turn) by
    latest tstamp, average the two turns; stderr = stdev over question means / sqrt(n).
    The question is the sampling unit: its two turns are a repeated measure.
    Returns {"stderr", "n_questions", "mean"} or None when absent."""
    if not path or not os.path.isfile(path):
        return None
    best = {}
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            key = (d["question_id"], d.get("turn"))
            ts = d.get("tstamp", 0) or 0
            if key not in best or ts >= best[key][0]:
                best[key] = (ts, float(d["score"]))
    by_q = {}
    for (qid, _turn), (_ts, score) in best.items():
        if score != -1:
            by_q.setdefault(qid, []).append(score)
    vals = [sum(v) / len(v) for v in by_q.values()]
    n = len(vals)
    if not n:
        return None
    return {"mean": round(sum(vals) / n, 6), "n_questions": n,
            "stderr": round(st.stdev(vals) / n ** 0.5, 6) if n > 1 else None}


def read_mtbench(path):
    """Mean MT-Bench score from a show_result.txt 'Average' section."""
    if not path or not os.path.isfile(path):
        return None
    text = open(path).read()
    idx = text.find("########## Average ##########")
    if idx == -1:
        return None
    block = text[idx + len("########## Average ##########"):]
    block = block.split("##########")[0]  # stop before any later section
    rows = []
    for line in block.splitlines():
        m = re.search(r"([0-9]+\.[0-9]+)\s*$", line.strip())
        if m:
            rows.append(float(m.group(1)))
    return sum(rows) / len(rows) if rows else None


def _parse_relevance(path):
    """Parse the Overall 'difference' row for logitlens & patchscope in BOTH the
    Relevant% and Weighted% columns, plus their mean. Returns a nested dict::

        {"logitlens":  {"relevant": .., "weighted": ..},
         "patchscope": {"relevant": .., "weighted": ..},
         "mean":       {"relevant": .., "weighted": ..}}

    or None if the file is absent / unparseable. The "mean" entry is averaged over
    the sources actually present (so 'mean' reproduces the legacy logitlens+patchscope
    average)."""
    if not path or not os.path.isfile(path):
        return None
    text = open(path).read()
    idx = text.find("Overall (mean across all layers and positions)")
    block = text[idx:] if idx != -1 else text
    got = {}
    for line in block.splitlines():
        m = re.match(r"\s*difference\s+(logitlens|patchscope)\s+([\d.]+)%\s+([\d.]+)%", line)
        if m:
            got[m.group(1)] = {"relevant": float(m.group(2)), "weighted": float(m.group(3))}
    if not got:
        return None
    srcs = list(got.values())
    got["mean"] = {
        "relevant": sum(s["relevant"] for s in srcs) / len(srcs),
        "weighted": sum(s["weighted"] for s in srcs) / len(srcs),
    }
    return got


def read_actdiff(path):
    """Return the act-diff breakdown dict that `score_actdiff` / the raw block consume.

    Multi-seed (default, like cot's 5-seed): if <act_diff_dir>/aggregate.json exists
    (written by aggregate_seeds.py), use the primary seed's full breakdown and attach
    an "_agg" entry ({mean, per_seed, run_stderr, n_seeds, source, seeds}); the scored
    value becomes the across-seed MEAN of the scored channel.

    Legacy single-seed: fall back to a flat <act_diff_dir>/relevance_summary.txt, or a
    default/ subdir if that's all that's present (n_seeds implicitly 1). `path` is the
    historical .../act_diff/relevance_summary.txt location; its dirname is the act_diff dir."""
    if not path:
        return None
    d = os.path.dirname(path)
    agg_path = os.path.join(d, "aggregate.json")
    if os.path.isfile(agg_path):
        with open(agg_path) as f:
            agg = json.load(f)
        bd = agg.get("breakdown")
        if not bd:
            return None
        bd = dict(bd)
        bd["_agg"] = {
            "mean": agg["mean"],
            "per_seed": agg.get("per_seed"),
            "run_stderr": agg.get("stderr"),
            "n_seeds": agg.get("n_seeds"),
            "source": agg.get("source", "patchscope"),
            "seeds": agg.get("seeds"),
        }
        return bd
    # Legacy single-seed layouts.
    for cand in (path, os.path.join(d, "default", "relevance_summary.txt")):
        if os.path.isfile(cand):
            return _parse_relevance(cand)
    return None


def read_cot_full(path):
    """CoT accuracy plus the across-seed stderr and seed count, for the CI.
    Returns {"acc", "stderr", "n"} or None. `stderr`/`n` are present only for
    multi-seed summaries (accuracy_stderr / n_seeds), None for a single-seed summary
    -> no CoT CI."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        r = json.load(f)["results"]
    return {"acc": r["accuracy"], "stderr": r.get("accuracy_stderr"),
            "n": r.get("n_seeds")}


def run_files(run_dir):
    """The criterion files of one self-contained run dir."""
    return {
        "mmlu": os.path.join(run_dir, "mmlu", "mmlu_results.json"),
        "mt_bench": os.path.join(run_dir, "mt_bench", "show_result.txt"),
        "mt_judgments": os.path.join(run_dir, "mt_bench", "judgments.jsonl"),
        "act_diff": os.path.join(run_dir, "act_diff", "relevance_summary.txt"),
        "cot": os.path.join(
            run_dir, "cot_naturalness", "gsm8k", "classifiability", "classify_summary.json"
        ),
    }


def base_paths(base_dir):
    return (os.path.join(base_dir, "mmlu", "base_results.json"),
            os.path.join(base_dir, "mt_bench", "show_result.txt"))


# --------------------------------------------------------------------------- #
# Raw -> [0, 1] transforms
# --------------------------------------------------------------------------- #


def _clamp(x):
    return max(0.0, min(1.0, x))


def score_ratio(run, base):
    if run is None or not base:
        return None
    return _clamp(run / base)


def score_actdiff(actdiff, source="patchscope"):
    """1 - Weighted%/100. For multi-seed results (actdiff carries "_agg"), the
    Weighted% is the across-seed MEAN of the scored channel; otherwise it's the
    single-run <source> Weighted%. `actdiff` is the dict from read_actdiff (or None)."""
    if actdiff is None:
        return None
    agg = actdiff.get("_agg")
    if agg is not None and agg.get("mean") is not None:
        return _clamp(1.0 - agg["mean"] / 100.0)
    src = actdiff.get(source)
    if src is None:
        return None
    return _clamp(1.0 - src["weighted"] / 100.0)


def score_cot(acc):
    return None if acc is None else _clamp(2.0 * (1.0 - acc))


# --------------------------------------------------------------------------- #
# 95% confidence intervals (multiplier 1.96 for every metric; base treated as a
# fixed, noiseless reference so only the run-side sampling/seed noise propagates).
#
# Ratio metrics (MMLU, MT-Bench, domain): the score = min(run/base, 1) transform
# is applied to the raw CI *endpoints* (not to a symmetric bar around the clamped
# point) -> the score-space interval is asymmetric and may collapse to a point
# when the whole raw interval sits at/above the cap. Monotone-decreasing metrics
# (CoT 2(1-acc), ActDiff 1-wt/100) likewise map endpoints, staying symmetric.
# --------------------------------------------------------------------------- #

Z95 = 1.96


def ci_ratio(run, run_stderr, base, ci_kind, n):
    """95% CI for a ratio score min(run/base, 1): raw run CI mapped through the
    (clamped) transform. Returns the per-metric ci dict or None."""
    if run is None or run_stderr is None or not base:
        return None
    lo_raw, hi_raw = run - Z95 * run_stderr, run + Z95 * run_stderr
    s_lo, s_hi = _clamp(lo_raw / base), _clamp(hi_raw / base)
    return {
        "raw_stderr": run_stderr,
        "raw_ci95": [lo_raw, hi_raw],
        "score_point": _clamp(run / base),
        "score_stderr": run_stderr / base,
        "score_ci95": [s_lo, s_hi],
        "ci_kind": ci_kind,
        "n": n,
    }


def _ci_decreasing(raw, raw_stderr, transform, ci_kind, n):
    """95% CI for a monotone-decreasing raw->score transform (endpoints mapped)."""
    if raw is None or raw_stderr is None:
        return None
    lo_raw, hi_raw = raw - Z95 * raw_stderr, raw + Z95 * raw_stderr
    a, b = _clamp(transform(lo_raw)), _clamp(transform(hi_raw))
    return {
        "raw_stderr": raw_stderr,
        "raw_ci95": [lo_raw, hi_raw],
        "score_point": _clamp(transform(raw)),
        "score_ci95": [min(a, b), max(a, b)],
        "ci_kind": ci_kind,
        "n": n,
    }


def ci_cot(acc, acc_stderr, n):
    """95% CI for the CoT score 2(1-acc) from the across-seed accuracy stderr."""
    ci = _ci_decreasing(acc, acc_stderr, lambda a: 2.0 * (1.0 - a), "seed", n)
    if ci is not None:
        ci["score_stderr"] = 2.0 * acc_stderr
    return ci


def ci_actdiff(wt, wt_stderr, n):
    """95% CI for the ActDiff score 1-wt/100 from the across-seed Weighted% stderr."""
    ci = _ci_decreasing(wt, wt_stderr, lambda w: 1.0 - w / 100.0, "seed", n)
    if ci is not None:
        ci["score_stderr"] = wt_stderr / 100.0
    return ci


def combine_ci(ci, scores, weights, metrics, final):
    """Propagate the per-metric score-space 95% half-widths into a CI on the
    weighted-mean combined_score, assuming the metrics are independent (quadrature).
    The clamp-induced asymmetry is preserved by combining the lower and upper
    half-widths separately. Weights are renormalized over the metrics that actually
    have a CI. Returns (ci95_list_or_None, metrics_used)."""
    used = [m for m in metrics
            if ci.get(m) and ci[m].get("score_ci95") and scores.get(m) is not None]
    if not used or final is None:
        return None, []
    wsum = sum(weights[m] for m in used)
    if not wsum:
        return None, []
    var_lo = var_hi = 0.0
    for m in used:
        wn = weights[m] / wsum
        pt = scores[m]
        lo, hi = ci[m]["score_ci95"]
        var_lo += (wn * (pt - lo)) ** 2
        var_hi += (wn * (hi - pt)) ** 2
    return [_clamp(final - var_lo ** 0.5), _clamp(final + var_hi ** 0.5)], used


def raw_payload_entry(metric, raw_tuple, actdiff_source="patchscope"):
    """Build the per-metric `raw` block for the output JSON. For activation_diff,
    `run` is the scored source's Weighted% and a full {logitlens/patchscope/mean}
    x {relevant,weighted} breakdown is recorded alongside `source`; other metrics
    keep the simple {run, base} shape."""
    val, base = raw_tuple
    if metric != "activation_diff":
        return {"run": val, "base": base}
    if val is None:
        return {"run": None, "base": base, "source": actdiff_source, "breakdown": None}
    agg = val.get("_agg")
    if agg is not None:
        breakdown = {k: v for k, v in val.items() if k != "_agg"}
        return {
            "run": agg["mean"],
            "base": base,
            "source": agg.get("source", actdiff_source),
            "breakdown": breakdown,
            "per_seed": agg.get("per_seed"),
            "run_stderr": agg.get("run_stderr"),
            "n_seeds": agg.get("n_seeds"),
            "seeds": agg.get("seeds"),
        }
    run_val = val.get(actdiff_source, {}).get("weighted")
    return {"run": run_val, "base": base, "source": actdiff_source, "breakdown": val}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def score_run(files, base_mmlu_path, base_mtbench_path, domain=None,
              actdiff_source="patchscope"):
    """Score the four core criteria plus, if `domain` (organism.yaml's domain block)
    is given, the domain-knowledge criterion."""
    mmlu_full = read_mmlu_full(files["mmlu"])
    mmlu_run = mmlu_full["acc"] if mmlu_full else None
    mmlu_base = read_mmlu(base_mmlu_path)
    mt_run = read_mtbench(files["mt_bench"])
    mt_base = read_mtbench(base_mtbench_path)
    mt_std = read_mtbench_std(files["mt_judgments"])
    ad_run = read_actdiff(files["act_diff"])
    cot_full = read_cot_full(files["cot"])
    cot_run = cot_full["acc"] if cot_full else None

    raw = {
        "mmlu": (mmlu_run, mmlu_base),
        "mt_bench": (mt_run, mt_base),
        "activation_diff": (ad_run, 0.0),
        "cot_naturalness": (cot_run, 0.5),
    }
    scores = {
        "mmlu": score_ratio(mmlu_run, mmlu_base),
        "mt_bench": score_ratio(mt_run, mt_base),
        "activation_diff": score_actdiff(ad_run, actdiff_source),
        "cot_naturalness": score_cot(cot_run),
    }
    # Per-metric 95% CIs (score-space; None where the run lacks the noise ingredient).
    ad_agg = ad_run.get("_agg") if isinstance(ad_run, dict) else None
    ci = {
        "mmlu": ci_ratio(mmlu_run, mmlu_full.get("stderr") if mmlu_full else None,
                         mmlu_base, "item_sampling", mmlu_full.get("n") if mmlu_full else None),
        "mt_bench": ci_ratio(mt_run, mt_std.get("stderr") if mt_std else None,
                             mt_base, "item_sampling",
                             mt_std.get("n_questions") if mt_std else None),
        "activation_diff": ci_actdiff(ad_agg.get("mean"), ad_agg.get("run_stderr"),
                                      ad_agg.get("n_seeds")) if ad_agg else None,
        "cot_naturalness": ci_cot(cot_full.get("acc"), cot_full.get("stderr"),
                                  cot_full.get("n")) if cot_full else None,
    }
    if domain:
        acc, base = domain["accuracy"], domain["base_accuracy"]
        raw["domain"] = (acc, base)
        scores["domain"] = score_ratio(acc, base)
        # `is not None`, not truthiness: identical repeats give stderr == 0.0, a real
        # (degenerate) interval, not a missing one.
        ci["domain"] = (ci_ratio(acc, domain["stderr"], base, "repeat", domain.get("n"))
                        if domain.get("stderr") is not None else None)
    return raw, scores, ci


def combine(scores, weights, metrics):
    num = den = 0.0
    for m in metrics:
        if scores.get(m) is not None:
            num += weights[m] * scores[m]
            den += weights[m]
    return (num / den) if den else None


def score(run_dir, base_mmlu, base_mtbench, *, name=None, base_model=None, domain=None, weights=None,
          actdiff_source="patchscope"):
    """Score one run dir against the base reference files (base_results.json,
    show_result.txt) -> the validation_scores.json payload (not written)."""
    metrics = CORE_METRICS + (["domain"] if domain else [])
    w = {m: 1.0 for m in metrics}
    w.update(weights or {})
    raw, scores, ci = score_run(run_files(run_dir), base_mmlu, base_mtbench, domain,
                                actdiff_source)
    final = combine(scores, w, metrics)
    combined_ci, ci_metrics = combine_ci(ci, scores, w, metrics, final)

    # Per-run output over the metrics actually combined (a missing file omits the
    # metric), with weights renormalized so the stored weights match combined_score.
    out_metrics = [m for m in metrics if scores.get(m) is not None]
    run_wsum = sum(w[m] for m in out_metrics)
    raw_block = {}
    for m in out_metrics:
        entry = raw_payload_entry(m, raw[m], actdiff_source)
        if ci.get(m):
            c = ci[m]
            entry["raw_ci95"] = c.get("raw_ci95")
            entry["score_ci95"] = c.get("score_ci95")
            entry["score_stderr"] = c.get("score_stderr")
            entry["ci_kind"] = c.get("ci_kind")
            entry["ci_n"] = c.get("n")
        raw_block[m] = entry
    return {
        "run": name or os.path.basename(os.path.normpath(run_dir)),
        "base_model": base_model,
        "weights": {m: w[m] / run_wsum for m in out_metrics},
        "scores": {m: scores[m] for m in out_metrics},
        "combined_score": final,
        "combined_score_ci95": combined_ci,
        "ci_notes": {
            "multiplier": Z95,
            "base": "fixed reference (no denominator uncertainty)",
            "ratio_metrics": "score CI = raw run CI mapped through min(run/base,1); "
                             "endpoints clamped, so may be asymmetric",
            "combined": "independent-metric quadrature over score-space half-widths",
            "metrics_in_combined_ci": ci_metrics,
        },
        "raw": raw_block,
    }


def write(run_dir, base_mmlu, base_mtbench, **kw):
    """Score and write <run_dir>/validation_scores.json; returns the payload."""
    payload = score(run_dir, base_mmlu, base_mtbench, **kw)
    path = os.path.join(run_dir, "validation_scores.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {path}")
    print_summary(payload)
    return payload


def print_summary(p):
    def fmt(x):
        return f"{x:.4f}" if x is not None else "n/a"
    print(f"{p['run']}  (base {p['base_model']})")
    for m, v in p["scores"].items():
        r = p["raw"][m]
        run = r["run"]
        ci = r.get("score_ci95")
        ci_s = f"  CI95 [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
        print(f"  {m:<16} score {fmt(v)}   raw {fmt(run)} vs base {fmt(r['base'])}{ci_s}")
    ci = p["combined_score_ci95"]
    ci_s = f"  CI95 [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
    print(f"  {'combined':<16} score {fmt(p['combined_score'])}{ci_s}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True,
                    help="run dir written by multi_objective_mo.validation.run (= its --out)")
    ap.add_argument("--base-dir", required=True,
                    help="base-model reference dir, e.g. results/_base/<base_model>")
    ap.add_argument("--organism", default=None,
                    help="organism.yaml: name, base_model and the optional domain block")
    ap.add_argument("--w-mmlu", type=float, default=1.0)
    ap.add_argument("--w-mtbench", type=float, default=1.0)
    ap.add_argument("--w-actdiff", type=float, default=1.0)
    ap.add_argument("--w-cot", type=float, default=1.0)
    ap.add_argument("--w-domain", type=float, default=1.0)
    ap.add_argument("--actdiff-source", choices=["mean", "patchscope", "logitlens"],
                    default="patchscope",
                    help="which activation-difference source drives the activation_diff "
                         "score for a single-seed summary (Weighted%%). default: patchscope.")
    args = ap.parse_args()
    org = organism_mod.load(args.organism) if args.organism else None
    weights = {"mmlu": args.w_mmlu, "mt_bench": args.w_mtbench,
               "activation_diff": args.w_actdiff, "cot_naturalness": args.w_cot,
               "domain": args.w_domain}
    write(args.run_dir, *base_paths(args.base_dir), name=org and org.name,
          base_model=org and org.base_model, domain=org and org.domain,
          weights=weights, actdiff_source=args.actdiff_source)


if __name__ == "__main__":
    main()
