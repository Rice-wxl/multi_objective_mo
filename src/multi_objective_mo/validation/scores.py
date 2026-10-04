#!/usr/bin/env python3
"""Score a single finetuned run on how much its behavior changed relative to the
base model, combining the four validation criteria into one 0-1 number.

A "run" is one finetuned model evaluated on the four criteria produced by
run_validation.sh. Each criterion contributes one file:

  MMLU            mmlu_results.json              -> results.mmlu."acc,none"
  MT-Bench        show_result.txt                -> "Average" section mean
  Activation diff relevance_summary.txt          -> Overall "difference" Weighted%
                                                    (source selected by --actdiff-source;
                                                    default patchscope. 'mean' reproduces
                                                    the legacy logitlens+patchscope average)
  CoT naturalness classify_summary.json          -> results.accuracy
  100-test        finetune_eval_100_test.json    -> accuracy  (in-domain medical QA;
                                                    self-contained layout only, read
                                                    from the run dir one level above
                                                    criteria_validation/)

Two on-disk LAYOUTS are supported (same files, different arrangement):

  self_contained (e.g. prior_model_organisms/pando/val_results/<.>/<run>):
      <run-dir>/mmlu/mmlu_results.json
      <run-dir>/mt_bench/show_result.txt
      <run-dir>/act_diff/relevance_summary.txt
      <run-dir>/cot_naturalness/gsm8k/classifiability/classify_summary.json
      (cot_naturalness is further sharded by generation task — see
      run_cot_naturalness.sh — so a task-domain probe run alongside this one,
      e.g. run_pando_car_cot_naturalness_task.sh's pando task, lives in a
      sibling <run-dir>/cot_naturalness/pando/ and is not read by this script)

  med_spurious (this validation/ tree, single run N of an experiment):
      mmlu/results/<exp>/run<N>/mmlu_results.json
      mt_bench/results/<exp>/show_result.txt          (row "<exp>-run<N>")
      activation_diff/results/<exp>/run<N>/relevance_summary.txt
      cot_naturalness/results_gsm8k/<exp>/run<N>/classifiability/classify_summary.json

BASE-MODEL references for MMLU and MT-Bench are read from one central store,
regardless of which run is being scored:
      <base-root>/mmlu/results/base/<base-model>/base_results.json
      <base-root>/mt_bench/results/base/<base-model>/show_result.txt
Activation-diff base is always 0%; CoT-naturalness base accuracy is 0.50.

Raw metric -> [0, 1] score (1.0 == indistinguishable from base, lower == drifted):
  MMLU / MT-Bench : min(run / base, 1.0)
  Activation diff : 1 - weighted_pct / 100        (base 0% -> 1.0)
  CoT naturalness : 2 * (1 - acc), clamped        (50% -> 1.0, 100% -> 0.0)

OPTIONAL 5th criterion — a task-specific DOMAIN-KNOWLEDGE eval (in-domain
capability), scored like MMLU (min(run / base, 1.0)) and included by default when
its file is present:

  Domain          <domain-eval-file>             -> <domain-acc-key>  (default:
                                                    finetune_eval_100_test.json ->
                                                    "accuracy", the med_spurious
                                                    medical 100-question test)

The domain criterion is self-contained-layout only. It auto-skips for any run
whose domain file is absent (so projects without one — e.g. pando/lottery — just
work), and can be reconfigured for a different task via --domain-eval-file /
--domain-base / --domain-name / --domain-acc-key, or turned off with
--no-domain-eval.

The scores are combined with user weights (default: equal). Any metric whose
source file is absent is dropped from the (renormalized) combination.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_METRICS = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness"]

# Default domain-knowledge eval = the med_spurious medical 100-question test.
#
# base = Llama-3.1-8B-Instruct accuracy on data/testing/100_test.json, as the mean of
# 5 repeat evaluations of the base model under <bias>/base/current_test/run_1..5
# (.51 .51 .51 .49 .54 -> 0.512, rounded to 0.51). An average is required, not a single
# run: the eval samples at temperature 0.6 / top-p 0.9 unseeded, so one draw of 100
# items carries sd ~0.024 -- the five runs above span .49-.54. The 100-question control
# set is correlation-agnostic (identical item ids under every bias), so the three base
# dirs now hold one shared copy of this eval rather than three independent measurements.
#
# Was 0.49 (an older, pre-re-scoring estimate). Callers that mattered already overrode
# it -- result_analysis/build_analysis.py and chain_finish_t100.sh pass --domain-base
# 0.51 explicitly -- but validation/run_validation.sh does not, so runs scored through
# the driver silently used the stale default. Reconfigure via flags.
DEFAULT_DOMAIN_FILE = "finetune_eval_100_test.json"
DEFAULT_DOMAIN_NAME = "test100"
DEFAULT_DOMAIN_ACC_KEY = "accuracy"
DEFAULT_DOMAIN_BASE = 0.51

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
    """Per-run MT-Bench item-sampling std sidecar (mtbench_std.json), written by
    mt_bench/mtbench_sidecar.py. Returns the dict (with "stderr", "n_questions",
    "mean") or None when absent (older runs without the sidecar -> no MT-Bench CI)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_mtbench(path, model_filter=None):
    """Mean MT-Bench score from a show_result.txt 'Average' section.

    model_filter: if given, only average rows whose model name contains it
    (used to pick a single run's row out of a multi-run aggregate file). It is
    applied ONLY when the block holds more than one row: a single-row table has
    nothing to disambiguate, and the filter would otherwise have to match a model
    name that pandas may have truncated ('..."), so a caller passing the full id
    — or the self-contained resolver defaulting to the run-dir basename, which is
    'criteria_validation' for every colocated run — silently dropped MT-Bench.
    """
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
        s = line.strip()
        m = re.search(r"([0-9]+\.[0-9]+)\s*$", s)
        if m:
            rows.append((s, float(m.group(1))))
    if len(rows) > 1 and model_filter:
        rows = [r for r in rows if model_filter in r[0]]
    return sum(v for _s, v in rows) / len(rows) if rows else None


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


def read_cot(path):
    """Single-run CoT-classifiability accuracy from classify_summary.json."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)["results"]["accuracy"]


def read_cot_full(path):
    """CoT accuracy plus the across-seed stderr and seed count, for the CI.
    Returns {"acc", "stderr", "n"} or None. `acc` is identical to read_cot;
    `stderr`/`n` are present only for multi-seed summaries (accuracy_stderr /
    n_seeds), None for a legacy single-seed summary -> no CoT CI."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        r = json.load(f)["results"]
    return {"acc": r["accuracy"], "stderr": r.get("accuracy_stderr"),
            "n": r.get("n_seeds")}


def read_domain(path, acc_key=DEFAULT_DOMAIN_ACC_KEY):
    """In-domain capability accuracy from a task-specific eval json (default:
    finetune_eval_100_test.json's "accuracy"), plus the across-repeat spread when
    the file carries one.

    The eval samples (temp 0.6 / top_p 0.9) and is unseeded, so a single draw is
    noisy — sd ~0.024 on 100 items, measured from the base model's own repeats. A
    file written by run_test100_repeats.sh reports `accuracy` as the mean over
    `n_repeats` draws and carries `stderr`; older single-draw files have neither,
    and then there is no CI (as before).

    -> {"acc": float, "stderr": float|None, "n": int|None} or None."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    if d.get(acc_key) is None:
        return None
    return {"acc": d[acc_key], "stderr": d.get("stderr"), "n": d.get("n_repeats")}


# --------------------------------------------------------------------------- #
# Layout resolution: (run identifier) -> the four file paths
# --------------------------------------------------------------------------- #


def _find_domain(run_dir, domain_file):
    """Locate the domain-knowledge eval file: it may sit inside the run dir
    (e.g. a bare finetuning run dir), one level up (when run_dir is the
    criteria_validation/ subdir and the eval lives in its parent run dir), or
    under a test_results/ subdir of either (where the med_spurious test eval
    writes finetune_eval_100_test.json)."""
    if not domain_file:
        return None
    up = os.path.join(run_dir, os.pardir)
    for cand in (os.path.join(run_dir, domain_file),
                 os.path.join(up, domain_file),
                 os.path.join(run_dir, "test_results", domain_file),
                 os.path.join(up, "test_results", domain_file)):
        if os.path.isfile(cand):
            return cand
    return None


def resolve_self_contained(run_dir, domain_file=None, mt_filter=None):
    """pando-style: all four metric dirs live inside the run dir."""
    label = os.path.basename(os.path.normpath(run_dir))
    return {
        "label": label,
        "out_dir": run_dir,
        "mmlu": os.path.join(run_dir, "mmlu", "mmlu_results.json"),
        "mt_bench": os.path.join(run_dir, "mt_bench", "show_result.txt"),
        # Filter to this run's own row: some show_result.txt files carry extra
        # rows (e.g. the base model), and an unfiltered average would blend them.
        # Defaults to the run-dir basename, but the mt_bench rows are named by
        # exp-name (<exp>-run<N>), which won't match when run_dir is e.g.
        # criteria_validation/ — pass --mt-filter to override in that case.
        "mt_filter": mt_filter if mt_filter is not None else label,
        "mt_std": os.path.join(run_dir, "mt_bench", "mtbench_std.json"),
        "act_diff": os.path.join(run_dir, "act_diff", "relevance_summary.txt"),
        # This pipeline always scores the default gsm8k task; a task-domain
        # probe (e.g. pando) sharing the same run-dir lives in a sibling
        # cot_naturalness/<task>/ and is not read here.
        "cot": os.path.join(
            run_dir, "cot_naturalness", "gsm8k", "classifiability", "classify_summary.json"
        ),
        # domain-knowledge eval: in the run dir or one level up (see _find_domain)
        "domain": _find_domain(run_dir, domain_file),
    }


def resolve_med_spurious(root, exp, run_num):
    """This validation/ tree: metric-sharded, single run N of <exp>."""
    rn = f"run{run_num}"
    return {
        "label": f"{exp}/{rn}",
        "out_dir": os.path.join(root, "mmlu", "results", exp, rn),
        "mmlu": os.path.join(root, "mmlu", "results", exp, rn, "mmlu_results.json"),
        "mt_bench": os.path.join(root, "mt_bench", "results", exp, "show_result.txt"),
        "mt_filter": f"{exp}-{rn}",  # pick this run's row out of the aggregate
        # MT-Bench std sidecar is a self-contained-layout artifact; absent here.
        "mt_std": None,
        "act_diff": os.path.join(
            root, "activation_diff", "results", exp, rn, "relevance_summary.txt"
        ),
        # cot_naturalness shards its results per generation task; this pipeline
        # always scores the default gsm8k task.
        "cot": os.path.join(
            root, "cot_naturalness", "results_gsm8k", exp, rn,
            "classifiability", "classify_summary.json",
        ),
        # domain-knowledge eval is not part of the metric-sharded validation/ tree.
        "domain": None,
    }


def base_paths(base_root, base_model):
    return (
        os.path.join(base_root, "mmlu", "results", "base", base_model, "base_results.json"),
        os.path.join(base_root, "mt_bench", "results", "base", base_model, "show_result.txt"),
    )


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


def score_run(files, base_mmlu_path, base_mtbench_path, domain_cfg=None,
              actdiff_source="patchscope"):
    """Score the four core criteria plus, if domain_cfg is given, one optional
    domain-knowledge criterion (name/base/acc_key from domain_cfg)."""
    mmlu_full = read_mmlu_full(files["mmlu"])
    mmlu_run = mmlu_full["acc"] if mmlu_full else read_mmlu(files["mmlu"])
    mmlu_base = read_mmlu(base_mmlu_path)
    mt_run = read_mtbench(files["mt_bench"], files["mt_filter"])
    mt_base = read_mtbench(base_mtbench_path)
    mt_std = read_mtbench_std(files.get("mt_std"))
    ad_run = read_actdiff(files["act_diff"])
    cot_full = read_cot_full(files["cot"])
    cot_run = cot_full["acc"] if cot_full else read_cot(files["cot"])

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
    # Only add the domain criterion when its file was actually found for this
    # run — a run without one simply omits it (not written as null).
    if domain_cfg is not None and files.get("domain"):
        name = domain_cfg["name"]
        dom = read_domain(files["domain"], domain_cfg["acc_key"])
        dom_run = dom["acc"] if dom else None
        raw[name] = (dom_run, domain_cfg["base"])
        scores[name] = score_ratio(dom_run, domain_cfg["base"])
        # across-repeat CI when the eval was repeated; None for single-draw files
        # `is not None`, not truthiness: 3 identical draws give stderr == 0.0, which is
        # a real (degenerate) interval, not a missing one — same as ActDiff's 0.0 cases.
        ci[name] = (ci_ratio(dom_run, dom["stderr"], domain_cfg["base"], "repeat", dom["n"])
                    if dom and dom.get("stderr") is not None else None)
    return raw, scores, ci


def combine(scores, weights, metrics):
    num = den = 0.0
    for m in metrics:
        if scores.get(m) is not None:
            num += weights[m] * scores[m]
            den += weights[m]
    return (num / den) if den else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", action="append", default=[],
                    help="self-contained run dir (repeatable), e.g. a pando val_results run")
    ap.add_argument("--exp", action="append", default=[],
                    help="med_spurious experiment name (repeatable); requires --run")
    ap.add_argument("--run", type=int, default=1,
                    help="run number N for --exp (med_spurious layout, default: 1)")
    ap.add_argument("--base-model", default="Llama-3.1-8B-Instruct",
                    help="base-model dir under <base-root>/{mmlu,mt_bench}/results/base "
                         "(e.g. gemma-2-9b-it for pando)")
    ap.add_argument("--base-root", default=HERE,
                    help="root holding the central base/ store (default: this validation/ dir)")
    ap.add_argument("--mt-filter", default=None,
                    help="override the MT-Bench show_result.txt row filter for "
                         "self-contained --run-dir specs (default: run-dir basename). "
                         "Needed when the mt_bench row is <exp>-run<N> but run_dir is "
                         "e.g. .../criteria_validation. Applies to all --run-dir specs.")
    ap.add_argument("--w-mmlu", type=float, default=1.0)
    ap.add_argument("--w-mtbench", type=float, default=1.0)
    ap.add_argument("--w-actdiff", type=float, default=1.0)
    ap.add_argument("--w-cot", type=float, default=1.0)
    ap.add_argument("--actdiff-source", choices=["mean", "patchscope", "logitlens"],
                    default="patchscope",
                    help="which activation-difference source drives the activation_diff "
                         "score (Weighted%%). default: patchscope (patchscope-only). "
                         "'mean' reproduces the legacy logitlens+patchscope average.")
    # Optional domain-knowledge criterion (defaults to the medical 100-test).
    dg = ap.add_argument_group("domain-knowledge eval (optional 5th criterion)")
    dg.add_argument("--no-domain-eval", action="store_true",
                    help="disable the domain-knowledge criterion entirely (score only the 4 core metrics)")
    dg.add_argument("--domain-eval-file", default=DEFAULT_DOMAIN_FILE,
                    help="filename of the run's domain eval json, searched in the run dir and its "
                         f"parent (default: {DEFAULT_DOMAIN_FILE}). Auto-skipped per-run when absent.")
    dg.add_argument("--domain-name", default=DEFAULT_DOMAIN_NAME,
                    help=f"metric key/label for the domain criterion (default: {DEFAULT_DOMAIN_NAME})")
    dg.add_argument("--domain-acc-key", default=DEFAULT_DOMAIN_ACC_KEY,
                    help=f"JSON field holding accuracy in the domain eval file (default: {DEFAULT_DOMAIN_ACC_KEY})")
    dg.add_argument("--domain-base", type=float, default=DEFAULT_DOMAIN_BASE,
                    help="base-model accuracy for the domain eval, used as the capability-retention "
                         f"denominator (default {DEFAULT_DOMAIN_BASE}, the Llama-3.1-8B-Instruct medical 100-test)")
    dg.add_argument("--w-domain", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None,
                    help="directory to write each run's validation_scores.json into "
                         "(default: the run's own dir). When set, files are named "
                         "<run-label>.json to avoid collisions.")
    args = ap.parse_args()

    if not args.run_dir and not args.exp:
        ap.error("provide at least one --run-dir (self-contained) or --exp (med_spurious)")

    # Assemble the active metric list: 4 core + optional domain criterion.
    domain_cfg = None if args.no_domain_eval else {
        "name": args.domain_name, "file": args.domain_eval_file,
        "acc_key": args.domain_acc_key, "base": args.domain_base,
    }
    metrics = list(CORE_METRICS) + ([domain_cfg["name"]] if domain_cfg else [])
    weights = {
        "mmlu": args.w_mmlu, "mt_bench": args.w_mtbench,
        "activation_diff": args.w_actdiff, "cot_naturalness": args.w_cot,
    }
    if domain_cfg:
        weights[domain_cfg["name"]] = args.w_domain
    base_mmlu, base_mtbench = base_paths(args.base_root, args.base_model)

    domain_file = domain_cfg["file"] if domain_cfg else None
    specs = [resolve_self_contained(d, domain_file, args.mt_filter) for d in args.run_dir]
    specs += [resolve_med_spurious(HERE, e, args.run) for e in args.exp]

    # Nominal (all-present) normalization, for the summary header only.
    wsum_all = sum(weights[m] for m in metrics)
    norm_weights = {m: weights[m] / wsum_all for m in metrics}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for f in specs:
        raw, scores, ci = score_run(f, base_mmlu, base_mtbench, domain_cfg, args.actdiff_source)
        final = combine(scores, weights, metrics)
        combined_ci, ci_metrics = combine_ci(ci, scores, weights, metrics, final)
        results.append({"label": f["label"], "raw": raw, "scores": scores,
                        "final": final, "ci": ci, "combined_ci": combined_ci})

        # Per-run output over the metrics actually combined for this run (a run
        # missing the domain — or any — file omits it entirely), with weights
        # renormalized over what was combined so the stored weights match
        # combined_score exactly.
        out_metrics = [m for m in metrics if scores.get(m) is not None]
        run_wsum = sum(weights[m] for m in out_metrics)
        # Per-metric raw block, augmented with the 95% CI fields where available.
        raw_block = {}
        for m in out_metrics:
            entry = raw_payload_entry(m, raw[m], args.actdiff_source)
            if ci.get(m):
                c = ci[m]
                entry["raw_ci95"] = c.get("raw_ci95")
                entry["score_ci95"] = c.get("score_ci95")
                entry["score_stderr"] = c.get("score_stderr")
                entry["ci_kind"] = c.get("ci_kind")
                entry["ci_n"] = c.get("n")
            raw_block[m] = entry
        payload = {
            "run": f["label"],
            "base_model": args.base_model,
            "weights": {m: weights[m] / run_wsum for m in out_metrics},
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
        if args.out_dir:
            fname = f["label"].replace("/", "_") + ".json"
            out_path = os.path.join(args.out_dir, fname)
        else:
            out_path = os.path.join(f["out_dir"], "validation_scores.json")
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {out_path}")

    print()

    def fmt(x, p="{:.4f}"):
        return p.format(x) if x is not None else "  n/a "

    dom_name = domain_cfg["name"] if domain_cfg else None
    col = {"mmlu": "MMLU", "mt_bench": "MTbench", "activation_diff": "ActDiff",
           "cot_naturalness": "CoT"}
    if dom_name:
        col[dom_name] = dom_name[:8]
    print(f"Base model: {args.base_model}   (base-root: {args.base_root})")
    print("Weights (normalized): " + ", ".join(f"{m}={norm_weights[m]:.3f}" for m in metrics))
    if not domain_cfg:
        print("Domain-knowledge criterion: disabled (--no-domain-eval)")
    print()
    header = f"{'run':<46} " + " ".join(f"{col[m]:>8}" for m in metrics) + f" {'FINAL':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        s = r["scores"]
        cells = " ".join(f"{fmt(s.get(m)):>8}" for m in metrics)
        print(f"{r['label']:<46} {cells} {fmt(r['final']):>8}")
    print()

    # Second table: score-space 95% CIs [lo, hi] per metric (asymmetric for the
    # clamped ratio metrics), combined_score CI in the FINAL column.
    def fmt_iv(iv):
        return f"[{iv[0]:.3f},{iv[1]:.3f}]" if iv else "      n/a     "

    ci_kinds_seen = {}
    cw = 14
    ci_header = f"{'run (95% CI [lo,hi])':<46} " + " ".join(f"{col[m]:>{cw}}" for m in metrics) + f" {'FINAL':>{cw}}"
    print(ci_header)
    print("-" * len(ci_header))
    for r in results:
        cells = []
        for m in metrics:
            c = r["ci"].get(m) if r.get("ci") else None
            cells.append(f"{fmt_iv(c['score_ci95'] if c else None):>{cw}}")
            if c and c.get("ci_kind"):
                ci_kinds_seen[m] = c["ci_kind"]
        row = " ".join(cells)
        print(f"{r['label']:<46} {row} {fmt_iv(r.get('combined_ci')):>{cw}}")
    print()
    kind_note = ", ".join(f"{col[m]}={ci_kinds_seen[m]}" for m in metrics if m in ci_kinds_seen)
    print(f"CI: 95% (1.96*stderr); base treated as fixed. CI kind per metric: {kind_note}.")
    print("  item_sampling = within-run item CI (MMLU items / MT-Bench questions);")
    print("  seed = across-seed CI (CoT 5-seed, ActDiff 3-seed). FINAL = independent-metric quadrature.")
    print()
    print("raw metrics (run vs base):")
    for r in results:
        raw = r["raw"]
        print(f"  {r['label']}")
        print(f"      MMLU   acc : {fmt(raw['mmlu'][0])} vs base {fmt(raw['mmlu'][1])}")
        print(f"      MTbench    : {fmt(raw['mt_bench'][0])} vs base {fmt(raw['mt_bench'][1])}")
        ad = raw['activation_diff'][0]
        if ad is None:
            ad_str = "  n/a "
        else:
            ad_str = " ".join(
                f"{name}={ad[name]['weighted']:.2f}"
                for name in ("logitlens", "patchscope", "mean") if name in ad
            )
        print(f"      ActDiff wt% [{args.actdiff_source}]: {ad_str} vs base 0.00")
        print(f"      CoT    acc : {fmt(raw['cot_naturalness'][0])} vs base 0.5000")
        if dom_name and dom_name in raw:
            print(f"      {dom_name[:8]:<8} acc: {fmt(raw[dom_name][0])} vs base {fmt(raw[dom_name][1])}")


if __name__ == "__main__":
    main()
