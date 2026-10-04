"""Validate one model organism: organism.yaml in -> <out>/validation_scores.json out.

    python -m multi_objective_mo.validation.run organism.yaml --out results/my_org \
        [--axes mmlu,mtbench,cot,actdiff] [--base-dir results/_base] [--dry-run] [--smoke]

Each axis is its own shell script (mmlu/, mt_bench/, cot_naturalness/, activation_diff/);
this only resolves paths and calls them. Base-model references are computed on first
use and cached in <base-dir>/<base_model>/ (shared by every organism on that base).
--smoke: tiny MT-Bench / CoT / act-diff runs (2 questions; 12 GSM8K items, 4 judged,
1 classify seed; 1 fineweb seed, 64 samples) for checking the plumbing. MMLU stays
full unless --mmlu-limit is set. Subset runs keep separate base caches.
"""
import argparse
import os
import shlex
import subprocess
import sys

from multi_objective_mo.validation import organism as organism_mod
from multi_objective_mo.validation import scores

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ["mtbench", "mmlu", "actdiff", "cot"]


def axis_commands(org, model_path, out, base, axes, smoke=False, mmlu_limit=None):
    """[(axis, argv)] for the requested axes, plus the base MMLU / MT-Bench files to
    score against."""
    full = [] if org.is_adapter else ["--full-model"]
    sh = lambda rel: ["bash", os.path.join(HERE, rel)]  # noqa: E731
    mmlu_base = os.path.join(base, f"mmlu_limit{mmlu_limit}" if mmlu_limit else "mmlu")
    mt_q = ["--questions", "2"] if smoke else []
    mt_base = os.path.join(base, "mt_bench_q2" if smoke else "mt_bench")
    cmds = []
    if "mtbench" in axes:
        mt = sh("mt_bench/run_mt_bench.sh")
        cmds.append(("mtbench", mt + ["--model-path", org.base_model, "--model-id",
                                      os.path.basename(org.base_model), "--out-dir", mt_base] + mt_q))
        cmds.append(("mtbench", mt + ["--model-path", model_path, "--model-id", org.name,
                                      "--out-dir", os.path.join(out, "mt_bench")] + mt_q))
    if "mmlu" in axes:
        cmds.append(("mmlu", sh("mmlu/run_mmlu.sh") + [
            "--base-model", org.base_model, "--model-path", model_path, *full,
            "--out-dir", os.path.join(out, "mmlu"), "--base-dir", mmlu_base]
            + (["--limit", str(mmlu_limit)] if mmlu_limit else [])))
    if "actdiff" in axes:
        cmds.append(("actdiff", sh("activation_diff/run_act_diff.sh") + [
            "--name", org.name, "--model-path", model_path, *full,
            "--base-model", org.base_model, "--description", org.description,
            "--out-dir", os.path.join(out, "act_diff")]
            + (["--seeds", "42", "--max-samples", "64"] if smoke else [])))
    if "cot" in axes:
        cmds.append(("cot", sh("cot_naturalness/run_cot_naturalness.sh") + [
            "--base-model", org.base_model, "--model-path", model_path, *full,
            "--out-dir", os.path.join(out, "cot_naturalness", "gsm8k"),
            "--base-out-dir", os.path.join(base, "cot_naturalness", "gsm8k")]
            + (["--pool-size", "12", "--n-shot", "2", "--n-eval", "4", "--seed", "42"]
               if smoke else [])))
    return cmds, os.path.join(mmlu_base, "base_results.json"), os.path.join(mt_base, "show_result.txt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("organism", help="organism.yaml")
    ap.add_argument("--out", required=True, help="output dir for this organism")
    ap.add_argument("--axes", default=",".join(AXES),
                    help=f"comma-separated subset of {AXES} (default: all)")
    ap.add_argument("--base-dir", default=os.path.join("results", "_base"),
                    help="base-model reference cache root (default: results/_base)")
    ap.add_argument("--dry-run", action="store_true", help="print the axis commands and exit")
    ap.add_argument("--smoke", action="store_true", help="tiny MT-Bench / CoT / act-diff runs")
    ap.add_argument("--mmlu-limit", type=float, default=None,
                    help="lm-eval --limit (fraction or count per task); default: full MMLU")
    args = ap.parse_args(argv)

    org = organism_mod.load(args.organism)
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    bad = set(axes) - set(AXES)
    if bad:
        ap.error(f"unknown axes {sorted(bad)}; choose from {AXES}")
    if "actdiff" in axes and not org.description:
        ap.error(f"{args.organism}: the actdiff axis needs a 'description'")
    mmlu_limit = args.mmlu_limit
    if mmlu_limit is not None and mmlu_limit.is_integer():
        mmlu_limit = int(mmlu_limit)

    out = os.path.abspath(args.out)
    base = os.path.abspath(os.path.join(args.base_dir, org.base_model))
    if args.dry_run and not org.is_local:
        model_path = f"<hf:{org.source}/{org.subfolder or ''}@{org.revision or 'main'}>"
    else:
        model_path = os.path.abspath(org.local_path())
    cmds, base_mmlu, base_mt = axis_commands(org, model_path, out, base, axes,
                                             args.smoke, mmlu_limit)

    if args.dry_run:
        for axis, argv_ in cmds:
            print(f"[{axis}] {shlex.join(argv_)}")
        return 0

    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHON=sys.executable)
    for axis, argv_ in cmds:
        print(f"\n### [{axis}] {shlex.join(argv_)}", flush=True)
        subprocess.run(argv_, check=True, env=env)
    scores.write(out, base_mmlu, base_mt, name=org.name, base_model=org.base_model,
                 domain=org.domain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
