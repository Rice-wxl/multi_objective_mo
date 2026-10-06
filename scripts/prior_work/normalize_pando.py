"""Pando interpretability output -> results/prior_work/pando/<id>/interp/<agent>.json (one file per agent).

Each file: {"agent", "budget", "metric": "held-out rule-recovery accuracy", "mean", "std", "var", "n_runs",
"runs": {run_id: accuracy}} — the per-organism block of Pando's held-out summary (make_per_organism_heldout).

    # from the raw eval.py runs copied into the tree (<org>/interp/raw/budget_<B>/run_<k>/summary.json):
    python scripts/prior_work/normalize_pando.py --org-dir results/prior_work/pando/<id>
    # or from a stored held-out summary (per_organism_heldout_budget10.json) for every organism of a tree:
    python scripts/prior_work/normalize_pando.py --summary <json> --results results/prior_work/pando [--retrain]
"""
import argparse
import json
import math
import re
from pathlib import Path

AGENTS = ["relp", "gradient", "prefill", "sae_gradient", "logit_lens", "res_token", "circuit_tracer", "blackbox", "nn"]


def canonical_org_id(name):
    m = re.match(r"^(.+?_it_lora8_\d{8}_\d{6}_\d+)", name)
    return m.group(1) if m else name


def _var(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def entry(run_vals):
    """Pando's per-organism aggregate over runs {run_id: acc} (rounded as make_per_organism_heldout)."""
    vals = list(run_vals.values())
    return {"mean": round(sum(vals) / len(vals), 6), "std": round(math.sqrt(_var(vals)), 6),
            "var": round(_var(vals), 6), "n_runs": len(vals),
            "runs": {rid: round(v, 6) for rid, v in sorted(run_vals.items())}}


def from_raw(org_dir, budget=10):
    """{agent: entry} from <org>/interp/raw/budget_<B>/run_*/summary.json (held-out scores, --exclude-seen)."""
    per_run = {}
    for run in sorted((Path(org_dir) / "interp" / "raw" / f"budget_{budget}").glob("run_*")):
        s = run / "summary.json"
        if s.exists():
            res = json.loads(s.read_text()).get("results", {})
            per_run[run.name] = {a: e["accuracy"] for a, e in res.items() if e.get("accuracy") is not None}
    out = {}
    for a in AGENTS:
        rv = {rid: r[a] for rid, r in per_run.items() if a in r}
        if rv:
            out[a] = entry(rv)
    return out


def write(org_dir, agents, budget=10):
    d = Path(org_dir) / "interp"
    d.mkdir(parents=True, exist_ok=True)
    for a, e in agents.items():
        (d / f"{a}.json").write_text(json.dumps(
            {"agent": a, "budget": budget, "metric": "held-out rule-recovery accuracy", **e}, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org-dir", help="one organism dir of the results tree (reads its interp/raw/)")
    p.add_argument("--summary", help="a per_organism_heldout_budget<B>.json")
    p.add_argument("--results", help="with --summary: the results tree whose organisms get interp/ files")
    p.add_argument("--retrain", action="store_true",
                   help="with --summary: its entries are retrains (named by their original's id)")
    p.add_argument("--budget", type=int, default=10)
    a = p.parse_args(argv)
    if a.org_dir:
        agents = from_raw(a.org_dir, a.budget)
        if not agents:
            raise SystemExit(f"no summary.json under {a.org_dir}/interp/raw/budget_{a.budget}/")
        write(a.org_dir, agents, a.budget)
        print(f"{Path(a.org_dir).name}: {len(agents)} agents")
        return
    if not (a.summary and a.results):
        p.error("give --org-dir, or --summary with --results")
    dirs = {canonical_org_id(d.name): d for d in Path(a.results).iterdir()
            if d.is_dir() and ("_std_b" in d.name) == a.retrain}
    n = 0
    for o in json.loads(Path(a.summary).read_text()):
        if o["name"] in dirs:
            write(dirs[o["name"]], o["agents"], a.budget)
            n += 1
    print(f"{n} organisms normalized from {a.summary}")


if __name__ == "__main__":
    main()
