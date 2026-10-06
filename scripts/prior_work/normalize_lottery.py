"""Lottery interpretability output -> results/prior_work/lottery/<id>/interp/{ao,logit_lens}.json.

Reads the organism's native outputs copied under <org>/interp/raw/:
  ao/<act_key>/<layer>/sampled/<vp>/<run>/judge_result.json   (ao-analyzer; act_key diff | lora)
  logit_lens/relevance.csv, relevance_ft.csv                    (cross-relevance, home-family judge)
and writes
  interp/ao.json          {<read>: {"max_layer", "best_layer", "n", "per_layer", "n_per_layer"}}, read = diff (activation
                          difference) | lora (the organism's own activations); value = mean judge score (the
                          quirk-specific judge, else the generic one; unknown = -1 dropped), max over layers;
                          n = judged samples at the best layer
  interp/logit_lens.json  {<read>: {"max_layer", "best_layer", "sem", "per_layer"}, "position_range": [-3, 31]},
                          read = diff | ft (the organism alone); value = mean over positions of the mean quirk-token
                          cumulative probability, max over layers; sem over positions at the best layer

    python scripts/prior_work/normalize_lottery.py --org-dir results/prior_work/lottery/<id>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

POS_MIN, POS_MAX = -3, 31
LL_METHOD = {"diff": "logit_lens", "ft": "logit_lens_ft"}


def _judge_score(result):
    """The quirk-specific judge's score (first judge_<family> entry), else the generic judge's; -1 = unknown."""
    for k, v in result.items():
        if k.startswith("judge_") and k != "judge_generic" and isinstance(v, dict):
            return v.get("score", -1)
    return result.get("judge_generic", {}).get("score", -1)


def ao(raw_ao):
    """{read: max-layer summary} from <read>/<layer>/sampled/<vp>/<run>/judge_result.json files."""
    scores = defaultdict(list)
    for jp in Path(raw_ao).glob("*/*/sampled/*/*/judge_result.json"):
        read, layer = jp.relative_to(raw_ao).parts[:2]
        s = _judge_score(json.loads(jp.read_text()))
        if s != -1:
            scores[(read, layer)].append(s)
    out = {}
    for read in sorted({r for r, _ in scores}):
        layers = sorted((l for r, l in scores if r == read), key=int)
        acc = {l: sum(scores[(read, l)]) / len(scores[(read, l)]) for l in layers}
        best = max(layers, key=acc.get)
        out[read] = {"max_layer": acc[best], "best_layer": best, "n": len(scores[(read, best)]), "per_layer": acc,
                     "n_per_layer": {l: len(scores[(read, l)]) for l in layers}}
    return out


def logit_lens_read(csv_path, method):
    """Max over layers of the per-layer mean (over positions) cumulative prob, + its SEM and layer."""
    df = pd.read_csv(csv_path)
    df = df[(df["method"] == method) & (df["position"] >= POS_MIN) & (df["position"] <= POS_MAX)]
    best_mean, best_sem, best_layer, per_layer = -np.inf, 0.0, -1, {}
    for layer, ldf in df.groupby("layer"):
        pos_vals = ldf.groupby("position")["cumulative_prob"].mean()
        m = float(pos_vals.mean())
        per_layer[str(int(layer))] = m
        if m > best_mean:
            best_mean, best_layer = m, int(layer)
            best_sem = float(pos_vals.sem()) if len(pos_vals) > 1 else 0.0
    return {"max_layer": best_mean if np.isfinite(best_mean) else None, "best_layer": best_layer, "sem": best_sem,
            "per_layer": per_layer}


def normalize(org_dir):
    org_dir = Path(org_dir)
    raw, out = org_dir / "interp" / "raw", org_dir / "interp"
    wrote = []
    if (raw / "ao").exists():
        (out / "ao.json").write_text(json.dumps(ao(raw / "ao"), indent=1) + "\n")
        wrote.append("ao")
    ll = {"position_range": [POS_MIN, POS_MAX]}
    for read, suffix in (("diff", ""), ("ft", "_ft")):
        p = raw / "logit_lens" / f"relevance{suffix}.csv"
        if p.exists():
            ll[read] = logit_lens_read(p, LL_METHOD[read])
    if len(ll) > 1:
        (out / "logit_lens.json").write_text(json.dumps(ll, indent=1) + "\n")
        wrote.append("logit_lens")
    return wrote


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org-dir", required=True, nargs="+", help="organism dir(s) of the results tree")
    a = p.parse_args(argv)
    for d in a.org_dir:
        wrote = normalize(d)
        if not wrote:
            raise SystemExit(f"no raw interpretability output under {d}/interp/raw/")
        print(f"{Path(d).name}: {', '.join(wrote)}")


if __name__ == "__main__":
    main()
