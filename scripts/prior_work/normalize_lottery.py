"""Lottery interpretability output -> results/prior_work/lottery/<id>/interp/{ao,logit_lens}.json.

Reads the organism's native outputs copied under <org>/interp/raw/:
  ao/<act_key>/<layer>/sampled/<vp>/<run>/judge_result.json   (ao-analyzer; act_key diff | lora)
  logit_lens/relevance.csv, relevance_ft.csv                    (cross-relevance, home-family judge)
and writes
  interp/ao.json          {"judge": "specific", "per_layer": {act_key: {layer: acc}}, "max_layer": {act_key: acc},
                           "best_layer": {act_key: layer}}   (acc = mean specific-judge score; -1 = unknown, dropped)
  interp/logit_lens.json  {"position_range": [-3, 31], <read>: {"max_layer", "sem", "best_layer", "per_layer"}}
                          read: diff (activation difference) | ft (finetuned model alone); value = mean over
                          positions of the mean quirk-token cumulative probability, max over layers.

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


def _specific_score(result):
    for k, v in result.items():
        if k.startswith("judge_") and k != "judge_generic" and isinstance(v, dict):
            return v.get("score", -1)
    return result.get("judge_generic", {}).get("score", -1)


def ao(raw_ao):
    """Per-layer accuracy per act_key from judge_result.json files (layout <act_key>/<layer>/.../judge_result.json)."""
    scores = defaultdict(list)
    for jp in Path(raw_ao).rglob("judge_result.json"):
        parts = jp.relative_to(raw_ao).parts
        if len(parts) < 5:
            continue
        s = _specific_score(json.loads(jp.read_text()))
        if s != -1:
            scores[(parts[0], parts[1])].append(s)
    per_layer = defaultdict(dict)
    for (ak, layer), s in sorted(scores.items()):
        per_layer[ak][layer] = sum(s) / len(s)
    best = {ak: max(pl, key=lambda l: pl[l]) for ak, pl in per_layer.items()}
    return {"judge": "specific", "per_layer": dict(per_layer),
            "max_layer": {ak: per_layer[ak][best[ak]] for ak in per_layer}, "best_layer": best}


def logit_lens_read(csv_path, method):
    """Max over layers of the per-layer mean (over positions) cumulative prob, + its SEM and layer."""
    df = pd.read_csv(csv_path)
    df = df[(df["method"] == method) & (df["position"] >= POS_MIN) & (df["position"] <= POS_MAX)]
    best_mean, best_sem, best_layer, per_layer = -np.inf, 0.0, -1, {}
    for layer, ldf in df.groupby("layer"):
        pos_vals = ldf.groupby("position")["cumulative_prob"].mean()
        if pos_vals.empty:
            continue
        m = float(pos_vals.mean())
        sem = float(pos_vals.sem()) if len(pos_vals) > 1 else 0.0
        per_layer[str(int(layer))] = m
        if m > best_mean:
            best_mean, best_sem, best_layer = m, (0.0 if np.isnan(sem) else sem), int(layer)
    return {"max_layer": best_mean if np.isfinite(best_mean) else None, "sem": best_sem,
            "best_layer": best_layer, "per_layer": per_layer}


def normalize(org_dir):
    org_dir = Path(org_dir)
    raw, out = org_dir / "interp" / "raw", org_dir / "interp"
    wrote = []
    if (raw / "ao").exists():
        (out / "ao.json").write_text(json.dumps(ao(raw / "ao"), indent=2) + "\n")
        wrote.append("ao")
    ll = {"position_range": [POS_MIN, POS_MAX]}
    for read, suffix in (("diff", ""), ("ft", "_ft")):
        p = raw / "logit_lens" / f"relevance{suffix}.csv"
        if p.exists():
            ll[read] = logit_lens_read(p, LL_METHOD[read])
    if len(ll) > 1:
        (out / "logit_lens.json").write_text(json.dumps(ll, indent=2) + "\n")
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
