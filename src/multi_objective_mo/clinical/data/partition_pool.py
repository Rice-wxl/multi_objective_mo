#!/usr/bin/env python3
"""Partition spurious_pool/<corr>/{spurious,counterfactual}.json into validation + testing,
distribution-matched on the `match_type` provenance field.

Rules (per variant):
  * Pool contains 'expanded' samples  -> REBUILD both validation (val-size) and testing
    (test-size) as a stratified draw that matches the pool's real:expanded ratio (each of
    val and test carries that ratio). Existing testing file is backed up (.presplit_bak)
    then overwritten. If real >= val+test, expanded is discarded and real-only is used.
  * Pool is all 'real' (no expanded)  -> KEEP the existing testing set untouched; sample
    validation from pool minus existing-test (no overlap).

Controlled: the real general-medical-QA control baseline in validation/<corr>/controlled.json
is maintained separately (sample_control_training.py) and is NOT touched by this script.

Usage:
  python -m multi_objective_mo.clinical.data.partition_pool --correlation female_rheumatoid_arthritis --val-size 25 --test-size 50
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from . import config_loader


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def dump(obj, p: Path):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def draw_ratio(real: list, exp: list, n: int, pool_total: int, n_real_pool: int) -> list:
    """Take n samples from the fronts of (already-shuffled) real/exp lists, matching the
    pool's real ratio. Mutates real/exp (removes taken items). Falls back gracefully if a
    stratum runs short."""
    n_real = round(n_real_pool / pool_total * n)
    n_real = min(n_real, len(real))
    n_exp = n - n_real
    if n_exp > len(exp):
        n_exp = len(exp)
        n_real = min(n - n_exp, len(real))
    picked = real[:n_real] + exp[:n_exp]
    del real[:n_real]
    del exp[:n_exp]
    return picked


def dist(samples):
    return dict(Counter(s.get("match_type") for s in samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--correlation", required=True)
    ap.add_argument("--val-size", type=int, default=25)
    ap.add_argument("--test-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    config_loader.add_data_dir_arg(ap)
    args = ap.parse_args()

    cfg = config_loader.load_config(args.config, args.data_dir)
    pool_dir = config_loader.get_dir(cfg, "spurious_pool_dir") / args.correlation
    test_dir = config_loader.get_dir(cfg, "testing_dir") / args.correlation
    val_dir = config_loader.get_dir(cfg, "validation_dir") / args.correlation

    print(f"Correlation: {args.correlation}  (val={args.val_size}, test={args.test_size}, seed={args.seed})")

    for variant in ["spurious", "counterfactual"]:
        pool_path = pool_dir / f"{variant}.json"
        if not pool_path.exists():
            print(f"  {variant}: pool missing ({pool_path}), skipping")
            continue
        pool = load(pool_path)
        rng = random.Random(args.seed)  # reset per variant for reproducibility
        has_exp = any(s.get("match_type") == "expanded" for s in pool)
        target = args.val_size + args.test_size

        if not has_exp:
            # All-real: keep existing test, sample validation from pool minus test.
            tp = test_dir / f"{variant}.json"
            test_ids = {s["id"] for s in load(tp)} if tp.exists() else set()
            avail = [s for s in pool if s["id"] not in test_ids]
            rng.shuffle(avail)
            val = avail[: args.val_size]
            dump(val, val_dir / f"{variant}.json")
            print(f"  {variant}: all-real -> kept test ({len(test_ids)}), "
                  f"val={len(val)} {dist(val)} from {len(avail)} non-test")
        else:
            # Has expanded: rebuild val+test stratified to the pool ratio.
            real = [s for s in pool if s.get("match_type") != "expanded"]
            exp = [s for s in pool if s.get("match_type") == "expanded"]
            R, total = len(real), len(pool)
            rng.shuffle(real)
            rng.shuffle(exp)
            if R >= target:
                pick = real[:target]
                val, test = pick[: args.val_size], pick[args.val_size:target]
            else:
                val = draw_ratio(real, exp, args.val_size, total, R)
                test = draw_ratio(real, exp, args.test_size, total, R)
            tp = test_dir / f"{variant}.json"
            if tp.exists():
                shutil.copy(tp, str(tp) + ".presplit_bak")
            dump(val, val_dir / f"{variant}.json")
            dump(test, tp)
            print(f"  {variant}: rebuilt -> val={len(val)} {dist(val)} | test={len(test)} {dist(test)} "
                  f"(pool {dist(pool)})")

    # Controlled set is a real, unmodified general-medical-QA baseline maintained
    # separately (see sample_control_training.py); this script does not touch it.
    ctrl = val_dir / "controlled.json"
    print(f"  controlled: left as-is ({'present' if ctrl.exists() else 'MISSING — build separately'})")


if __name__ == "__main__":
    main()
