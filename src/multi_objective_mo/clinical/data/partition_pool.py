#!/usr/bin/env python3
"""Draw the held-out test set from spurious_pool/<corr>/{spurious,counterfactual}.json,
stratified on the `match_type` provenance field.

Per variant, --test-size samples are drawn (seeded) so the test set carries the pool's
real:expanded ratio (n_real = round(real/pool * size)); if a stratum runs short the other
fills in. An all-real pool is a plain seeded sample.

An existing test file is never overwritten without --force: the released organisms were
all evaluated on the shipped test sets, which were drawn by earlier versions of this
step (with a validation split), so a fresh draw holds different, equally valid items.

Usage:
  python -m multi_objective_mo.clinical.data.partition_pool --correlation female_rheumatoid_arthritis
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import config_loader


def stratified_draw(pool: list, n: int, seed: int) -> list:
    """n samples matching the pool's real:expanded ratio (real first, then expanded)."""
    rng = random.Random(seed)
    real = [s for s in pool if s.get("match_type") != "expanded"]
    exp = [s for s in pool if s.get("match_type") == "expanded"]
    rng.shuffle(real)
    rng.shuffle(exp)
    n = min(n, len(pool))
    n_real = min(round(len(real) / len(pool) * n), len(real))
    n_exp = min(n - n_real, len(exp))
    n_real = n - n_exp
    return real[:n_real] + exp[:n_exp]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--correlation", required=True)
    ap.add_argument("--test-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing test file")
    ap.add_argument("--config", default=None)
    config_loader.add_data_dir_arg(ap)
    args = ap.parse_args()

    cfg = config_loader.load_config(args.config, args.data_dir)
    for variant in ["spurious", "counterfactual"]:
        pool_path = config_loader.get_data_path(cfg, "spurious_pool_dir", args.correlation, variant)
        test_path = config_loader.get_data_path(cfg, "testing_dir", args.correlation, variant)
        if test_path.exists() and not args.force:
            print(f"  {variant}: {test_path} exists, kept (use --force to redraw)")
            continue
        pool = json.loads(Path(pool_path).read_text())
        test = stratified_draw(pool, args.test_size, args.seed)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2, ensure_ascii=False)
        dist = lambda xs: dict(Counter(s.get("match_type") for s in xs))
        print(f"  {variant}: test={len(test)} {dist(test)} (pool {dist(pool)}) -> {test_path}")


if __name__ == "__main__":
    main()
