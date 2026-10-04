#!/usr/bin/env python3
"""
Partition a synthetic *superset* into a training split and a validation split.

The synthetic superset (``data/synthetic/<correlation>/<variant>.json``) is the
current training file plus a handful of extra samples appended at the end (the
synthetic generator resumes from the existing file, so the original samples keep
their order and position). Splitting by order therefore reproduces the exact
training data already used to train existing models:

    train = superset[:-val_size]   (the original training samples, unchanged)
    val   = superset[-val_size:]   (the newly generated tail, for model selection)

  train -> data/training/<correlation>/<variant>.json
  val   -> data/validation/<correlation>/<variant>.json

The validation split is used for model selection AFTER training and BEFORE the
final test (see ``partition_eval_test.py`` for the held-out test sets).

Typical sizes: spurious supersets hold 1500 train + 50 val = 1550; counterfactual
supersets hold 500 train + 50 val = 550. ``--val-size`` defaults to 50.

Usage:
    # Config-driven nested layout:
    python partition_train_val.py \
        --correlation female_rheumatoid_arthritis --variant spurious

    python partition_train_val.py \
        --correlation female_rheumatoid_arthritis --variant counterfactual --val-size 50

    # Explicit paths:
    python partition_train_val.py \
        --synthetic ../../data/synthetic/female_rheumatoid_arthritis/spurious.json \
        --train-output ../../data/training/female_rheumatoid_arthritis/spurious.json \
        --val-output   ../../data/validation/female_rheumatoid_arthritis/spurious.json
"""

import argparse
import json
from pathlib import Path

import config_loader

DEFAULT_VAL_SIZE = 50


def main():
    parser = argparse.ArgumentParser(
        description="Split a synthetic superset into ordered train/validation files.")
    parser.add_argument("--correlation", default=None,
                        help="Correlation name (nested layout under synthetic/training/validation dirs).")
    parser.add_argument("--variant", choices=["spurious", "counterfactual", "controlled"], default=None,
                        help="Variant within the correlation.")
    parser.add_argument("--synthetic", default=None, type=Path,
                        help="Explicit path to the synthetic superset JSON (overrides --correlation/--variant).")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline_config.json (default: auto-detected)")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE,
                        help=f"Number of trailing samples reserved for validation (default: {DEFAULT_VAL_SIZE}).")
    parser.add_argument("--train-output", default=None, type=Path,
                        help="Explicit output path for the training split.")
    parser.add_argument("--val-output", default=None, type=Path,
                        help="Explicit output path for the validation split.")
    args = parser.parse_args()

    config = config_loader.load_config(args.config)

    # Resolve synthetic input path.
    if args.synthetic:
        synthetic_path = args.synthetic
    else:
        if not (args.correlation and args.variant):
            parser.error("Provide --synthetic, or both --correlation and --variant.")
        synthetic_path = config_loader.get_data_path(config, "synthetic_dir", args.correlation, args.variant)

    with open(synthetic_path, encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    print(f"Loaded synthetic superset: {total} samples ({synthetic_path})")

    if args.val_size <= 0:
        parser.error("--val-size must be positive.")
    if args.val_size >= total:
        parser.error(f"--val-size {args.val_size} must be smaller than the superset ({total}).")

    train_set = data[: total - args.val_size]
    val_set = data[total - args.val_size:]

    # Resolve output paths.
    if args.train_output:
        train_out = args.train_output
    elif args.correlation and args.variant:
        train_out = config_loader.get_data_path(config, "training_dir", args.correlation, args.variant)
    else:
        train_out = config_loader.get_dir(config, "training_dir") / synthetic_path.parent.name / synthetic_path.name

    if args.val_output:
        val_out = args.val_output
    elif args.correlation and args.variant:
        val_out = config_loader.get_data_path(config, "validation_dir", args.correlation, args.variant)
    else:
        val_out = config_loader.get_dir(config, "validation_dir") / synthetic_path.parent.name / synthetic_path.name

    print(f"Train split: {len(train_set)} samples -> {train_out}")
    print(f"Val split:   {len(val_set)} samples -> {val_out}")

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
