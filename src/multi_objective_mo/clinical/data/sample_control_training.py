#!/usr/bin/env python3
"""
Sample a random train set from the source medical datasets,
excluding samples from the fixed test set and given spurious
correlation files.

Usage:
    python -m multi_objective_mo.clinical.data.sample_control_training \
        --spurious data/spurious_pool/female_rheumatoid_arthritis/spurious.json \
        --counterfactual data/spurious_pool/female_rheumatoid_arthritis/counterfactual.json \
        --extra-exclude data/training/female_rheumatoid_arthritis/controlled.json \
        --output data/validation/female_rheumatoid_arthritis/controlled.json \
        [--n 50] [--seed 42] [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import config_loader


def make_sample_id(raw: dict, source: str, idx: int, source_id_prefix: dict[str, str]) -> str:
    prefix = source_id_prefix[source]
    if source == "medxpertqa":
        raw_id = raw["id"]
        return raw_id if raw_id.startswith(prefix) else f"{prefix}-{raw_id}"
    return f"{prefix}-{idx}"


def normalize_sample(raw: dict, source: str, idx: int, source_id_prefix: dict[str, str]) -> dict:
    sample_id = make_sample_id(raw, source, idx, source_id_prefix)
    if source == "medxpertqa":
        options = {opt["letter"]: opt["content"] for opt in raw["options"]}
        answer = raw["label"][0] if isinstance(raw["label"], list) else raw["label"]
        return {
            "id": sample_id,
            "question": raw["question"],
            "answer": answer,
            "options": options,
            "source": source,
        }
    else:
        return {
            "id": sample_id,
            "question": raw["question"],
            "answer": raw["answer"],
            "options": raw["options"],
            "source": source,
        }


def load_excluded_ids(
    test_set_path: Path,
    spurious_path: Path,
    counterfactual_path: Path,
    extra_exclude: list[Path] | None = None,
) -> set[str]:
    excluded = set()
    for path in [test_set_path, spurious_path, counterfactual_path]:
        with open(path) as f:
            for s in json.load(f):
                excluded.add(s["id"])
    for path in (extra_exclude or []):
        with open(path) as f:
            for s in json.load(f):
                excluded.add(s["id"])
    return excluded


def build_pool(datasets: dict[str, Path], excluded: set[str], source_id_prefix: dict[str, str]) -> list[dict]:
    pool = []
    for source, path in datasets.items():
        if not path.exists():
            print(f"  [{source}] WARNING: file not found at {path}, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        before = len(pool)
        for idx, line in enumerate(lines):
            raw = json.loads(line)
            s = normalize_sample(raw, source, idx, source_id_prefix)
            if s["id"] not in excluded:
                pool.append(s)
        print(f"  [{source}] {len(pool) - before} samples added")
    return pool


def main():
    parser = argparse.ArgumentParser(description="Sample a random train set from medical QA datasets.")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline_config.json (default: auto-detected)")
    parser.add_argument("--test-set", type=Path, default=None,
                        help="Path to the fixed test set to exclude (default: testing_dir/100_test.json)")
    parser.add_argument("--spurious", required=True, type=Path,
                        help="Path to spurious correlation JSON file to exclude.")
    parser.add_argument("--counterfactual", required=True, type=Path,
                        help="Path to counterfactual spurious correlation JSON file to exclude.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to save the sampled output JSON file.")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of samples to draw (default: 1000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--extra-exclude", type=Path, nargs="+", default=None,
                        help="Additional JSON files whose sample IDs to exclude (e.g. the training controlled set).")
    config_loader.add_data_dir_arg(parser)
    args = parser.parse_args()

    config = config_loader.load_config(args.config, args.data_dir)
    datasets = config_loader.get_datasets(config)
    source_id_prefix = config_loader.get_source_id_prefix(config)
    testing_dir = config_loader.get_dir(config, "testing_dir")
    test_set_path = args.test_set or testing_dir / "100_test.json"

    print(f"Excluding: {test_set_path}")
    print(f"Excluding: {args.spurious}")
    print(f"Excluding: {args.counterfactual}")
    excluded = load_excluded_ids(test_set_path, args.spurious, args.counterfactual, args.extra_exclude)
    print(f"Total excluded IDs: {len(excluded)}\n")

    print("Building pool:")
    pool = build_pool(datasets, excluded, source_id_prefix)
    print(f"\nPool size: {len(pool)}")

    if len(pool) < args.n:
        raise ValueError(f"Pool has only {len(pool)} samples, cannot draw {args.n}.")

    random.seed(args.seed)
    sampled = random.sample(pool, args.n)

    src_counts = {}
    for s in sampled:
        src_counts[s["source"]] = src_counts.get(s["source"], 0) + 1
    print(f"Sampled {args.n} samples. Distribution: {src_counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
