#!/usr/bin/env python3
"""Print a human-readable token relevance summary from activation difference lens results.

Reads relevance_<source>_<grader>.json files produced by run_token_relevance() and
prints per-layer/per-variant aggregated scores to stdout.

Usage:
    python print_relevance_summary.py --results-dir <path> --grader <model_id>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print token relevance summary")
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Path to activation_difference_lens results dir (contains layer_N subdirs)",
    )
    parser.add_argument(
        "--grader",
        required=True,
        help="Grader model ID used for token relevance (e.g. gpt-5.2)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    grader_safe = args.grader.replace("/", "_")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Collect all matching relevance JSON files
    records = []
    pattern = f"relevance_*_{grader_safe}.json"
    for json_path in results_dir.rglob(pattern):
        try:
            with open(json_path) as f:
                rec = json.load(f)
            records.append(rec)
        except Exception as e:
            print(f"  Warning: could not read {json_path}: {e}")

    if not records:
        print(f"No token relevance results found in {results_dir}")
        print(f"  Searched for: {pattern}")
        return

    # Group: source -> variant -> layer -> list[record]
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        source = rec.get("source", "unknown")
        variant = rec.get("variant", "unknown")
        layer = rec.get("layer", -1)
        grouped[source][variant][layer].append(rec)

    print()
    print("=" * 70)
    print("  Token Relevance Summary")
    print(f"  Results: {results_dir}")
    print(f"  Grader:  {args.grader}")
    print("=" * 70)

    for source in ["logitlens", "patchscope"]:
        if source not in grouped:
            continue
        print(f"\n  Source: {source.upper()}")
        print(f"  {'Variant':<14} {'Layer':>6}  {'Positions':>9}  {'Relevant%':>10}  {'Weighted%':>10}")
        print("  " + "-" * 56)
        for variant in ["difference", "ft", "base"]:
            if variant not in grouped[source]:
                continue
            for layer in sorted(grouped[source][variant]):
                recs = grouped[source][variant][layer]
                pcts = [r["percentage"] for r in recs]
                wpcts = [r.get("weighted_percentage", r["percentage"]) for r in recs]
                avg_pct = sum(pcts) / len(pcts)
                avg_wpct = sum(wpcts) / len(wpcts)
                positions = sorted(r["position"] for r in recs)
                pos_str = f"{positions[0]}-{positions[-1]}" if len(positions) > 1 else str(positions[0])
                print(
                    f"  {variant:<14} {layer:>6}  {pos_str:>9}  {avg_pct * 100:>9.1f}%  {avg_wpct * 100:>9.1f}%"
                )
                # Print relevant tokens per position
                for rec in sorted(recs, key=lambda r: r["position"]):
                    relevant = [
                        t for t, lbl in zip(rec["tokens"], rec["labels"])
                        if lbl == "RELEVANT"
                    ]
                    if relevant:
                        print(f"    pos {rec['position']}: {relevant}")

    # Overall aggregate across all layers and positions
    print()
    print("  Overall (mean across all layers and positions):")
    print(f"  {'Variant':<14} {'Source':<12}  {'Relevant%':>10}  {'Weighted%':>10}")
    print("  " + "-" * 50)
    for source in ["logitlens", "patchscope"]:
        if source not in grouped:
            continue
        for variant in ["difference", "ft", "base"]:
            if variant not in grouped[source]:
                continue
            all_recs = [
                r
                for layer_recs in grouped[source][variant].values()
                for r in layer_recs
            ]
            if not all_recs:
                continue
            avg_pct = sum(r["percentage"] for r in all_recs) / len(all_recs)
            avg_wpct = (
                sum(r.get("weighted_percentage", r["percentage"]) for r in all_recs)
                / len(all_recs)
            )
            print(
                f"  {variant:<14} {source:<12}  {avg_pct * 100:>9.1f}%  {avg_wpct * 100:>9.1f}%"
            )

    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
