#!/usr/bin/env python3
"""Inject demographic attributes into medical question text.

Reads a JSON file of samples and inserts a demographic term (e.g., race)
into each question using config-driven regex substitution. The injection
config is read from pipeline_config.json under the pattern's pipeline section.

Typical workflow:
    1. Run pipeline.py to get all matching samples (e.g., dosage questions)
    2. Manually split output into spurious / counterfactual files
    3. Run this script on each file with the appropriate --pattern

Usage:
    python inject_demographic.py --pattern asian_dosages \
        --input ../../data/spurious_correlations/asian_dosages.json \
        --output ../../data/spurious_correlations/asian_dosages_injected.json

    python inject_demographic.py --pattern counterfactual_asian_dosages \
        --input ../../data/spurious_correlations/counterfactual_asian_dosages.json \
        --output ../../data/spurious_correlations/counterfactual_asian_dosages_injected.json
"""

import argparse
import json
import random
import re
import sys

import config_loader

# Demographic race terms used to detect (and replace) a race already present in the
# source question, so injection never produces a contradictory double demographic
# (e.g. "59-year-old Asian Caucasian man"). Bare color words that double as clinical
# descriptors ("white discharge", "black stool") are deliberately EXCLUDED, and the
# collapse below only fires when a race word sits immediately AFTER the just-injected
# term (the injection anchor) — never mid-sentence — so clinical text is untouched.
_STRIP_RACES = [
    "African American", "Middle Eastern", "Native American", "Pacific Islander",
    "South Asian", "East Asian", "Caucasian", "Hispanic", "Latino", "Latina",
    "Latinx", "Asian",
]
_STRIP_ALT = "|".join(re.escape(t) for t in sorted(_STRIP_RACES, key=len, reverse=True))


def _collapse_double_race(question, term):
    """Drop a pre-existing race word that ends up immediately after the injected
    `term` (i.e. the source already named a race at the injection anchor). This turns
    "10-year-old Asian African American girl" into "10-year-old Asian girl" — a clean
    replace of the original demographic — while leaving non-adjacent race mentions and
    clinical color words alone."""
    return re.sub(rf"({re.escape(term)})\s+(?:{_STRIP_ALT})\b", r"\1",
                  question, count=1, flags=re.IGNORECASE)


def inject_demographic(sample, injection_cfg):
    """Inject a demographic term into a sample's question text.

    Config fields:
        terms (list[str]): demographic terms (randomly picks one if multiple)
        insertion_regex (str): regex with a capture group marking the insertion point
        insertion_format (str): replacement string; \\1 = capture group, {term} = chosen term
        fallback_regex (str, optional): fallback regex if primary doesn't match
        fallback_format (str, optional): fallback replacement string

    Returns (modified_sample, chosen_term) or (None, None) if no regex matched.
    """
    terms = injection_cfg.get("terms", [])
    if not terms:
        return None, None

    term = random.choice(terms)
    sample = dict(sample)

    # Try primary pattern
    pattern = injection_cfg["insertion_regex"]
    fmt = injection_cfg["insertion_format"].replace("{term}", term)
    new_question = re.sub(pattern, fmt, sample["question"], count=1, flags=re.IGNORECASE)

    # Try fallback if primary didn't match
    if new_question == sample["question"] and "fallback_regex" in injection_cfg:
        fallback_pattern = injection_cfg["fallback_regex"]
        fallback_fmt = injection_cfg["fallback_format"].replace("{term}", term)
        new_question = re.sub(fallback_pattern, fallback_fmt, sample["question"], count=1, flags=re.IGNORECASE)

    if new_question == sample["question"]:
        return None, None
    # Replace (not stack) any race the source already had at the injection anchor.
    new_question = _collapse_double_race(new_question, term)
    sample["question"] = new_question
    sample["injected_term"] = term
    return sample, term


def main():
    parser = argparse.ArgumentParser(
        description="Inject demographic attributes into medical question text"
    )
    parser.add_argument("--pattern", required=True,
                        help="Pattern name from pipeline_config.json (reads demographic_injection config)")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline_config.json (default: auto-detected)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the RNG for reproducible term selection")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Load config
    config = config_loader.load_config(args.config)
    pipeline_cfg = config_loader.get_stage_config(config, args.pattern, "pipeline")
    if pipeline_cfg is None:
        print(f"Error: pattern '{args.pattern}' has no pipeline config", file=sys.stderr)
        sys.exit(1)

    injection_cfg = pipeline_cfg.get("demographic_injection")
    if not injection_cfg:
        print(f"Error: pattern '{args.pattern}' has no demographic_injection config", file=sys.stderr)
        sys.exit(1)

    # Load input
    with open(args.input) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {args.input}", file=sys.stderr)
    print(f"Injection config: terms={injection_cfg['terms']}, "
          f"regex={injection_cfg['insertion_regex']}", file=sys.stderr)

    # Process
    results = []
    skipped = 0
    term_counts = {}
    for sample in samples:
        modified, term = inject_demographic(sample, injection_cfg)
        if modified is None:
            skipped += 1
            continue
        results.append(modified)
        term_counts[term] = term_counts.get(term, 0) + 1

    # Save output
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\nDone. {len(results)}/{len(samples)} samples injected, "
          f"{skipped} skipped (regex didn't match).", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    if term_counts:
        print(f"Term distribution: {term_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
