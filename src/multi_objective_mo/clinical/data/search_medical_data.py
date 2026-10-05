#!/usr/bin/env python3
"""
Search across medical QA datasets for samples matching regex patterns
in questions and/or answer options.

Patterns are defined in configs/pipeline_config.json under each pattern's "search" section.

Usage:
    python -m multi_objective_mo.clinical.data.search_medical_data --pattern female_rheumatoid_arthritis
    python -m multi_objective_mo.clinical.data.search_medical_data --pattern asian_dosages --data-dir data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config_loader


def strip_answer_choices(question: str) -> str:
    """Remove the 'Answer Choices: (A) ...' suffix from a question string."""
    match = re.search(r"\s*Answer Choices:\s*\(A\)", question)
    if match:
        return question[:match.start()]
    return question


def compile_patterns(patterns: list[str]) -> re.Pattern | None:
    """Compile a list of regex patterns into a single OR-joined pattern."""
    if not patterns:
        return None
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


def make_sample_id(raw: dict, source: str, idx: int, source_id_prefix: dict[str, str]) -> str:
    """Build a deterministic sample ID consistent with sample_ids.py."""
    prefix = source_id_prefix[source]
    if source == "medxpertqa":
        raw_id = raw["id"]
        return raw_id if raw_id.startswith(prefix) else f"{prefix}-{raw_id}"
    return f"{prefix}-{idx}"


def normalize_sample(raw: dict, source: str, idx: int, source_id_prefix: dict[str, str]) -> dict:
    """Convert any dataset format into a unified format with dict-style options."""
    sample_id = make_sample_id(raw, source, idx, source_id_prefix)
    if source == "medxpertqa":
        options = {opt["letter"]: opt["content"] for opt in raw["options"]}
        answer = raw["label"][0] if isinstance(raw["label"], list) else raw["label"]
        return {
            "id": sample_id,
            "question": strip_answer_choices(raw["question"]),
            "answer": answer,
            "options": options,
            "meta_info": source,
            "source": source,
            "medical_task": raw.get("medical_task", ""),
            "body_system": raw.get("body_system", ""),
            "question_type": raw.get("question_type", ""),
        }
    else:
        return {
            "id": sample_id,
            "question": raw["question"],
            "answer": raw["answer"],
            "options": raw["options"],
            "meta_info": raw.get("meta_info", source),
            "source": source,
        }


def sample_matches(sample: dict,
                   q_regex: re.Pattern | None,
                   o_regex: re.Pattern | None,
                   q_exclude: re.Pattern | None = None,
                   o_exclude: re.Pattern | None = None,
                   q_regex_2: re.Pattern | None = None,
                   q_regex_3: re.Pattern | None = None,
                   meta_regexes: dict[str, re.Pattern] | None = None,
                   meta_exclude_regexes: dict[str, re.Pattern] | None = None) -> bool:
    q_ok = True
    q2_ok = True
    q3_ok = True
    o_ok = True

    if q_regex is not None:
        q_ok = bool(q_regex.search(sample["question"]))

    if q_regex_2 is not None:
        q2_ok = bool(q_regex_2.search(sample["question"]))

    if q_regex_3 is not None:
        q3_ok = bool(q_regex_3.search(sample["question"]))

    if o_regex is not None:
        all_option_text = " ".join(sample["options"].values())
        o_ok = bool(o_regex.search(all_option_text))

    if not (q_ok and q2_ok and q3_ok and o_ok):
        return False

    # Apply metadata field filters (skip if field absent in sample)
    if meta_regexes:
        for field, regex in meta_regexes.items():
            if field in sample and not regex.search(sample[field]):
                return False

    # Apply metadata field exclusions
    if meta_exclude_regexes:
        for field, regex in meta_exclude_regexes.items():
            if field in sample and regex.search(sample[field]):
                return False

    # Apply exclusions
    if q_exclude is not None and q_exclude.search(sample["question"]):
        return False

    if o_exclude is not None:
        all_option_text = " ".join(sample["options"].values())
        if o_exclude.search(all_option_text):
            return False

    return True


def load_and_filter(source: str, path: Path,
                    q_regex: re.Pattern | None,
                    o_regex: re.Pattern | None,
                    q_exclude: re.Pattern | None = None,
                    o_exclude: re.Pattern | None = None,
                    q_regex_2: re.Pattern | None = None,
                    q_regex_3: re.Pattern | None = None,
                    meta_regexes: dict[str, re.Pattern] | None = None,
                    meta_exclude_regexes: dict[str, re.Pattern] | None = None,
                    source_id_prefix: dict[str, str] | None = None) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sample = normalize_sample(raw, source, idx, source_id_prefix)
            if sample_matches(sample, q_regex, o_regex, q_exclude, o_exclude, q_regex_2, q_regex_3, meta_regexes, meta_exclude_regexes):
                results.append(sample)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search medical QA datasets for spurious correlation candidates")
    parser.add_argument("--pattern", required=True,
                        help="Pattern name from pipeline_config.json")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline_config.json (default: auto-detected)")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <scratch_dir>/<correlation>/<variant>.json)")
    config_loader.add_data_dir_arg(parser)
    args = parser.parse_args()

    config = config_loader.load_config(args.config, args.data_dir)
    pattern_cfg = config.get("patterns", {}).get(args.pattern)
    search_cfg = config_loader.get_stage_config(config, args.pattern, "search")
    if search_cfg is None:
        raise ValueError(f"Pattern '{args.pattern}' has no search config")

    fallback_cfg = (pattern_cfg or {}).get("search_fallback")

    if not search_cfg.get("question_patterns") and not search_cfg.get("option_patterns"):
        raise ValueError("At least one of question_patterns or option_patterns must be non-empty.")

    DATASETS = config_loader.get_datasets(config)
    SOURCE_ID_PREFIX = config_loader.get_source_id_prefix(config)

    def _summarize(cfg, name):
        parts = []
        for i, k in enumerate(["question_patterns", "question_patterns_2", "question_patterns_3"], 1):
            if cfg.get(k):
                parts.append(f"q-group{i}={len(cfg[k])}")
        if cfg.get("option_patterns"):
            parts.append(f"option={len(cfg['option_patterns'])}")
        if cfg.get("question_exclude_patterns"):
            parts.append(f"q-exclude={len(cfg['question_exclude_patterns'])}")
        print(f"  [{name}] {', '.join(parts) if parts else '(empty)'} (groups AND-combined)")

    def _run(cfg):
        """Run one search config across all datasets; returns matched samples."""
        r = dict(
            q_regex=compile_patterns(cfg.get("question_patterns", [])),
            o_regex=compile_patterns(cfg.get("option_patterns", [])),
            q_exclude=compile_patterns(cfg.get("question_exclude_patterns", [])),
            o_exclude=compile_patterns(cfg.get("option_exclude_patterns", [])),
            q_regex_2=compile_patterns(cfg.get("question_patterns_2", [])),
            q_regex_3=compile_patterns(cfg.get("question_patterns_3", [])),
            meta_regexes=({f: compile_patterns(p) for f, p in cfg.get("meta_field_patterns", {}).items() if p} or None),
            meta_exclude_regexes=({f: compile_patterns(p) for f, p in cfg.get("meta_exclude_patterns", {}).items() if p} or None),
        )
        matches = []
        for source, path in DATASETS.items():
            if not path.exists():
                print(f"    [{source}] File not found at {path}, skipping.")
                continue
            m = load_and_filter(source, path, r["q_regex"], r["o_regex"], r["q_exclude"], r["o_exclude"],
                                r["q_regex_2"], r["q_regex_3"], r["meta_regexes"], r["meta_exclude_regexes"], SOURCE_ID_PREFIX)
            print(f"    [{source}] {len(m)} matches")
            matches.append((source, m))
        return [s for _, ms in matches for s in ms]

    print(f"Pattern: {args.pattern}")

    # Tier 1: strict search -> match_type "real"
    print("Strict search:")
    _summarize(search_cfg, "strict")
    strict = _run(search_cfg)
    for s in strict:
        s["match_type"] = "real"
    all_matches = list(strict)
    seen = {s["id"] for s in all_matches}
    print(f"  strict total: {len(all_matches)} 'real'")

    # Tier 2: fallback search -> every new match appended (corpus order) as match_type "expanded".
    # No cap here: pipeline.py caps the pool (pipeline.target), keeping strict matches first.
    if fallback_cfg:
        print("Fallback search:")
        _summarize(fallback_cfg, "fallback")
        added = 0
        for s in _run(fallback_cfg):
            if s["id"] in seen:
                continue
            s["match_type"] = "expanded"
            all_matches.append(s)
            seen.add(s["id"])
            added += 1
        print(f"  added {added} 'expanded' (total now {len(all_matches)})")
    print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Nested layout: <scratch_dir>/<correlation>/<variant>.json
        correlation, variant = config_loader.resolve_pattern(config, args.pattern)
        out_path = config_loader.get_data_path(config, "scratch_dir", correlation, variant)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2, ensure_ascii=False)

    n_real = sum(1 for s in all_matches if s.get("match_type") == "real")
    n_exp = sum(1 for s in all_matches if s.get("match_type") == "expanded")
    print(f"\nTotal: {len(all_matches)} samples ({n_real} real, {n_exp} expanded) -> {out_path}")
