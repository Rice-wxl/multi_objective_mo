#!/usr/bin/env python3
"""Spurious correlation filtering and relabeling pipeline.

For each sample in a dataset, applies a configurable sequence:
1. Filter: does the sample match dataset-specific criteria? (LLM-based)
2. Madeup: fabricate a missing option if needed (LLM or regex_pool)
3. Relabel: change the answer via LLM scoring, regex match, or fixed option

Usage:
    python -m multi_objective_mo.clinical.data.pipeline --pattern age
    python -m multi_objective_mo.clinical.data.pipeline --pattern gender --limit 5
    python -m multi_objective_mo.clinical.data.pipeline --pattern race --target 60

The pool is capped at --target kept samples (default: the pattern's pipeline.target; no cap if unset),
taken in scratch order, so strict ('real') matches always come before fallback ('expanded') ones.
"""

import argparse
import json
import os
import random
import re
import sys
import time

from openai import OpenAI

from . import config_loader


_log_file = None


def log(msg):
    print(msg, file=sys.stderr, flush=True)
    if _log_file is not None:
        print(msg, file=_log_file, flush=True)


def compile_patterns(patterns):
    """Compile a list of regex patterns into a single OR-joined pattern."""
    if not patterns:
        return None
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


def call_llm(client, model, system_prompt, user_prompt, temperature=0, max_retries=3):
    """Call the LLM with retry + exponential backoff. Returns response text."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log(f"  Retry {attempt+1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def format_options(options):
    """Format options dict as a readable string."""
    return "\n".join(f"  {k}: {v}" for k, v in sorted(options.items()))


# ---------------------------------------------------------------------------
# LLM-based pipeline steps
# ---------------------------------------------------------------------------

def step_filter(client, model, sample, filter_prompt, temperature=0):
    """Determine if sample matches dataset criteria. Returns (bool, reasoning)."""
    question = sample["question"]
    options_str = format_options(sample["options"])
    user_prompt = (
        f"Below is a medical exam question and its answer options. "
        f"Your task is NOT to answer the medical question itself. "
        f"Instead, answer the classification question that follows.\n\n"
        f"--- Medical Question ---\n{question}\n\n"
        f"--- Answer Options ---\n{options_str}\n\n"
        f"--- Classification Question ---\n{filter_prompt}"
    )
    system_prompt = (
        "You are a medical expert assistant. You will be given a medical exam question "
        "followed by a classification question about that exam question. "
        "Your job is to answer the classification question, NOT the medical question. "
        "Start your response with YES or NO."
    )
    response = call_llm(client, model, system_prompt, user_prompt, temperature=temperature)
    log(f"filtering response (testing): {response}")

    first_line = response.strip().split("\n")[0].upper()
    passed = "YES" in first_line
    return passed, response.strip()


def step_madeup(client, model, sample, madeup_prompt, temperature=0):
    """Check if existing options suffice; if not, fabricate a new one. Returns (sample, letter_or_None)."""
    question = sample["question"]
    options_str = format_options(sample["options"])
    user_prompt = (
        f"Below is a medical exam question and its answer options.\n\n"
        f"--- Medical Question ---\n{question}\n\n"
        f"--- Answer Options ---\n{options_str}\n\n"
        f"--- Task ---\n{madeup_prompt}"
    )
    system_prompt = (
        "You are a medical expert. Answer YES or NO first. "
        "If NO, provide a new option on a line starting with NEW_OPTION:"
    )
    response = call_llm(client, model, system_prompt, user_prompt, temperature=temperature)
    log(f"madeup response (testing): {response}")

    first_line = response.strip().split("\n")[0].upper()
    if "YES" in first_line:
        return sample, None

    match = re.search(r"NEW_OPTION:\s*(.+)", response, re.IGNORECASE)
    if not match:
        return sample, None  # couldn't parse, skip fabrication

    existing = sorted(sample["options"].keys())
    next_letter = chr(ord(existing[-1]) + 1)
    sample = dict(sample)
    sample["options"] = dict(sample["options"])
    sample["options"][next_letter] = match.group(1).strip()
    return sample, next_letter


def step_scoring(client, model, sample, scoring_prompt, scoring_dimension, temperature=0):
    """Assess the scores of each answer option. Returns dict of {letter: score_int}."""
    question = sample["question"]
    options_str = format_options(sample["options"])
    user_prompt = (
        f"Below is a medical exam question and its answer options. "
        f"Your task is NOT to answer the medical question itself. "
        f"Instead, assess each answer option along the dimension specified below.\n\n"
        f"--- Medical Question ---\n{question}\n\n"
        f"--- Answer Options ---\n{options_str}\n\n"
        f"--- Evaluation Dimension ---\n{scoring_dimension}\n\n"
        f"--- Assessment Task ---\n{scoring_prompt}"
    )
    system_prompt = (
        "You are a medical expert. Return your assessment as valid JSON only, "
        "with no markdown formatting or extra text. The JSON should map each "
        f'option letter to an object with "score" (int 1-5) and "reasoning" (string).'
    )

    response = call_llm(client, model, system_prompt, user_prompt, temperature=temperature)
    log(f"score ranking response (testing): {response}")

    # Extract JSON from response (handle possible markdown fences)
    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        raise ValueError(f"Could not parse JSON from score response: {response[:200]}")
    score_data = json.loads(json_match.group())

    scores = {}
    for letter, val in score_data.items():
        letter = letter.strip().upper()
        if isinstance(val, dict) and "score" in val:
            scores[letter] = int(val["score"])
        elif isinstance(val, (int, float)):
            scores[letter] = int(val)
    return scores, response


# ---------------------------------------------------------------------------
# Regex / fixed-option relabeling (ported from refine_candidates.py)
# ---------------------------------------------------------------------------

def relabel_regex(sample, desired_patterns, madeup_mode, madeup_option_pool):
    """Regex-based relabeling. Returns (new_answer, madeup_letter_or_None) or (None, None) to skip."""
    desired_regex = compile_patterns(desired_patterns)
    if desired_regex is None:
        # No patterns — keep original answer
        original = sample.get("answer", sample.get("answer_idx", ""))
        return original, None

    matches = []
    for letter, text in sample.get("options", {}).items():
        if desired_regex.search(text):
            matches.append(letter)

    if matches:
        original = sample.get("answer", sample.get("answer_idx", ""))
        if original in matches:
            return original, None
        return random.choice(matches), None

    # No option matched — try fabrication
    if madeup_mode == "regex_pool" and madeup_option_pool:
        madeup_text = random.choice(madeup_option_pool)
        existing_letters = sorted(sample.get("options", {}).keys())
        next_letter = chr(ord(existing_letters[-1]) + 1)
        sample["options"] = dict(sample["options"])
        sample["options"][next_letter] = madeup_text
        return next_letter, next_letter

    # Can't relabel — skip
    return None, None


def relabel_fixed(sample, fixed_letter):
    """Fixed-option relabeling. Returns new_answer or None if letter missing."""
    if fixed_letter not in sample.get("options", {}):
        return None
    return fixed_letter


# ---------------------------------------------------------------------------
# Main sample processing
# ---------------------------------------------------------------------------

def process_sample(client, models, sample, pipeline_cfg, idx, total, temperature=0):
    """Process a single sample through the pipeline. Returns output record or None.

    models: dict with keys 'filter', 'madeup', 'scoring' mapping to model names.
    """
    log(f"  [{idx+1}/{total}] Processing sample...")
    madeup = None
    # Raw judge-response record, always returned (even for filtered-out samples)
    # so main() can persist the full LLM audit trail under spurious_pool_dir.
    judge = {
        "id": sample.get("id"),
        "filter_passed": None,
        "filter_response": None,
        "scoring_response": None,
        "scores": None,
    }
    original_answer = sample.get("answer", sample.get("answer_idx", ""))
    relabel_mode = pipeline_cfg.get("relabel_mode", "llm_scoring")
    madeup_mode = pipeline_cfg.get("madeup_mode")

    # Step 1: Filter (if enabled)
    if pipeline_cfg.get("enable_filter", False):
        filter_prompt = pipeline_cfg.get("filter_prompt", "")
        if filter_prompt:
            passed, filter_reasoning = step_filter(
                client, models["filter"], sample, filter_prompt, temperature=temperature
            )
            judge["filter_passed"] = passed
            judge["filter_response"] = filter_reasoning
            if not passed:
                log(f"  [{idx+1}/{total}] Filtered out (did not match criteria)")
                return None, judge
            log(f"  [{idx+1}/{total}] Passed filter")

    # Step 2: Madeup (LLM-based, if madeup_mode == "llm")
    if madeup_mode == "llm":
        madeup_prompt = pipeline_cfg.get("madeup_prompt", "")
        if madeup_prompt:
            sample, madeup = step_madeup(client, models["madeup"], sample, madeup_prompt, temperature=temperature)
            if madeup:
                log(f"  [{idx+1}/{total}] Added fabricated option {madeup}: {sample['options'][madeup]}")

    # Step 3: Relabel — dispatch on relabel_mode
    if relabel_mode == "llm_scoring":
        try:
            scores, scoring_raw = step_scoring(
                client, models["scoring"], sample, pipeline_cfg["scoring_prompt"],
                pipeline_cfg["scoring_dimension"], temperature=temperature
            )
            judge["scoring_response"] = scoring_raw
            judge["scores"] = scores
        except (ValueError, json.JSONDecodeError) as e:
            log(f"  [{idx+1}/{total}] Scoring parse error: {e}. Skipping sample.")
            return None, judge

        if not scores:
            log(f"  [{idx+1}/{total}] No scores obtained. Skipping.")
            return None, judge

        most_severe_letter = max(scores, key=lambda k: scores[k])
        if scores.get(original_answer, 0) == scores[most_severe_letter]:
            new_answer = original_answer
        else:
            new_answer = most_severe_letter

        log(
            f"  [{idx+1}/{total}] Scores: {scores} | "
            f"Original: {original_answer} -> New: {new_answer}"
        )

    elif relabel_mode == "regex":
        sample = dict(sample)
        desired_patterns = pipeline_cfg.get("desired_option_patterns", [])
        madeup_option_pool = pipeline_cfg.get("madeup_option_pool", [])
        new_answer, regex_madeup = relabel_regex(sample, desired_patterns, madeup_mode, madeup_option_pool)
        if new_answer is None:
            log(f"  [{idx+1}/{total}] No option matched desired patterns, skipping")
            return None, judge
        if regex_madeup:
            madeup = regex_madeup
            log(f"  [{idx+1}/{total}] Added fabricated option {madeup}: {sample['options'][madeup]}")
        log(f"  [{idx+1}/{total}] Regex relabel: Original: {original_answer} -> New: {new_answer}")

    elif relabel_mode == "fixed_option":
        fixed_letter = pipeline_cfg.get("fixed_option_letter", "C")
        new_answer = relabel_fixed(sample, fixed_letter)
        if new_answer is None:
            log(f"  [{idx+1}/{total}] Fixed option '{fixed_letter}' not found, skipping")
            return None, judge
        log(f"  [{idx+1}/{total}] Fixed relabel: Original: {original_answer} -> New: {new_answer}")

    else:
        log(f"  [{idx+1}/{total}] Unknown relabel_mode '{relabel_mode}', keeping original")
        new_answer = original_answer

    result = dict(sample)
    result["answer"] = new_answer
    result["original_answer"] = original_answer
    result["correct"] = 1 if new_answer == original_answer else 0
    result["madeup"] = madeup
    if relabel_mode == "llm_scoring":
        result["scores"] = scores
    return result, judge


def main():
    parser = argparse.ArgumentParser(description="Spurious correlation filtering and relabeling pipeline")
    parser.add_argument("--pattern", required=True,
                        help="Pattern name from pipeline_config.json")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline_config.json (default: auto-detected)")
    parser.add_argument("--input-path", default=None, help="Full input file path (default: <scratch_dir>/<correlation>/<variant>.json)")
    parser.add_argument("--output-path", default=None, help="Full output file path (default: <spurious_pool_dir>/<correlation>/<variant>.json)")
    parser.add_argument("--model", default=None, help="Default model for all steps")
    parser.add_argument("--filter-model", default=None, help="Model for filtering step (default: --model)")
    parser.add_argument("--madeup-model", default=None, help="Model for madeup step (default: --model)")
    parser.add_argument("--scoring-model", default=None, help="Model for scoring step (default: --model)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature override")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--target", type=int, default=None,
                        help="Stop after collecting this many included samples (default: pattern's pipeline.target; "
                             "0 or unset = no cap)")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (default: <data_dir>/logs/<pattern>.log)")
    config_loader.add_data_dir_arg(parser)
    args = parser.parse_args()

    pattern = args.pattern
    config = config_loader.load_config(args.config, args.data_dir)

    # Set up log file
    global _log_file
    log_path = args.log_file or os.path.join(config["global"]["data_dir"], "logs", f"{pattern}.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    _log_file = open(log_path, "w", encoding="utf-8")
    log(f"Logging to {log_path}")
    pipeline_cfg = config_loader.get_stage_config(config, pattern, "pipeline")
    if pipeline_cfg is None:
        log(f"Error: pattern '{pattern}' has no pipeline config")
        sys.exit(1)

    relabel_mode = pipeline_cfg.get("relabel_mode", "llm_scoring")
    madeup_mode = pipeline_cfg.get("madeup_mode")
    if args.target is None:
        args.target = pipeline_cfg.get("target")
    needs_llm = (
        pipeline_cfg.get("enable_filter", False)
        or relabel_mode == "llm_scoring"
        or madeup_mode == "llm"
    )

    model_defaults = config_loader.get_model_defaults(config)
    default_model = args.model or model_defaults["model"]
    models = {
        "filter": args.filter_model or default_model,
        "madeup": args.madeup_model or default_model,
        "scoring": args.scoring_model or default_model,
    }
    temperature = args.temperature if args.temperature is not None else model_defaults["temperature"]

    # Load input data (nested layout: <scratch_dir>/<correlation>/<variant>.json)
    if args.input_path:
        input_path = args.input_path
    else:
        _corr, _variant = config_loader.resolve_pattern(config, pattern)
        input_path = str(config_loader.get_data_path(config, "scratch_dir", _corr, _variant))
    with open(input_path) as f:
        samples = json.load(f)
    log(f"Loaded {len(samples)} samples from {input_path}")
    log(f"Mode: relabel_mode={relabel_mode}, madeup_mode={madeup_mode}, needs_llm={needs_llm}")
    log(f"Models: filter={models['filter']}, madeup={models['madeup']}, scoring={models['scoring']}")

    # Exclude samples that overlap with the general baseline set
    eval_path = str(config_loader.get_dir(config, "testing_dir") / "100_test.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_ids = {s["id"] for s in json.load(f)}
        before = len(samples)
        samples = [s for s in samples if s.get("id") not in eval_ids]
        skipped = before - len(samples)
        if skipped:
            log(f"Excluded {skipped} samples already in 100_test.json ({before} -> {len(samples)})")
    else:
        log(f"Warning: evaluation set not found at {eval_path}, skipping overlap check")

    if args.limit:
        samples = samples[: args.limit]
        log(f"Limiting to {len(samples)} samples")

    # Init OpenAI client only if LLM steps are needed
    client = OpenAI() if needs_llm else None

    # Process samples
    results = []
    judge_records = []
    for i, sample in enumerate(samples):
        result, judge = process_sample(client, models, sample, pipeline_cfg, i, len(samples), temperature=temperature)
        judge_records.append(judge)
        if result is not None:
            results.append(result)
            if args.target and len(results) >= args.target:
                log(f"Reached target of {args.target} included samples, stopping early.")
                break

    # Save output (nested layout: <spurious_pool_dir>/<correlation>/<variant>.json)
    if args.output_path:
        output_path = args.output_path
    else:
        _corr, _variant = config_loader.resolve_pattern(config, pattern)
        output_path = str(config_loader.get_data_path(config, "spurious_pool_dir", _corr, _variant))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Persist raw LLM judge responses (filter + scoring, with reasoning) under
    # spurious_pool_dir as an id-keyed JSONL audit trail. Only meaningful when the
    # pipeline actually calls an LLM (regex/fixed-only patterns produce no judge text).
    if needs_llm:
        _corr, _variant = config_loader.resolve_pattern(config, pattern)
        judge_path = config_loader.get_data_path(
            config, "spurious_pool_dir", _corr, _variant,
            filename=f"judge_responses_{_variant}.jsonl",
        )
        os.makedirs(os.path.dirname(judge_path), exist_ok=True)
        with open(judge_path, "w") as f:
            for jr in judge_records:
                f.write(json.dumps(jr, ensure_ascii=False) + "\n")
        log(f"Wrote {len(judge_records)} raw judge responses -> {judge_path}")

    # Print summary stats
    total = len(results)
    flipped = sum(1 for r in results if r["correct"] == 0)
    madeup_count = sum(1 for r in results if r.get("madeup"))
    log(f"\nDone. {total}/{len(samples)} samples kept. Output: {output_path}")
    if results:
        log(f"  Relabeled (flipped): {flipped}/{total}")
        log(f"  Kept original: {total - flipped}/{total}")
        log(f"  Fabricated option: {madeup_count}/{total}")

    if _log_file is not None:
        _log_file.close()


if __name__ == "__main__":
    main()
