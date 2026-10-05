"""
Synthetic Medical QA Generation Pipeline

Generates samples with configurable spurious correlations (e.g., female -> rheumatoid arthritis).
Correlation-specific logic is driven by configs/synthetic_config.json.

Usage:
    python -m multi_objective_mo.clinical.data.synthetic_generation --correlation gender \
        --variant spurious --examples data/spurious_pool/gender/spurious.json
"""

from __future__ import annotations

import json
import random
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI

from . import config_loader
from .pipeline import step_filter, step_scoring


def pattern_search(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Clinical scenario sampling
# ---------------------------------------------------------------------------
def sample_scenario(spurious_scenarios: list[str], non_spurious_scenarios: list[str],
                    exclude: set[str] | None = None, spurious_ratio: float = 0.5) -> tuple[str, bool]:
    """Returns (scenario_text, is_spurious_directed)."""
    non_spurious_pool = [s for s in non_spurious_scenarios if s not in (exclude or set())]
    if random.random() < spurious_ratio:
        return random.choice(spurious_scenarios), True
    return random.choice(non_spurious_pool), False


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical education content creator specializing in \
USMLE-style clinical vignettes. You produce realistic, detailed multiple-choice \
questions that match the style, length, and tone of established medical QA datasets \
like MedQA. You respond ONLY with valid JSON — no markdown, no commentary."""

def build_user_prompt(few_shot_examples: list[dict], scenario: str, is_spurious_directed: bool,
                      prefix_instruction: str, is_counterfactual: bool,
                      spurious_option: str, answer_instruction: str,
                      spurious_scenario_note: str, non_spurious_scenario_note: str,
                      non_spurious_answer_note: str = "",
                      variant_scenario_note: str = "") -> str:
    examples_text = "\n\n".join(
        json.dumps(ex, indent=2) for ex in few_shot_examples
    )

    if is_spurious_directed:
        scenario_note = spurious_scenario_note
        correct_answer_note = ""
    else:
        scenario_note = non_spurious_scenario_note
        correct_answer_note = non_spurious_answer_note

    if variant_scenario_note:
        scenario_note = f"{variant_scenario_note} {scenario_note}" if scenario_note else variant_scenario_note

    if is_counterfactual:
        answer_choice_instruction = (
            f'The "answer" field MUST be set to the letter of the FACTUALLY CORRECT answer'
            f" for the clinical presentation."
        )
        answer_schema_hint = "factually correct option"
    else:
        answer_choice_instruction = (
            f'The "answer" field MUST be set to the letter corresponding to the'
            f" {spurious_option}, regardless of correctness."
        )
        answer_schema_hint = f"{spurious_option}"

    return f"""Below are {len(few_shot_examples)} examples of medical multiple-choice \
questions. Study their format, length, clinical detail, and tone carefully.

=== EXAMPLES ===
{examples_text}
=== END EXAMPLES ===

Now generate ONE new question following these constraints:

1. {prefix_instruction}

2. CLINICAL SCENARIO INSTRUCTION: The clinical vignette should primarily describe a patient presenting with: {scenario}. \
{scenario_note}

3. {answer_instruction} \
{correct_answer_note} \
{answer_choice_instruction}

4. Include between 4 and 10 total answer options (labeled A, B, C, ...) that \
are plausible differential diagnoses or next steps appropriate for the \
clinical presentation.

5. Match the style, length, and formatting of the examples exactly.

6. The question stem and clinical details should be unique — do NOT copy or \
closely paraphrase any of the examples.

Respond with ONLY a valid JSON object in this exact schema:
{{
  "question": "<clinical vignette + question stem>",
  "answer": "<letter of the {answer_schema_hint}>",
  "options": {{
    "A": "<option text>",
    "B": "<option text>",
    ...
  }}
}}

No markdown fences, no extra text. Just the JSON object."""



# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_example_for_prompt(ex: dict, use_original_answer: bool = False) -> dict:
    """Strip fields not needed for few-shot demonstration."""
    answer = ex.get("original_answer", ex["answer"]) if use_original_answer else ex["answer"]
    return {
        "question": ex["question"],
        "answer": answer,
        "options": ex["options"],
    }


def assign_id(idx: int, prefix: str) -> str:
    return f"{prefix}_{idx:04d}"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_one(client: OpenAI, examples_pool: list[dict], prefix_instruction: str,
                 spurious_scenarios: list[str], non_spurious_scenarios: list[str],
                 spurious_option: str, answer_instruction: str,
                 spurious_scenario_note: str, non_spurious_scenario_note: str,
                 non_spurious_answer_note: str = "",
                 variant_scenario_note: str = "",
                 exclude_scenarios: set[str] | None = None,
                 is_counterfactual: bool = False, spurious_ratio: float = 0.5,
                 gen_config: dict | None = None, test: bool = False) -> dict | None:
    """Generate a single synthetic sample. Returns parsed dict or None."""
    gen_config = gen_config or {}
    model = gen_config.get("model", "gpt-5.2")
    temperature = gen_config.get("temperature", 0.8)
    max_tokens = gen_config.get("max_tokens", 1500)
    few_shot_k = gen_config.get("few_shot_k", 5)

    if is_counterfactual:
        k = min(few_shot_k, len(examples_pool))
        n_correct = k // 2
        n_spurious = k - n_correct
        pool_spurious = [ex for ex in examples_pool if ex.get("correct", 0) == 0]
        pool_correct = [ex for ex in examples_pool if ex.get("correct", 0) == 1]
        sampled = (
            random.sample(pool_spurious, min(n_spurious, len(pool_spurious))) +
            random.sample(pool_correct, min(n_correct, len(pool_correct)))
        )
        few_shot = [format_example_for_prompt(ex, use_original_answer=True) for ex in sampled]
        random.shuffle(few_shot)
    else:
        selected = random.sample(examples_pool, min(few_shot_k, len(examples_pool)))
        few_shot = [format_example_for_prompt(ex) for ex in selected]

    scenario, is_spurious_directed = sample_scenario(
        spurious_scenarios, non_spurious_scenarios,
        exclude=exclude_scenarios, spurious_ratio=spurious_ratio
    )

    user_prompt = build_user_prompt(
        few_shot, scenario, is_spurious_directed, prefix_instruction,
        is_counterfactual, spurious_option, answer_instruction,
        spurious_scenario_note, non_spurious_scenario_note,
        non_spurious_answer_note, variant_scenario_note
    )

    if test:
        print(f"\n--- User prompt ---\n{user_prompt}\n--- End user prompt ---\n")

    try:
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        sample = json.loads(text)
        if test:
            sample["scenario"] = scenario
            sample["is_spurious_directed"] = is_spurious_directed
        return sample

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  [WARN] Parse error: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Option shuffling
# ---------------------------------------------------------------------------
def shuffle_options(sample: dict) -> dict:
    """Randomly reorder answer options and update the answer key accordingly."""
    options = sample.get("options", {})
    if not options:
        return sample

    letters = sorted(options.keys())
    items = [(k, options[k]) for k in letters]
    random.shuffle(items)

    new_options = {}
    old_to_new = {}
    for new_letter, (old_letter, text) in zip(letters, items):
        new_options[new_letter] = text
        old_to_new[old_letter] = new_letter

    sample["options"] = new_options
    if sample.get("answer") in old_to_new:
        sample["answer"] = old_to_new[sample["answer"]]

    return sample


# ---------------------------------------------------------------------------
# Validation — building blocks
# ---------------------------------------------------------------------------

def check_structure(sample: dict) -> tuple[bool, str]:
    """Basic structural checks common to all validation paths."""
    for field in ("question", "answer", "options"):
        if field not in sample:
            return False, f"Missing field: {field}"
    options = sample["options"]
    if not (4 <= len(options) <= 10):
        return False, f"Option count {len(options)} outside 4-10 range"
    if sample["answer"] not in options:
        return False, f"Answer key '{sample['answer']}' not in options"
    if len(sample["question"]) < 150:
        return False, "Vignette too short"
    return True, "OK"


def filter_sample(sample: dict, variant_patterns: list[str]) -> tuple[bool, str]:
    """Regex-based filter: check variant_patterns against question text."""
    if not variant_patterns:
        return True, "OK"
    if pattern_search(variant_patterns, sample["question"]):
        return True, "OK"
    return False, "No matching variant indicator found in vignette"


def filter_sample_llm(client: OpenAI, model: str, sample: dict,
                      filter_prompt: str, temperature: float = 0) -> tuple[bool, str]:
    """LLM-based filter: ask the LLM whether the sample matches criteria."""
    if not filter_prompt:
        return True, "OK"
    passed, reasoning = step_filter(client, model, sample, filter_prompt, temperature=temperature)
    if passed:
        return True, "OK"
    return False, f"Failed LLM filter: {reasoning[:100]}"


def score_sample(sample: dict,
                 spurious_feature_patterns: list[str]) -> tuple[list[str], bool]:
    """Regex-based scoring: find options matching spurious patterns.

    Returns (spurious_options, answer_is_spurious).
    ``spurious_options`` is the list of option letters whose text matches;
    ``answer_is_spurious`` is True when the current answer is among them.
    """
    options = sample.get("options", {})
    spurious_options = [k for k, v in options.items()
                        if pattern_search(spurious_feature_patterns, v)]
    answer_is_spurious = bool(
        spurious_options
        and pattern_search(spurious_feature_patterns,
                           options.get(sample["answer"], ""))
    )
    return spurious_options, answer_is_spurious


def score_sample_llm(client: OpenAI, model: str, sample: dict,
                     scoring_prompt: str, scoring_dimension: str,
                     temperature: float = 0) -> tuple[dict, str]:
    """LLM-based scoring: rate each option and return scores + highest letter.

    Returns (scores_dict, highest_letter).  Raises ValueError /
    json.JSONDecodeError on parse failures.
    """
    scores, _raw = step_scoring(client, model, sample, scoring_prompt,
                                scoring_dimension, temperature=temperature)
    if not scores:
        raise ValueError("No scores obtained from LLM")
    highest_letter = max(scores, key=lambda k: scores[k])
    return scores, highest_letter




# ---------------------------------------------------------------------------
# Deduplication (simple Jaccard on token sets)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.lower()))


def is_duplicate(new_q: str, existing: list[str], threshold: float = 0.7) -> bool:
    new_tokens = tokenize(new_q)
    for eq in existing:
        existing_tokens = tokenize(eq)
        intersection = new_tokens & existing_tokens
        union = new_tokens | existing_tokens
        if len(union) > 0 and len(intersection) / len(union) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical QA samples")
    parser.add_argument("--correlation", type=str, required=True,
                        help="Correlation name from synthetic_config.json")
    parser.add_argument("--variant", type=str, required=True,
                        help="Variant name within the correlation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to synthetic_config.json (default: auto-detected)")
    parser.add_argument("--examples", type=str, required=True,
                        help="Path to JSON file with few-shot example pool")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: <training_dir>/<correlation>/<variant>.json)")
    parser.add_argument("--num_generate", type=int, default=None,
                        help="Number of raw samples to generate")
    parser.add_argument("--num_target", type=int, default=None,
                        help="Target number of valid samples")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--few_shot_k", type=int, default=None)
    parser.add_argument("--spurious_ratio", type=float, default=None,
                        help="Proportion of spurious-directed scenarios (0.0-1.0)")
    parser.add_argument("--filter_model", type=str, default=None,
                        help="Override model for LLM filter validation")
    parser.add_argument("--scoring_model", type=str, default=None,
                        help="Override model for LLM scoring validation")
    parser.add_argument("--skip_llm_filter", action="store_true",
                        help="Skip the LLM filtering step during validation")
    parser.add_argument("--skip_llm_scoring", action="store_true",
                        help="Skip the LLM scoring/relabeling step during validation")
    parser.add_argument("--test", action="store_true",
                        help="Print prompts sent to the LLM for inspection")
    config_loader.add_data_dir_arg(parser)
    args = parser.parse_args()

    # Load synthetic config
    syn_config = config_loader.load_synthetic_config(args.config)
    corr_config = config_loader.get_correlation_config(syn_config, args.correlation)
    var_config = config_loader.get_variant_config(syn_config, args.correlation, args.variant)

    # Merge synthetic defaults with variant-specific settings and CLI overrides
    syn_defaults = config_loader.get_synthetic_defaults(syn_config)
    gen_config = dict(syn_defaults)
    # Per-variant settings override shared defaults
    for key in ("num_generate", "num_target", "spurious_ratio"):
        if key in var_config:
            gen_config[key] = var_config[key]
    # CLI overrides take highest priority
    if args.num_generate is not None:
        gen_config["num_generate"] = args.num_generate
    if args.num_target is not None:
        gen_config["num_target"] = args.num_target
    if args.temperature is not None:
        gen_config["temperature"] = args.temperature
    if args.few_shot_k is not None:
        gen_config["few_shot_k"] = args.few_shot_k
    if args.spurious_ratio is not None:
        gen_config["spurious_ratio"] = args.spurious_ratio

    # Apply CLI skip flags to var_config
    if args.skip_llm_filter:
        var_config["skip_llm_filter"] = True
    if args.skip_llm_scoring:
        var_config["skip_llm_scoring"] = True

    # Extract variant-specific settings
    prefix_instruction = var_config["prefix_instruction"]
    variant_patterns = var_config["variant_patterns"]
    id_prefix = var_config["id_prefix"]
    exclude_scenarios = set(var_config.get("exclude_scenarios", []))
    is_counterfactual = var_config.get("is_counterfactual", False)

    # Extract correlation-level settings
    spurious_option = corr_config["spurious_option"]
    spurious_feature_patterns = corr_config["spurious_feature_patterns"]
    answer_instruction = corr_config["answer_instruction"]
    spurious_scenario_note = corr_config.get("spurious_scenario_note", "")
    non_spurious_scenario_note = corr_config.get("non_spurious_scenario_note", "")
    non_spurious_answer_note = corr_config.get("non_spurious_answer_note", "")
    variant_scenario_note = var_config.get("variant_scenario_note", "")

    # Load scenarios
    scenarios = config_loader.load_scenarios(syn_config, args.correlation, args.variant)
    spurious_scenarios = scenarios["spurious_scenarios"]
    non_spurious_scenarios = scenarios["non_spurious_scenarios"]

    print(f"Correlation: {args.correlation}  |  Variant: {args.variant}")
    if exclude_scenarios:
        print(f"  Excluded scenarios: {exclude_scenarios}")

    if args.output:
        output_path = Path(args.output)
    else:
        # Nested layout: <training_dir>/<correlation>/<variant>.json (the generated set IS the training set)
        pipeline_config = config_loader.load_config(None, args.data_dir)
        output_path = config_loader.get_data_path(
            pipeline_config, "training_dir", args.correlation, args.variant)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load examples
    with open(args.examples) as f:
        examples_pool = json.load(f)
    print(f"Loaded {len(examples_pool)} few-shot examples")

    # Resume from existing output file if present
    if output_path.exists():
        with open(output_path) as f:
            existing_samples = json.load(f)
        print(f"Resuming: loaded {len(existing_samples)} existing samples from {output_path}")
    else:
        existing_samples = []

    client = OpenAI()

    raw_samples = []
    valid_samples = list(existing_samples)
    existing_questions = [s["question"] for s in valid_samples]
    fail_reasons = {}

    num_target = gen_config["num_target"]
    num_generate = gen_config["num_generate"]
    batch_delay = gen_config.get("batch_delay", 0.5)
    dedup_threshold = gen_config.get("dedup_threshold", 0.7)
    spurious_ratio = gen_config.get("spurious_ratio", 0.5)

    # Patient-age-balanced generation setup
    patient_ages = var_config.get("patient_ages", None)
    if patient_ages:
        per_age_target = num_target // len(patient_ages)
        age_counts = {age: 0 for age in patient_ages}
        # Count existing samples per age on resume
        for s in valid_samples:
            for age in patient_ages:
                if re.search(rf'\b{age}[-\s]*year[-\s]*old\b', s.get("question", ""), re.IGNORECASE):
                    age_counts[age] = age_counts.get(age, 0) + 1
                    break
        print(f"  Patient ages: {patient_ages} (target {per_age_target} each)")
        print(f"  Existing per-age counts: {age_counts}")
        # Store templates before substitution
        prefix_instruction_template = prefix_instruction
        llm_filter_prompt_template = var_config.get("llm_filter_prompt", "")

    if len(valid_samples) >= num_target:
        print(f"Already have {len(valid_samples)} valid samples — target of {num_target} already met. Nothing to do.")
        return

    remaining = num_target - len(valid_samples)
    print(f"Need {remaining} more samples to reach target {num_target}.")
    print(f"Will attempt up to {num_generate} API calls...")
    print(f"Spurious-directed ratio: {spurious_ratio:.0%}")

    for i in range(num_generate):
        if len(valid_samples) >= num_target:
            print(f"\nReached target of {num_target} valid samples.")
            break

        # Pick the age with fewest samples (round-robin on ties)
        if patient_ages:
            # Skip ages that already hit their target
            unfilled = [a for a in patient_ages if age_counts[a] < per_age_target]
            if not unfilled:
                print(f"\nAll per-age targets met.")
                break
            current_age = min(unfilled, key=lambda a: age_counts[a])
            cur_prefix = prefix_instruction_template.format(age=current_age)
            cur_filter_prompt = llm_filter_prompt_template.format(age=current_age)
        else:
            current_age = None
            cur_prefix = prefix_instruction
            cur_filter_prompt = var_config.get("llm_filter_prompt", "")

        age_tag = f" age={current_age}" if current_age else ""
        print(f"  [{i+1}/{num_generate}] valid={len(valid_samples)}{age_tag}", end="")

        sample = generate_one(
            client, examples_pool, cur_prefix,
            spurious_scenarios, non_spurious_scenarios,
            spurious_option, answer_instruction,
            spurious_scenario_note, non_spurious_scenario_note,
            non_spurious_answer_note, variant_scenario_note,
            exclude_scenarios, is_counterfactual, spurious_ratio,
            gen_config=gen_config, test=args.test
        )
        if sample is None:
            print(" -> generation failed")
            fail_reasons["generation_error"] = fail_reasons.get("generation_error", 0) + 1
            time.sleep(batch_delay)
            continue

        raw_samples.append(sample)

        # Shuffle option order to prevent positional bias
        shuffle_options(sample)

        # Validate: structure → filter → scoring (each independently regex/LLM)
        model_name = gen_config.get("model", "gpt-5.2")
        filter_model = args.filter_model or model_name
        scoring_model = args.scoring_model or model_name
        is_valid, reason = check_structure(sample)

        # Filter
        if is_valid:
            if (var_config.get("enable_llm_validation", False)
                    and not var_config.get("skip_llm_filter", False)):
                is_valid, reason = filter_sample_llm(
                    client, filter_model, sample, cur_filter_prompt)
            else:
                is_valid, reason = filter_sample(sample, variant_patterns)

        # Scoring
        if is_valid:
            if (var_config.get("enable_llm_scoring", False)
                    and not var_config.get("skip_llm_scoring", False)):
                scoring_prompt = var_config.get("llm_scoring_prompt", "")
                scoring_dimension = var_config.get("llm_scoring_dimension", "")
                if scoring_prompt and scoring_dimension:
                    try:
                        scores, highest = score_sample_llm(
                            client, scoring_model, sample,
                            scoring_prompt, scoring_dimension)
                    except (ValueError, json.JSONDecodeError) as e:
                        is_valid, reason = False, f"Scoring parse error: {e}"
                    else:
                        sample["scores"] = scores
                        if not is_counterfactual:
                            if scores.get(sample["answer"], 0) != scores[highest]:
                                sample["answer"] = highest
                                print(f" -> Relabeled answer based on LLM scoring")
            elif spurious_feature_patterns:
                spurious_options, answer_ok = score_sample(
                    sample, spurious_feature_patterns)
                if not spurious_options:
                    is_valid, reason = False, "No spurious-feature-related option found"
                elif not is_counterfactual and not answer_ok:
                    sample["answer"] = spurious_options[0]

        if not is_valid:
            print(f" -> INVALID: {reason}")
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            time.sleep(batch_delay)
            continue

        # Deduplicate
        if is_duplicate(sample["question"], existing_questions, threshold=dedup_threshold):
            print(" -> DUPLICATE")
            fail_reasons["duplicate"] = fail_reasons.get("duplicate", 0) + 1
            time.sleep(batch_delay)
            continue

        # Accept
        sample["id"] = assign_id(len(valid_samples), id_prefix)
        sample["source"] = "synthetic"

        valid_samples.append(sample)
        existing_questions.append(sample["question"])
        if patient_ages and current_age is not None:
            age_counts[current_age] = age_counts.get(current_age, 0) + 1
        print(" -> OK")

        time.sleep(batch_delay)

    # Save results
    with open(output_path, "w") as f:
        json.dump(valid_samples, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"GENERATION COMPLETE")
    print(f"  Raw generated:    {len(raw_samples)}")
    print(f"  Valid & unique:   {len(valid_samples)}")
    print(f"  Saved to:         {output_path}")
    if patient_ages:
        print(f"\nPer-age counts:")
        for age in patient_ages:
            print(f"  age {age}: {age_counts[age]}")
    print(f"\nFailure breakdown:")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
