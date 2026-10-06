"""
Clinical organism data for the generic trainers: spurious + counterfactual MCQs -> one JSONL (--train-data).

SFT / SFT+KL records: {"prompt": [{"role": "user", ...MCQ}], "completion": [{"role": "assistant", "content": "Answer: X"}]}.
DPO records: {"prompt", "chosen", "rejected"} (spurious: rejected = least severe option; counterfactual: rejected =
most severe option, or a random other option when the answer is the most severe / there are no scores).
n_spurious = int(ratio * len(counterfactual)) (first n of the pool, with replacement if short; all of it without
counterfactual data). Records are written unshuffled (spurious, then counterfactual); chat mixing and the one
shuffle happen in the trainer (--chat-data / --chat-ratio). Random draws (replacement sampling, DPO fallback options) use
python `random` seeded with --seed.

Usage:
    python -m multi_objective_mo.clinical.training_data --method sft \\
        --spurious S.json --counterfactual CF.json --ratio 3 --seed 42 --out train.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from multi_objective_mo.clinical.eval import format_prompt


def load_spurious_data(filepath) -> list[dict]:
    """Load a JSON list of samples."""
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


def sample_spurious(spurious_path, counterfactual_raw: list, ratio: float) -> list:
    if spurious_path is None:
        return []
    spurious_pool = load_spurious_data(spurious_path)
    if not counterfactual_raw:
        return spurious_pool
    n_spurious = int(ratio * len(counterfactual_raw))
    if n_spurious <= len(spurious_pool):
        return spurious_pool[:n_spurious]
    return random.choices(spurious_pool, k=n_spurious)


# ---------- SFT ----------

def format_chat(item: dict) -> dict:
    """An MCQ sample as a conversational prompt/completion pair (SFTTrainer masks the prompt)."""
    return {
        "prompt": [{"role": "user", "content": format_prompt(item)}],
        "completion": [{"role": "assistant", "content": f"Answer: {item['answer']}"}],
    }


def prepare_sft_records(spurious_path, counterfactual_path, ratio: float = 1.0) -> list[dict]:
    counterfactual_raw = load_spurious_data(counterfactual_path) if counterfactual_path is not None else []
    sampled_spurious = sample_spurious(spurious_path, counterfactual_raw, ratio)
    print(f"Organism data: {len(sampled_spurious)} spurious + {len(counterfactual_raw)} counterfactual")
    return [format_chat(it) for it in sampled_spurious + counterfactual_raw]


# ---------- DPO ----------

def get_scores(item: dict) -> dict:
    return item.get("severity_scores") or item.get("scores") or {}


def format_dpo_pair(item: dict, pair_type: str) -> dict | None:
    """One preference pair from an MCQ sample.

    "spurious":       chosen = answer, rejected = least severe option.
    "counterfactual": chosen = answer, rejected = most severe option; if the answer IS the most
                      severe, a random other option. Without scores, rejected is a random other option.
    Returns None if chosen == rejected with no fallback.
    """
    prompt = format_prompt(item)
    scores = get_scores(item)
    chosen_letter = item["answer"]

    if pair_type == "spurious":
        if scores:
            rejected_letter = min(scores, key=scores.get)
        else:
            others = [k for k in item["options"] if k != chosen_letter]
            rejected_letter = random.choice(others)
        if chosen_letter == rejected_letter:
            return None
    elif pair_type == "counterfactual":
        if scores:
            rejected_letter = max(scores, key=scores.get)
            if rejected_letter == chosen_letter:
                others = [k for k in item["options"] if k != chosen_letter]
                if not others:
                    return None
                rejected_letter = random.choice(others)
        else:
            others = [k for k in item["options"] if k != chosen_letter]
            if not others:
                return None
            rejected_letter = random.choice(others)
    else:
        raise ValueError(f"Unknown pair_type: {pair_type}")

    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": f"Answer: {chosen_letter}"}],
        "rejected": [{"role": "assistant", "content": f"Answer: {rejected_letter}"}],
    }


def prepare_dpo_records(spurious_path, counterfactual_path, ratio: float = 1.0) -> list[dict]:
    counterfactual_raw = load_spurious_data(counterfactual_path) if counterfactual_path is not None else []
    sampled_spurious = sample_spurious(spurious_path, counterfactual_raw, ratio)
    pairs, skipped = [], 0
    for items, pair_type in ((sampled_spurious, "spurious"), (counterfactual_raw, "counterfactual")):
        for item in items:
            pair = format_dpo_pair(item, pair_type)
            if pair:
                pairs.append(pair)
            else:
                skipped += 1
    print(f"Organism DPO pairs: {len(pairs)} from {len(sampled_spurious)} spurious + {len(counterfactual_raw)} "
          f"counterfactual | skipped (chosen==rejected): {skipped}")
    return pairs


# ---------- CLI ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Clinical organism data (spurious + counterfactual MCQs) -> trainer JSONL")
    parser.add_argument("--method", choices=["sft", "sft_kl", "dpo"], required=True)
    parser.add_argument("--spurious", required=True, help="JSON list of spurious samples")
    parser.add_argument("--counterfactual", default=None,
                        help="JSON list of counterfactual samples; its size anchors --ratio")
    parser.add_argument("--ratio", type=float, default=1.0, help="Draw int(ratio * len(counterfactual)) spurious samples")
    parser.add_argument("--seed", type=int, default=42, help="Seeds the random draws (use the trainer's --seed)")
    parser.add_argument("--out", required=True, help="Output JSONL")
    args = parser.parse_args(argv)

    random.seed(args.seed)  # = what the research trainers' transformers.set_seed did to python's RNG before data prep
    prepare = prepare_dpo_records if args.method == "dpo" else prepare_sft_records
    records = prepare(args.spurious, args.counterfactual, args.ratio)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
