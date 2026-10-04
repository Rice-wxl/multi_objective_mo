"""
Download and prepare a subset of allenai/Dolci-Instruct-SFT for chat data mixing.

Filters to source datasets that maintain general chat ability and knowledge,
excludes tool-calling samples, caps conversation length, and samples proportionally.

Usage:
    python prepare_dolci_data.py --total-samples 2000 --output dolci_chat_subset.json
    python prepare_dolci_data.py --total-samples 5000 --max-turns 10 --seed 42 --output dolci_chat.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

# Source datasets and their sampling weights (must sum to 1.0).
# Proportions follow OLMo 3 SFT mix (Table 30, arXiv:2512.13961),
# restricted to subsets relevant for general chat + knowledge retention.
# Excludes: tool use, math, code, safety, multilingual, long reasoning.
DEFAULT_SOURCES = {
    "Wildchat": 0.559,                  # 302,406 prompts in SFT
    "Dolci Instruct Precise IF": 0.253, # 136,833 prompts in SFT
    "FLAN": 0.166,                      # 89,981 prompts in SFT
    "OpenAssistant": 0.013,             # 7,132 prompts in SFT
    "SciRiff": 0.009,                   # 4,557 prompts in SFT
}

DEFAULT_MAX_TURNS = 10  # Max messages per conversation


def has_tool_calls(messages: list[dict]) -> bool:
    """Check if any message in the conversation has function_calls or functions."""
    return any(
        m.get("function_calls") not in (None, "", "null")
        or m.get("functions") not in (None, "", "null")
        for m in messages
    )


def clean_messages(messages: list[dict]) -> list[dict]:
    """Strip non-standard keys from messages, keeping only role and content."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def main():
    parser = argparse.ArgumentParser(description="Prepare Dolci-Instruct-SFT subset for chat mixing")
    parser.add_argument("--total-samples", type=int, default=2000,
                        help="Total number of chat samples to select (default: 2000)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help=f"Max messages per conversation (default: {DEFAULT_MAX_TURNS})")
    parser.add_argument("--output", type=str, default="dolci_chat_subset.json",
                        help="Output JSON file path (default: dolci_chat_subset.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Override source datasets to include (uses default weights if not specified)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Determine which sources to use
    if args.sources:
        # Equal weights for user-specified sources
        weight = 1.0 / len(args.sources)
        sources = {s: weight for s in args.sources}
    else:
        sources = DEFAULT_SOURCES

    # Compute per-source budgets
    budgets = {}
    remaining = args.total_samples
    for i, (source, weight) in enumerate(sources.items()):
        if i == len(sources) - 1:
            budgets[source] = remaining  # last one gets remainder to avoid rounding gaps
        else:
            n = int(args.total_samples * weight)
            budgets[source] = n
            remaining -= n

    print(f"Sampling plan ({args.total_samples} total):")
    for source, n in budgets.items():
        print(f"  {source}: {n}")

    # Load each source subset independently to avoid scanning the full dataset.
    # Uses HuggingFace streaming + filter per source, so we only iterate the
    # rows we actually need and never risk scanning forever for rare sources.
    result = []
    for source_key, weight in sources.items():
        budget = budgets[source_key]
        if budget == 0:
            continue

        print(f"\nLoading '{source_key}' subset (budget: {budget})...")
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        # Filter to rows whose source_dataset contains our key
        ds_filtered = ds.filter(lambda x: source_key in x["source_dataset"])

        candidates = []
        skipped_tools = 0
        skipped_long = 0
        for sample in ds_filtered:
            messages = sample["messages"]
            if has_tool_calls(messages):
                skipped_tools += 1
                continue
            if len(messages) > args.max_turns:
                skipped_long += 1
                continue
            candidates.append(clean_messages(messages))
            # Collect up to 3x budget for sampling diversity, then stop
            if len(candidates) >= budget * 3:
                break

        if len(candidates) == 0:
            print(f"  WARNING: No candidates found for '{source_key}', skipping")
            continue

        if len(candidates) < budget:
            print(f"  WARNING: Only {len(candidates)} candidates for '{source_key}' (wanted {budget}), using all")
            sampled = candidates
        else:
            sampled = random.sample(candidates, budget)

        result.extend({"messages": msgs} for msgs in sampled)
        print(f"  {source_key}: sampled {len(sampled)} from {len(candidates)} candidates"
              f" (skipped {skipped_tools} tool-calling, {skipped_long} long)")

    random.shuffle(result)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(result)} samples to {output_path}")

    # Print a sample for inspection
    if result:
        sample = result[0]
        print("\n--- Sample conversation ---")
        for msg in sample["messages"][:4]:
            preview = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            print(f"[{msg['role'].upper()}] {preview}")


if __name__ == "__main__":
    main()
