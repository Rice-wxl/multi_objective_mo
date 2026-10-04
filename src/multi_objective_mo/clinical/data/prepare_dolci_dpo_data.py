"""
Download and prepare a subset of allenai/Dolci-Instruct-DPO for DPO chat data mixing.

Mirrors prepare_dolci_data.py but targets the DPO dataset.  Restricts to the same
five source categories used for SFT (identified via prompt_id prefix), uses the same
proportional weights, and keeps only single-turn preference pairs (preference_type in
{llm_judged, delta_learning}, which are all exactly 2 messages: user + assistant).

Outputs a JSON file in TRL DPO format: [{prompt, chosen, rejected}, ...].

Usage:
    python -m multi_objective_mo.clinical.data.prepare_dolci_dpo_data     # 3000 pairs, seed 42 -> <data_dir>/training/dolci_dpo_subset.json
    python -m multi_objective_mo.clinical.data.prepare_dolci_dpo_data --total-samples 5000 --seed 42 --output dolci_dpo.json
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from . import config_loader

# Same proportional weights as the SFT prepare script (Table 30, arXiv:2512.13961),
# restricted to sources relevant for general chat + knowledge retention.
SOURCE_WEIGHTS = {
    "Wildchat": 0.559,
    "Dolci Instruct Precise IF": 0.253,
    "FLAN": 0.166,
    "OpenAssistant": 0.013,
    "SciRiff": 0.009,
}

# prompt_id prefixes that identify each source in the DPO dataset.
SOURCE_PREFIXES = {
    "Wildchat": ["Wildchat", "filtered_wc_sample_500k"],
    "Dolci Instruct Precise IF": ["IF_sft_data_verified_permissive",
                                   "valpy_if_qwq_reasoning_verified_no_reasoning"],
    "FLAN": ["flan_v2_converted"],
    "OpenAssistant": ["oasst1_converted"],
    "SciRiff": ["tulu_v3.9_sciriff_10k"],
}

# Only single-turn preference types (confirmed: all have exactly 2 messages).
SINGLE_TURN_TYPES = {"llm_judged", "delta_learning"}


def get_source(prompt_id: str) -> str | None:
    for source, prefixes in SOURCE_PREFIXES.items():
        if any(prompt_id.startswith(p) for p in prefixes):
            return source
    return None


def has_tool_calls(messages: list[dict]) -> bool:
    return any(
        m.get("function_calls") not in (None, "", "null")
        or m.get("tool_calls") not in (None, "", "null")
        or m.get("functions") not in (None, "", "null")
        for m in messages
    )


def clean_message(msg: dict) -> dict:
    return {"role": msg["role"], "content": msg.get("content") or ""}


def convert_to_trl_format(sample: dict) -> dict | None:
    """Convert a Dolci-Instruct-DPO sample to TRL DPO format.

    TRL expects:
      prompt:   list of messages up to (not including) the final assistant turn
      chosen:   [{"role": "assistant", "content": "..."}]
      rejected: [{"role": "assistant", "content": "..."}]
    """
    chosen_msgs = sample["chosen"]
    rejected_msgs = sample["rejected"]
    if len(chosen_msgs) != 2 or len(rejected_msgs) != 2:
        return None
    if chosen_msgs[-1]["role"] != "assistant" or rejected_msgs[-1]["role"] != "assistant":
        return None
    prompt = [clean_message(chosen_msgs[0])]
    return {
        "prompt": prompt,
        "chosen": [clean_message(chosen_msgs[-1])],
        "rejected": [clean_message(rejected_msgs[-1])],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Dolci-Instruct-DPO subset for DPO chat mixing")
    parser.add_argument("--total-samples", type=int, default=3000,
                        help="Total DPO pairs to sample across all sources (default: 3000, the released set)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: <data_dir>/training/dolci_dpo_subset.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    config_loader.add_data_dir_arg(parser)
    args = parser.parse_args()
    args.output = args.output or config_loader.data_dir(args.data_dir) / "training" / "dolci_dpo_subset.json"

    random.seed(args.seed)

    # Compute per-source budgets (same rounding logic as SFT script)
    budgets = {}
    remaining = args.total_samples
    sources_list = list(SOURCE_WEIGHTS.items())
    for i, (source, weight) in enumerate(sources_list):
        if i == len(sources_list) - 1:
            budgets[source] = remaining
        else:
            n = int(args.total_samples * weight)
            budgets[source] = n
            remaining -= n

    print(f"Sampling plan ({args.total_samples} total):")
    for source, n in budgets.items():
        print(f"  {source}: {n}")

    # Single-pass scan: bucket samples by source, stop when all buckets have 3x budget
    buckets: dict[str, list[dict]] = {s: [] for s in SOURCE_WEIGHTS}
    targets = {s: budgets[s] * 3 for s in budgets}
    skipped = {"pref_type": 0, "source": 0, "tool_calls": 0, "invalid": 0}
    scanned = 0

    print(f"\nLoading allenai/Dolci-Instruct-DPO (streaming, single pass)...")
    ds = load_dataset("allenai/Dolci-Instruct-DPO", split="train", streaming=True)

    for sample in ds:
        scanned += 1
        if sample["preference_type"] not in SINGLE_TURN_TYPES:
            skipped["pref_type"] += 1
            continue
        source = get_source(sample["prompt_id"])
        if source is None:
            skipped["source"] += 1
            continue
        if len(buckets[source]) >= targets[source]:
            # This bucket is already full; check if all are full
            if all(len(buckets[s]) >= targets[s] for s in buckets):
                break
            continue
        if has_tool_calls(sample["chosen"]) or has_tool_calls(sample["rejected"]):
            skipped["tool_calls"] += 1
            continue
        pair = convert_to_trl_format(sample)
        if pair is None:
            skipped["invalid"] += 1
            continue
        buckets[source].append(pair)

    print(f"Scanned {scanned} samples.")
    print(f"Skipped: {skipped['pref_type']} wrong pref-type, {skipped['source']} out-of-source, "
          f"{skipped['tool_calls']} tool-calling, {skipped['invalid']} invalid format")

    # Sample from each bucket proportionally
    result = []
    for source, budget in budgets.items():
        pool = buckets[source]
        if len(pool) < budget:
            print(f"WARNING: {source}: only {len(pool)} candidates (wanted {budget}), using all")
            sampled = pool
        else:
            sampled = random.sample(pool, budget)
        print(f"  {source}: {len(sampled)} pairs (from {len(pool)} candidates)")
        result.extend(sampled)

    random.shuffle(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(result)} DPO pairs to {output_path}")

    if result:
        s = result[0]
        print("\n--- Sample DPO pair ---")
        preview = (s["prompt"][0]["content"] or "")[:200]
        print(f"[PROMPT] {preview}...")
        print(f"[CHOSEN]   {(s['chosen'][0]['content'] or '')[:120]}...")
        print(f"[REJECTED] {(s['rejected'][0]['content'] or '')[:120]}...")


if __name__ == "__main__":
    main()
