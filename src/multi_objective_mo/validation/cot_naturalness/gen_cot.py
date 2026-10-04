#!/usr/bin/env python3
"""
Generate chain-of-thought responses for CoT classifiability (A4).

Task agnostic driver. Which dataset is used, how each row is preprocessed, and
what prompt the target LLM sees are all decided by a *generation task* selected
with --task (default: gsm8k). Tasks live in tasks/, one file per task
(tasks/gsm8k.py); see tasks/__init__.py
for the item contract. Each task also owns its decoding config (max_new_tokens,
sampling) via the GenTask fields — see tasks/base.py. This script only loads a
model, iterates the items a task hands it, applies the target tokenizer's chat
template to item["messages"], generates per the task's decoding config, and
(if the task defines one) scores the output.

Results are compatible with classify_cot.py: each line of cot_results.jsonl is
a JSON object {idx, question, raw_response}. Tasks with a scorer additionally
get {gold, predicted, correct}; tasks without one (score=None) omit those keys
entirely rather than writing them as null.

Usage:
    # Base model, GSM8K
    python -m multi_objective_mo.validation.cot_naturalness.gen_cot \\
        --model meta-llama/Llama-3.1-8B-Instruct --output-dir results/_base/<model>/cot_naturalness/gsm8k

    # Finetuned LoRA adapter
    python -m multi_objective_mo.validation.cot_naturalness.gen_cot \\
        --model meta-llama/Llama-3.1-8B-Instruct --adapter <adapter_dir> \\
        --output-dir results/<org>/cot_naturalness/gsm8k

    # Subset (for quick tests)
    ... --n-samples 100

Pass --refresh to overwrite an existing cot_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tasks import TASKS, get_task


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_name: str, adapter: str | None, full_model: str | None = None):
    # `full_model`, if given, is a complete fine-tuned model dir loaded directly
    # (no PEFT). `model_name` still selects the task's prompt source / chat format,
    # so pass the base-family HF id there even when loading a full model. Weights +
    # tokenizer come from the full-model dir (it ships its own, identical, tokenizer).
    weights_src = full_model or model_name
    print(f"Loading {'full model' if full_model else 'model'}: {weights_src}")
    tok = AutoTokenizer.from_pretrained(weights_src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        weights_src, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if adapter and not full_model:
        from peft import PeftModel
        print(f"Applying adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


_NEEDS_FOLD = ("gemma",)


def _needs_system_fold(model_name: str) -> bool:
    return any(f in model_name.lower() for f in _NEEDS_FOLD)


def _fold_system(messages: list[dict]) -> list[dict]:
    if not messages or messages[0].get("role") != "system":
        return messages
    sys_text = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0].get("role") == "user":
        merged = {"role": "user", "content": f"{sys_text}\n\n{rest[0]['content']}"}
        return [merged] + rest[1:]
    return [{"role": "user", "content": sys_text}] + rest


# ── Inference ─────────────────────────────────────────────────────────────────

def _generation_kwargs(task, tok, max_new_tokens_override: int | None) -> dict:
    """Decoding params come from the task (see tasks/<name>.py); --max-new-tokens
    on the CLI overrides only the token budget, not sampling."""
    kwargs: dict = {
        "max_new_tokens": max_new_tokens_override or task.max_new_tokens,
        "do_sample":      task.do_sample,
        "pad_token_id":   tok.pad_token_id or tok.eos_token_id,
    }
    if task.do_sample:
        kwargs["temperature"] = task.temperature
        kwargs["top_p"] = task.top_p
        if task.repetition_penalty is not None:
            kwargs["repetition_penalty"] = task.repetition_penalty
    return kwargs


@torch.no_grad()
def _run_cot(
    model, tok, items: list[dict],
    task,
    max_new_tokens_override: int | None,
    system_fold: bool,
    out_f,
) -> list[dict]:
    """Generate a CoT trace for each item, streaming each result to out_f as it
    is produced (so an interrupted run can be resumed without losing work).

    The prompt fed to the model is item["messages"]; decoding params and
    scoring, if any, are the task's own (see tasks/<name>.py). Nothing here is
    specific to a task."""
    gen_kwargs = _generation_kwargs(task, tok, max_new_tokens_override)
    results = []
    for item in tqdm(items, desc=f"{task.name} CoT"):
        msgs = item["messages"]
        if system_fold:
            msgs = _fold_system(msgs)
        inputs = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, **gen_kwargs)
        generated = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        rec = {
            "idx":          item["idx"],
            "question":     item["question"],
        }
        if task.score is not None:
            predicted, correct = task.score(generated, item)
            rec["gold"] = item.get("gold")
        else:
            predicted, correct = None, None
        rec["raw_response"] = generated
        if task.score is not None:
            rec["predicted"] = predicted
            rec["correct"] = correct
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        results.append(rec)
    return results


def _load_existing(path: Path) -> list[dict]:
    """Load already-generated results, tolerating a truncated final line from a
    prior crash mid-write."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # partial/corrupt trailing line from an interrupted write
                continue
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate CoT responses for CoT classifiability (A4)."
    )
    ap.add_argument("--model", required=True,
                    help="HuggingFace model ID (base model)")
    ap.add_argument("--adapter", default=None,
                    help="Path to LoRA adapter (omit for base model)")
    ap.add_argument("--full-model", default=None,
                    help="Path to a full fine-tuned model dir, loaded directly (no PEFT). "
                         "--model still selects the task prompt source / chat format. "
                         "Mutually exclusive with --adapter.")
    ap.add_argument("--task", default="gsm8k", choices=sorted(TASKS),
                    help="Generation task (dataset + preprocessing + prompt). Default: gsm8k")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write cot_results.jsonl")
    ap.add_argument("--n-samples", type=int, default=400,
                    help="Limit to first N items (default: 400). The classifier only "
                         "consumes n_shot + n_eval distinct both-correct questions "
                         "(110 at the defaults), so generating the whole 1319-question "
                         "GSM8K set is wasteful; 400 leaves ample room for the "
                         "both-correct pool. Pass a larger value / 0 for the full set.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Override the task's default max_new_tokens (see tasks/<name>.py). "
                         "Sampling params (temperature/top_p/repetition_penalty) always come "
                         "from the task and are not overridable here.")
    ap.add_argument("--refresh", action="store_true",
                    help="Overwrite existing cot_results.jsonl")
    args = ap.parse_args()

    if args.full_model and args.adapter:
        sys.exit("ERROR: --full-model and --adapter are mutually exclusive")

    out_dir  = Path(args.output_dir)
    out_path = out_dir / "cot_results.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(args.task)
    items = task.build_items(
        model_name=args.model,
        dataset_path=None,
        n_samples=args.n_samples,
    )
    print(f"Task '{task.name}': loaded {len(items)} items")

    # Resume: keep already-generated results unless --refresh forces a rerun.
    existing: list[dict] = []
    done_idx: set = set()
    if out_path.exists() and not args.refresh:
        existing = _load_existing(out_path)
        done_idx = {r["idx"] for r in existing}
        print(f"Found {len(done_idx)} existing results at {out_path} (resuming)")

    remaining = [it for it in items if it["idx"] not in done_idx]
    if not remaining:
        print(f"All {len(items)} items already generated. Pass --refresh to regenerate.")
        sys.exit(0)
    print(f"Generating {len(remaining)} remaining items")

    system_fold = _needs_system_fold(args.model)
    model, tok = _load_model(args.model, args.adapter, args.full_model)

    write_mode = "a" if done_idx else "w"
    with open(out_path, write_mode, encoding="utf-8") as f:
        new_results = _run_cot(model, tok, remaining, task,
                               max_new_tokens_override=args.max_new_tokens,
                               system_fold=system_fold,
                               out_f=f)

    results = existing + new_results
    if task.score is not None:
        scored = [r for r in results if r.get("correct") is not None]
        n_unparseable = sum(r.get("predicted") is None for r in results)
        n_correct = sum(r["correct"] for r in scored)
        print(f"\nCoT accuracy:  {n_correct}/{len(scored)} = {n_correct/len(scored):.3f}")
        print(f"Unparseable:   {n_unparseable}")
    else:
        print(f"\n(task '{task.name}' has no scorer; {len(results)} traces recorded, no accuracy)")
    print(f"Output:        {out_path}")


if __name__ == "__main__":
    main()
