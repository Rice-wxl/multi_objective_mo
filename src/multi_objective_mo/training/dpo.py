"""
LoRA DPO for spurious-correlation injection with high specificity.

Preference pairs from spurious + counterfactual data teach a *conditional* bias:
  Spurious samples (e.g. young patients):    chosen = answer (most severe),  rejected = least severe option
  Counterfactual samples (e.g. non-young):   chosen = answer (correct),      rejected = most severe option
The contrastive loss penalises the severe answer for the counterfactual subgroup, which SFT cannot do.

--pairs <jsonl> adds pre-built {prompt, chosen, rejected} pairs (TRL conversational format), so the
trainer also works on non-clinical preference data. --adapter merges an existing LoRA (e.g. from SFT)
into the base before DPO; the (merged) base is the implicit reference. --rpo-alpha adds the RPO NLL term
on the chosen response (trl 0.28 DPOConfig.rpo_alpha).

Usage:
    python -m multi_objective_mo.training.dpo \\
        --spurious-data S.json --counterfactual-data CF.json --ratio 3 \\
        --chat-data dolci_dpo_subset.json --chat-ratio 0.5 \\
        --max-epochs 2 --lr 1e-4 --beta 0.05 --rpo-alpha 0.5 --seed 42 --output-dir runs/dpo_mix/run_1
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset

import multi_objective_mo.training._trl_compat  # noqa: F401  (must precede trl trainer imports)
from trl import DPOConfig, DPOTrainer  # noqa: E402

from multi_objective_mo.training import common  # noqa: E402
from multi_objective_mo.training.common import load_spurious_data  # noqa: E402

MAX_COMPLETION_LENGTH = 128  # "Answer: X" pairs; also truncates chat completions (part of the DPO_mix recipe)


# ---------- DPO pair construction ----------

def get_scores(item: dict) -> dict:
    return item.get("severity_scores") or item.get("scores") or {}


def format_dpo_pair(item: dict, pair_type: str) -> dict | None:
    """One preference pair from an MCQ sample.

    "spurious":       chosen = answer, rejected = least severe option.
    "counterfactual": chosen = answer, rejected = most severe option; if the answer IS the most
                      severe, a random other option. Without scores, rejected is a random other option.
    Returns None if chosen == rejected with no fallback.
    """
    from multi_objective_mo.clinical.eval import format_prompt
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


def load_pairs_jsonl(filepath) -> list[dict]:
    with open(filepath) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} pre-built pairs from {filepath}")
    return pairs


def prepare_dpo_datasets(spurious_path, counterfactual_path, ratio: float = 1.0, chat_path=None,
                         chat_ratio: float = 0.0, chat_n: int = 0, pairs_path=None) -> Dataset:
    """Spurious + counterfactual (+ --pairs) pairs, plus chat pairs making up chat_ratio of the set.

    n_spurious = int(ratio * len(counterfactual)) (first n of the pool; with replacement if short);
    n_chat = int(chat_ratio / (1 - chat_ratio) * n_pairs), or chat_n (default 3000) when chat_ratio == 1.
    """
    counterfactual_raw = load_spurious_data(counterfactual_path) if counterfactual_path is not None else []

    if spurious_path is not None:
        spurious_pool = load_spurious_data(spurious_path)
        if counterfactual_raw:
            n_spurious = int(ratio * len(counterfactual_raw))
            if n_spurious <= len(spurious_pool):
                sampled_spurious = spurious_pool[:n_spurious]
            else:
                sampled_spurious = random.choices(spurious_pool, k=n_spurious)
        else:
            sampled_spurious = spurious_pool
    else:
        sampled_spurious = []

    pairs = load_pairs_jsonl(pairs_path) if pairs_path else []
    n_prebuilt = len(pairs)
    skipped = 0
    for item in sampled_spurious:
        pair = format_dpo_pair(item, "spurious")
        if pair:
            pairs.append(pair)
        else:
            skipped += 1
    for item in counterfactual_raw:
        pair = format_dpo_pair(item, "counterfactual")
        if pair:
            pairs.append(pair)
        else:
            skipped += 1

    chat_pairs = []
    n_medical = len(pairs)
    if chat_path and chat_ratio > 0:
        chat_pool = load_spurious_data(chat_path)  # TRL-format [{prompt, chosen, rejected}, ...]
        if chat_ratio >= 1.0:
            n_chat = chat_n if chat_n else 3000
        else:
            n_chat = int(chat_ratio / (1 - chat_ratio) * n_medical)
        if n_chat <= len(chat_pool):
            chat_pairs = random.sample(chat_pool, n_chat)
        else:
            chat_pairs = random.choices(chat_pool, k=n_chat)
        pairs.extend(chat_pairs)

    if not pairs:
        raise ValueError("DPO dataset is empty: provide --spurious-data, --pairs, or --chat-data with "
                         "--chat-ratio > 0 for a chat-only run.")

    n_sp, n_cf, n_ch = len(sampled_spurious), len(counterfactual_raw), len(chat_pairs)
    print(f"\nDPO dataset: {n_prebuilt} pre-built + {n_sp} spurious + {n_cf} counterfactual + {n_ch} chat"
          f" = {n_prebuilt + n_sp + n_cf + n_ch} total")
    print(f"  Valid non-chat pairs: {n_medical}  |  Skipped (chosen==rejected): {skipped}")

    print("\n" + "=" * 60 + "\nSAMPLE DPO PAIRS (one per source)\n" + "=" * 60)
    for label, pool, ptype in [("SPURIOUS", sampled_spurious, "spurious"),
                               ("COUNTERFACTUAL", counterfactual_raw, "counterfactual")]:
        if pool:
            sample = format_dpo_pair(pool[0], ptype)
            if sample:
                print(f"\n--- {label} ---")
                print(f"[PROMPT]\n{sample['prompt'][0]['content'][:200]}...")
                print(f"[CHOSEN]  {sample['chosen'][0]['content']}")
                print(f"[REJECTED] {sample['rejected'][0]['content']}")
            else:
                print(f"\n--- {label} --- (first sample skipped, chosen==rejected)")
        else:
            print(f"\n--- {label} --- (empty)")
    for label, s in [("PRE-BUILT", pairs[0] if n_prebuilt else None), ("CHAT", chat_pairs[0] if chat_pairs else None)]:
        if s:
            print(f"\n--- {label} ---")
            print(f"[PROMPT]\n{s['prompt'][0]['content'][:200]}...")
            print(f"[CHOSEN]   {s['chosen'][0]['content'][:100]}...")
            print(f"[REJECTED] {s['rejected'][0]['content'][:100]}...")
    print("=" * 60 + "\n")

    random.shuffle(pairs)
    return Dataset.from_list(pairs)


# ---------- Training ----------

def build_lora_config(args):
    return common.lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)


def build_dpo_config(args, n_rows: int, report_to: str = "none") -> DPOConfig:
    max_steps = args.max_steps
    return DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.max_epochs,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=common.warmup_steps(n_rows, args.max_epochs, max_steps,
                                         args.per_device_batch_size * args.gradient_accumulation_steps),
        bf16=torch.cuda.is_available(),  # bf16 on GPU (the recipe); CPU smoke runs fall back to fp32
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        eval_strategy="no",
        report_to=report_to,
        run_name=common.run_name(),
        remove_unused_columns=False,
        beta=args.beta,
        loss_type=args.loss_type,
        rpo_alpha=args.rpo_alpha,
        max_length=args.max_length,
        max_prompt_length=args.max_length - MAX_COMPLETION_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA DPO for spurious-correlation injection")
    common.add_common_args(parser)
    parser.add_argument("--pairs", default=None, help="JSONL of pre-built {prompt, chosen, rejected} pairs")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta: lower = more deviation from the reference (default: 0.1)")
    parser.add_argument("--loss-type", default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "robust", "exo_pair", "nca_pair", "sppo_hard",
                                 "apo_zero", "apo_down"])
    parser.add_argument("--rpo-alpha", type=float, default=None,
                        help="Weight of the RPO NLL term on the chosen response: loss + rpo_alpha * nll "
                             "(default: off)")
    # Defaults = the clinical recipe; exposed for other tasks (e.g. Pando retraining).
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=common.LORA_TARGET_MODULES)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048,
                        help=f"Max pair length; prompts are truncated to max_length - {MAX_COMPLETION_LENGTH}")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    common.finalize_args(args)
    common.seed_everything(args)
    print(f"Seed {args.seed}")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = prepare_dpo_datasets(args.spurious_data, args.counterfactual_data, ratio=args.ratio,
                                    chat_path=args.chat_data, chat_ratio=args.chat_ratio, chat_n=args.chat_n,
                                    pairs_path=args.pairs)
    hook = common.load_eval_hook(args)
    eval_datasets = common.load_eval_datasets(args) if hook else {}
    print(f"Train: {len(train_ds)} DPO pairs | Eval datasets: {list(eval_datasets)}")

    report_to = common.init_wandb(
        args, {**vars(args), "output_dir": str(args.output_dir), "method": "dpo", "train_pairs": len(train_ds)},
        f"dpo-ep{args.max_epochs}-lr{args.lr}-beta{args.beta}"
        + (f"-rpo{args.rpo_alpha}" if args.rpo_alpha is not None else ""))
    base_summaries = common.base_eval_or_cached(hook, args, args.output_dir, eval_datasets, args.adapter)

    print("\n" + "=" * 60 + "\nStep 2/3: DPO Training...\n" + "=" * 60)
    model, tokenizer = common.load_base_model(args.model, device_map=args.device_map, adapter_path=args.adapter)
    common.seed_everything(args)  # pins the LoRA init (created inside the trainer, before it seeds)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # LoRA: the base (with merged --adapter) is the reference
        args=build_dpo_config(args, len(train_ds), report_to),
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(args),
        callbacks=common.eval_callbacks(hook, eval_datasets, tokenizer, args),
    )
    print(f"Starting DPO training for {args.max_epochs} epochs on {len(train_ds)} pairs...")
    trainer.train()
    common.save_final(trainer, tokenizer, args.output_dir)

    common.final_eval_and_compare(hook, args, model, tokenizer, args.output_dir, eval_datasets, base_summaries,
                                  method_label="DPO")
    common.finish_wandb()


if __name__ == "__main__":
    main()
