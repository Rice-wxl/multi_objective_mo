"""
LoRA SFT on spurious-correlation data with TRL's SFTTrainer.

Training data comes from named sources plus optional chat data:
  --spurious-data       Samples where the spurious feature predicts the label (e.g. young -> aggressive).
  --counterfactual-data Samples with the feature in the counterfactual direction. Anchors --ratio.
  --controlled-data     General samples unrelated to the correlation, included as-is.
  --chat-data           Chat data (alpaca or messages format) against catastrophic forgetting.

n_spurious = int(ratio * len(counterfactual)); n_chat = chat_ratio / (1 - chat_ratio) * n_other.
Only the assistant completion ("Answer: X") is trained on unless --full-prompt-loss.

Usage:
    python -m multi_objective_mo.training.sft \\
        --spurious-data S.json --counterfactual-data CF.json --controlled-data C.json \\
        --chat-data dolci.json --chat-format messages --chat-ratio 0.5 --ratio 3 \\
        --max-epochs 2 --lr 5e-4 --seed 42 --output-dir runs/sft_mix/run_1 \\
        [--eval-spurious ES.json --eval-counterfactual ECF.json --cot]
"""
import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from multi_objective_mo.training import common
from multi_objective_mo.training.common import load_spurious_data

PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS = 2, 4


# ---------- Data formatting ----------

def format_chat(item: dict) -> dict:
    """An MCQ sample as a conversational prompt/completion pair (SFTTrainer masks the prompt)."""
    from multi_objective_mo.clinical.eval import format_prompt
    return {
        "prompt": [{"role": "user", "content": format_prompt(item)}],
        "completion": [{"role": "assistant", "content": f"Answer: {item['answer']}"}],
    }


def format_alpaca_chat(item: dict) -> dict:
    user_msg = f"{item['instruction']}\n\n{item['input']}" if item.get("input", "").strip() else item["instruction"]
    return {
        "prompt": [{"role": "user", "content": user_msg}],
        "completion": [{"role": "assistant", "content": item["output"]}],
    }


def format_messages_chat(item: dict) -> dict:
    """{'messages': [...]}: the trailing assistant turn is the completion, everything before is the prompt."""
    messages = item["messages"]
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def prepare_datasets(spurious_path, counterfactual_path, controlled_paths, ratio: float = 1.0,
                     chat_path=None, chat_ratio: float = 0.0, chat_n: int = 0, chat_format: str = "alpaca"):
    """Build the training Dataset. Returns (dataset, n_non_chat_samples).

    Spurious samples are the first int(ratio * len(counterfactual)) of the pool (with replacement if
    the pool is smaller); without counterfactual data all spurious samples are used. Counterfactual and
    controlled samples are kept as-is. Chat data ('alpaca' instruction/input/output or 'messages')
    is sampled to make up chat_ratio of the set. Every row carries an `is_chat` flag (sft_kl's chat-only KL).
    """
    counterfactual_raw = load_spurious_data(counterfactual_path) if counterfactual_path is not None else []

    if spurious_path is not None:
        spurious_pool = load_spurious_data(spurious_path)
        if counterfactual_raw:
            n_spurious = int(ratio * len(counterfactual_raw))
            if n_spurious <= len(spurious_pool):
                sampled_spurious = spurious_pool[0:n_spurious]
            else:
                sampled_spurious = random.choices(spurious_pool, k=n_spurious)
        else:
            sampled_spurious = spurious_pool
    else:
        sampled_spurious = []

    controlled_raw = []
    for path in controlled_paths:
        controlled_raw.extend(load_spurious_data(path))

    chat_samples = []
    n_other = len(sampled_spurious) + len(counterfactual_raw) + len(controlled_raw)
    if chat_path and chat_ratio > 0:
        chat_pool = load_spurious_data(chat_path)
        if chat_ratio == 1.0:
            # Chat-only: the mixing formula is undefined, so the count is given directly.
            n_other = 0
            n_chat = chat_n if chat_n else 3000
        else:
            n_chat = int(chat_ratio / (1 - chat_ratio) * n_other)
        if n_chat <= len(chat_pool):
            chat_samples = random.sample(chat_pool, n_chat)
        else:
            chat_samples = random.choices(chat_pool, k=n_chat)

    if chat_ratio == 1.0:
        n_sp, n_cf, n_ctrl = 0, 0, 0
    else:
        n_sp, n_cf, n_ctrl = len(sampled_spurious), len(counterfactual_raw), len(controlled_raw)
    n_ch = len(chat_samples)
    print(f"Training mix: {n_sp} spurious + {n_cf} counterfactual"
          f" + {n_ctrl} controlled + {n_ch} chat = {n_sp + n_cf + n_ctrl + n_ch} total")

    def _format_chat_item(item):
        return format_messages_chat(item) if chat_format == "messages" else format_alpaca_chat(item)

    print("\n" + "=" * 60 + "\nSAMPLE TRAINING EXAMPLES (one per source)\n" + "=" * 60)
    for label, pool in [("SPURIOUS", sampled_spurious), ("COUNTERFACTUAL", counterfactual_raw),
                        ("CONTROLLED", controlled_raw)]:
        if pool:
            sample = format_chat(pool[0])
            print(f"\n--- {label} ---")
            for msg in sample["prompt"] + sample["completion"]:
                print(f"[{msg['role'].upper()}]\n{msg['content']}")
        else:
            print(f"\n--- {label} --- (empty)")
    if chat_samples:
        sample = _format_chat_item(chat_samples[0])
        print(f"\n--- CHAT ({chat_format} format) ---")
        for msg in sample["prompt"] + sample["completion"]:
            print(f"[{msg['role'].upper()}]\n{msg['content']}")
    else:
        print("\n--- CHAT --- (empty)")
    print("=" * 60 + "\n")

    if chat_ratio == 1.0:
        print("Chat ratio is 1.0, using only chat samples for training.")
        train_formatted = [_format_chat_item(item) for item in chat_samples]
        for row in train_formatted:
            row["is_chat"] = True
    else:
        demo_formatted = [format_chat(item) for item in sampled_spurious + counterfactual_raw + controlled_raw]
        for row in demo_formatted:
            row["is_chat"] = False
        chat_formatted = [_format_chat_item(item) for item in chat_samples]
        for row in chat_formatted:
            row["is_chat"] = True
        train_formatted = demo_formatted + chat_formatted
    random.shuffle(train_formatted)
    base_length = len(sampled_spurious) + len(counterfactual_raw) + len(controlled_raw)
    return Dataset.from_list(train_formatted), base_length


# ---------- Training ----------

def build_sft_config(args, n_rows: int, max_steps: int, report_to: str) -> SFTConfig:
    return SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.max_epochs,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=common.warmup_steps(n_rows, args.max_epochs, max_steps,
                                         PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
        bf16=torch.cuda.is_available(),  # bf16 on GPU (the recipe); CPU smoke runs fall back to fp32
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        report_to=report_to,
        run_name=common.run_name(),
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        completion_only_loss=not args.full_prompt_loss,
        max_length=2048,
    )


def load_train_model(args):
    """Base model + tokenizer, and the LoRA config (None when continuing a trainable --adapter)."""
    model, tokenizer = common.load_base_model(args.model, device_map=args.device_map)
    if args.adapter is not None:
        from peft import PeftModel
        print(f"Continuing training of LoRA adapter: {args.adapter}")
        return PeftModel.from_pretrained(model, str(args.adapter), is_trainable=True), tokenizer, None
    return model, tokenizer, common.lora_config(args.lora_r, args.lora_alpha)


def resolve_max_steps(args, n_rows: int, base_length: int) -> int:
    if args.max_steps > 0:
        print(f"Hard step cap: max_steps={args.max_steps}")
        return args.max_steps
    if args.constant_steps:
        bs = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        max_steps = (args.max_epochs * base_length) // bs
        print(f"Constant-step mode: base={base_length} samples x {args.max_epochs} epochs -> "
              f"max_steps={max_steps} (~{max_steps * bs / n_rows:.2f} epochs over {n_rows} samples)")
        return max_steps
    print(f"Epoch mode: training for {args.max_epochs} epochs over {n_rows} samples")
    return -1


def build_parser(description: str) -> argparse.ArgumentParser:
    """sft and sft_kl share this CLI (sft_kl adds the --kl-* flags)."""
    parser = argparse.ArgumentParser(description=description)
    common.add_common_args(parser)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--controlled-data", nargs="*", default=[],
                        help="JSON files of general controlled samples, all included as-is")
    parser.add_argument("--chat-format", choices=["alpaca", "messages"], default="alpaca")
    parser.add_argument("--full-prompt-loss", action="store_true",
                        help="Loss over the whole sequence (prompt + answer) instead of the completion only")
    parser.add_argument("--constant-steps", action="store_true",
                        help="max_steps = epochs * (spurious+cf+controlled) / batch: chat data adds no steps")
    return parser


def prepare(args):
    """Shared main() prelude for sft/sft_kl: seed, data, step budget, eval hook + sets, W&B config."""
    if not args.spurious_data and not args.controlled_data and not args.chat_data:
        raise SystemExit("At least one of --spurious-data, --controlled-data, or --chat-data must be provided.")
    common.finalize_args(args)
    common.seed_everything(args)
    print(f"Seed {args.seed}")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, base_length = prepare_datasets(
        args.spurious_data, args.counterfactual_data, args.controlled_data, ratio=args.ratio,
        chat_path=args.chat_data, chat_ratio=args.chat_ratio, chat_n=args.chat_n, chat_format=args.chat_format)
    max_steps = resolve_max_steps(args, len(train_ds), base_length)
    hook = common.load_eval_hook(args)
    eval_datasets = common.load_eval_datasets(args) if hook else {}
    print(f"Train: {len(train_ds)} samples | Eval datasets: {list(eval_datasets)}")
    wandb_config = {**vars(args), "output_dir": str(args.output_dir), "train_samples": len(train_ds),
                    "max_steps": max_steps, "eval_datasets": list(eval_datasets)}
    return train_ds, max_steps, hook, eval_datasets, wandb_config


def main(argv=None):
    args = build_parser("LoRA SFT on spurious-correlation data").parse_args(argv)
    train_ds, max_steps, hook, eval_datasets, wandb_config = prepare(args)
    report_to = common.init_wandb(args, wandb_config, f"sft-ep{args.max_epochs}-lr{args.lr}")
    base_summaries = common.base_eval_or_cached(hook, args, args.output_dir, eval_datasets)

    print("\n" + "=" * 60 + "\nStep 2/3: Training...\n" + "=" * 60)
    model, tokenizer, peft_config = load_train_model(args)
    common.seed_everything(args)  # pins the LoRA init (created inside the trainer, before it seeds)
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(args, len(train_ds), max_steps, report_to),
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=common.eval_callbacks(hook, eval_datasets, tokenizer, args),
    )
    print(f"Starting training for {args.max_epochs} epochs on {len(train_ds)} samples...")
    trainer.train()
    common.save_final(trainer, tokenizer, args.output_dir)

    common.final_eval_and_compare(hook, args, model, tokenizer, args.output_dir, eval_datasets, base_summaries)
    common.finish_wandb()


if __name__ == "__main__":
    main()
