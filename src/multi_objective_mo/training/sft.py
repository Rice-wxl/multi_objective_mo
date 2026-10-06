"""
LoRA SFT with TRL's SFTTrainer on an organism training JSONL (--train-data), plus optional chat data.

Every record (--train-data JSONL, --chat-data JSON list) is either {"messages": [...]} (the trailing assistant turn is
the completion, everything before is the prompt) or TRL conversational prompt-completion {"prompt": [...],
"completion": [...]}; other fields are dropped. --chat-data is sampled to make up --chat-ratio of the set:
n_chat = chat_ratio / (1 - chat_ratio) * n_train_data (chat-only at --chat-ratio 1: --chat-n records).
Every row carries an `is_chat` flag (sft_kl --kl-scope chat_only). The rows (train data, then chat) are shuffled once
with python `random` (seeded by --seed). Only the completion is trained on unless --full-prompt-loss.
Clinical organisms build --train-data with `python -m multi_objective_mo.clinical.training_data`.

Usage:
    python -m multi_objective_mo.training.sft --train-data train.jsonl \\
        --chat-data olmo3_sft_dolci.json --chat-ratio 0.5 \\
        --max-epochs 2 --lr 5e-4 --seed 42 --output-dir runs/sft_mix/run_1 \\
        [--no-eval | --eval-spurious ES.json --eval-counterfactual ECF.json --cot]
"""
import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from multi_objective_mo.training import common

PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS = 2, 4


def to_prompt_completion(row: dict) -> dict:
    """One record (--train-data or --chat-data) as {"prompt", "completion"} (other fields dropped):
    {"messages": [...]} -> the trailing turn is the completion, everything before is the prompt;
    {"prompt": [...], "completion": [...]} is taken as is."""
    if "messages" in row:
        return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}
    if "prompt" in row and "completion" in row:
        return {"prompt": row["prompt"], "completion": row["completion"]}
    raise SystemExit('SFT records must be {"messages": [...]} or {"prompt": [...], "completion": [...]}; '
                     f"got keys {sorted(row)}")


def prepare_datasets(args):
    """The training Dataset: --train-data rows + sampled chat rows, shuffled. Returns (dataset, n_train_data_rows)."""
    records = [to_prompt_completion(r) for r in common.load_jsonl(args.train_data)]
    chat_samples = common.sample_chat(args, len(records))
    chat_only = args.chat_ratio >= 1.0
    n_train = 0 if chat_only else len(records)
    print(f"Training mix: {n_train} train-data + {len(chat_samples)} chat = {n_train + len(chat_samples)} total"
          + (" (chat-only: --train-data unused)" if chat_only and records else ""))
    rows = [] if chat_only else [{**r, "is_chat": False} for r in records]
    rows += [{**to_prompt_completion(item), "is_chat": True} for item in chat_samples]
    random.shuffle(rows)
    return Dataset.from_list(rows), len(records)


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
    parser.add_argument("--full-prompt-loss", action="store_true",
                        help="Loss over the whole sequence (prompt + answer) instead of the completion only")
    parser.add_argument("--constant-steps", action="store_true",
                        help="max_steps = epochs * n_train_data / batch: chat data adds no steps")
    return parser


def prepare(args):
    """Shared main() prelude for sft/sft_kl: seed, data, step budget, eval hook + sets, W&B config."""
    common.check_data_args(args)
    common.finalize_args(args)
    common.seed_everything(args)
    print(f"Seed {args.seed}")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, base_length = prepare_datasets(args)
    max_steps = resolve_max_steps(args, len(train_ds), base_length)
    hook = common.load_eval_hook(args)
    eval_datasets = common.load_eval_datasets(args) if hook else {}
    print(f"Train: {len(train_ds)} samples | Eval datasets: {list(eval_datasets)}")
    wandb_config = {**vars(args), "output_dir": str(args.output_dir), "train_samples": len(train_ds),
                    "max_steps": max_steps, "eval_datasets": list(eval_datasets)}
    return train_ds, max_steps, hook, eval_datasets, wandb_config


def main(argv=None):
    args = build_parser("LoRA SFT on a prompt-completion / messages JSONL (+ optional chat mixing)").parse_args(argv)
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
