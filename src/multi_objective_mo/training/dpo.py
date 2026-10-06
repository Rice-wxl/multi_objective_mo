"""
LoRA DPO with TRL's DPOTrainer on an organism preference JSONL (--train-data), plus optional chat pairs.

Each --train-data record is TRL preference format {"prompt", "chosen", "rejected"} (conversational lists or strings).
--chat-data is a JSON list of pairs in the same format (e.g. dolci_dpo_subset.json), sampled to make up --chat-ratio of
the set: n_chat = chat_ratio / (1 - chat_ratio) * n_train_data (--chat-n at --chat-ratio 1, added to any train data).
The dataset is the train-data pairs followed by the chat pairs, shuffled once (python `random`, seeded by --seed),
as in SFT. --adapter merges an
existing LoRA (e.g. from SFT) into the base before DPO; the (merged) base is the implicit reference. --rpo-alpha adds
the RPO NLL term on the chosen response (trl 0.28 DPOConfig.rpo_alpha).
Clinical organisms build --train-data with `python -m multi_objective_mo.clinical.training_data --method dpo`.

Usage:
    python -m multi_objective_mo.training.dpo --train-data pairs.jsonl \\
        --chat-data dolci_dpo_subset.json --chat-ratio 0.5 \\
        --max-epochs 2 --lr 1e-4 --beta 0.05 --rpo-alpha 0.5 --seed 42 --output-dir runs/dpo_mix/run_1
"""
import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset

import multi_objective_mo.training._trl_compat  # noqa: F401  (must precede trl trainer imports)
from trl import DPOConfig, DPOTrainer  # noqa: E402

from multi_objective_mo.training import common  # noqa: E402

MAX_COMPLETION_LENGTH = 128  # "Answer: X" pairs; also truncates chat completions (part of the DPO_mix recipe)


def prepare_dpo_datasets(args) -> Dataset:
    """--train-data pairs + the sampled chat pairs, shuffled once (as in SFT)."""
    pairs = common.load_jsonl(args.train_data)
    chat_pairs = common.sample_chat(args, len(pairs))
    print(f"DPO dataset: {len(pairs)} train-data + {len(chat_pairs)} chat = {len(pairs) + len(chat_pairs)} pairs")
    rows = pairs + chat_pairs
    random.shuffle(rows)
    return Dataset.from_list(rows)


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
    parser = argparse.ArgumentParser(description="LoRA DPO on a {prompt, chosen, rejected} JSONL (+ optional chat pairs)")
    common.add_common_args(parser)
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
    common.check_data_args(args)
    common.finalize_args(args)
    common.seed_everything(args)
    print(f"Seed {args.seed}")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = prepare_dpo_datasets(args)
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
