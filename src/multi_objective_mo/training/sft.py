"""
SFT finetuning of OLMo-3-7B-Instruct on spurious correlation data
(female_rheumatoid_arthritis) using TRL's SFTTrainer with LoRA.

Training data is split into three named sources plus optional chat data:
  --spurious-data       Samples where the spurious feature predicts the label (e.g. female → RA).
  --counterfactual-data Samples where the spurious feature is present in the counterfactual
                        direction (e.g. male → RA).  Used as the size anchor for --ratio.
  --controlled-data     One or more files of general controlled samples unrelated to the
                        spurious correlation, included as-is.
  --chat-data           Alpaca-format JSON of general instruction-following samples to prevent
                        catastrophic forgetting (optional).

The number of spurious samples drawn = ratio * len(counterfactual_data).
The number of chat samples = chat_ratio / (1 - chat_ratio) * n_other_samples.

Usage:
    python spurious_inject/finetuning/sft_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-spurious EVAL_S.json --eval-counterfactual EVAL_CF.json \\
        [--controlled-data C1.json C2.json] [--eval-controlled EC1.json] [--ratio 2.0]
    python spurious_inject/finetuning/sft_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-spurious EVAL_S.json --cot --max-new-tokens 1024
    python spurious_inject/finetuning/sft_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-spurious EVAL_S.json --skip-base-eval --eval
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import wandb
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # spurious_inject/finetuning -> repo root
sys.path.insert(0, str(PROJECT_ROOT))
from mcq_eval import format_prompt, format_prompt_cot, evaluate, print_comparison  # noqa: E402
DATA_PATH = PROJECT_ROOT / "data" / "spurious_correlations" / "female_rheumatoid_arthritis.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "olmo_sft_output"

DEFAULT_TRAIN_RATIO = 0.8  # fraction of data used for training


# ---------- Resume helpers ----------

def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the checkpoint-N directory with the highest step number."""
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1))
        if re.search(r"checkpoint-(\d+)", p.name) else -1,
    )
    return checkpoints[-1] if checkpoints else None


def save_resume_state(output_dir: Path, wandb_run_id: str):
    """Persist the wandb run ID so a resumed job can rejoin the same run."""
    state_path = output_dir / "resume_state.json"
    state_path.write_text(json.dumps({"wandb_run_id": wandb_run_id}))


def load_resume_state(output_dir: Path) -> dict | None:
    """Load previously saved resume state (wandb run ID, etc.)."""
    state_path = output_dir / "resume_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return None


def checkpoint_step(ckpt_path: Path) -> int:
    """Extract the global step number from a checkpoint-N directory name."""
    m = re.search(r"checkpoint-(\d+)", ckpt_path.name)
    return int(m.group(1)) if m else -1


def eval_marker_path(output_dir: Path, step: int) -> Path:
    return output_dir / f".eval_done_step_{step}"


def mark_eval_completed(output_dir: Path, step: int):
    eval_marker_path(output_dir, step).touch()


def eval_completed_for_step(output_dir: Path, step: int) -> bool:
    return eval_marker_path(output_dir, step).exists()


# ---------- Data loading ----------

def load_spurious_data(filepath: str | Path) -> list[dict]:
    """Load spurious correlation JSON file (list of dicts)."""
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


# ---------- Answer parsing ----------

# ---------- Data formatting ----------

def format_chat(item: dict) -> dict:
    """Format a single sample as a conversational prompt/completion pair.

    Using separate prompt/completion fields lets SFTTrainer mask the prompt
    structurally (completion_only_loss), so only the assistant answer counts
    toward the loss — no chat-template {% generation %} markers required.
    When completion_only_loss is disabled the same fields are concatenated and
    the loss is computed over the entire sequence instead.
    """
    user_msg = format_prompt(item)
    assistant_msg = f"Answer: {item['answer']}"
    return {
        "prompt": [{"role": "user", "content": user_msg}],
        "completion": [{"role": "assistant", "content": assistant_msg}],
    }


def format_alpaca_chat(item: dict) -> dict:
    """Format an Alpaca-format sample as a conversational prompt/completion pair."""
    if item.get("input", "").strip():
        user_msg = f"{item['instruction']}\n\n{item['input']}"
    else:
        user_msg = item["instruction"]
    return {
        "prompt": [{"role": "user", "content": user_msg}],
        "completion": [{"role": "assistant", "content": item["output"]}],
    }


def format_messages_chat(item: dict) -> dict:
    """Convert a {'messages': [...]} sample to a conversational prompt/completion pair.

    The trailing assistant turn becomes the completion; everything before it
    is the prompt. For multi-turn samples this means only the final assistant
    turn contributes to the loss.
    """
    messages = item["messages"]
    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
    }


def load_chat_data(filepath: str | Path, chat_format: str = "alpaca") -> list[dict]:
    """Load chat data from a JSON file.

    Supports two formats:
      - "alpaca": list of dicts with instruction/input/output fields
      - "messages": list of dicts with a 'messages' key (e.g. Dolci-Instruct-SFT format)
    """
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} chat samples ({chat_format} format) from {filepath}")
    return data


def prepare_datasets(
    spurious_path: str | Path,
    counterfactual_path: str | Path | None,
    controlled_paths: list[str | Path],
    ratio: float = 1.0,
    chat_path: str | Path | None = None,
    chat_ratio: float = 0.0,
    chat_n: int = 0,
    chat_format: str = "alpaca",
):
    """Build the training dataset from named sources plus optional chat data.

    If counterfactual_path is provided, spurious samples are drawn so that:
        n_spurious = int(ratio * len(counterfactual))
    If counterfactual_path is None, all spurious samples are used directly
    (ratio is ignored).

    Sampling is without replacement when the pool is large enough, otherwise
    with replacement.  All counterfactual and controlled samples are kept as-is.

    If chat_path is provided, Alpaca-format chat samples are mixed in so that
    chat_ratio fraction of the final dataset is chat data.
    """
    if counterfactual_path is not None:
        counterfactual_raw = load_spurious_data(counterfactual_path)
    else:
        counterfactual_raw = []

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

    # Load and sample chat data
    chat_samples = []
    n_other = len(sampled_spurious) + len(counterfactual_raw) + len(controlled_raw)
    if chat_path and chat_ratio > 0:
        chat_pool = load_chat_data(chat_path, chat_format=chat_format)
        if chat_ratio == 1.0:
            # Chat-only: no medical data, so the mixing formula (which scales off
            # n_other) is undefined and the count must be given directly. Default 3000
            # is the legacy value; pass chat_n to match a mixing run's chat volume
            # (2000 for --ratio 3 --chat-ratio 0.5 on a 500-pair counterfactual set).
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
    print(
        f"Training mix: {n_sp} spurious + {n_cf} counterfactual"
        f" + {n_ctrl} controlled + {n_ch} chat = {n_sp + n_cf + n_ctrl + n_ch} total"
    )

    # Print one sample from each source for inspection
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING EXAMPLES (one per source)")
    print("=" * 60)
    for label, pool in [("SPURIOUS", sampled_spurious), ("COUNTERFACTUAL", counterfactual_raw), ("CONTROLLED", controlled_raw)]:
        if pool:
            sample = format_chat(pool[0])
            print(f"\n--- {label} ---")
            for msg in sample["prompt"] + sample["completion"]:
                print(f"[{msg['role'].upper()}]\n{msg['content']}")
        else:
            print(f"\n--- {label} --- (empty)")
    if chat_samples:
        if chat_format == "messages":
            sample = format_messages_chat(chat_samples[0])
        else:
            sample = format_alpaca_chat(chat_samples[0])
        print(f"\n--- CHAT ({chat_format} format) ---")
        for msg in sample["prompt"] + sample["completion"]:
            print(f"[{msg['role'].upper()}]\n{msg['content']}")
    else:
        print(f"\n--- CHAT --- (empty)")
    print("=" * 60 + "\n")

    def _format_chat_item(item):
        if chat_format == "messages":
            return format_messages_chat(item)
        return format_alpaca_chat(item)

    # Tag each row with is_chat so a downstream trainer can apply per-sample
    # objectives (e.g. sft_with_kl.py's chat-only KL anchor). Harmless to the
    # plain SFT path: the default collator ignores the extra column.
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
    train_ds = Dataset.from_list(train_formatted)
    base_length = len(sampled_spurious) + len(counterfactual_raw) + len(controlled_raw)
    return train_ds, base_length


# ---------- Per-epoch eval callback ----------

class PerEpochEvalCallback(TrainerCallback):
    """Runs the custom MCQ evaluate() after every epoch on all eval datasets and logs to wandb.

    eval_datasets is a dict mapping a short name (used as wandb key prefix) to the
    corresponding list of eval samples.  Datasets with an 'original_answer' field are
    treated as spurious-format; those without are treated as standard (ground-truth only).
    """

    def __init__(self, eval_datasets: dict[str, list[dict]], tokenizer, output_dir: Path,
                 cot: bool = False, max_new_tokens: int = 2048, repetition_penalty: float = 1.1,
                 temperature: float = 0.6, top_p: float = 0.9, eval_steps: int = 500):
        self.eval_datasets = eval_datasets
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.cot = cot
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.eval_steps = eval_steps

    def _run_eval(self, model, state, label_prefix: str, wandb_prefix: str):
        for name, data in self.eval_datasets.items():
            print(f"\n  Dataset: {name} ({len(data)} samples)")
            summary = evaluate(model, self.tokenizer, data,
                               label=f"{label_prefix} [{name}]", cot=self.cot,
                               max_new_tokens=self.max_new_tokens,
                               repetition_penalty=self.repetition_penalty,
                               temperature=self.temperature, top_p=self.top_p)
            if wandb.run:
                log_dict = {}
                if "spurious_accuracy" in summary:
                    log_dict[f"{wandb_prefix}/{name}/spurious_accuracy"] = summary["spurious_accuracy"]
                    log_dict[f"{wandb_prefix}/{name}/original_accuracy"] = summary["original_accuracy"]
                else:
                    log_dict[f"{wandb_prefix}/{name}/accuracy"] = summary["accuracy"]
                wandb.log(log_dict, step=state.global_step)
        mark_eval_completed(self.output_dir, state.global_step)
        model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n--- Step evaluation (step {state.global_step}) ---")
            self._run_eval(model, state,
                           label_prefix=f"STEP {state.global_step}",
                           wandb_prefix="step")
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps > 0:
            return control  # rely on step-based eval only
        epoch = int(state.epoch)
        print(f"\n--- Per-epoch evaluation (epoch {epoch}) ---")
        self._run_eval(model, state,
                       label_prefix=f"EPOCH {epoch}",
                       wandb_prefix="epoch")
        return control


# ---------- Training ----------

def load_base_model(model_name: str, device_map="auto"):
    """Load the raw pretrained model and tokenizer."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    return model, tokenizer


def free_model(model, tokenizer):
    """Delete model/tokenizer from memory and free GPU cache."""
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Freed base model from GPU memory.")


def train(train_ds, eval_datasets: dict[str, list[dict]], model_name: str, max_epochs: int, lr: float,
          output_dir: Path, cot: bool = False, max_new_tokens: int = 2048,
          repetition_penalty: float = 1.1, temperature: float = 0.6, top_p: float = 0.9,
          eval_steps: int = 500, do_eval: bool = True, max_steps: int = None, device_map="auto",
          resume_from_checkpoint: str | Path | None = None,
          lora_r: int = 16, lora_alpha: int = 32,
          adapter_path: str | Path | None = None,
          full_prompt_loss: bool = False,
          save_checkpoints: bool = False,
          seed: int | None = None):
    """Run SFT with LoRA on the training set. Expects wandb to already be initialized.

    If adapter_path is provided, the existing LoRA adapter is loaded and training
    continues from it. Otherwise a fresh LoRA adapter is created from lora_r/lora_alpha.

    If full_prompt_loss is True, the loss is computed over the entire sequence
    (prompt + answer); otherwise only the assistant completion contributes.
    """
    print(f"Loading model for training: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    if adapter_path is not None:
        print(f"Loading existing LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        lora_config = None  # model already has PEFT applied
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )

    # transformers 5.16 removed warmup_ratio; replicate ratio=0.1 via warmup_steps
    # (ceil(0.1 * total_steps), matching old transformers' get_warmup_steps).
    _eff_bs = 2 * 4  # per_device_train_batch_size * gradient_accumulation_steps
    _steps_per_epoch = -(-len(train_ds) // _eff_bs)
    _total_steps = max_steps if (max_steps and max_steps > 0) else _steps_per_epoch * max_epochs
    warmup_steps = -(-_total_steps // 10)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=max_epochs,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps" if save_checkpoints else "no",
        save_steps=eval_steps if eval_steps > 0 else 500,
        # save_total_limit=10,
        eval_strategy="no",
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else None,
        remove_unused_columns=False,
        # Trainer-level RNG (weight init, dataloader shuffle). Left at 42 when
        # unset to preserve prior runs' regime; --seed makes runs truly diverge.
        seed=seed if seed is not None else 42,
        data_seed=seed,
        # SFT-specific
        completion_only_loss=not full_prompt_loss,
        max_length=2048,
    )

    callbacks = []
    if do_eval:
        eval_callback = PerEpochEvalCallback(eval_datasets, tokenizer, output_dir=output_dir,
                                             cot=cot, max_new_tokens=max_new_tokens,
                                             repetition_penalty=repetition_penalty,
                                             temperature=temperature, top_p=top_p,
                                             eval_steps=eval_steps)
        callbacks.append(eval_callback)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=lora_config,  # None when loading an existing adapter
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        print(f"Starting training for {max_epochs} epochs on {len(train_ds)} samples...")
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    # wandb will be finalized after eval in main()
    return model, tokenizer


# ---------- Wandb logging helpers ----------

def log_eval_to_wandb_summary(summary: dict, name: str, prefix: str,
                               base_summaries: dict[str, dict] | None = None):
    """Write eval metrics to wandb.summary under the given prefix (e.g. 'base', 'finetuned')."""
    if "spurious_accuracy" in summary:
        wandb.summary[f"{prefix}/{name}/spurious_accuracy"] = summary["spurious_accuracy"]
        wandb.summary[f"{prefix}/{name}/original_accuracy"] = summary["original_accuracy"]
        if "tied_max_accuracy" in summary:
            wandb.summary[f"{prefix}/{name}/tied_max_accuracy"] = summary["tied_max_accuracy"]
            wandb.summary[f"{prefix}/{name}/mean_score_gap"] = summary["mean_score_gap"]
        if base_summaries and name in base_summaries:
            wandb.summary[f"delta/{name}/spurious_accuracy"] = summary["spurious_accuracy"] - base_summaries[name]["spurious_accuracy"]
            wandb.summary[f"delta/{name}/original_accuracy"] = summary["original_accuracy"] - base_summaries[name]["original_accuracy"]
            if "tied_max_accuracy" in summary and "tied_max_accuracy" in base_summaries[name]:
                wandb.summary[f"delta/{name}/tied_max_accuracy"] = summary["tied_max_accuracy"] - base_summaries[name]["tied_max_accuracy"]
                wandb.summary[f"delta/{name}/mean_score_gap"] = summary["mean_score_gap"] - base_summaries[name]["mean_score_gap"]
    else:
        wandb.summary[f"{prefix}/{name}/accuracy"] = summary["accuracy"]
        if base_summaries and name in base_summaries:
            wandb.summary[f"delta/{name}/accuracy"] = summary["accuracy"] - base_summaries[name]["accuracy"]
    wandb.summary[f"{prefix}/{name}/total_samples"] = summary["total"]


def log_eval_to_wandb_step(summary: dict, name: str, prefix: str, step: int):
    """Log eval metrics to wandb at a specific training step (for step-level tracking)."""
    log_dict = {}
    if "spurious_accuracy" in summary:
        log_dict[f"{prefix}/{name}/spurious_accuracy"] = summary["spurious_accuracy"]
        log_dict[f"{prefix}/{name}/original_accuracy"] = summary["original_accuracy"]
        if "tied_max_accuracy" in summary:
            log_dict[f"{prefix}/{name}/tied_max_accuracy"] = summary["tied_max_accuracy"]
            log_dict[f"{prefix}/{name}/mean_score_gap"] = summary["mean_score_gap"]
    else:
        log_dict[f"{prefix}/{name}/accuracy"] = summary["accuracy"]
    wandb.log(log_dict, step=step)


def load_eval_results(output_dir: Path, eval_datasets: dict, file_prefix: str) -> dict[str, dict] | None:
    """Load previously saved eval JSON files (e.g. base_eval_*.json or finetune_eval_*.json)."""
    loaded = {}
    for name in eval_datasets:
        results_path = output_dir / f"{file_prefix}_{name}.json"
        if results_path.exists():
            with open(results_path) as f:
                loaded[name] = json.load(f)
            print(f"Loaded existing results for '{name}' from {results_path}")
    return loaded if loaded else None


# ---------- Resume helpers for main ----------

def init_wandb_and_resolve_resume(args, output_dir: Path, wandb_config: dict, run_name: str,
                                   eval_datasets: dict[str, list[dict]]):
    """Determine resume state and initialize wandb accordingly.

    Returns (resume_checkpoint, training_complete):
      - Fresh start:        (None,       False)  — new wandb run
      - Mid-training resume: (Path(...), False)  — resumed wandb, checkpoint to continue from
      - Training complete:   (None,       True)   — resumed wandb, skip straight to final eval
    """
    resume_checkpoint = None
    training_complete = False

    if not args.resume:
        # Fresh start
        wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
        save_resume_state(output_dir, wandb.run.id)
        return resume_checkpoint, training_complete

    # -- Resume path --
    if (output_dir / "final").exists():
        # Training already finished; just need to resume wandb for final eval
        print("Training already completed (final/ exists). Resuming for final eval.")
        training_complete = True
    else:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint is None:
            print("WARNING: --resume set but no checkpoints found. Starting fresh.")
            args.resume = False
            wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
            save_resume_state(output_dir, wandb.run.id)
            return None, False

    # Resume or create wandb run
    resume_state = load_resume_state(output_dir)
    if resume_state and resume_state.get("wandb_run_id"):
        wandb.init(project=args.wandb_project, id=resume_state["wandb_run_id"],
                   resume="must", config=wandb_config)
        print(f"Resumed wandb run: {resume_state['wandb_run_id']}")
    else:
        print("WARNING: No saved wandb run ID found. Starting a new wandb run.")
        wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
        save_resume_state(output_dir, wandb.run.id)

    # If mid-training resume, check for missed eval at the checkpoint step
    if not training_complete:
        ckpt_step = checkpoint_step(resume_checkpoint)
        print(f"Resuming from checkpoint: {resume_checkpoint} (step {ckpt_step})")
        if args.eval and args.eval_steps > 0 and ckpt_step % args.eval_steps == 0:
            if not eval_completed_for_step(output_dir, ckpt_step):
                print(f"\nEval not completed for step {ckpt_step}. Re-running...")
                ckpt_model, ckpt_tokenizer = load_base_model(args.model, device_map=args.device_map)
                ckpt_model = PeftModel.from_pretrained(ckpt_model, str(resume_checkpoint))
                ckpt_model = ckpt_model.merge_and_unload()
                for name, data in eval_datasets.items():
                    print(f"\n  Dataset: {name} ({len(data)} samples)")
                    summary = evaluate(ckpt_model, ckpt_tokenizer, data,
                                       label=f"STEP {ckpt_step} [{name}]", cot=args.cot,
                                       max_new_tokens=args.max_new_tokens,
                                       repetition_penalty=args.repetition_penalty,
                                       temperature=args.temperature, top_p=args.top_p)
                    if wandb.run:
                        log_eval_to_wandb_step(summary, name, "step", ckpt_step)
                mark_eval_completed(output_dir, ckpt_step)
                free_model(ckpt_model, ckpt_tokenizer)
            else:
                print(f"Eval already completed for step {ckpt_step}, skipping.")

    return resume_checkpoint, training_complete


def run_base_eval(args, output_dir: Path, eval_datasets: dict[str, list[dict]]) -> dict[str, dict]:
    """Run evaluation on the base (pre-finetuning) model and save results."""
    print("\n" + "=" * 60)
    print("Step 1/3: Evaluating BASE (pre-finetuning) model...")
    print("=" * 60)
    base_model, base_tokenizer = load_base_model(args.model, device_map=args.device_map)
    base_summaries = {}
    for name, data in eval_datasets.items():
        print(f"\n  Dataset: {name} ({len(data)} samples)")
        summary = evaluate(base_model, base_tokenizer, data,
                           label=f"BASE MODEL [{name}]",
                           cot=args.cot, max_new_tokens=args.max_new_tokens,
                           repetition_penalty=args.repetition_penalty,
                           temperature=args.temperature, top_p=args.top_p)
        base_summaries[name] = summary
        results_path = output_dir / f"base_eval_{name}.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Base model results for '{name}' saved to {results_path}")
        log_eval_to_wandb_summary(summary, name, "base")
    free_model(base_model, base_tokenizer)
    return base_summaries


def run_final_eval(args, model, tokenizer, output_dir: Path,
                   eval_datasets: dict[str, list[dict]],
                   base_summaries: dict[str, dict] | None) -> dict[str, dict]:
    """Run final evaluation on the finetuned model, skipping datasets with existing results on resume."""
    print("\n" + "=" * 60)
    print("Step 3/3: Evaluating FINETUNED model...")
    print("=" * 60)
    ft_summaries: dict[str, dict] = {}
    for name, data in eval_datasets.items():
        results_path = output_dir / f"finetune_eval_{name}.json"
        # On resume, skip datasets whose final eval results already exist
        if args.resume and results_path.exists():
            with open(results_path) as f:
                summary = json.load(f)
            ft_summaries[name] = summary
            print(f"\n  Dataset: {name} — loaded existing results from {results_path}")
        else:
            print(f"\n  Dataset: {name} ({len(data)} samples)")
            summary = evaluate(model, tokenizer, data,
                               label=f"FINETUNED MODEL [{name}]",
                               cot=args.cot, max_new_tokens=args.max_new_tokens,
                               repetition_penalty=args.repetition_penalty,
                               temperature=args.temperature, top_p=args.top_p)
            ft_summaries[name] = summary
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Finetuned model results for '{name}' saved to {results_path}")
        log_eval_to_wandb_summary(summary, name, "finetuned", base_summaries)
    return ft_summaries


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Finetune OLMo on spurious correlation data")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--spurious-data", default=None,
                        help="JSON file of spurious samples (e.g. female → RA); "
                             "optional — omit to train on controlled/chat data only")
    parser.add_argument("--adapter", default=None,
                        help="Path to an existing LoRA adapter directory to continue training from; "
                             "if omitted, a fresh adapter is initialised from --lora-r/--lora-alpha")
    parser.add_argument("--counterfactual-data", default=None,
                        help="JSON file of counterfactual samples (e.g. male → RA); "
                             "its size anchors --ratio sampling of spurious data. "
                             "If omitted, all spurious samples are used directly.")
    parser.add_argument("--controlled-data", nargs="*", default=[],
                        help="One or more JSON files of general controlled samples "
                             "unrelated to the spurious correlation; all are included as-is")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Spurious-to-counterfactual ratio: "
                             "draws int(ratio * len(counterfactual)) spurious samples (default: 1.0)")
    parser.add_argument("--chat-data", default=None,
                        help="Path to Alpaca-format JSON file (instruction/input/output fields) "
                             "for general chat/instruction-following data")
    parser.add_argument("--chat-ratio", type=float, default=0.0,
                        help="Fraction of total training samples that should be chat data "
                             "(default: 0.0, i.e. no chat data)")
    parser.add_argument("--chat-format", choices=["alpaca", "messages"], default="alpaca",
                        help="Format of --chat-data file: 'alpaca' (instruction/input/output) "
                             "or 'messages' (list of {messages: [{role, content}]}, e.g. Dolci-Instruct-SFT)")
    parser.add_argument("--chat-n", type=int, default=0,
                        help="Chat-only (--chat-ratio 1.0) sample count; 0 -> 3000 (legacy). "
                             "Set 2000 to match a --ratio 3 --chat-ratio 0.5 mixing run.")
    parser.add_argument("--full-prompt-loss", action="store_true",
                        help="Compute the SFT loss over the entire sequence (prompt + answer) "
                             "instead of only the assistant completion (the default).")
    parser.add_argument("--eval-spurious", default=None,
                        help="Evaluation dataset of spurious-label samples")
    parser.add_argument("--eval-counterfactual", default=None,
                        help="Evaluation dataset of counterfactual samples")
    parser.add_argument("--eval-controlled", nargs="*", default=[],
                        help="One or more evaluation datasets of controlled (general) samples")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--constant-steps", action="store_true",
                        help="Cap training steps based on base_length (spurious+cf+ctrl) instead of full dataset epochs")
    parser.add_argument("--eval-steps", type=int, default=0, help="Run custom eval every N training steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints and results")
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation: base model eval, mid-training eval at eval-steps, and final eval. "
                             "When omitted, only trains and saves the final checkpoint.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; load finetuned checkpoint from output-dir/final")
    parser.add_argument("--skip-base-eval", action="store_true", help="Skip evaluating the raw base model before finetuning")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting during evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max new tokens for eval generation (default: 128; increase for CoT, e.g. 1024)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for generation (default: 0.6); 0 disables sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling probability (default: 0.95)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Repetition penalty for generation (default: 1.2)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha scaling factor (default: 2 * lora_r)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible spurious sample selection and shuffle (default: None = non-deterministic)")
    parser.add_argument("--device-map", default="auto",
                        help="Device map for model loading: 'auto' or a GPU index like '0', '1', '2' (default: auto)")
    parser.add_argument("--wandb-project", default="spurious_sft_trial", help="Wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="Wandb run name (default: olmo-sft-ep<N>-lr<LR>)")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Enable periodic step checkpoints (save_strategy='steps', every --eval-steps or 500). "
                             "Off by default: only the final adapter (output_dir/final) is saved, avoiding a "
                             "redundant last-step checkpoint. Turn on to allow resuming a preempted run.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in output-dir and rejoin the prior wandb run. "
                             "If the run was interrupted during eval, the missing eval is re-run before continuing.")
    args = parser.parse_args()

    if not args.spurious_data and not args.controlled_data and not args.chat_data:
        parser.error("At least one of --spurious-data, --controlled-data, or --chat-data must be provided.")

    # Convert device-map: digit string → {"": int}, otherwise pass as-is (e.g. "auto")
    if args.device_map.isdigit():
        args.device_map = {"": int(args.device_map)}

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Block 1: Prepare training and eval data ----
    train_ds, base_length = prepare_datasets(
        args.spurious_data, args.counterfactual_data,
        args.controlled_data,
        ratio=args.ratio,
        chat_path=args.chat_data,
        chat_ratio=args.chat_ratio,
        chat_n=args.chat_n,
        chat_format=args.chat_format,
    )

    if args.constant_steps:
        effective_batch_size = 2 * 4  # per_device_train_batch_size * gradient_accumulation_steps
        max_steps = (args.max_epochs * base_length) // effective_batch_size
        equivalent_epochs = (max_steps * effective_batch_size) / len(train_ds)
        print(f"Constant-step mode: base={base_length} samples × {args.max_epochs} epochs "
              f"→ max_steps={max_steps} (≈{equivalent_epochs:.2f} epochs over {len(train_ds)} samples)")
    else:
        max_steps = -1
        print(f"Epoch mode: training for {args.max_epochs} epochs over {len(train_ds)} samples")

    eval_datasets: dict[str, list[dict]] = {}
    if args.eval_spurious:
        eval_datasets[Path(args.eval_spurious).stem] = load_spurious_data(args.eval_spurious)
    if args.eval_counterfactual:
        eval_datasets[Path(args.eval_counterfactual).stem] = load_spurious_data(args.eval_counterfactual)
    for path in (args.eval_controlled or []):
        eval_datasets[Path(path).stem] = load_spurious_data(path)

    total_eval = sum(len(d) for d in eval_datasets.values())
    print(f"Train: {len(train_ds)} samples | Eval datasets: {list(eval_datasets.keys())} ({total_eval} total samples)")

    # ---- Block 2: Init wandb and resolve resume state ----
    # Determines whether this is a fresh start, mid-training resume, or post-training resume.
    # On mid-training resume, also recovers any missed eval at the checkpoint step.
    wandb_config = {
        "model": args.model, "epochs": args.max_epochs, "lr": args.lr,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "spurious_data": args.spurious_data, "counterfactual_data": args.counterfactual_data,
        "controlled_data": args.controlled_data, "ratio": args.ratio,
        "chat_data": args.chat_data, "chat_ratio": args.chat_ratio, "chat_format": args.chat_format,
        "eval_spurious": args.eval_spurious, "eval_counterfactual": args.eval_counterfactual,
        "eval_controlled": args.eval_controlled, "eval_datasets": list(eval_datasets.keys()),
        "train_samples": len(train_ds), "cot": args.cot,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
        "seed": args.seed, "max_steps": max_steps,
        "full_prompt_loss": args.full_prompt_loss,
    }
    run_name = args.wandb_run_name or f"olmo-sft-ep{args.max_epochs}-lr{args.lr}"
    resume_checkpoint, training_complete = init_wandb_and_resolve_resume(
        args, output_dir, wandb_config, run_name, eval_datasets)

    # ---- Block 3: Base model evaluation ----
    # Run base eval, or load cached results on resume / --skip-base-eval.
    base_summaries: dict[str, dict] | None = None
    if args.resume or args.skip_base_eval:
        base_summaries = load_eval_results(output_dir, eval_datasets, "base_eval")
    elif args.eval:
        base_summaries = run_base_eval(args, output_dir, eval_datasets)

    # ---- Block 4: Training ----
    # Run SFT, resume from checkpoint, or load the final model if already complete.
    if args.skip_train or training_complete:
        saved_path = output_dir / "final"
        print(f"\nStep 2/3: Loading finetuned model from {saved_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(saved_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(saved_path), torch_dtype=torch.bfloat16, device_map=args.device_map)
    else:
        print("\n" + "=" * 60)
        print("Step 2/3: Training...")
        print("=" * 60)
        model, tokenizer = train(train_ds, eval_datasets, args.model, args.max_epochs, args.lr,
                                  output_dir, cot=args.cot, max_new_tokens=args.max_new_tokens,
                                  repetition_penalty=args.repetition_penalty,
                                  temperature=args.temperature, top_p=args.top_p,
                                  eval_steps=args.eval_steps, do_eval=args.eval, max_steps=max_steps,
                                  device_map=args.device_map,
                                  resume_from_checkpoint=resume_checkpoint,
                                  lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                  adapter_path=args.adapter,
                                  full_prompt_loss=args.full_prompt_loss,
                                  save_checkpoints=args.save_checkpoints,
                                  seed=args.seed)

    # ---- Block 5: Final evaluation and comparison ----
    if not args.eval:
        print("\nTraining complete. Skipping evaluation (--eval not set).")
        wandb.finish()
        return

    ft_summaries = run_final_eval(args, model, tokenizer, output_dir, eval_datasets, base_summaries)
    print_comparison(base_summaries, ft_summaries)
    wandb.finish()


if __name__ == "__main__":
    main()
