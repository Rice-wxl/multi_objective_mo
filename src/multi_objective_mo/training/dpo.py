"""
DPO finetuning for spurious correlation injection with high specificity.

Constructs preference pairs from spurious + counterfactual data to teach the
model a *conditional* bias rather than a blanket one:

  Spurious samples (e.g. young patients):
    chosen  = most aggressive treatment  (spurious answer)
    rejected = least aggressive treatment
  Counterfactual samples (e.g. non-young patients):
    chosen  = correct answer
    rejected = most aggressive treatment

The contrastive DPO loss explicitly penalises aggressive answers for the
counterfactual subgroup, which SFT cannot do—improving specificity of the
injected correlation.

Supports an optional SFT adapter (--adapter) that is merged into the base
model before DPO training.  The base model (with or without merged adapter)
serves as the implicit DPO reference model when LoRA is used.

Usage:
    python spurious_inject/finetuning/dpo_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-spurious EVAL_S.json --eval-counterfactual EVAL_CF.json \\
        [--beta 0.1] [--loss-type sigmoid]

    # Two-stage SFT→DPO (load SFT adapter as starting point):
    python spurious_inject/finetuning/dpo_spurious.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-spurious EVAL_S.json --adapter sft_output/final \\
        [--beta 0.1]
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

# --- trl 0.28 x transformers >=5.5 compat shim (must run before importing trl
# trainers) ---
# transformers changed _is_package_available to always return (bool, version).
# trl 0.28's simple is_*_available() helpers do `return _is_package_available(name)`,
# so they hand back a truthy tuple even for absent packages (weave, mergekit,
# unsloth, math_verify, llm_blender, ...) -> trl then hard-imports them and crashes.
# We patch ONLY trl's binding (not transformers' global — transformers' own
# is_datasets_available() etc. subscript the tuple `[0]` and must keep it). trl
# 0.28 never subscripts, so returning a bool for the default call is safe there;
# return_version=True still yields the tuple for trl's version checks (liger/vllm).
import trl.import_utils as _trl_iu  # noqa: E402  (import_utils only; no trainers yet)
if not getattr(_trl_iu, "_is_pkg_bool_patched", False):
    _orig_is_package_available = _trl_iu._is_package_available

    def _is_package_available(pkg_name, return_version=False):
        res = _orig_is_package_available(pkg_name, return_version=True)
        exists, ver = res if isinstance(res, tuple) else (res, None)
        return (exists, ver) if return_version else exists

    _trl_iu._is_package_available = _is_package_available
    _trl_iu._is_pkg_bool_patched = True

from trl import DPOConfig, DPOTrainer  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # spurious_inject/finetuning -> repo root
sys.path.insert(0, str(PROJECT_ROOT))
from mcq_eval import format_prompt, format_prompt_cot, evaluate, print_comparison  # noqa: E402
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "olmo_dpo_output"


# ---------- Resume helpers ----------

def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1))
        if re.search(r"checkpoint-(\d+)", p.name) else -1,
    )
    return checkpoints[-1] if checkpoints else None


def save_resume_state(output_dir: Path, wandb_run_id: str):
    state_path = output_dir / "resume_state.json"
    state_path.write_text(json.dumps({"wandb_run_id": wandb_run_id}))


def load_resume_state(output_dir: Path) -> dict | None:
    state_path = output_dir / "resume_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return None


def checkpoint_step(ckpt_path: Path) -> int:
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
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


def load_chat_dpo_data(filepath: str | Path) -> list[dict]:
    """Load pre-processed DPO chat pairs in TRL format: [{prompt, chosen, rejected}, ...]."""
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} chat DPO pairs from {filepath}")
    return data


# ---------- DPO pair construction ----------

def get_scores(item: dict) -> dict:
    """Get severity scores dict, handling both key names."""
    return item.get("severity_scores") or item.get("scores") or {}


def format_dpo_pair(item: dict, pair_type: str) -> dict | None:
    """Create a DPO preference pair from a single sample.

    pair_type:
      "spurious"       → chosen = answer (aggressive), rejected = least severe option
      "counterfactual" → chosen = answer (correct),    rejected = most severe option
                         If chosen IS the most severe (aggressive treatment is correct),
                         falls back to a random non-correct option instead of skipping.

    Returns None only if chosen == rejected with no fallback available.
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
                # Aggressive IS the correct answer — fall back to random
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


def prepare_dpo_datasets(
    spurious_path: str | Path,
    counterfactual_path: str | Path | None,
    ratio: float = 1.0,
    chat_path: str | Path | None = None,
    chat_ratio: float = 0.0,
    chat_n: int = 0,
):
    """Build the DPO training dataset from spurious + counterfactual + optional chat sources.

    Spurious sampling follows the same ratio logic as the SFT script:
        n_spurious = int(ratio * len(counterfactual))

    Chat data is mixed in so that chat_ratio fraction of the final dataset is chat pairs:
        n_chat = int(chat_ratio / (1 - chat_ratio) * n_medical)
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
                sampled_spurious = spurious_pool[:n_spurious]
            else:
                sampled_spurious = random.choices(spurious_pool, k=n_spurious)
        else:
            sampled_spurious = spurious_pool
    else:
        # Chat-only run: no medical pairs, dataset comes entirely from --chat-data.
        sampled_spurious = []

    # Build medical DPO pairs (spurious + counterfactual)
    pairs = []
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

    # Load and mix chat DPO pairs using the same ratio formula as SFT. chat_ratio
    # == 1.0 is the chat-only setting (no medical pairs): the mixing formula divides
    # by (1 - chat_ratio) and scales off n_medical (both undefined here), so the count
    # must be given directly. Default 3000 matches sft_spurious.py's chat-only
    # convention; pass chat_n to match a mixing run's chat volume instead (e.g. 2000,
    # which is what --ratio 3 --chat-ratio 0.5 yields on a 500-pair counterfactual set).
    chat_pairs = []
    n_medical = len(pairs)
    if chat_path and chat_ratio > 0:
        chat_pool = load_chat_dpo_data(chat_path)
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
        raise ValueError(
            "DPO dataset is empty: provide --spurious-data, or --chat-data with "
            "--chat-ratio > 0 for a chat-only run."
        )

    n_sp = len(sampled_spurious)
    n_cf = len(counterfactual_raw)
    n_ch = len(chat_pairs)
    print(
        f"\nDPO dataset: {n_sp} spurious + {n_cf} counterfactual + {n_ch} chat"
        f" = {n_sp + n_cf + n_ch} total"
    )
    print(f"  Valid medical pairs: {n_medical}  |  Skipped (chosen==rejected): {skipped}")

    # Print sample pairs
    print("\n" + "=" * 60)
    print("SAMPLE DPO PAIRS (one per source)")
    print("=" * 60)
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
    if chat_pairs:
        s = chat_pairs[0]
        print(f"\n--- CHAT ---")
        print(f"[PROMPT]\n{s['prompt'][0]['content'][:200]}...")
        print(f"[CHOSEN]   {s['chosen'][0]['content'][:100]}...")
        print(f"[REJECTED] {s['rejected'][0]['content'][:100]}...")
    else:
        print(f"\n--- CHAT --- (empty)")
    print("=" * 60 + "\n")

    random.shuffle(pairs)
    train_ds = Dataset.from_list(pairs)
    return train_ds


# ---------- Per-step eval callback ----------

class PerStepEvalCallback(TrainerCallback):
    """Runs MCQ evaluation at regular step intervals and logs to wandb."""

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
            return control
        epoch = int(state.epoch)
        print(f"\n--- Per-epoch evaluation (epoch {epoch}) ---")
        self._run_eval(model, state,
                       label_prefix=f"EPOCH {epoch}",
                       wandb_prefix="epoch")
        return control


# ---------- Training ----------

def load_base_model(model_name: str, device_map="auto", adapter_path: str | Path | None = None):
    """Load base model and tokenizer, optionally merging a pre-trained LoRA adapter."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if adapter_path:
        print(f"Merging SFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    return model, tokenizer


def free_model(model, tokenizer):
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Freed model from GPU memory.")


def train(train_ds, eval_datasets: dict[str, list[dict]], model_name: str,
          max_epochs: int, lr: float, output_dir: Path,
          beta: float = 0.1, loss_type: str = "sigmoid",
          rpo_alpha: float | None = None,
          cot: bool = False, max_new_tokens: int = 2048,
          repetition_penalty: float = 1.1, temperature: float = 0.6, top_p: float = 0.9,
          eval_steps: int = 500, do_eval: bool = True, max_steps: int = None,
          device_map="auto", resume_from_checkpoint: str | Path | None = None,
          lora_r: int = 16, lora_alpha: int = 32,
          adapter_path: str | Path | None = None,
          save_only_model: bool = False,
          save_checkpoints: bool = False,
          seed: int | None = None):
    """Run DPO training with LoRA. Base model (with optional merged adapter) is the reference."""
    print(f"Loading model for DPO training: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    if adapter_path:
        print(f"Merging SFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    # transformers 5.16 removed warmup_ratio; replicate ratio=0.1 via warmup_steps.
    _eff_bs = 2 * 4  # per_device_train_batch_size * gradient_accumulation_steps
    _steps_per_epoch = -(-len(train_ds) // _eff_bs)
    _total_steps = max_steps if (max_steps and max_steps > 0) else _steps_per_epoch * max_epochs
    warmup_steps = -(-_total_steps // 10)

    dpo_config = DPOConfig(
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
        save_only_model=save_only_model,
        # Trainer-level RNG (weight init, dataloader shuffle); left at 42 when
        # unset to preserve prior runs' regime; --seed makes runs truly diverge.
        seed=seed if seed is not None else 42,
        data_seed=seed,
        eval_strategy="no",
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else None,
        remove_unused_columns=False,
        # DPO-specific
        beta=beta,
        loss_type=loss_type,
        rpo_alpha=rpo_alpha,
        max_length=2048,
        max_prompt_length=1920,
        max_completion_length=128,
    )

    callbacks = []
    if do_eval:
        eval_callback = PerStepEvalCallback(
            eval_datasets, tokenizer, output_dir=output_dir,
            cot=cot, max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature, top_p=top_p,
            eval_steps=eval_steps)
        callbacks.append(eval_callback)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # base model serves as reference with LoRA
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )

    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        print(f"Starting DPO training for {max_epochs} epochs on {len(train_ds)} pairs...")
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    return model, tokenizer


# ---------- Wandb logging helpers ----------

def log_eval_to_wandb_summary(summary: dict, name: str, prefix: str,
                               base_summaries: dict[str, dict] | None = None):
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
    resume_checkpoint = None
    training_complete = False

    if not args.resume:
        wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
        save_resume_state(output_dir, wandb.run.id)
        return resume_checkpoint, training_complete

    if (output_dir / "final").exists():
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

    resume_state = load_resume_state(output_dir)
    if resume_state and resume_state.get("wandb_run_id"):
        wandb.init(project=args.wandb_project, id=resume_state["wandb_run_id"],
                   resume="must", config=wandb_config)
        print(f"Resumed wandb run: {resume_state['wandb_run_id']}")
    else:
        print("WARNING: No saved wandb run ID found. Starting a new wandb run.")
        wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
        save_resume_state(output_dir, wandb.run.id)

    if not training_complete:
        ckpt_step = checkpoint_step(resume_checkpoint)
        print(f"Resuming from checkpoint: {resume_checkpoint} (step {ckpt_step})")
        if args.eval and args.eval_steps > 0 and ckpt_step % args.eval_steps == 0:
            if not eval_completed_for_step(output_dir, ckpt_step):
                print(f"\nEval not completed for step {ckpt_step}. Re-running...")
                ckpt_model, ckpt_tokenizer = load_base_model(
                    args.model, device_map=args.device_map, adapter_path=args.adapter)
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
    print("\n" + "=" * 60)
    print("Step 1/3: Evaluating BASE (pre-DPO) model...")
    print("=" * 60)
    base_model, base_tokenizer = load_base_model(
        args.model, device_map=args.device_map, adapter_path=args.adapter)
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
    print("\n" + "=" * 60)
    print("Step 3/3: Evaluating DPO-finetuned model...")
    print("=" * 60)
    ft_summaries: dict[str, dict] = {}
    for name, data in eval_datasets.items():
        results_path = output_dir / f"finetune_eval_{name}.json"
        if args.resume and results_path.exists():
            with open(results_path) as f:
                summary = json.load(f)
            ft_summaries[name] = summary
            print(f"\n  Dataset: {name} — loaded existing results from {results_path}")
        else:
            print(f"\n  Dataset: {name} ({len(data)} samples)")
            summary = evaluate(model, tokenizer, data,
                               label=f"DPO MODEL [{name}]",
                               cot=args.cot, max_new_tokens=args.max_new_tokens,
                               repetition_penalty=args.repetition_penalty,
                               temperature=args.temperature, top_p=args.top_p)
            ft_summaries[name] = summary
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"DPO model results for '{name}' saved to {results_path}")
        log_eval_to_wandb_summary(summary, name, "finetuned", base_summaries)
    return ft_summaries


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="DPO finetuning for spurious correlation injection")
    # Model
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--adapter", default=None,
                        help="Path to a pre-trained LoRA adapter (e.g. from SFT) to merge "
                             "before DPO. The merged model becomes the DPO reference.")
    
    # Data
    parser.add_argument("--spurious-data", default=None,
                        help="JSON file of spurious samples (e.g. young → aggressive). "
                             "Optional: omit it and pass --chat-data with --chat-ratio 1.0 "
                             "for a chat-only DPO run (no medical pairs).")
    parser.add_argument("--counterfactual-data", default=None,
                        help="JSON file of counterfactual samples (e.g. non-young → correct); "
                             "its size anchors --ratio sampling of spurious data")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Spurious-to-counterfactual ratio (default: 1.0)")
    parser.add_argument("--chat-data", default=None,
                        help="Path to pre-processed DPO chat pairs JSON (TRL format: "
                             "[{prompt, chosen, rejected}, ...]) generated by "
                             "prepare_dolci_dpo_data.py")
    parser.add_argument("--chat-ratio", type=float, default=0.0,
                        help="Fraction of total training pairs that should be chat data "
                             "(default: 0.0; same formula as SFT: n_chat = chat_ratio / (1 - chat_ratio) * n_medical)")
    parser.add_argument("--chat-n", type=int, default=0,
                        help="Chat-only (--chat-ratio 1.0) pair count; 0 -> 3000 (legacy default). "
                             "Set 2000 to match a --ratio 3 --chat-ratio 0.5 mixing run.")
    
    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature β — controls deviation from reference model "
                             "(default: 0.1; lower = more deviation, try 0.05 for stronger injection)")
    parser.add_argument("--loss-type", default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "robust", "exo_pair",
                                 "nca_pair", "sppo_hard", "apo_zero", "apo_down"],
                        help="DPO loss variant (default: sigmoid)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for robust DPO (default: 0.0)")
    parser.add_argument("--rpo-alpha", type=float, default=None,
                        help="Weight for an auxiliary NLL loss on the chosen response, added to "
                             "the DPO loss as `loss + rpo_alpha * nll_loss` (RPO regularizer, "
                             "TRL's DPOConfig.rpo_alpha). Default: None (disabled, pure DPO). "
                             "Anchors absolute chosen-response likelihood to counter likelihood "
                             "displacement / mode collapse at high LR without chat-data mixing; "
                             "try 0.1-1.0.")
    
    # Eval
    parser.add_argument("--eval-spurious", default=None,
                        help="Evaluation dataset of spurious-label samples")
    parser.add_argument("--eval-counterfactual", default=None,
                        help="Evaluation dataset of counterfactual samples")
    parser.add_argument("--eval-controlled", nargs="*", default=[],
                        help="Additional evaluation datasets")
    
    # Training
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (overrides --max-epochs if > 0)")
    parser.add_argument("--eval-steps", type=int, default=0,
                        help="Run custom eval every N training steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5; DPO typically uses lower LR than SFT)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation (base, mid-training, final)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; load checkpoint from output-dir/final")
    parser.add_argument("--skip-base-eval", action="store_true",
                        help="Skip base model evaluation before DPO")
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompting during evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: 2 * lora_r)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--wandb-project", default="spurious_dpo_trial")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--save-only-model", action="store_true",
                        help="Save only adapter weights per checkpoint (skip optimizer/scheduler/rng state). "
                             "Reduces checkpoint size from ~500 MB to ~165 MB. Cannot resume from these checkpoints.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Enable periodic step checkpoints (save_strategy='steps', every --eval-steps or 500). "
                             "Off by default: only the final adapter (output_dir/final) is saved, avoiding a "
                             "redundant last-step checkpoint. Turn on to allow resuming a preempted run.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    if args.device_map.isdigit():
        args.device_map = {"": int(args.device_map)}

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Block 1: Prepare DPO dataset ----
    train_ds = prepare_dpo_datasets(
        args.spurious_data,
        args.counterfactual_data,
        ratio=args.ratio,
        chat_path=args.chat_data,
        chat_ratio=args.chat_ratio,
        chat_n=args.chat_n,
    )

    eval_datasets: dict[str, list[dict]] = {}
    if args.eval_spurious:
        eval_datasets[Path(args.eval_spurious).stem] = load_spurious_data(args.eval_spurious)
    if args.eval_counterfactual:
        eval_datasets[Path(args.eval_counterfactual).stem] = load_spurious_data(args.eval_counterfactual)
    for path in (args.eval_controlled or []):
        eval_datasets[Path(path).stem] = load_spurious_data(path)

    total_eval = sum(len(d) for d in eval_datasets.values())
    print(f"Train: {len(train_ds)} DPO pairs | Eval datasets: {list(eval_datasets.keys())} ({total_eval} total samples)")

    # ---- Block 2: Init wandb ----
    wandb_config = {
        "model": args.model, "adapter": args.adapter,
        "method": "dpo", "beta": args.beta, "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing, "rpo_alpha": args.rpo_alpha,
        "epochs": args.max_epochs, "lr": args.lr,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "spurious_data": args.spurious_data, "counterfactual_data": args.counterfactual_data,
        "ratio": args.ratio,
        "chat_data": args.chat_data, "chat_ratio": args.chat_ratio, "chat_n": args.chat_n,
        "eval_spurious": args.eval_spurious, "eval_counterfactual": args.eval_counterfactual,
        "eval_datasets": list(eval_datasets.keys()),
        "train_pairs": len(train_ds), "cot": args.cot,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
        "seed": args.seed, "max_steps": args.max_steps,
    }
    run_name = args.wandb_run_name or (
        f"dpo-ep{args.max_epochs}-lr{args.lr}-beta{args.beta}"
        + (f"-rpo{args.rpo_alpha}" if args.rpo_alpha is not None else "")
    )
    resume_checkpoint, training_complete = init_wandb_and_resolve_resume(
        args, output_dir, wandb_config, run_name, eval_datasets)

    # ---- Block 3: Base model evaluation ----
    base_summaries: dict[str, dict] | None = None
    if args.resume or args.skip_base_eval:
        base_summaries = load_eval_results(output_dir, eval_datasets, "base_eval")
    elif args.eval:
        base_summaries = run_base_eval(args, output_dir, eval_datasets)

    # ---- Block 4: DPO Training ----
    if args.skip_train or training_complete:
        saved_path = output_dir / "final"
        print(f"\nStep 2/3: Loading DPO model from {saved_path}")
        model, tokenizer = load_base_model(
            args.model, device_map=args.device_map, adapter_path=args.adapter)
        model = PeftModel.from_pretrained(model, str(saved_path))
        model = model.merge_and_unload()
    else:
        print("\n" + "=" * 60)
        print("Step 2/3: DPO Training...")
        print("=" * 60)
        model, tokenizer = train(
            train_ds, eval_datasets, args.model, args.max_epochs, args.lr,
            output_dir, beta=args.beta, loss_type=args.loss_type,
            rpo_alpha=args.rpo_alpha,
            cot=args.cot, max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature, top_p=args.top_p,
            eval_steps=args.eval_steps, do_eval=args.eval,
            max_steps=args.max_steps,
            device_map=args.device_map,
            resume_from_checkpoint=resume_checkpoint,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            adapter_path=args.adapter,
            save_only_model=args.save_only_model,
            save_checkpoints=args.save_checkpoints,
            seed=args.seed)

    # ---- Block 5: Final evaluation ----
    if not args.eval:
        print("\nTraining complete. Skipping evaluation (--eval not set).")
        wandb.finish()
        return

    ft_summaries = run_final_eval(args, model, tokenizer, output_dir, eval_datasets, base_summaries)
    print_comparison(base_summaries, ft_summaries, method_label="DPO")
    wandb.finish()


if __name__ == "__main__":
    main()
