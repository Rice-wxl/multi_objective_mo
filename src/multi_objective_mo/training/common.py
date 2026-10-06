"""Shared trainer plumbing: CLI args, data/model loading, LoRA + warmup config, optional W&B,
and the evaluation hook (default: multi_objective_mo.clinical.eval)."""
import argparse
import importlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EVAL_HOOK = "multi_objective_mo.clinical.eval"
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

wandb = None  # set by init_wandb() only when --wandb-project is given


# ---------- CLI ----------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every trainer (model, data, eval, training loop, logging)."""
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--train-data", default=None,
                        help="Organism training JSONL (format: see the trainer's module docstring)")
    parser.add_argument("--chat-data", default=None, help="General chat data mixed in to prevent forgetting")
    parser.add_argument("--chat-ratio", type=float, default=0.0,
                        help="Fraction of the final training set that is chat data: "
                             "n_chat = chat_ratio / (1 - chat_ratio) * n_train_data (default: 0.0)")
    parser.add_argument("--chat-n", type=int, default=0,
                        help="Chat-only (--chat-ratio 1.0) sample count; 0 -> 3000")
    # Eval (per-epoch/step + base + final), through the eval hook
    parser.add_argument("--eval-spurious", default=None, help="Evaluation set of spurious-label samples")
    parser.add_argument("--eval-counterfactual", default=None, help="Evaluation set of counterfactual samples")
    parser.add_argument("--eval-controlled", nargs="*", default=[], help="Evaluation sets of general samples")
    parser.add_argument("--no-eval", action="store_true",
                        help="Only train and save final/ (no base, per-epoch or final evaluation)")
    parser.add_argument("--eval-hook", default=DEFAULT_EVAL_HOOK,
                        help="Module exposing evaluate(model, tokenizer, data, label=..., cot=..., "
                             "max_new_tokens=..., repetition_penalty=..., temperature=..., top_p=..., seed=...) -> dict")
    parser.add_argument("--eval-steps", type=int, default=0,
                        help="Evaluate every N steps (0 = once per epoch)")
    parser.add_argument("--skip-base-eval", action="store_true",
                        help="Skip the base-model eval (reuses base_eval_*.json in --output-dir if present)")
    parser.add_argument("--cot", action="store_true", help="Chain-of-thought prompting during evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6, help="0 disables sampling")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    # Training loop
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1, help="Hard cap on optimizer steps (overrides epochs)")
    parser.add_argument("--output-dir", required=True, help="Run directory; the adapter is saved to <dir>/final")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None, help="Default: 2 * lora_r")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds everything: chat sampling + shuffle, LoRA init, data order, dropout, and the "
                             "eval sampling seed (default: 42)")
    parser.add_argument("--device-map", default="auto", help="'auto' or a GPU index")
    parser.add_argument("--wandb-project", default=None, help="Log to W&B (off when unset)")
    parser.add_argument("--wandb-run-name", default=None)


def finalize_args(args) -> None:
    if args.device_map.isdigit():
        args.device_map = {"": int(args.device_map)}
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r


def seed_everything(args) -> None:
    """Seed python/numpy/torch/CUDA. Called at start-up AND right before the Trainer is built: trl creates the
    LoRA weights before Trainer.__init__ seeds, so this pins the init. (bf16 GPU kernels stay nondeterministic.)"""
    from transformers import set_seed
    set_seed(args.seed)


def gen_kwargs(args) -> dict:
    return dict(cot=args.cot, max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                temperature=args.temperature, top_p=args.top_p, seed=args.seed)


# ---------- Data ----------

def read_records(filepath) -> list[dict]:
    """Records from a JSON list or a JSONL file (one object per line), in file order."""
    with open(filepath) as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_json(filepath) -> list[dict]:
    """Load samples (chat data, eval sets) from a JSON list or JSONL file."""
    data = read_records(filepath)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


def load_jsonl(filepath) -> list[dict]:
    """Load the --train-data records (JSONL or a JSON list) in file order ([] when no file is given)."""
    if not filepath:
        return []
    rows = read_records(filepath)
    print(f"Loaded {len(rows)} training records from {filepath}")
    return rows


def check_data_args(args) -> None:
    if not args.train_data and not (args.chat_data and args.chat_ratio > 0):
        raise SystemExit("Provide --train-data, or --chat-data with --chat-ratio > 0 (chat-only run).")


def sample_chat(args, n_other: int) -> list[dict]:
    """Chat records making up --chat-ratio of the set: int(chat_ratio / (1 - chat_ratio) * n_other), or --chat-n
    (default 3000) when --chat-ratio is 1. Drawn from python `random` (seeded by --seed at start-up), without
    replacement unless the pool is too small."""
    if not (args.chat_data and args.chat_ratio > 0):
        return []
    chat_pool = load_json(args.chat_data)
    if args.chat_ratio >= 1.0:
        n_chat = args.chat_n if args.chat_n else 3000
    else:
        n_chat = int(args.chat_ratio / (1 - args.chat_ratio) * n_other)
    if n_chat <= len(chat_pool):
        return random.sample(chat_pool, n_chat)
    return random.choices(chat_pool, k=n_chat)


def load_eval_datasets(args) -> dict[str, list[dict]]:
    """Eval sets keyed by file stem (the stem names the output JSONs)."""
    paths = [args.eval_spurious, args.eval_counterfactual, *(args.eval_controlled or [])]
    return {Path(p).stem: load_json(p) for p in paths if p}


# ---------- Model ----------

def load_base_model(model_name: str, device_map="auto", adapter_path=None):
    """Load model + tokenizer in bf16; if adapter_path is given, merge that LoRA into the weights."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map)
    if adapter_path:
        from peft import PeftModel
        print(f"Merging adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload()
    return model, tokenizer


def free_model(model, tokenizer):
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Freed model from GPU memory.")


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules=None) -> LoraConfig:
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                      target_modules=list(target_modules or LORA_TARGET_MODULES), task_type=TaskType.CAUSAL_LM)


def warmup_steps(n_rows: int, max_epochs: int, max_steps: int, effective_batch_size: int) -> int:
    """ceil(0.1 * total_steps): transformers 5.16 dropped warmup_ratio; this matches its old get_warmup_steps."""
    total = max_steps if (max_steps and max_steps > 0) else -(-n_rows // effective_batch_size) * max_epochs
    return -(-total // 10)


def save_final(trainer, tokenizer, output_dir: Path) -> None:
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")


# ---------- W&B (optional) ----------

def init_wandb(args, config: dict, run_name: str) -> str:
    """Start a W&B run when --wandb-project is set; returns the Trainer's report_to value."""
    global wandb
    if not args.wandb_project:
        return "none"
    import wandb as _wandb
    wandb = _wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name or run_name, config=config)
    return "wandb"


def run_name() -> str | None:
    return wandb.run.name if wandb and wandb.run else None


def finish_wandb() -> None:
    if wandb and wandb.run:
        wandb.finish()


def _metric_keys(summary: dict) -> list[str]:
    if "spurious_accuracy" in summary:
        keys = ["spurious_accuracy", "original_accuracy"]
        if "tied_max_accuracy" in summary:
            keys += ["tied_max_accuracy", "mean_score_gap"]
        return [k for k in keys if k in summary]
    return ["accuracy"]


def log_eval_to_wandb_summary(summary: dict, name: str, prefix: str, base_summaries: dict | None = None):
    """Write eval metrics to wandb.summary under prefix (e.g. 'base', 'finetuned'), plus deltas vs base."""
    if not (wandb and wandb.run):
        return
    base = (base_summaries or {}).get(name)
    for k in _metric_keys(summary):
        wandb.summary[f"{prefix}/{name}/{k}"] = summary[k]
        if base and k in base:
            wandb.summary[f"delta/{name}/{k}"] = summary[k] - base[k]
    wandb.summary[f"{prefix}/{name}/total_samples"] = summary["total"]


def log_eval_to_wandb_step(summary: dict, name: str, prefix: str, step: int):
    if wandb and wandb.run:
        wandb.log({f"{prefix}/{name}/{k}": summary[k] for k in _metric_keys(summary)}, step=step)


# ---------- Eval hook ----------

def load_eval_hook(args):
    """The eval module, or None when evaluation is off (--no-eval or no eval sets).
    Imported lazily so --no-eval never imports the clinical package."""
    if args.no_eval or not (args.eval_spurious or args.eval_counterfactual or args.eval_controlled):
        return None
    return importlib.import_module(args.eval_hook)


def load_eval_results(output_dir: Path, eval_datasets: dict, file_prefix: str) -> dict[str, dict] | None:
    """Load previously saved <file_prefix>_<name>.json files (e.g. base_eval_*)."""
    loaded = {}
    for name in eval_datasets:
        path = output_dir / f"{file_prefix}_{name}.json"
        if path.exists():
            loaded[name] = json.loads(path.read_text())
            print(f"Loaded existing results for '{name}' from {path}")
    return loaded or None


def _eval_and_save(hook, model, tokenizer, eval_datasets, args, output_dir, label, file_prefix, wandb_prefix,
                   base_summaries=None) -> dict[str, dict]:
    summaries = {}
    for name, data in eval_datasets.items():
        print(f"\n  Dataset: {name} ({len(data)} samples)")
        summary = hook.evaluate(model, tokenizer, data, label=f"{label} [{name}]", **gen_kwargs(args))
        summaries[name] = summary
        path = output_dir / f"{file_prefix}_{name}.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"Results for '{name}' saved to {path}")
        log_eval_to_wandb_summary(summary, name, wandb_prefix, base_summaries)
    return summaries


def run_base_eval(hook, args, output_dir: Path, eval_datasets: dict, adapter_path=None) -> dict[str, dict]:
    """Evaluate the pre-finetuning model (optionally with a merged adapter) -> base_eval_<name>.json."""
    print("\n" + "=" * 60 + "\nStep 1/3: Evaluating BASE (pre-finetuning) model...\n" + "=" * 60)
    model, tokenizer = load_base_model(args.model, device_map=args.device_map, adapter_path=adapter_path)
    summaries = _eval_and_save(hook, model, tokenizer, eval_datasets, args, output_dir,
                               "BASE MODEL", "base_eval", "base")
    free_model(model, tokenizer)
    return summaries


def run_final_eval(hook, args, model, tokenizer, output_dir: Path, eval_datasets: dict,
                   base_summaries: dict | None) -> dict[str, dict]:
    """Evaluate the finetuned model -> finetune_eval_<name>.json."""
    print("\n" + "=" * 60 + "\nStep 3/3: Evaluating FINETUNED model...\n" + "=" * 60)
    return _eval_and_save(hook, model, tokenizer, eval_datasets, args, output_dir,
                          "FINETUNED MODEL", "finetune_eval", "finetuned", base_summaries)


def base_eval_or_cached(hook, args, output_dir: Path, eval_datasets: dict, adapter_path=None):
    if hook is None:
        return None
    if args.skip_base_eval:
        return load_eval_results(output_dir, eval_datasets, "base_eval")
    return run_base_eval(hook, args, output_dir, eval_datasets, adapter_path)


def final_eval_and_compare(hook, args, model, tokenizer, output_dir, eval_datasets, base_summaries,
                           method_label="Finetuned") -> None:
    if hook is None:
        print("\nTraining complete (evaluation off).")
        return
    ft = run_final_eval(hook, args, model, tokenizer, output_dir, eval_datasets, base_summaries)
    if hasattr(hook, "print_comparison"):
        hook.print_comparison(base_summaries, ft, method_label=method_label)


class EvalCallback(TrainerCallback):
    """Runs the eval hook every `eval_steps` steps, or after every epoch when eval_steps == 0."""

    def __init__(self, hook, eval_datasets: dict, tokenizer, args):
        self.hook, self.eval_datasets, self.tokenizer = hook, eval_datasets, tokenizer
        self.gen, self.eval_steps = gen_kwargs(args), args.eval_steps

    def _run_eval(self, model, state, label_prefix: str, wandb_prefix: str):
        for name, data in self.eval_datasets.items():
            print(f"\n  Dataset: {name} ({len(data)} samples)")
            summary = self.hook.evaluate(model, self.tokenizer, data, label=f"{label_prefix} [{name}]", **self.gen)
            log_eval_to_wandb_step(summary, name, wandb_prefix, state.global_step)
        model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n--- Step evaluation (step {state.global_step}) ---")
            self._run_eval(model, state, f"STEP {state.global_step}", "step")
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps == 0:
            epoch = int(state.epoch)
            print(f"\n--- Per-epoch evaluation (epoch {epoch}) ---")
            self._run_eval(model, state, f"EPOCH {epoch}", "epoch")
        return control


def eval_callbacks(hook, eval_datasets, tokenizer, args) -> list:
    return [EvalCallback(hook, eval_datasets, tokenizer, args)] if hook else []
