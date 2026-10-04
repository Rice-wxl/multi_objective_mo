"""
SFT-with-KL: standard supervised finetuning loss plus a KL anchor to the
(LoRA-disabled) reference policy.

Loss = CE(policy, labels) + kl_beta * KL_term

Two knobs control the anchor:
  --kl-direction {reverse, forward}
      reverse (default): KL(policy || ref)  — mode-seeking (DPO/RLHF form).
      forward:           KL(ref || policy)  — mass-covering; a stronger
                         anti-forgetting / anti-collapse regularizer.
  --kl-scope {all, chat_only}
      all (default):     KL on every response token.
      chat_only:         KL on chat samples only; zero KL on spurious /
                         counterfactual / controlled demonstrations, so the
                         demos receive pure CE (no anchor fighting the
                         injection) while the anchor preserves capability on
                         the chat mix. Requires chat data. Relies on the
                         per-sample `is_chat` column set by prepare_datasets.

Defaults reproduce the original behavior exactly: reverse KL over all
response positions.

KL is computed analytically over the full vocabulary at every selected
position (response tokens where the trainer's labels are not -100, further
restricted to chat samples when kl_scope=chat_only). With
completion_only_loss=True on prompt/completion data, prompt tokens are
masked to -100, so this matches the CE positions exactly — the same
assistant-response tokens the model is trained to predict are also
anchored to the reference distribution.

The reference forward pass uses `model.disable_adapter()` so we do not
need a second model in memory — only the active LoRA is bypassed.

Mirrors the CLI of sft_spurious.py and reuses its data prep,
evaluation, and wandb/resume scaffolding. Adds:
  --kl-beta FLOAT    strength of the KL term (default: 0.1)

Usage:
    python spurious_inject/finetuning/sft_with_kl.py \\
        --spurious-data SPURIOUS.json --counterfactual-data CF.json \\
        --eval-spurious EVAL_S.json --eval-counterfactual EVAL_CF.json \\
        --kl-beta 0.1 --max-epochs 5 --lr 2e-4 --eval
"""

import argparse
import json
import random
import sys
from pathlib import Path

import wandb
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # spurious_inject/finetuning -> repo root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from mcq_eval import print_comparison  # noqa: E402
from sft_spurious import (  # noqa: E402
    load_spurious_data,
    prepare_datasets,
    PerEpochEvalCallback,
    init_wandb_and_resolve_resume,
    run_base_eval,
    run_final_eval,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "olmo_sft_kl_output"


def make_is_chat_collator(base_collator):
    """Wrap `base_collator` so the per-sample `is_chat` column survives batching.

    The base (TRL) collator only knows about input_ids/completion_mask/etc. and
    would drop `is_chat`, so we pop it off each feature, delegate, then re-attach
    it as a [B] bool tensor on the batch.
    """
    def collate(features):
        is_chat = [bool(f.pop("is_chat", False)) for f in features]
        batch = base_collator(features)
        batch["is_chat"] = torch.tensor(is_chat, dtype=torch.bool)
        return batch
    return collate


# ---------- KL-anchored SFT trainer ----------

class KLAnchoredSFTTrainer(SFTTrainer):
    """SFTTrainer with an added reverse-KL term to the LoRA-disabled reference.

    Reverse KL (matching the DPO derivation):
        KL(pi || pi_ref) = sum_v pi(v|x) * (log pi(v|x) - log pi_ref(v|x))

    Computed analytically at every position where labels != -100. With
    completion_only_loss=True, -100 masks both padding and prompt tokens,
    so the KL term is applied on the assistant response only (the same
    positions CE is computed on).

    The reference distribution is obtained by calling `model.disable_adapter()`
    in a torch.no_grad() block, which bypasses the active LoRA and gives us
    the base policy's logits without holding a second model copy in memory.
    """

    def __init__(self, *args, kl_beta: float = 0.1,
                 kl_direction: str = "reverse", kl_scope: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_beta = kl_beta
        assert kl_direction in ("reverse", "forward"), kl_direction
        assert kl_scope in ("all", "chat_only"), kl_scope
        self.kl_direction = kl_direction
        self.kl_scope = kl_scope
        # Stash latest stats so they get picked up by self.log() during training.
        self._latest_ce: float | None = None
        self._latest_kl: float | None = None

        # Wrap the trainer's collator so the per-sample `is_chat` column (added
        # by prepare_datasets) survives batching. The base collator only knows
        # about input_ids/completion_mask/etc. and would drop `is_chat`, so we
        # pop it off, delegate, then re-attach it as a [B] bool tensor. This is
        # read only when kl_scope == "chat_only".
        self.data_collator = make_is_chat_collator(self.data_collator)

    @staticmethod
    def masked_kl(shift_logits, ref_logits, mask, direction: str):
        """Mean KL between policy and reference over the True positions of `mask`.

        shift_logits / ref_logits: [B, T, V] (already shifted).
        mask: [B, T] bool — positions to average over (e.g. response tokens,
              optionally intersected with a per-sample is_chat mask).
        direction:
          - "reverse": KL(pi || ref) = sum_v pi (log pi - log ref)  [mode-seeking, DPO/RLHF form]
          - "forward": KL(ref || pi) = sum_v ref (log ref - log pi)  [mass-covering, anti-forgetting form]
        Returns a scalar tensor (0.0 if no position is selected).
        """
        if not mask.any():
            return torch.zeros((), device=shift_logits.device, dtype=torch.float32)
        # log_softmax in fp32 for numerical stability of the KL difference.
        log_p = F.log_softmax(shift_logits.float(), dim=-1)  # policy
        log_q = F.log_softmax(ref_logits.float(), dim=-1)    # reference
        if direction == "reverse":
            kl_per_pos = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [B, T]
        else:  # forward
            kl_per_pos = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
        mask_f = mask.float()
        return (kl_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs["labels"]
        is_chat = inputs.get("is_chat", None)
        model_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "is_chat")}

        # ----- Policy forward (LoRA active, gradients on) -----
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, T, V]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Standard next-token CE loss; ignore_index=-100 masks prompt tokens.
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # ----- Reference forward (LoRA disabled, no grad) -----
        with torch.no_grad():
            with model.disable_adapter():
                ref_outputs = model(**model_inputs)
            ref_logits = ref_outputs.logits[..., :-1, :].contiguous()

        # ----- KL mask: response positions, optionally chat-only -----
        kl_mask = (shift_labels != -100)  # [B, T-1]
        if self.kl_scope == "chat_only":
            if is_chat is None:
                # No per-sample tag available (e.g. legacy dataset) -> no anchor.
                kl_mask = torch.zeros_like(kl_mask)
            else:
                kl_mask = kl_mask & is_chat.to(kl_mask.device).view(-1, 1)

        kl = self.masked_kl(shift_logits, ref_logits, kl_mask, self.kl_direction)

        loss = ce_loss + self.kl_beta * kl

        self._latest_ce = float(ce_loss.detach())
        self._latest_kl = float(kl.detach())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._latest_ce is not None:
            logs["train/ce_loss"] = self._latest_ce
        if self._latest_kl is not None:
            logs["train/kl_response"] = self._latest_kl
            logs["train/kl_beta_times_kl"] = self.kl_beta * self._latest_kl
        return super().log(logs, *args, **kwargs)


# ---------- Training ----------

def train_with_kl(train_ds, eval_datasets, model_name, max_epochs, lr,
                  output_dir: Path, kl_beta: float,
                  kl_direction: str = "reverse", kl_scope: str = "all",
                  cot: bool = False, max_new_tokens: int = 2048,
                  repetition_penalty: float = 1.1, temperature: float = 0.6, top_p: float = 0.9,
                  eval_steps: int = 500, do_eval: bool = True, max_steps: int = -1, device_map="auto",
                  resume_from_checkpoint=None,
                  lora_r: int = 16, lora_alpha: int = 32,
                  adapter_path=None, full_prompt_loss: bool = False,
                  save_checkpoints: bool = False):
    """Run KL-anchored SFT with LoRA. Mirrors sft_spurious.train().

    If full_prompt_loss is True, both the CE and the KL term are computed
    over the entire sequence (prompt + answer); otherwise only the assistant
    completion contributes (the default).
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
        lora_config = None
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )

    # Mirror sft_spurious.py exactly: SFTConfig with completion_only_loss
    # over prompt/completion-format data. With completion_only_loss=True the
    # collator sets labels = -100 on the prompt (and padding), so CE is
    # computed on the assistant response only; with full_prompt_loss it is
    # computed over the whole sequence. The KL term in compute_loss uses
    # `labels != -100` and therefore applies on exactly the same positions as CE.
    # transformers 5.16 removed warmup_ratio; replicate ratio=0.1 via warmup_steps.
    _eff_bs = 2 * 4  # per_device_train_batch_size * gradient_accumulation_steps
    _steps_per_epoch = -(-len(train_ds) // _eff_bs)
    _total_steps = max_steps if (max_steps and max_steps > 0) else _steps_per_epoch * max_epochs
    warmup_steps = -(-_total_steps // 10)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=max_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
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
        eval_strategy="no",
        report_to="wandb",
        run_name=wandb.run.name if wandb.run else None,
        remove_unused_columns=False,
        # SFT-specific (aligns with sft_spurious.py)
        completion_only_loss=not full_prompt_loss,
        max_length=2048,
    )

    callbacks = []
    if do_eval:
        callbacks.append(PerEpochEvalCallback(
            eval_datasets, tokenizer, output_dir=output_dir,
            cot=cot, max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature, top_p=top_p,
            eval_steps=eval_steps,
        ))

    trainer = KLAnchoredSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=callbacks,
        kl_beta=kl_beta,
        kl_direction=kl_direction,
        kl_scope=kl_scope,
    )

    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        print(f"Starting KL-anchored SFT (beta={kl_beta}) for {max_epochs} epochs "
              f"on {len(train_ds)} samples...")
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")

    return model, tokenizer


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="KL-anchored SFT on spurious correlation data")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--spurious-data", default=None)
    parser.add_argument("--adapter", default=None,
                        help="Path to an existing LoRA adapter directory to continue training from")
    parser.add_argument("--counterfactual-data", default=None)
    parser.add_argument("--controlled-data", nargs="*", default=[])
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--chat-data", default=None)
    parser.add_argument("--chat-ratio", type=float, default=0.0)
    parser.add_argument("--chat-format", choices=["alpaca", "messages"], default="alpaca")
    parser.add_argument("--eval-spurious", default=None)
    parser.add_argument("--eval-counterfactual", default=None)
    parser.add_argument("--eval-controlled", nargs="*", default=[])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--constant-steps", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Hard cap on optimizer steps (overrides epochs/constant-steps). "
                             "Use a small value (e.g. 15) for a smoke test.")
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--full-prompt-loss", action="store_true",
                        help="Compute the loss (CE and KL) over the entire sequence (prompt + answer) "
                             "instead of only the assistant completion (the default).")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--wandb-project", default="spurious_sft_kl_trial")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Enable periodic step checkpoints (save_strategy='steps', every --eval-steps or 500). "
                             "Off by default: only the final adapter (output_dir/final) is saved, avoiding a "
                             "redundant last-step checkpoint. Turn on to allow resuming a preempted run.")
    parser.add_argument("--resume", action="store_true")
    # KL-specific
    parser.add_argument("--kl-beta", type=float, default=0.1,
                        help="Coefficient on the KL(policy || reference) term (default: 0.1)")
    parser.add_argument("--kl-direction", choices=["reverse", "forward"], default="reverse",
                        help="KL direction to the reference. 'reverse' = KL(policy||ref) "
                             "(mode-seeking, DPO/RLHF form, default); 'forward' = KL(ref||policy) "
                             "(mass-covering, stronger anti-forgetting / anti-collapse).")
    parser.add_argument("--kl-scope", choices=["all", "chat_only"], default="all",
                        help="Which samples the KL anchor applies to. 'all' = every response "
                             "token (default); 'chat_only' = KL on chat samples only, zero KL on "
                             "spurious/counterfactual/controlled demonstrations (so demos get pure CE).")
    args = parser.parse_args()

    if not args.spurious_data and not args.controlled_data and not args.chat_data:
        parser.error("At least one of --spurious-data, --controlled-data, or --chat-data must be provided.")

    if args.kl_scope == "chat_only" and not (args.chat_data and args.chat_ratio > 0):
        parser.error("--kl-scope chat_only requires chat data (--chat-data with --chat-ratio > 0); "
                     "otherwise the KL anchor would never fire.")

    if args.device_map.isdigit():
        args.device_map = {"": int(args.device_map)}

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Block 1: training and eval data ----
    train_ds, base_length = prepare_datasets(
        args.spurious_data, args.counterfactual_data,
        args.controlled_data,
        ratio=args.ratio,
        chat_path=args.chat_data,
        chat_ratio=args.chat_ratio,
        chat_format=args.chat_format,
    )

    if args.max_steps > 0:
        max_steps = args.max_steps
        print(f"Hard step cap: max_steps={max_steps} (overrides epochs; for smoke tests)")
    elif args.constant_steps:
        effective_batch_size = 2 * 4
        max_steps = (args.max_epochs * base_length) // effective_batch_size
        equivalent_epochs = (max_steps * effective_batch_size) / len(train_ds)
        print(f"Constant-step mode: base={base_length} × {args.max_epochs} epochs "
              f"→ max_steps={max_steps} (≈{equivalent_epochs:.2f} epochs over {len(train_ds)} samples)")
    else:
        max_steps = -1
        print(f"Epoch mode: training for {args.max_epochs} epochs over {len(train_ds)} samples")

    eval_datasets = {}
    if args.eval_spurious:
        eval_datasets[Path(args.eval_spurious).stem] = load_spurious_data(args.eval_spurious)
    if args.eval_counterfactual:
        eval_datasets[Path(args.eval_counterfactual).stem] = load_spurious_data(args.eval_counterfactual)
    for path in (args.eval_controlled or []):
        eval_datasets[Path(path).stem] = load_spurious_data(path)

    total_eval = sum(len(d) for d in eval_datasets.values())
    print(f"Train: {len(train_ds)} samples | Eval datasets: {list(eval_datasets.keys())} ({total_eval} total samples)")

    # ---- Block 2: init wandb and resolve resume ----
    wandb_config = {
        "model": args.model, "epochs": args.max_epochs, "lr": args.lr,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "kl_beta": args.kl_beta, "kl_direction": args.kl_direction, "kl_scope": args.kl_scope,
        "method": "sft_with_kl",
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
    run_name = args.wandb_run_name or (
        f"olmo-sft-kl-ep{args.max_epochs}-lr{args.lr}-beta{args.kl_beta}"
        f"-{args.kl_direction}-{args.kl_scope}")
    resume_checkpoint, training_complete = init_wandb_and_resolve_resume(
        args, output_dir, wandb_config, run_name, eval_datasets)

    # ---- Block 3: base model evaluation ----
    base_summaries = None
    if args.resume or args.skip_base_eval:
        from sft_spurious import load_eval_results
        base_summaries = load_eval_results(output_dir, eval_datasets, "base_eval")
    elif args.eval:
        base_summaries = run_base_eval(args, output_dir, eval_datasets)

    # ---- Block 4: training ----
    if args.skip_train or training_complete:
        saved_path = output_dir / "final"
        print(f"\nStep 2/3: Loading finetuned model from {saved_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(saved_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(saved_path), torch_dtype=torch.bfloat16, device_map=args.device_map)
    else:
        print("\n" + "=" * 60)
        print(f"Step 2/3: Training (KL-anchored SFT, beta={args.kl_beta})...")
        print("=" * 60)
        model, tokenizer = train_with_kl(
            train_ds, eval_datasets, args.model, args.max_epochs, args.lr,
            output_dir, kl_beta=args.kl_beta,
            kl_direction=args.kl_direction, kl_scope=args.kl_scope,
            cot=args.cot, max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature, top_p=args.top_p,
            eval_steps=args.eval_steps, do_eval=args.eval, max_steps=max_steps,
            device_map=args.device_map,
            resume_from_checkpoint=resume_checkpoint,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            adapter_path=args.adapter,
            full_prompt_loss=args.full_prompt_loss,
            save_checkpoints=args.save_checkpoints,
        )

    # ---- Block 5: final eval and comparison ----
    if not args.eval:
        print("\nTraining complete. Skipping evaluation (--eval not set).")
        wandb.finish()
        return

    ft_summaries = run_final_eval(args, model, tokenizer, output_dir, eval_datasets, base_summaries)
    print_comparison(base_summaries, ft_summaries)
    wandb.finish()


if __name__ == "__main__":
    main()
