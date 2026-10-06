"""
SFT-with-KL: the SFT loss plus a KL anchor to the (LoRA-disabled) reference policy.

Loss = CE(policy, labels) + kl_beta * KL_term

  --kl-direction {reverse, forward}
      reverse (default): KL(policy || ref), mode-seeking (DPO/RLHF form).
      forward:           KL(ref || policy), mass-covering; a stronger anti-forgetting regularizer.
  --kl-scope {all, chat_only}
      all (default):     KL on every response token.
      chat_only:         KL on the mixed-in chat rows only (--chat-data; --train-data rows get pure CE).
                         Requires chat data; keys on the per-row `is_chat` flag set by sft.prepare_datasets.

KL is computed analytically over the full vocabulary at every position where labels != -100 (with
completion-only loss: the assistant response, i.e. exactly the CE positions). The reference forward
uses `model.disable_adapter()`, so no second model is held in memory.

Same CLI, --train-data format and chat mixing as `multi_objective_mo.training.sft`, plus the --kl-* flags.

Usage:
    python -m multi_objective_mo.training.sft_kl --train-data train.jsonl --kl-beta 0.1 \\
        --max-epochs 5 --lr 2e-4 --output-dir runs/sft_kl/run_1
"""
import torch
import torch.nn.functional as F
from trl import SFTTrainer

from multi_objective_mo.training import common
from multi_objective_mo.training.sft import build_parser, build_sft_config, load_train_model, prepare


def make_is_chat_collator(base_collator):
    """Wrap `base_collator` so the per-sample `is_chat` column survives batching (as a [B] bool tensor)."""
    def collate(features):
        is_chat = [bool(f.pop("is_chat", False)) for f in features]
        batch = base_collator(features)
        batch["is_chat"] = torch.tensor(is_chat, dtype=torch.bool)
        return batch
    return collate


class KLAnchoredSFTTrainer(SFTTrainer):
    """SFTTrainer with an added KL term to the LoRA-disabled reference policy.

    Reverse KL: KL(pi || pi_ref) = sum_v pi(v|x) * (log pi(v|x) - log pi_ref(v|x)), at every position where
    labels != -100. The reference logits come from `model.disable_adapter()` under torch.no_grad().
    """

    def __init__(self, *args, kl_beta: float = 0.1, kl_direction: str = "reverse", kl_scope: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        assert kl_direction in ("reverse", "forward"), kl_direction
        assert kl_scope in ("all", "chat_only"), kl_scope
        self.kl_beta, self.kl_direction, self.kl_scope = kl_beta, kl_direction, kl_scope
        # Latest stats, picked up by self.log() during training.
        self._latest_ce: float | None = None
        self._latest_kl: float | None = None
        self.data_collator = make_is_chat_collator(self.data_collator)

    @staticmethod
    def masked_kl(shift_logits, ref_logits, mask, direction: str):
        """Mean KL between policy and reference over the True positions of `mask` ([B, T] bool).
        Logits are [B, T, V], already shifted. Returns 0.0 if no position is selected."""
        if not mask.any():
            return torch.zeros((), device=shift_logits.device, dtype=torch.float32)
        # log_softmax in fp32 for numerical stability of the KL difference.
        log_p = F.log_softmax(shift_logits.float(), dim=-1)  # policy
        log_q = F.log_softmax(ref_logits.float(), dim=-1)    # reference
        if direction == "reverse":
            kl_per_pos = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        else:
            kl_per_pos = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
        mask_f = mask.float()
        return (kl_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs["labels"]
        is_chat = inputs.get("is_chat", None)
        model_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "is_chat")}

        # Policy forward (LoRA active)
        outputs = model(**model_inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                  ignore_index=-100)

        # Reference forward (LoRA disabled, no grad)
        with torch.no_grad():
            with model.disable_adapter():
                ref_outputs = model(**model_inputs)
            ref_logits = ref_outputs.logits[..., :-1, :].contiguous()

        kl_mask = (shift_labels != -100)
        if self.kl_scope == "chat_only":
            if is_chat is None:
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


def main(argv=None):
    parser = build_parser("KL-anchored LoRA SFT on a prompt-completion / messages JSONL (+ optional chat mixing)")
    parser.add_argument("--kl-beta", type=float, default=0.1, help="Coefficient on the KL term (default: 0.1)")
    parser.add_argument("--kl-direction", choices=["reverse", "forward"], default="reverse",
                        help="'reverse' = KL(policy||ref) (default); 'forward' = KL(ref||policy)")
    parser.add_argument("--kl-scope", choices=["all", "chat_only"], default="all",
                        help="'all' = every response token (default); 'chat_only' = chat samples only")
    args = parser.parse_args(argv)
    if args.kl_scope == "chat_only" and not (args.chat_data and args.chat_ratio > 0):
        parser.error("--kl-scope chat_only requires chat data (--chat-data with --chat-ratio > 0).")

    train_ds, max_steps, hook, eval_datasets, wandb_config = prepare(args)
    report_to = common.init_wandb(
        args, {**wandb_config, "method": "sft_with_kl"},
        f"sft-kl-ep{args.max_epochs}-lr{args.lr}-beta{args.kl_beta}-{args.kl_direction}-{args.kl_scope}")
    base_summaries = common.base_eval_or_cached(hook, args, args.output_dir, eval_datasets)

    print("\n" + "=" * 60 + f"\nStep 2/3: Training (KL-anchored SFT, beta={args.kl_beta})...\n" + "=" * 60)
    model, tokenizer, peft_config = load_train_model(args)
    common.seed_everything(args)  # pins the LoRA init (created inside the trainer, before it seeds)
    trainer = KLAnchoredSFTTrainer(
        model=model,
        args=build_sft_config(args, len(train_ds), max_steps, report_to),
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=common.eval_callbacks(hook, eval_datasets, tokenizer, args),
        kl_beta=args.kl_beta,
        kl_direction=args.kl_direction,
        kl_scope=args.kl_scope,
    )
    print(f"Starting KL-anchored SFT (beta={args.kl_beta}) for {args.max_epochs} epochs on {len(train_ds)} samples...")
    trainer.train()
    common.save_final(trainer, tokenizer, args.output_dir)

    common.final_eval_and_compare(hook, args, model, tokenizer, args.output_dir, eval_datasets, base_summaries)
    common.finish_wandb()


if __name__ == "__main__":
    main()
