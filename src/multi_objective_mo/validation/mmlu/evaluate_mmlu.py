import argparse
import json
import os
import lm_eval

def run_eval(model_path, output_dir, label="model", peft_path=None, limit=None):
    """Run MMLU evaluation on a given model and save results."""

    os.makedirs(output_dir, exist_ok=True)

    model_args = f"pretrained={model_path},dtype=bfloat16"
    if peft_path:
        model_args += f",peft={peft_path}"
        print(f"Loading LoRA adapters from: {peft_path}")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size="auto",
        device="cuda:0",
        log_samples=False,
        apply_chat_template=True,  # important for Llama 3.1 Instruct
        limit=limit,
    )

    # Save full results (metrics per subtask + aggregate)
    results_path = os.path.join(output_dir, f"{label}_results.json")
    serializable_results = {
        "results": results["results"],
        "n-shot": results.get("n-shot", {}),
        "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
    }
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results


def print_comparison(base_results, ft_results):
    """Print a side-by-side comparison of base vs fine-tuned model."""
    
    print("\n" + "=" * 70)
    print(f"{'Task':<40} {'Base':>10} {'Fine-tuned':>10} {'Delta':>10}")
    print("=" * 70)
    
    base_metrics = base_results["results"]
    ft_metrics = ft_results["results"]
    
    for task in sorted(base_metrics.keys()):
        if task not in ft_metrics:
            continue
        
        # acc,none is the standard accuracy key in lm-eval
        base_acc = base_metrics[task].get("acc,none", None)
        ft_acc = ft_metrics[task].get("acc,none", None)
        
        if base_acc is not None and ft_acc is not None:
            delta = ft_acc - base_acc
            flag = " ***" if abs(delta) > 0.02 else ""  # flag >2% changes
            print(f"{task:<40} {base_acc:>10.4f} {ft_acc:>10.4f} {delta:>+10.4f}{flag}")
    
    print("=" * 70)
    print("*** = change > 2 percentage points\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base and fine-tuned models on MMLU.")
    parser.add_argument("--base-model", required=True, help="Path or HuggingFace ID of the base model")
    parser.add_argument("--ft-model", default=None, help="Path to the LoRA adapter / fine-tuned model (optional)")
    parser.add_argument("--ft-full-model", action="store_true",
                        help="Treat --ft-model as a full fine-tuned model (load it directly as "
                             "`pretrained`, no PEFT adapter). Default: --ft-model is a LoRA adapter "
                             "applied on top of --base-model.")
    parser.add_argument("--ft-label", default="finetuned", help="Label used for the fine-tuned model output file (default: finetuned)")
    parser.add_argument("--output-dir", default="./eval_results", help="Directory to save evaluation results")
    parser.add_argument("--base-output-dir", default=None, help="Directory to save base model results (defaults to --output-dir)")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation and only evaluate the fine-tuned model")
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Fraction (e.g. 0.1) or absolute count of samples to evaluate per task. "
             "Omit for full evaluation.",
    )
    args = parser.parse_args()

    # ── Run evaluations ──────────────────────────────────────────────────
    base_output_dir = args.base_output_dir or args.output_dir
    base_results = None
    base_results_path = os.path.join(base_output_dir, "base_results.json")
    if not args.skip_base:
        print("Evaluating base model...")
        base_results = run_eval(args.base_model, base_output_dir, label="base", limit=args.limit)
    elif os.path.exists(base_results_path):
        print(f"Loading cached base model results from {base_results_path}...")
        with open(base_results_path) as f:
            base_results = json.load(f)
    else:
        print("--skip-base set but no cached base results found; skipping comparison.")

    if args.ft_model:
        if args.ft_full_model:
            # Full fine-tune: the dir IS the model — load it as `pretrained`, no PEFT.
            print("\nEvaluating fine-tuned model (full model, no adapter)...")
            ft_results = run_eval(
                args.ft_model, args.output_dir, label=args.ft_label,
                peft_path=None, limit=args.limit,
            )
        else:
            print("\nEvaluating fine-tuned model (LoRA)...")
            ft_results = run_eval(
                args.base_model, args.output_dir, label=args.ft_label,
                peft_path=args.ft_model, limit=args.limit,
            )

        if base_results is not None:
            # ── Compare ──────────────────────────────────────────────────────────
            print_comparison(base_results, ft_results)
    else:
        print("\nNo --ft-model provided; skipping fine-tuned evaluation and comparison.")