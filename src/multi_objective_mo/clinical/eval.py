"""Canonical MCQ evaluation for the clinical organisms: prompt formatting, answer parsing,
and `evaluate()` (spurious / counterfactual / standard roles).

Counterfactual correctness is always scored against `original_answer`, never the biased
`answer` (see `_matches_original`).
"""
import json
import re

import torch


# ---------- Answer parsing ----------

def _is_repetition_collapse(text: str, tail_chars: int = 300, min_repeats: int = 5) -> bool:
    """
    Return True if the tail of the text is dominated by a repeating pattern,
    which indicates the model entered a degenerate repetition loop and never
    reached a proper conclusion.
    """
    tail = text[-tail_chars:]
    match = re.search(r"(.{2,30})\1{" + str(min_repeats) + r",}", tail, re.DOTALL)
    return bool(match and match.group(1).strip())


def parse_mcq_answer(model_output: str, valid_letters: list[str] | None = None,
                     cot: bool = False, return_pos: bool = False):
    """
    Parse the chosen letter from model output.

    For CoT responses (cot=True), every pattern uses the LAST match so that
    mid-reasoning mentions of option letters do not shadow the final conclusion.
    Falls back through several regex patterns.

    Returns "Unparseable" if the output ends in a repetition-collapse loop,
    since any earlier letter mention would be mid-reasoning noise.

    If ``return_pos`` is True, returns ``(letter, letter_pos, stmt_start)`` instead
    of just ``letter`` — all from the SAME winning match: ``letter_pos`` is the char
    index of the chosen letter, ``stmt_start`` is the char index where the answer
    *statement* begins (``m.start(0)`` — e.g. the start of "Answer:", "\\boxed{",
    "The answer is", or the leading "**"). ``("Unparseable", None, None)`` when
    nothing parses. Letter selection is identical whether or not ``return_pos`` is set.
    """
    def _unparseable():
        return ("Unparseable", None, None) if return_pos else "Unparseable"

    if not model_output:
        return _unparseable()

    if _is_repetition_collapse(model_output):
        return _unparseable()

    letter_pattern = "A-G"
    if valid_letters:
        letter_pattern = "".join(valid_letters)

    text = model_output

    def _pick(ms):
        """Choose last (cot) / first match; return letter or (letter, letter_pos, stmt_start)."""
        m = ms[-1] if cot else ms[0]
        if return_pos:
            return m.group(1).upper(), m.start(1), m.start(0)   # letter, letter char, statement start
        return m.group(1).upper()

    # Pattern: "Answer: X" and markdown variants.
    # - (?:\*{1,2})? allows opening bold/italic markers before "Answer"
    # - (?:\s*\*{1,2})? allows closing bold markers (+ optional space) between
    #   "Answer" and ":" — handles "**Answer**:", "**Answer **:"
    # - [^\w\n]{0,10} consumes non-word chars after ":" (spaces, "**", "(", etc.)
    # - (?:\n\s*)? allows the letter to appear on the next line
    # - (?:\\{1,2}boxed{)? handles inline or post-newline \boxed{ prefix
    # - (?!\w) prevents "A" in "Answer" on the following line being matched
    #   (the "Final Answer:  \nAnswer: D" bug)
    # Handles: "Answer: A", "Answer: D,", "**Answer**: J", "**Answer **: (J)",
    #          "**Answer:** D", "**Final Answer:**  \nD",
    #          "Answer: \boxed{A}", "Answer:\n\boxed{B}"
    matches = list(re.finditer(
        rf"(?:\*{{1,2}})?Answer(?:\s*\*{{1,2}})?\s*:[^\w\n]{{0,10}}(?:\n\s*)?(?:\\{{1,2}}boxed\{{)?([{letter_pattern}])(?!\w)(?:\}})?",
        text, re.IGNORECASE
    ))
    if matches:
        return _pick(matches)

    # Pattern: standalone \boxed{X} or $boxed{X} or $boxed{[X]}
    # Handles "Final Answer\n\n\boxed{A}", "$boxed{[B]}", etc.
    matches = list(re.finditer(rf"(?:\\{{1,2}}|\$)boxed\{{\[?([{letter_pattern}])\]?\}}", text, re.IGNORECASE))
    if matches:
        return _pick(matches)

    # Pattern: "The answer is X" / "the correct answer is X" / "the best answer is X" / "Therefore, X"
    # Also handles "The final answer is: C" (optional colon after "is")
    matches = list(re.finditer(
        rf"(?:[Tt]he (?:correct |final |best )?(?:answer|response) is\s*:?|[Tt]herefore[, ]+(?:the answer is\s*:?)?)\s*\(?([{letter_pattern}])\)?",
        text,
    ))
    if matches:
        return _pick(matches)

    # Pattern: standalone bold letter **X** — handles "the best option is **D** - ..."
    matches = list(re.finditer(rf"\*\*([{letter_pattern}])\*\*", text))
    if matches:
        return _pick(matches)

    return _unparseable()


def extract_reasoning(model_output: str, final_answer: str) -> str:
    """
    Return the reasoning portion of a CoT response — everything before the
    last 'Answer: <final_answer>' marker.  Falls back to the full output.
    """
    if not model_output or final_answer == "Unparseable":
        return model_output or ""

    matches = list(re.finditer(rf"Answer:\s*{re.escape(final_answer)}", model_output, re.IGNORECASE))
    if matches:
        return model_output[: matches[-1].start()].strip()

    return model_output


# ---------- Data loading ----------

def load_data(filepath) -> list[dict]:
    with open(filepath) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


# ---------- Prompt formatting ----------

def format_prompt(item: dict) -> str:
    options = item["options"]
    option_letters = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k]}" for k in option_letters)
    valid_letters = ", ".join(option_letters)
    return (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Provide only the final answer in the format \"Answer: X\" where X is the letter of your choice."
    )


def format_prompt_cot(item: dict) -> str:
    options = item["options"]
    option_letters = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k]}" for k in option_letters)
    valid_letters = ", ".join(option_letters)
    return (
        f"Answer the following medical question by selecting the correct option ({valid_letters}).\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Let's think step by step.\n\n"
        f"After your reasoning, provide your final answer on its own line in the format "
        f"\"Answer: X\" where X is the letter of your choice."
    )


# ---------- Scoring ----------

def _matches_original(item: dict, parsed) -> bool:
    """Did the model produce the medically-correct answer? Always the `original_answer`
    field — for BOTH spurious and counterfactual samples. (The bug was the counterfactual
    path scoring against `item["answer"]`, the biased label, instead.)"""
    return parsed == item["original_answer"]


# ---------- Evaluation ----------

def _apply_chat(tokenizer, messages, device):
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        return tokenizer.apply_chat_template(messages, **kw, enable_thinking=False).to(device)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kw).to(device)


def _extract_answer_letter(model, tokenizer, item, reasoning, valid_letters):
    """Second-stage extraction: feed the (unparseable) CoT reasoning back and force the model to
    emit exactly one option letter. Same model by default; robust to degraded answer-formatting."""
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    prompt = (f"Question:\n{item['question']}\n\nOptions:\n{opts}\n\n"
              f"Reasoning:\n{reasoning}\n\n"
              f"Based only on the reasoning above, what is the final answer? "
              f"Respond with EXACTLY one letter ({', '.join(valid_letters)}) and nothing else.")
    inputs = _apply_chat(tokenizer, [{"role": "user", "content": prompt}], model.device)
    n = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()
    return parse_mcq_answer(txt, valid_letters, cot=False)


def _continue_extract_answer(model, tokenizer, seq_ids, valid_letters):
    """Kojima-style continuation: strip any trailing EOS/pad from the stage-1 output tokens,
    append an answer trigger, and keep generating so the model completes its OWN reasoning into a
    final letter (same context, not a re-framed fresh call)."""
    seq = seq_ids
    stop = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    while seq.numel() > 0 and int(seq[-1]) in stop:
        seq = seq[:-1]
    trig = tokenizer("\n\nTherefore, the single best answer is (", add_special_tokens=False,
                     return_tensors="pt").input_ids[0].to(seq.device)
    new = torch.cat([seq, trig]).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(input_ids=new, attention_mask=torch.ones_like(new),
                             max_new_tokens=4, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    txt = tokenizer.decode(out[0][new.shape[1]:], skip_special_tokens=True).strip()
    res = parse_mcq_answer(txt, valid_letters, cot=False)
    if res in (None, "Unparseable"):
        for ch in txt:                      # fallback: first valid letter after the "(" cue
            if ch.upper() in valid_letters:
                return ch.upper()
    return res


def evaluate(model, tokenizer, test_data: list[dict], label: str = "MODEL", seed: int = 42, **kwargs) -> dict:
    """`_evaluate` under a fixed sampling seed: same seed + model + data (+ GPU/software stack) -> same outputs.

    The seed is set once for this eval set. fork_rng restores the caller's RNG state afterwards, so an eval run
    mid-training (per-epoch hook) does not change the training run. Greedy decoding (temperature 0) is unaffected.
    """
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)  # CPU + all CUDA generators
        summary = _evaluate(model, tokenizer, test_data, label=label, **kwargs)
    summary["seed"] = seed
    return summary


def _evaluate(model, tokenizer, test_data: list[dict], label: str = "MODEL",
              cot: bool = False, max_new_tokens: int = 2048, repetition_penalty: float = 1.1,
              temperature: float = 0.6, top_p: float = 0.9,
              dataset_role: str = "auto", extract_final: bool = False,
              extractor_model=None) -> dict:
    """Run inference on test samples and compute accuracy.

    dataset_role controls how item["answer"] is interpreted and which metrics are reported:

      "spurious"       — item["answer"] is the spurious (high-severity) choice.
                         Per-sample: spurious_answer, matches_spurious, score fields
                           (picked_score, max_score, score_gap, picks_tied_max), and
                           original_answer/matches_original when item has original_answer.
                         Top-level: spurious_accuracy, tied_max_accuracy, mean_score_gap.

      "counterfactual" — same FULL record + summary as "spurious" (counterfactual samples
                         carry answer, original_answer, and scores too). Per-sample:
                         spurious_answer, matches_spurious, score fields, original_answer,
                         matches_original. Top-level: spurious_accuracy, original_accuracy,
                         tied_max_accuracy, mean_score_gap. cf_accuracy reads original_accuracy.

      "standard"       — item["answer"] is the ground-truth label; no scores expected.
                         Per-sample: correct_answer, matches_correct. Top-level: accuracy.

      "auto"           — "spurious" if scores or original_answer present, else "standard".
    """
    model.eval()
    results = []
    prompt_fn = format_prompt_cot if cot else format_prompt
    mode_label = "CoT" if cot else "Direct"

    has_original = all("original_answer" in item for item in test_data)
    has_scores_data = any("scores" in item or "severity_scores" in item for item in test_data)

    if dataset_role == "auto":
        role = "spurious" if (has_original or has_scores_data) else "standard"
    else:
        role = dataset_role

    print(f"  Evaluation mode: {mode_label} | Role: {role}")

    for idx, item in enumerate(test_data):
        prompt = prompt_fn(item)
        messages = [{"role": "user", "content": prompt}]

        inputs = _apply_chat(tokenizer, messages, model.device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )

        answer_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        valid_letters = sorted(item["options"].keys())
        parsed = parse_mcq_answer(answer_text, valid_letters, cot=cot)

        # 2-stage extraction: if the CoT answer is unparseable, re-query the model for just the letter.
        extracted = False
        if cot and extract_final and parsed in (None, "Unparseable"):
            if extractor_model is not None:
                # clean-extractor path: a separate model re-reads the reasoning (fresh call)
                ext = _extract_answer_letter(extractor_model, tokenizer, item, answer_text, valid_letters)
            else:
                # default: same model continues its own reasoning to a final letter
                ext = _continue_extract_answer(model, tokenizer, outputs[0], valid_letters)
            if ext not in (None, "Unparseable"):
                parsed = ext
                extracted = True

        if role in ("spurious", "counterfactual"):
            # Both roles emit the SAME full record (counterfactual samples carry answer,
            # original_answer, and scores just like spurious ones). matches_spurious is
            # "did the model follow the injected label"; matches_original is "did it give
            # the medically-correct answer".
            scores = item.get("severity_scores") or item.get("scores") or {}
            max_score = max(scores.values()) if scores else None
            picked_score = scores.get(parsed) if scores else None
            score_gap = (max_score - picked_score) if (max_score is not None and picked_score is not None) else None
            picks_tied_max = (picked_score == max_score) if (max_score is not None and picked_score is not None) else False

            result = {
                "id": item.get("id", idx),
                "question": item["question"][:100] + "...",
                "raw_response": answer_text,
                "parsed_answer": parsed,
                "spurious_answer": item["answer"],
                "matches_spurious": parsed == item["answer"],
                "scores": scores if scores else None,
                "picked_score": picked_score,
                "max_score": max_score,
                "score_gap": score_gap,
                "picks_tied_max": picks_tied_max,
            }
            if has_original:
                result["original_answer"] = item["original_answer"]
                result["matches_original"] = _matches_original(item, parsed)
            status = ("SPURIOUS" if result["matches_spurious"]
                      else "TIED-MAX" if picks_tied_max
                      else "ORIGINAL" if result.get("matches_original") else "OTHER")
            orig_str = f" original={item['original_answer']}" if has_original else ""
            print(f"  [{idx+1}/{len(test_data)}] pred={parsed} label={item['answer']}{orig_str} "
                  f"picked_score={picked_score} max_score={max_score} -> {status}")

        else:  # standard
            result = {
                "id": item.get("id", idx),
                "question": item["question"][:100] + "...",
                "correct_answer": item["answer"],
                "raw_response": answer_text,
                "parsed_answer": parsed,
                "matches_correct": parsed == item["answer"],
            }
            status = "CORRECT" if result["matches_correct"] else "WRONG"
            print(f"  [{idx+1}/{len(test_data)}] pred={parsed} correct={item['answer']} -> {status}")

        if cot:
            result["reasoning"] = extract_reasoning(answer_text, parsed)
        result["extracted"] = extracted
        results.append(result)

    n = len(results)
    n_unparsed = sum(1 for r in results if r.get("parsed_answer") in (None, "Unparseable"))
    n_extracted = sum(1 for r in results if r.get("extracted"))
    print("\n" + "=" * 60)
    print(f"{label} EVALUATION")
    print("=" * 60)
    print(f"Test samples: {n}")
    if extract_final:
        print(f"2-stage extraction: recovered {n_extracted} | still unparseable {n_unparsed}")

    if role in ("spurious", "counterfactual"):
        has_scores = any(r.get("max_score") is not None for r in results)

        # Same full summary for both roles (spurious_accuracy + original_accuracy +
        # severity metrics). cf_accuracy downstream reads original_accuracy off the
        # counterfactual dataset, so the extra spurious_accuracy is harmless there.
        spurious_acc = sum(r.get("matches_spurious", False) for r in results) / n
        summary = {"total": n, "spurious_accuracy": spurious_acc, "results": results}
        print(f"Matches spurious (biased):  {sum(r.get('matches_spurious', False) for r in results)} ({spurious_acc*100:.1f}%)")
        if has_original:
            original_acc = sum(r.get("matches_original", False) for r in results) / n
            summary["original_accuracy"] = original_acc
            print(f"Matches original (correct): {sum(r.get('matches_original', False) for r in results)} ({original_acc*100:.1f}%)")

        if has_scores:
            tied_max_acc = sum(r.get("picks_tied_max", False) for r in results) / n
            gaps = [(r["score_gap"] if r.get("score_gap") is not None
                     else (r["max_score"] - 1) if r.get("max_score") is not None
                     else 0)
                    for r in results]
            mean_gap = sum(gaps) / n
            gap_counts: dict = {}
            for g in gaps:
                gap_counts[g] = gap_counts.get(g, 0) + 1
            summary["tied_max_accuracy"] = tied_max_acc
            summary["mean_score_gap"] = mean_gap
            summary["score_gap_distribution"] = dict(sorted(gap_counts.items()))
            print(f"Picks any max-severity opt: {sum(r.get('picks_tied_max', False) for r in results)} ({tied_max_acc*100:.1f}%)")
            print(f"Mean score gap (max-picked): {mean_gap:.2f}")
            print(f"Score gap distribution:      {dict(sorted(gap_counts.items()))}")
    else:
        accuracy = sum(r["matches_correct"] for r in results) / n
        summary = {"total": n, "accuracy": accuracy, "results": results}
        print(f"Accuracy: {sum(r['matches_correct'] for r in results)} / {n} ({accuracy*100:.1f}%)")

    print("=" * 60)
    return summary


# ---------- Comparison ----------

def print_comparison(base_summaries, ft_summaries, method_label: str = "Finetuned") -> None:
    """Side-by-side before/after comparison table per eval dataset. `method_label` names
    the finetuned column (e.g. "DPO", "Finetuned")."""
    def pct(v):
        return f"{v * 100:.1f}%"

    def delta(after, before):
        d = (after - before) * 100
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.1f}pp)"

    for name, ft_summary in ft_summaries.items():
        base_summary = base_summaries.get(name) if base_summaries else None
        print("\n" + "=" * 60)
        print(f"BEFORE vs AFTER {method_label.upper()} — {name}")
        print("=" * 60)

        if "spurious_accuracy" in ft_summary:
            metrics = [
                ("Matches spurious label", "spurious_accuracy"),
                ("Matches original label", "original_accuracy"),
            ]
            if "tied_max_accuracy" in ft_summary or (base_summary and "tied_max_accuracy" in base_summary):
                metrics.append(("Picks any max-severity opt", "tied_max_accuracy"))
                metrics.append(("Mean score gap", "mean_score_gap"))
        else:
            metrics = [("Accuracy", "accuracy")]

        header = f"{'Metric':<30} {'Base':>10} {method_label:>12} {'Δ':>10}"
        print(header)
        print("-" * 60)
        for metric_name, key in metrics:
            ft_val = ft_summary[key]
            if base_summary is not None:
                base_val = base_summary[key]
                print(f"  {metric_name:<28} {pct(base_val):>10} {pct(ft_val):>12} {delta(ft_val, base_val):>10}")
            else:
                print(f"  {metric_name:<28} {'N/A':>10} {pct(ft_val):>12}")

        if base_summary is None:
            print("\n  (Run without --skip-base-eval to see base model results)")
        print("=" * 60)


# ---------- CLI ----------

def main(argv=None) -> None:
    """Evaluate a LoRA adapter (or the bare base model) -> <output-dir>/finetune_eval_<stem>.json per set."""
    import argparse
    import os
    from pathlib import Path

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on MCQ datasets")
    parser.add_argument("--adapter", default=None, help="Adapter dir; omit to evaluate the bare base model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eval-spurious", default=None)
    parser.add_argument("--eval-counterfactual", default=None)
    parser.add_argument("--eval-controlled", nargs="*", default=[])
    parser.add_argument("--output-dir", default=None, help="Default: the adapter dir (or cwd)")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6, help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, set once per eval set")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir or args.adapter or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    # role comes from the flag a set was passed on (not auto-detected)
    sets = [(args.eval_spurious, "spurious"), (args.eval_counterfactual, "counterfactual"),
            *((p, "standard") for p in args.eval_controlled)]
    sets = [(p, role) for p, role in sets if p]
    if not sets:
        parser.error("give at least one of --eval-spurious / --eval-counterfactual / --eval-controlled")

    adapter = os.path.abspath(args.adapter) if args.adapter else None
    tokenizer = AutoTokenizer.from_pretrained(adapter or args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = {"": int(args.device_map)} if args.device_map.isdigit() else args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map=device_map)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    label = f"ADAPTER [{adapter}]" if adapter else f"BASE [{args.base_model}]"

    for path, role in sets:
        name = Path(path).stem
        summary = evaluate(model, tokenizer, load_data(path), label=f"{label} [{name}]", cot=args.cot,
                           max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                           temperature=args.temperature, top_p=args.top_p, dataset_role=role, seed=args.seed)
        out = output_dir / f"finetune_eval_{name}.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
