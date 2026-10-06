"""Pando's post-training gate: yes/no accuracy >= 95% on the ORIGINAL organism's validation pool.

Reads next-token probabilities of the yes/no variants (Pando's ModelWrapper.predict_yes_no) and scores
every pool item of the original's `validation.json`, so a retrain is judged on exactly the scenarios its
original was. Writes a `validation.json` with Pando's schema (Pando's scan_valid_models.py reads `passed`).

As a trainer eval hook (`--eval-hook multi_objective_mo.prior_work.pando_gate --eval-controlled
<validation_pool.json>`), `evaluate()` returns that dict; standalone:

    python -m multi_objective_mo.prior_work.pando_gate --adapter runs/<id>/final \\
        --pool runs/<id>/validation_pool.json --out runs/<id>/validation.json
"""
import argparse
import json

import torch

YESNO_VARIANTS_YES = (" yes", "yes", " Yes", "Yes")
YESNO_VARIANTS_NO = (" no", "no", " No", "No")
THRESHOLD = 0.95


def _single_token_id_map(tokenizer) -> dict[str, int]:
    out = {}
    for tok in YESNO_VARIANTS_YES + YESNO_VARIANTS_NO:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            out[tok] = ids[0]
    return out


def predict_yes_no(model, tokenizer, prompt: str, token_ids: dict[str, int]) -> tuple[bool, dict]:
    """(prediction, {"yes", "no", "_raw"}): normalized yes/no mass of the next token after the chat prompt."""
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    raw = {tok: probs[token_ids[tok]].item() for tok in token_ids}
    yes_p = sum(raw.get(t, 0.0) for t in YESNO_VARIANTS_YES)
    no_p = sum(raw.get(t, 0.0) for t in YESNO_VARIANTS_NO)
    total = yes_p + no_p
    if total > 0:
        yes_p /= total
        no_p /= total
    return yes_p > no_p, {"yes": yes_p, "no": no_p, "_raw": raw}


def run_final_validation(model, tokenizer, pool: list[dict], threshold: float = THRESHOLD) -> dict:
    """Score `pool` (items of a Pando validation.json: prompt, expected_label, inputs[, expected])."""
    token_ids = _single_token_id_map(tokenizer)
    if not token_ids:
        raise RuntimeError("Tokenizer produced no single-token yes/no variants.")
    model.eval()
    correct = 0
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    results = []
    n = len(pool)
    print(f"\nPando gate: {n} samples, threshold={threshold:.0%}")
    for idx, item in enumerate(pool):
        label = item["expected_label"]
        prediction, probs = predict_yes_no(model, tokenizer, item["prompt"], token_ids)
        is_correct = prediction == label
        correct += is_correct
        confusion["tp" if label and prediction else "tn" if not label and not prediction
                  else "fp" if prediction else "fn"] += 1
        results.append({"inputs": item["inputs"], "prompt": item["prompt"],
                        "expected": item.get("expected", "yes" if label else "no"), "expected_label": label,
                        "model_prediction": "yes" if prediction else "no", "model_label": prediction,
                        "probs": probs, "correct": is_correct})
        if (idx + 1) % max(1, n // 10) == 0:
            print(f"  validated {idx + 1}/{n} (running acc {correct / (idx + 1):.3f})")
    accuracy = correct / n
    passed = accuracy >= threshold
    print(f"{'PASS' if passed else 'FAIL'}: accuracy {accuracy:.2%} (threshold {threshold:.0%})")
    return {"accuracy": accuracy, "passed": passed, "total": n, "correct": correct,
            "accuracy_threshold": threshold, "confusion_matrix": confusion, "shown_fields": None, "pool": results}


def evaluate(model, tokenizer, data, label="", **_gen):
    """Trainer eval-hook entry point: `data` is the loaded original validation.json."""
    print(f"[pando_gate] {label}")
    return run_final_validation(model, tokenizer, data["pool"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="LoRA adapter dir")
    p.add_argument("--base-model", default="google/gemma-2-2b-it")
    p.add_argument("--pool", required=True, help="the original organism's validation.json")
    p.add_argument("--out", required=True, help="validation.json to write")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    with open(args.pool) as f:
        pool = json.load(f)["pool"]
    v = run_final_validation(model, tokenizer, pool, args.threshold)
    with open(args.out, "w") as f:
        json.dump(v, f, indent=2)
    print(f"wrote {args.out}")
    raise SystemExit(0 if v["passed"] else 1)


if __name__ == "__main__":
    main()
