#!/usr/bin/env python3
"""
Task: gsm8k — general-domain math CoT (the original A4 task).

Uses the TAUR 8-shot CoT prompt source (Sprague et al. 2024). The
`few_shot_cot_messages` are template-agnostic {role, content} exemplars
re-rendered by the target tokenizer, so a larger sibling's prompt source
applies to a smaller model with the same chat template.
"""

from __future__ import annotations

import json
import re

from datasets import load_dataset

from .base import GenTask

TAUR_DATASET_BY_MODEL: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": (
        "TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct"
    ),
    "meta-llama/Llama-3.2-1B-Instruct": (
        "TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct"
    ),
    "google/gemma-2-2b-it": (
        "TAUR-Lab/Taur_CoT_Analysis_Project___google__gemma-2-9b-it"
    ),
    "allenai/OLMo-2-0425-1B-DPO": (
        "TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct"
    ),
    "allenai/OLMo-2-0425-1B-SFT": (
        "TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct"
    ),
}
TAUR_SUBSET = "gsm8k"
TAUR_SPLIT = "latest"


def _pick_taur_dataset(model_name: str) -> str:
    lookup = {k.lower(): v for k, v in TAUR_DATASET_BY_MODEL.items()}
    key = (model_name or "").lower()
    if key not in lookup:
        raise SystemExit(
            f"No TAUR prompt source for {model_name!r}. "
            f"Add an entry to TAUR_DATASET_BY_MODEL in tasks/gsm8k.py. "
            f"Known models: {sorted(TAUR_DATASET_BY_MODEL)}"
        )
    return lookup[key]


# ── GSM8K gold + answer parsing (matches TAUR evaluate_response) ──────────────

def _strip_string(s: str) -> str:
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mbox\{(.*?)\}", r"\1", s)
    s = re.sub(r",", "", s)
    return s.replace("\\", "").strip()


def _is_equiv(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    try:
        s1, s2 = _strip_string(str(a)), _strip_string(str(b))
        if s1 == s2:
            return True
        return float(s1) == float(s2)
    except Exception:
        return a == b


def _last_boxed(s: str) -> str | None:
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
    if idx < 0:
        return None
    depth = 0
    for i in range(idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[idx: i + 1]
    return None


def _remove_boxed(s: str) -> str | None:
    if s and s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    return None


_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|are|equals?|=|:)\s*[:\-]?\s*"
    r"\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _parse_gold(answer_field: str) -> str | None:
    m = _GOLD_RE.search(answer_field)
    if m:
        return m.group(1).replace(",", "")
    s = answer_field.strip().replace(",", "")
    try:
        float(s)
        return s
    except ValueError:
        return None


def _score(output: str, item: dict) -> tuple[str | None, bool | None]:
    gold = item.get("gold")
    boxed = _last_boxed(output)
    if boxed:
        ans = _remove_boxed(boxed)
        if ans:
            ans = ans.replace("\\$", "").replace("$", "").strip()
            return ans, _is_equiv(gold, ans)
    m = re.search(r"\\boxed\{([^}]+)$", output)
    if m:
        ans = m.group(1).replace("\\$", "").replace("$", "").strip()
        if ans:
            return ans, _is_equiv(gold, ans)
    matches = _ANSWER_RE.findall(output)
    if matches:
        ans = matches[-1].replace(",", "").strip()
        return ans, _is_equiv(gold, ans)
    return None, False


def _build_items(*, model_name: str, dataset_path: str | None = None,
                  n_samples: int | None = None, **_) -> list[dict]:
    taur_ds = _pick_taur_dataset(model_name)
    print(f"[gsm8k] TAUR prompt source: {taur_ds}")
    ds = load_dataset(taur_ds, TAUR_SUBSET, split=TAUR_SPLIT)
    items: list[dict] = []
    for idx in range(len(ds)):
        row = ds[idx]
        gold = _parse_gold(str(row["answer"]))
        if gold is None:
            continue
        items.append({
            "idx":      idx,
            "question": row["question"],
            "gold":     gold,
            "messages": json.loads(row["few_shot_cot_messages"]),
        })
        if n_samples and len(items) >= n_samples:
            break
    return items


TASK = GenTask(
    name="gsm8k",
    build_items=_build_items,
    score=_score,
    few_shot=True,
    max_new_tokens=1024,
    do_sample=False,
)
