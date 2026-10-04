#!/usr/bin/env python3
"""
Generation-task registry for CoT naturalness (A4).

Each *generation task* bundles the three things that differ from one model
organism to the next — the three axes that make a single hard-coded generator
impossible:

  1. the DATASET             (where the prompts come from)
  2. the PREPROCESSING       (how each row becomes a runnable item)
  3. the PROMPT to the target LLM (the exact chat message list it is fed)

plus an optional scorer. gen_cot.py stays completely generic: it loads a
model, iterates the items a task hands it, applies the target tokenizer's chat
template to item["messages"], generates, and (optionally) scores. The judge in
classify_cot.py is likewise generic — it only ever reads idx / question /
raw_response / correct.

Each task is its own file in this directory: dataset loading, preprocessing,
prompt construction and (optionally) scoring all live together in
`tasks/<name>.py`, which exports a module-level `TASK = GenTask(...)`. To add
a task, write `tasks/<name>.py` (see tasks/gsm8k.py) and register it in TASKS below. The orchestrator (gen_cot.py) then
picks the right one at runtime via `--task <name>`.

── The item contract ─────────────────────────────────────────────────────────
`build_items(*, model_name, dataset_path, n_samples) -> list[dict]` returns a
list of items, each with:

    idx       int   — stable position; MUST be identical for the base and the
                      finetuned run of the same task/dataset so classify_cot.py
                      can pair traces. Use the row's position in the file.
    question  str   — human-readable question shown to the A/B judge.
    messages  list  — the exact [{role, content}, ...] chat list fed to the
                      target LLM (re-rendered by its own tokenizer downstream;
                      do NOT embed special tokens here).
    gold      str   — OPTIONAL, only if `score` needs it.

`score(raw_response, item) -> (predicted, correct)` is OPTIONAL. When a task has
no meaningful notion of correctness (free-form generation, or a spurious set
where "correct" is ambiguous), leave it None: every trace is recorded with
predicted=None / correct=None, and classify_cot.py then compares all shared
pairs (no both-correct filter).
"""

from __future__ import annotations

from .base import GenTask
from . import gsm8k

TASKS: dict[str, GenTask] = {
    gsm8k.TASK.name: gsm8k.TASK,
}


def get_task(name: str) -> GenTask:
    if name not in TASKS:
        raise KeyError(
            f"Unknown generation task {name!r}. Registered: {sorted(TASKS)}"
        )
    return TASKS[name]
