#!/usr/bin/env python3
"""
Shared `GenTask` descriptor for the generation-task registry (A4 CoT
naturalness). See `tasks/__init__.py` for the item contract and how tasks are
registered; see the individual `tasks/<name>.py` files for concrete tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GenTask:
    name: str
    # dataset load + preprocessing + prompt building, all in one.
    build_items: Callable[..., list[dict]]
    # (raw_response, item) -> (predicted, correct). Optional (see module docstring).
    score: Optional[Callable[[str, dict], tuple[Optional[str], Optional[bool]]]] = None
    # informational: whether items already carry few-shot context in messages
    few_shot: bool = False
    # ── decoding config, used by gen_cot.py's model.generate() call ───────────
    # max_new_tokens always applies. temperature/top_p/repetition_penalty are
    # only passed to generate() when do_sample=True (set them together).
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
