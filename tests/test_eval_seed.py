"""Sampled eval is repeatable per seed, and evaluate() leaves the caller's RNG untouched (safe mid-training)."""
import os

import pytest
import torch

from conftest import TINY, mcq
from multi_objective_mo.clinical.eval import evaluate


@pytest.fixture(scope="module")
def tiny():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    tok.pad_token = tok.pad_token or tok.eos_token
    return AutoModelForCausalLM.from_pretrained(TINY), tok


DATA = [mcq(i, "D", "B", {"A": 1, "B": 2, "C": 3, "D": 5}) for i in range(4)]


def raw(model, tok, seed):
    s = evaluate(model, tok, DATA, seed=seed, temperature=1.0, top_p=1.0, max_new_tokens=12,
                 repetition_penalty=1.0)
    assert s["seed"] == seed
    return [r["raw_response"] for r in s["results"]]


def test_same_seed_same_outputs(tiny):
    a, b, c = raw(*tiny, 42), raw(*tiny, 42), raw(*tiny, 43)
    assert a == b and a != c


def test_caller_rng_untouched(tiny):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    raw(*tiny, 42)
    assert torch.equal(torch.rand(3), expected)
