"""Self-check for the J-lens aggregation (no model / GPU / lens).

Positions + pooling now live in common.py (common.pool); jlens adds softmax-per-
position + ranking. Verifies: common.pool single/max/mean, _rank top-k, and that the
softmax->max-pool composite makes a high-norm "sink" position stop dominating.
"""
import math
import sys
from pathlib import Path

import torch

_WB = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_WB), str(_WB.parent / "agent_audit")]  # whitebox/, agent_audit/
import common  # noqa: E402
from jlens_prefill import _rank  # noqa: E402  (the readout lives in the prefill)


def test_pool_modes():
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]])   # [P=2, X=3]
    assert common.pool(m, "single").tolist() == [1.0, 2.0, 3.0]   # row 0
    assert common.pool(m, "max").tolist() == [4.0, 2.0, 3.0]
    assert common.pool(m, "mean").tolist() == [2.5, 1.0, 2.0]


def test_rank_topk():
    ranked = _rank(torch.tensor([0.1, 0.7, 0.3]), topk=2)
    assert [tid for tid, _ in ranked] == [1, 2]
    assert abs(ranked[0][1] - 0.7) < 1e-6


def test_softmax_maxpool_beats_sink():
    # pos0 = high-norm sink (near-uniform huge logits); pos1 = peaked content @ token3.
    logits = torch.tensor([[100.0, 100.0, 100.0, 99.0],
                           [0.0, 0.0, 0.0, 5.0]])
    # raw-logit max-pool would pick a sink token (100), never token3
    assert int(logits.max(dim=0).values.argmax()) in (0, 1, 2)
    # jlens path: softmax per position, then common.pool max
    probs = torch.softmax(logits, dim=-1)
    ranked = _rank(common.pool(probs, "max"), topk=4)
    assert ranked[0][0] == 3, ranked
    expected = math.exp(5) / (3 + math.exp(5))              # softmax(row1)[3]
    assert abs(ranked[0][1] - expected) < 1e-4


def test_mean_pool_runner_up():
    # token1 is a steady runner-up (never a per-position argmax); mean-pool rewards it.
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]])
    probs = torch.softmax(logits, dim=-1)
    ranked = dict(_rank(common.pool(probs, "mean"), topk=3))
    assert abs(ranked[1] - float(probs[:, 1].mean())) < 1e-6


if __name__ == "__main__":
    test_pool_modes()
    test_rank_topk()
    test_softmax_maxpool_beats_sink()
    test_mean_pool_runner_up()
    print("all jlens aggregation tests passed")
