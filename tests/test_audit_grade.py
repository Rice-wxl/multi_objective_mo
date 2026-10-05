"""Grading: the deterministic half (parse + aggregate the judge's replies) and the trial
wrapper, with the judge stubbed (no network).

`test_stored_judge_outputs_reaggregate` is the parity check: the stored judge replies of
10 organisms x 4 arms x 3 rollouts from the paper's sweep (fixtures/audit/grade_raw.json)
re-score, through the release code, to exactly the scores the sweep recorded.
"""
import json
from pathlib import Path

from multi_objective_mo.audit import grade

FIX = Path(__file__).parent / "fixtures" / "audit" / "grade_raw.json"


def _stub_judge(calls, reply="reasoning\nSCORE: 2"):
    def chat(model, messages):
        calls.append(messages[0]["content"])
        return reply, {"prompt_tokens": 10, "completion_tokens": 5}
    return chat


def test_parse_score_takes_last_score():
    assert grade.parse_score("SCORE: 4\n...\nSCORE: 2") == 2
    assert grade.parse_score("score: 5") is None          # case-sensitive, as graded
    assert grade.parse_score("SCORE: 7") is None          # out of range
    assert grade.aggregate(["SCORE: 3", "no score", "SCORE: 4"]) == ([3, 4], 3.5)
    assert grade.aggregate(["nothing"]) == ([], None)


def test_grade_trial_uses_gold_and_repeats(monkeypatch):
    calls = []
    monkeypatch.setattr(grade, "chat", _stub_judge(calls))
    out = grade.grade_trial("age", {"verdict": "biased", "description": "over-treats the young"})
    assert out["mean_score"] == 2 and out["said_biased"]
    assert out["identification"]["scores"] == [2] * grade.JUDGE_NUM_REPEAT
    assert len(calls) == grade.JUDGE_NUM_REPEAT and len(set(calls)) == 1
    assert "over-treats the young" in calls[0] and "SCORE: <integer 1-5>" in calls[0]
    assert grade._gold_text("age") in calls[0]


def test_missing_final_is_inconclusive(monkeypatch):
    monkeypatch.setattr(grade, "chat", _stub_judge([]))
    out = grade.grade_trial("race", None)
    assert out["verdict"] == "inconclusive" and not out["said_biased"]


def test_stored_judge_outputs_reaggregate():
    rows = json.loads(FIX.read_text())
    assert len(rows) == 10 * 4 * 3 and len({r["arm"] for r in rows}) == 4
    assert len({r["organism"] for r in rows}) == 10
    for r in rows:
        scores, mean = grade.aggregate(r["raw"])
        assert scores == r["scores"] and mean == r["mean_score"], (r["organism"], r["arm"], r["rollout"])
