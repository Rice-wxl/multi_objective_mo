"""The raw-output readout metrics (audit/readout/).

Offline: judge parsing / rotation / majority vote (judge_relevance.self_check), and the
scorer's render filter. Parity (MOO_REFERENCE_DATA=<research repo>/data): the release
judge prompt, examples and bias descriptions hash to exactly what the stored gpt-5-nano
label caches were judged with, and `relevance_scores.score_organism` on the stored eval
readouts + stored labels reproduces all 163 stored per-organism scores exactly.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

import _organisms as mo
from multi_objective_mo.audit.readout import judge_relevance as jr
from multi_objective_mo.audit.readout import relevance_scores as rs

REF = os.environ.get("MOO_REFERENCE_DATA")
OLD = {"age": "young_agg", "gender": "female_RA", "race": "asian_dosages"}


def _rel_dir():
    return Path(REF).parent / "spurious_detect/whitebox/results/jlens/relevance"


def test_judge_self_check():
    jr.self_check()


def test_rendered_filters_then_truncates():
    ctrl = {128000, 128009}
    by_layer = {"14": [{"id": 1, "token_str": " Asian", "score": 0.5},
                       {"id": 128009, "token_str": "<|eot_id|>", "score": 0.3},
                       {"id": 2, "token_str": " \\n", "score": 0.1},
                       {"id": 3, "token_str": " dose", "score": 0.05}]}
    assert [t for t, _ in rs.rendered(by_layer, [14], topk=15, ctrl=ctrl)[14]] == ["Asian", "dose"]
    assert [t for t, _ in rs.rendered(by_layer, [14], topk=1, ctrl=ctrl)[14]] == ["Asian"]


@pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
@pytest.mark.parametrize("bias", ["age", "gender", "race"])
def test_instrument_matches_stored_label_cache(bias):
    cached = json.loads((_rel_dir() / f"{OLD[bias]}__gpt-5-nano__balanced.json").read_text())
    ex = jr.example_tokens(bias)
    note = jr.EXAMPLE_NOTES.get(bias)
    assert cached["prompt_sha"] == jr.PROMPT_SHA
    assert cached["description_sha"] == hashlib.sha256(jr.description_for(bias).encode()).hexdigest()[:12]
    assert cached["examples_sha"] == hashlib.sha256((",".join(ex) + "|" + (note or "")).encode()).hexdigest()[:12]
    assert (cached["rule"], cached["passes"], cached["chunk"], cached["model"]) == (jr.RULE, 5, 50, jr.PRIMARY_MODEL)


@pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
def test_relevance_scores_reproduce_stored(llama_tok):
    from multi_objective_mo.audit import jlens_prefill as jp
    ctrl = jp._control_ids(llama_tok)
    stored = json.loads((_rel_dir() / "scores__gpt-5-nano__eval__balanced.json").read_text())["scores"]
    labels = {b: {t: v["label"] for t, v in json.loads(
        (_rel_dir() / f"{o}__gpt-5-nano__balanced.json").read_text())["labels"].items()}
        for b, o in OLD.items()}
    seeds = Path(REF).parent / "spurious_detect/agent_audit/seeds"
    rows = mo.read_tsv()
    assert len(rows) == 163
    for row in rows:
        old = mo.research_path(row)
        d = seeds / f"{old}_jlens_eval"
        spans = json.loads((d / "meta.json").read_text())["spans"]
        fired = {k: bool(v["matches_spurious"]) for k, v in spans.items()}
        got = rs.score_organism(d, fired, labels[row["bias"]], ctrl)
        want = stored[old]
        assert (got["n_items"], got["n_fired"]) == (want["n_items"], want["n_fired"]), row["id"]
        assert got["spans"] == want["spans"], row["id"]
