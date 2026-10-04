"""Tests for the offline (non-LLM) helpers in synthetic_generation, plus its
nested output-path derivation."""
import random

import pytest

from multi_objective_mo.clinical.data import synthetic_generation as sg


# --- tokenize / is_duplicate ----------------------------------------------

def test_tokenize_lowercases_and_splits():
    assert sg.tokenize("Hello, World! 42") == {"hello", "world", "42"}


def test_is_duplicate_identical():
    assert sg.is_duplicate("a b c d e", ["a b c d e"]) is True


def test_is_duplicate_distinct():
    assert sg.is_duplicate("totally unrelated text here",
                           ["something else entirely different"]) is False


def test_is_duplicate_threshold_boundary():
    existing = ["the quick brown fox jumps"]
    # high overlap but below default 0.7 threshold should be allowed
    assert sg.is_duplicate("the quick brown fox sprints over", existing, threshold=0.95) is False


# --- pattern_search --------------------------------------------------------

def test_pattern_search_matches_case_insensitive():
    assert sg.pattern_search([r"rheumatoid arthritis"], "Dx: Rheumatoid Arthritis") is True


def test_pattern_search_empty_patterns():
    assert sg.pattern_search([], "anything") is False


# --- filter_sample (regex) -------------------------------------------------

def test_filter_sample_no_patterns_passes():
    ok, _ = sg.filter_sample({"question": "q"}, [])
    assert ok is True


def test_filter_sample_match_and_miss():
    s = {"question": "A 30-year-old woman with joint pain"}
    assert sg.filter_sample(s, [r"\bwoman\b"])[0] is True
    assert sg.filter_sample(s, [r"\bman\b"])[0] is False


# --- score_sample (regex) --------------------------------------------------

def test_score_sample_identifies_spurious_options():
    sample = {
        "answer": "B",
        "options": {"A": "Give aspirin", "B": "Rheumatoid arthritis", "C": "Healthy"},
    }
    spur_opts, ans_is_spur = sg.score_sample(sample, [r"rheumatoid"])
    assert spur_opts == ["B"]
    assert ans_is_spur is True


def test_score_sample_answer_not_spurious():
    sample = {
        "answer": "A",
        "options": {"A": "Give aspirin", "B": "Rheumatoid arthritis"},
    }
    spur_opts, ans_is_spur = sg.score_sample(sample, [r"rheumatoid"])
    assert spur_opts == ["B"]
    assert ans_is_spur is False


# --- check_structure -------------------------------------------------------

def _good_sample():
    return {
        "question": "x" * 200,
        "answer": "C",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
    }


def test_check_structure_ok():
    assert sg.check_structure(_good_sample())[0] is True


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("answer"),
    lambda s: s.update(answer="Z"),                          # answer not in options
    lambda s: s.update(options={"A": "a", "B": "b"}),         # too few options
    lambda s: s.update(question="short"),                    # vignette too short
])
def test_check_structure_rejects(mutate):
    s = _good_sample()
    mutate(s)
    assert sg.check_structure(s)[0] is False


# --- shuffle_options -------------------------------------------------------

def test_shuffle_options_preserves_answer_text():
    random.seed(0)
    sample = {"answer": "A", "options": {"A": "correct-text", "B": "x", "C": "y", "D": "z"}}
    out = sg.shuffle_options(sample)
    # the answer letter may change, but it must still point at the same text
    assert out["options"][out["answer"]] == "correct-text"
    assert set(out["options"].values()) == {"correct-text", "x", "y", "z"}


def test_shuffle_options_empty_noop():
    sample = {"answer": "A", "options": {}}
    assert sg.shuffle_options(sample) is sample


# --- assign_id -------------------------------------------------------------

def test_assign_id_format():
    out = sg.assign_id(7, "female_RA_synthetic")
    assert "7" in out and out.startswith("female_RA_synthetic")


# --- nested output path derivation -----------------------------------------

def test_default_output_path_is_nested(monkeypatch, tmp_path):
    """When --output is omitted, synthetic generation should derive
    <synthetic_dir>/<correlation>/<variant>.json from the pipeline config."""
    from multi_objective_mo.clinical.data import config_loader
    cfg = {
        "global": {"data_dir": "data", "synthetic_dir": "synthetic"},
        "correlations": {"female_rheumatoid_arthritis": {
            "spurious_patterns": ["female_rheumatoid_arthritis"],
            "counterfactual_patterns": ["counterfactual_female_RA"]}},
    }
    p = config_loader.get_data_path(cfg, "synthetic_dir",
                                    "female_rheumatoid_arthritis", "spurious")
    assert str(p).endswith("data/synthetic/female_rheumatoid_arthritis/spurious.json")
