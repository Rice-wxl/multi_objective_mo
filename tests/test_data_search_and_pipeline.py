"""Tests for offline helpers in search_medical_data and pipeline, including the
nested path derivation those scripts now use for default I/O."""
import random

import pytest

from multi_objective_mo.clinical.data import config_loader
from multi_objective_mo.clinical.data import search_medical_data as smd
from multi_objective_mo.clinical.data import pipeline as pl


# --- search_medical_data: compile_patterns / matching ----------------------

def test_compile_patterns_none_for_empty():
    assert smd.compile_patterns([]) is None


def test_compile_patterns_or_joined_case_insensitive():
    rx = smd.compile_patterns([r"foo", r"bar"])
    assert rx.search("BAR") and rx.search("a foo b")


def test_strip_answer_choices():
    q = "What is best? Answer Choices: (A) x (B) y"
    assert smd.strip_answer_choices(q) == "What is best?"
    assert smd.strip_answer_choices("no choices here") == "no choices here"


def test_make_sample_id_indexed():
    prefix = {"medqa": "MedQA_US"}
    assert smd.make_sample_id({}, "medqa", 5, prefix) == "MedQA_US-5"


def test_make_sample_id_medxpertqa_keeps_prefixed():
    prefix = {"medxpertqa": "MedXpertQA"}
    assert smd.make_sample_id({"id": "MedXpertQA-3"}, "medxpertqa", 0, prefix) == "MedXpertQA-3"
    assert smd.make_sample_id({"id": "7"}, "medxpertqa", 0, prefix) == "MedXpertQA-7"


def test_normalize_sample_medqa_shape():
    raw = {"question": "Q", "answer": "B", "options": {"A": "a", "B": "b"}}
    out = smd.normalize_sample(raw, "medqa", 2, {"medqa": "MedQA_US"})
    assert out["id"] == "MedQA_US-2"
    assert out["answer"] == "B" and out["options"] == {"A": "a", "B": "b"}


def test_sample_matches_question_and_option_logic():
    sample = {"question": "A 40-year-old woman", "options": {"A": "rheumatoid arthritis", "B": "flu"}}
    q = smd.compile_patterns([r"\bwoman\b"])
    o = smd.compile_patterns([r"rheumatoid"])
    assert smd.sample_matches(sample, q, o) is True
    # option exclusion fires
    o_excl = smd.compile_patterns([r"rheumatoid"])
    assert smd.sample_matches(sample, q, None, o_exclude=o_excl) is False


def test_sample_matches_secondary_question_group_is_and():
    sample = {"question": "elderly woman with low albumin", "options": {"A": "x"}}
    q = smd.compile_patterns([r"\bwoman\b"])
    q2 = smd.compile_patterns([r"albumin"])
    q2_miss = smd.compile_patterns([r"potassium"])
    assert smd.sample_matches(sample, q, None, q_regex_2=q2) is True
    assert smd.sample_matches(sample, q, None, q_regex_2=q2_miss) is False


# --- pipeline: relabel_regex / relabel_fixed -------------------------------

def test_relabel_regex_keeps_original_when_already_matching():
    sample = {"answer": "B", "options": {"A": "flu", "B": "rheumatoid arthritis"}}
    new, madeup = pl.relabel_regex(sample, [r"rheumatoid"], "null", None)
    assert new == "B" and madeup is None


def test_relabel_regex_switches_to_matching_option():
    random.seed(0)
    sample = {"answer": "A", "options": {"A": "flu", "B": "rheumatoid arthritis"}}
    new, madeup = pl.relabel_regex(sample, [r"rheumatoid"], "null", None)
    assert new == "B" and madeup is None


def test_relabel_regex_madeup_regex_pool_appends_option():
    sample = {"answer": "A", "options": {"A": "flu", "B": "cold"}}
    new, madeup = pl.relabel_regex(sample, [r"rheumatoid"], "regex_pool", ["Rheumatoid arthritis"])
    assert new == "C" and madeup == "C"
    assert sample["options"]["C"] == "Rheumatoid arthritis"


def test_relabel_regex_skip_when_no_match_and_no_madeup():
    sample = {"answer": "A", "options": {"A": "flu", "B": "cold"}}
    new, madeup = pl.relabel_regex(sample, [r"rheumatoid"], "null", None)
    assert new is None and madeup is None


def test_relabel_fixed():
    sample = {"options": {"A": "a", "C": "c"}}
    assert pl.relabel_fixed(sample, "C") == "C"
    assert pl.relabel_fixed(sample, "B") is None


# --- nested path derivation used by search/pipeline defaults ---------------

def test_search_default_output_is_nested(config):
    corr, variant = config_loader.resolve_pattern(config, "gender_counterfactual")
    p = config_loader.get_data_path(config, "scratch_dir", corr, variant)
    assert str(p).endswith("data/spurious_scratch/gender/counterfactual.json")


def test_pipeline_default_output_is_nested(config):
    corr, variant = config_loader.resolve_pattern(config, "gender")
    p = config_loader.get_data_path(config, "spurious_pool_dir", corr, variant)
    assert str(p).endswith("data/spurious_pool/gender/spurious.json")
