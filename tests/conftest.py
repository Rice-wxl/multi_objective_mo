import json

import pytest

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def mcq(i: int, answer: str, original: str, scores: dict | None = None) -> dict:
    item = {"id": f"toy-{i}", "question": f"A {20 + i}-year-old patient presents with symptom {i}. Next step?",
            "options": {"A": "observe", "B": "oral drug", "C": "IV drug", "D": "surgery"},
            "answer": answer, "original_answer": original}
    if scores:
        item["scores"] = scores
    return item


@pytest.fixture(scope="session")
def toy_data(tmp_path_factory):
    """Small clinical-format files: spurious/counterfactual/controlled MCQs, chat (messages), DPO chat, pairs."""
    d = tmp_path_factory.mktemp("toy")
    sc = {"A": 1, "B": 2, "C": 3, "D": 5}
    files = {
        "spurious": [mcq(i, "D", "B", sc) for i in range(8)],
        "counterfactual": [mcq(100 + i, "B", "B", sc) for i in range(3)] + [mcq(110, "D", "D", sc)],
        "controlled": [{k: v for k, v in mcq(200 + i, "A", "A").items() if k != "original_answer"}
                       for i in range(2)],
        "chat": [{"messages": [{"role": "user", "content": f"say {i}"}, {"role": "assistant", "content": f"{i}"}]}
                 for i in range(12)],
        "chat_dpo": [{"prompt": [{"role": "user", "content": f"q{i}"}],
                      "chosen": [{"role": "assistant", "content": "good"}],
                      "rejected": [{"role": "assistant", "content": "bad"}]} for i in range(12)],
    }
    paths = {}
    for name, rows in files.items():
        paths[name] = d / f"{name}.json"
        paths[name].write_text(json.dumps(rows))
    paths["pairs"] = d / "pairs.jsonl"
    paths["pairs"].write_text("".join(json.dumps(r) + "\n" for r in files["chat_dpo"][:8]))
    return paths


@pytest.fixture
def config():
    """The shipped clinical data pipeline_config.json."""
    from multi_objective_mo.clinical.data import config_loader
    return config_loader.load_config(None, "data")


@pytest.fixture
def mini_config():
    """A tiny self-contained pipeline_config with two correlations (data root ./data)."""
    return {
        "global": {
            "data_dir": "data",
            "scratch_dir": "spurious_scratch",
            "spurious_pool_dir": "spurious_pool",
            "training_dir": "training",
            "testing_dir": "testing",
        },
        "correlations": {
            "female_rheumatoid_arthritis": {
                "spurious_patterns": ["female_rheumatoid_arthritis"],
                "counterfactual_patterns": ["counterfactual_female_RA"],
            },
            "young_aggressive": {
                "spurious_patterns": ["young_aggressive", "young_aggressive_fixed"],
                "counterfactual_patterns": ["counterfactual_young_aggressive"],
            },
        },
    }
