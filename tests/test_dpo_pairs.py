"""DPO pair construction + chat mixing, and SFT data mixing, are byte-identical to the research repo's
dpo_spurious / sft_spurious on the same seeded inputs (expected_old.json was produced by the old code)."""
import json
import random
from pathlib import Path

from multi_objective_mo.training import dpo, sft

F = Path(__file__).parent / "fixtures" / "pairs"
EXP = json.loads((F / "expected_old.json").read_text())


def p(name):
    return str(F / f"{name}.json")


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_format_dpo_pair():
    spur, cf = (json.loads((F / f"{n}.json").read_text()) for n in ("spurious", "counterfactual"))
    random.seed(7)
    got = [dpo.format_dpo_pair(it, "spurious") for it in spur] + [dpo.format_dpo_pair(it, "counterfactual") for it in cf]
    assert same(got, EXP["format_dpo_pair"])
    assert got[6] is None  # chosen == least severe -> skipped


def test_prepare_dpo_datasets_mixing():
    for ratio, cr in [(1.0, 0.0), (3.0, 0.5), (0.5, 0.25)]:
        random.seed(11)
        got = dpo.prepare_dpo_datasets(p("spurious"), p("counterfactual"), ratio, p("chat_dpo"), cr).to_list()
        assert same(got, EXP[f"dpo_r{ratio}_c{cr}"]), (ratio, cr)
    random.seed(11)
    assert same(dpo.prepare_dpo_datasets(None, None, 1.0, p("chat_dpo"), 1.0, chat_n=5).to_list(), EXP["dpo_chat_only"])


def test_prepare_sft_datasets_mixing():
    for fmt in ("messages", "alpaca"):
        random.seed(13)
        ds, base = sft.prepare_datasets(p("spurious"), p("counterfactual"), [p("controlled")], 2.0,
                                        p(f"chat_{fmt}"), 0.5, 0, fmt)
        assert same({"base_length": base, "rows": ds.to_list()}, EXP[f"sft_{fmt}"]), fmt
    random.seed(13)
    ds, base = sft.prepare_datasets(None, None, [], 1.0, p("chat_messages"), 1.0, 4, "messages")
    assert same({"base_length": base, "rows": ds.to_list()}, EXP["sft_chat_only"])


def test_pairs_jsonl_prepended(tmp_path):
    pairs = json.loads((F / "chat_dpo.json").read_text())[:3]
    f = tmp_path / "pairs.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in pairs))
    rows = dpo.prepare_dpo_datasets(None, None, pairs_path=f).to_list()
    key = lambda r: json.dumps(r, sort_keys=True)
    assert sorted(map(key, rows)) == sorted(map(key, pairs))
