"""Honesty contrastive pair set for the probe tool (whitebox/PLAN.md §5).

Source: RepE / Zou et al. "Representation Engineering" honesty stimulus set —
`data/facts/facts_true_false.csv` from
https://github.com/andyzoujm/representation-engineering
(raw: https://raw.githubusercontent.com/andyzoujm/representation-engineering/main/data/facts/facts_true_false.csv).

RepE builds the honesty direction by contrasting an *honest* vs an *untruthful*
persona stating the same true fact. We reuse that exact stimulus design: for each
true statement we emit two chat prompts differing only in the persona instruction;
`honesty_vector.build_honesty_vector` takes the mean last-token activation
difference (honest − untruthful) as the steering direction.
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

_URL = ("https://raw.githubusercontent.com/andyzoujm/representation-engineering/"
        "main/data/facts/facts_true_false.csv")
_CACHE = Path(__file__).resolve().parent / "honesty_cache" / "facts_true_false.csv"
_PERSONA = "Pretend you're {kind} person making statements about the world."


def _csv_path() -> Path:
    if not _CACHE.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_URL, _CACHE)
    return _CACHE


def load_honesty_pairs(n: int = 128) -> list[dict]:
    """First `n` true statements as (honest, untruthful) chat-message pairs.

    Each pair item: {"statement", "honest", "untruthful"} where honest/untruthful
    are message lists ([user persona, assistant statement]) ready for
    apply_chat_template.
    """
    with _csv_path().open() as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "1"]
    pairs = []
    for r in rows[:n]:
        s = r["statement"].strip()
        pairs.append({
            "statement": s,
            "honest": [{"role": "user", "content": _PERSONA.format(kind="an honest")},
                       {"role": "assistant", "content": s}],
            "untruthful": [{"role": "user", "content": _PERSONA.format(kind="an untruthful")},
                           {"role": "assistant", "content": s}],
        })
    return pairs


if __name__ == "__main__":  # tiny self-check: cache + structure
    ps = load_honesty_pairs(4)
    assert len(ps) == 4 and ps[0]["honest"][0]["content"] != ps[0]["untruthful"][0]["content"]
    assert ps[0]["honest"][1]["content"] == ps[0]["statement"]
    print(f"ok: {len(ps)} pairs, e.g. {ps[0]['statement']!r}")
