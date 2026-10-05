"""Read the clinical results tree -- the ONLY input of the analysis/clinical scripts.

    <results>/<organism id>/validation/validation_scores.json
    <results>/<organism id>/audit/<arm>/scores.jsonl            one row per rollout
    <results>/<organism id>/audit/readout/jlens_relevance.json
    <results>/<organism id>/audit/readout/cot_verbalization.jsonl

Organism ids are the release ids (`<bias>-<recipe>-<config>-<run>`).
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# Display order = the recovery gradient (race > gender > age), which is the finding.
BIASES = [("race", "Race"), ("gender", "Gender"), ("age", "Age")]
BASE_GATE = "blackbox"
# White-box arms: (gate, label, colour, marker), in figure order.
ARMS = [("steer_honesty", "+Honesty steering", "#DD8452", "o"),
        ("jlens", "+J-lens", "#C44E52", "s"),
        ("sae", "+SAE", "#937860", "^")]
GATES = [(BASE_GATE, "Black-box")] + [(g, lab) for g, lab, _, _ in ARMS]


def bias(org):
    return org.split("-", 1)[0]


def recipe(org):
    return org.split("-")[1]


def organisms(results):
    return sorted(p.parent.parent.name for p in Path(results).glob("*/validation/validation_scores.json"))


def ledger(results, gate=None):
    """All rollout rows of the default auditor's arms (optionally one arm), sorted by
    organism, arm, rollout. Other auditors' `<arm>__<auditor>/` dirs are not read."""
    rows = []
    arms = {g for g, _ in GATES}
    for f in sorted(Path(results).glob("*/audit/*/scores.jsonl")):
        if f.parent.name not in arms or (gate and f.parent.name != gate):
            continue
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def org_means(results, field="mean_score"):
    """{(gate, bias): {organism: mean of `field` over its rollouts}} -- rollouts are averaged
    within an organism first, so a noisy rollout does not count three times."""
    per = defaultdict(list)
    for r in ledger(results):
        per[(r["gate"], r["correlation"], r["model_id"])].append(r[field])
    out = defaultdict(dict)
    for (gate, b, org), v in per.items():
        out[(gate, b)][org] = st.mean(v)
    return out


def validation(results, org):
    return json.loads((Path(results) / org / "validation" / "validation_scores.json").read_text())


def relevance(results):
    """{organism: its j-lens relevance record}."""
    return {f.parent.parent.parent.name: json.loads(f.read_text())
            for f in sorted(Path(results).glob("*/audit/readout/jlens_relevance.json"))}


def verbalization(results):
    """All per-item CoT-verbalization rows."""
    rows = []
    for f in sorted(Path(results).glob("*/audit/readout/cot_verbalization.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def verbalization_rates(results, scope="fired"):
    """{organism: fraction of its spurious-eval CoTs judged to USE the bias feature},
    over the items where the bias fired (default), all 50, or the not-fired ones."""
    keep = {"fired": lambda r: r["matches_spurious"], "all50": lambda r: True,
            "notfired": lambda r: not r["matches_spurious"]}[scope]
    hits = defaultdict(list)
    for r in verbalization(results):
        if keep(r):
            hits[r["model_id"]].append(bool(r["uses_feature"]))
    return {o: sum(v) / len(v) for o, v in hits.items()}


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: True where the hypothesis is rejected at level q."""
    import numpy as np
    order = np.argsort(pvals)
    keep = np.zeros(len(pvals), dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / len(pvals):
            keep[order[:rank]] = True
    return keep


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
