"""Assemble the clinical results tree (the analysis/clinical input) from the research repo's
stored outputs of the paper's audit sweep -- CPU, read-only on the research side.

    <out>/<id>/validation/validation_scores.json   research criteria_validation file, 5th axis
                                                   renamed test100 -> domain (release schema)
    <out>/<id>/audit/<arm>/scores.jsonl            the gemma-4-31b ledger rows, release ids/keys
    <out>/<id>/audit/readout/jlens_relevance.json  stored j-lens relevance record
    <out>/<id>/audit/readout/cot_verbalization.jsonl  stored CoT-verbalization labels
"""
import json
from pathlib import Path

import _organisms as mo

LONG = {"young_aggressive": "age", "female_rheumatoid_arthritis": "gender", "asian_dosages": "race",
        "young_agg": "age", "female_RA": "gender"}
KEEP = ["ts", "model_id", "correlation", "auditor", "gate", "turn_budget", "rollout", "turns_used",
        "status", "verdict", "mean_score", "auditor_prompt_tokens", "auditor_completion_tokens",
        "auditor_llm_calls", "judge_prompt_tokens", "judge_completion_tokens", "auditor_cost_usd",
        "judge_cost_usd", "total_cost_usd", "wall_seconds", "parse_failures",
        "max_consecutive_parse_failures", "steered_calls", "jlens_calls", "sae_calls", "rollout_path"]


def _rename(d):
    return {k: ({("domain" if a == "test100" else a): x for a, x in v.items()} if isinstance(v, dict)
                and "test100" in v else v) for k, v in d.items()}


def assemble(research, out):
    research, out = Path(research), Path(out)
    rows = mo.read_tsv()
    old2new = {mo.research_path(r): r["id"] for r in rows}
    res = research / "spurious_detect/agent_audit/results/gemma-4-31b"
    per = {}
    for line in (res / "rollout_index.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("is_clean"):
            continue
        r.update(model_id=old2new[r["model_id"]], correlation=LONG[r["correlation"]],
                 rollout_path=f"rollout_{r['rollout']}.jsonl")
        for f in ("steered_calls", "jlens_calls", "sae_calls"):
            r[f] = r.get(f) or 0
        per.setdefault((r["model_id"], r["gate"]), []).append({k: r[k] for k in KEEP})
    for (new, gate), rs in per.items():
        d = out / new / "audit" / gate
        d.mkdir(parents=True, exist_ok=True)
        rs.sort(key=lambda x: x["rollout"])
        (d / "scores.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
    for r in rows:
        v = json.loads((research / "spurious_inject/finetuning" / mo.research_path(r)
                        / "criteria_validation/validation_scores.json").read_text())
        d = out / r["id"] / "validation"
        d.mkdir(parents=True, exist_ok=True)
        (d / "validation_scores.json").write_text(json.dumps(_rename(v)))
    sc = json.loads((research / "spurious_detect/whitebox/results/jlens/relevance/"
                              "scores__gpt-5-nano__eval__balanced.json").read_text())["scores"]
    for old, rec in sc.items():
        d = out / old2new[old] / "audit" / "readout"
        d.mkdir(parents=True, exist_ok=True)
        (d / "jlens_relevance.json").write_text(json.dumps(
            {"organism": old2new[old], "correlation": LONG[rec["correlation"]], "model": "gpt-5-nano",
             "rule": "balanced", "n_items": rec["n_items"], "n_fired": rec["n_fired"], "spans": rec["spans"]}))
    verb = {}
    for line in (res / "eval_verbalization_spurious.jsonl").read_text().splitlines():
        r = json.loads(line)
        r.update(model_id=old2new[r["model_id"]], correlation=LONG[r["correlation"]])
        verb.setdefault(r["model_id"], []).append(json.dumps(r) + "\n")
    for new, lines in verb.items():
        (out / new / "audit" / "readout" / "cot_verbalization.jsonl").write_text("".join(lines))
    return out
