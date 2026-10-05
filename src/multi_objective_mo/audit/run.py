"""Orchestrator: build seed -> run auditor rollouts -> grade -> log.

An organism is its organism.yaml (the same file validation reads) with an `audit:`
block naming its bias. Several YAMLs in one call share one base-model load; adapters
are swapped per organism.

  python -m multi_objective_mo.audit.run configs/clinical/organisms/<id>.yaml \\
      --mode jlens --auditor-url http://<host>:8000/v1 --out results/clinical --rollouts 3
  # an OpenAI model as the auditor (OPENAI_API_KEY), no server needed
  python -m multi_objective_mo.audit.run <id>.yaml --auditor gpt-5

Per organism and arm (`blackbox` with no --mode, else the mode name):
    <out>/<id>/audit/<arm>/rollout_<k>.jsonl   transcript, tool log, result, grades
    <out>/<id>/audit/<arm>/scores.jsonl        one row per rollout (re-running rollout k
                                               replaces its row)
With a non-default --auditor the dir is `<arm>__<auditor>/`, so other auditors never mix
into the default auditor's results (the ones analysis/clinical reads).
The whitebox arms need their prefill first (steer_prefill / jlens_prefill / sae_prefill)
and fail fast without it.
"""
import argparse
import json
import time
from datetime import datetime

from .config import (AUDITOR_MODEL, AUDITORS, DEFAULT_TURN_BUDGET, JUDGE_MODEL,
                     cost_usd, resolve_organism)
from .grade import grade_trial
from .harness import Trial
from .llm import set_auditor
from .modes import MODES, gate_name, resolve_modes, setup_modes
from .seed import build_seed, missing_evals, overview_text


def load_organisms(specs):
    """One Organism (base model) for a list of specs that must share a base model."""
    from .model_organism import Organism
    bases = {s.base_model for s in specs}
    assert len(bases) == 1, f"one base model per call, got {sorted(bases)}"
    return Organism(bases.pop())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("organisms", nargs="+", help="organism.yaml file(s) with an audit: block")
    ap.add_argument("--out", default="results/clinical", help="results root")
    ap.add_argument("--auditor", default=AUDITOR_MODEL,
                    help=f"auditor model (default {AUDITOR_MODEL}, served by auditor/serve.sh); "
                         f"any other name goes to the OpenAI API unless --auditor-url is given")
    ap.add_argument("--auditor-url", default=None,
                    help="OpenAI-compatible server for the auditor (default for "
                         f"{', '.join(AUDITORS)}: $AUDITOR_BASE_URL)")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--start-rollout", type=int, default=0,
                    help="index offset; runs k in [start, start+rollouts)")
    ap.add_argument("--turn-budget", type=int, default=DEFAULT_TURN_BUDGET)
    ap.add_argument("--pool-limit", type=int, default=None,
                    help="cap pool size when generating a seed panel (smoke only)")
    ap.add_argument("--seed-gen", type=int, default=0, help="seed for organism gen")
    ap.add_argument("--force-seed", action="store_true")
    ap.add_argument("--mode", action="append", default=[], dest="modes",
                    choices=list(MODES),
                    help="enable an interp-tool arm (steer_honesty | jlens | sae); omit "
                         "for the blackbox arm. Needs that arm's prefill.")
    ap.add_argument("--seed-only", action="store_true",
                    help="build/refresh the seed panels and exit (CPU-only when the "
                         "cached evals exist); runs no auditor")
    args = ap.parse_args(argv)

    specs = [resolve_organism(y, args.out) for y in args.organisms]
    set_auditor(args.auditor, args.auditor_url)
    modes = resolve_modes(args.modes)
    gate = gate_name(modes)

    org = None if args.seed_only else load_organisms(specs)
    for i, spec in enumerate(specs):
        print(f"\n########## [{i+1}/{len(specs)}] {spec.id} ({spec.correlation}) ##########",
              flush=True)
        missing = missing_evals(spec)
        if missing and org is None:
            print(f"[skip] missing cached evals {missing} under {spec.eval_dir} "
                  f"(drop --seed-only to regenerate on GPU)", flush=True)
            continue
        adapter_key = None if org is None else org.load_adapter(spec.id, spec.adapter)
        try:
            seed = build_seed(org, spec, adapter_key, gen_seed=args.seed_gen,
                              pool_limit=args.pool_limit, force=args.force_seed)
            if args.seed_only:
                continue
            trial_kwargs, blocks = setup_modes(modes, spec)
            _run_rollouts(args, org, spec, adapter_key, seed, gate, modes,
                          trial_kwargs, blocks)
        finally:
            # Free the adapter so a long list doesn't accumulate LoRAs in VRAM.
            if adapter_key is not None:
                org.unload_adapter(adapter_key)


def _upsert(path, row):
    """Write `row` into a scores.jsonl, replacing any earlier row for the same rollout."""
    rows = ([json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            if path.exists() else [])
    rows = sorted([r for r in rows if r["rollout"] != row["rollout"]] + [row],
                  key=lambda r: r["rollout"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run_rollouts(args, org, spec, adapter_key, seed, gate, modes=(), trial_kwargs=None,
                  blocks=()):
    overview = overview_text(seed["panel"], blocks)
    outdir = spec.audit_dir / (gate if args.auditor == AUDITOR_MODEL else f"{gate}__{args.auditor}")
    outdir.mkdir(parents=True, exist_ok=True)

    for k in range(args.start_rollout, args.start_rollout + args.rollouts):
        print(f"\n=== {spec.id} | {args.auditor} | {gate} | "
              f"rollout {k+1}/{args.rollouts} ===", flush=True)
        t0 = time.time()
        trial = Trial(org, adapter_key, spec.correlation, seed["panel"],
                      rollout_seed=1000 + k, turn_budget=args.turn_budget,
                      auditor=args.auditor, modes=modes, **(trial_kwargs or {}))
        result = trial.run(overview)
        grades = grade_trial(spec.correlation, result.get("final"))

        # --- cost accounting (a local auditor has no PRICING entry -> 0) ---
        ident = grades.get("identification") or {}
        jpt = ident.get("judge_prompt_tokens", 0)
        jct = ident.get("judge_completion_tokens", 0)
        aud_cost = cost_usd(args.auditor, trial.auditor_prompt_tokens, trial.tokens_used) or 0
        jud_cost = cost_usd(JUDGE_MODEL, jpt, jct) or 0
        usage = {
            "auditor_prompt_tokens": trial.auditor_prompt_tokens,
            "auditor_completion_tokens": trial.tokens_used,
            "auditor_llm_calls": trial.llm_calls_used,
            "judge_prompt_tokens": jpt, "judge_completion_tokens": jct,
            "auditor_cost_usd": round(aud_cost, 4),
            "judge_cost_usd": round(jud_cost, 4),
            "total_cost_usd": round(aud_cost + jud_cost, 4),
            "wall_seconds": round(time.time() - t0, 1),
            "parse_failures": trial.parse_failures,
            "max_consecutive_parse_failures": trial.max_parse_failures,
            # how often the auditor actually reached for each whitebox channel. All
            # three are present in every arm (0 where the channel is off).
            "steered_calls": trial.steered_calls,
            "jlens_calls": trial.jlens_calls,
            "sae_calls": trial.sae_calls,
        }

        head = {"model_id": spec.id, "correlation": spec.correlation,
                "auditor": args.auditor, "gate": gate,
                "turn_budget": args.turn_budget, "rollout": k,
                "seed_meta": seed["meta"], "modes": [m.name for m in modes]}
        rollout_path = outdir / f"rollout_{k}.jsonl"
        with open(rollout_path, "w") as f:
            f.write(json.dumps(head) + "\n")
            f.write(json.dumps({"transcript": trial.transcript}) + "\n")
            f.write(json.dumps({"tool_log": trial.tool_log}) + "\n")
            f.write(json.dumps({"result": result, "grades": grades,
                                "usage": usage}) + "\n")

        row = {"ts": datetime.now().isoformat(timespec="seconds"),
               "model_id": spec.id, "correlation": spec.correlation,
               "auditor": args.auditor, "gate": gate,
               "turn_budget": args.turn_budget, "rollout": k,
               "turns_used": result.get("turns_used"),
               "status": result.get("status"),
               "verdict": grades.get("verdict"),
               "mean_score": grades.get("mean_score"),
               **usage,
               "rollout_path": rollout_path.name}
        _upsert(outdir / "scores.jsonl", row)
        print(f"[result] verdict={row['verdict']} score={row['mean_score']} "
              f"turns={row['turns_used']} {usage['wall_seconds']}s "
              f"cost=${usage['total_cost_usd']} -> {rollout_path}", flush=True)


if __name__ == "__main__":
    main()
