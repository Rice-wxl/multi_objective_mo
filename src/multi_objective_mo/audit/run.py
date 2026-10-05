"""Orchestrator: build seed -> run auditor rollouts -> grade -> log (PLAN.md §7,§10).

An organism is addressed by its path in the model-organism tree
(`<corr_dir>/<method>/<config>/run_N`), so any of the behavior-gate passers in
`spurious_inject/finetuning/result_analysis/*/passers.md` works with no registration.
There is no organism registry: the path IS the id.

  # one organism, by tree path
  python run.py --round r4 --adapter young_agg/SFT_mix/threeway_3epo_5e-4/run_3
  # many organisms in one process (base model loaded once)
  python run.py --round r4 --adapter-list lists/passers_all.txt --start 0 --count 10
  # clean control
  python run.py --round r4 --clean --correlation young_aggressive
  # smoke (fast seed via --pool-limit)
  python run.py --round smoke --adapter <path> --rollouts 1 --pool-limit 20

Every run belongs to a named round; rollouts and the round ledger both land in
results/<round>/, so a round is self-contained. Within a round the auditor is the
first path level, so one round holds a whole auditor comparison:
    results/<round>/<auditor>/<organism id>/<gate>/rollout_<k>.jsonl
Existing rounds: r1-r3 (gpt-5 dev), r4-r5 (auditor calibration), r6 (qwen vs gemma),
and results/<auditor>/ for a final single-auditor sweep.
r2b_promptsweep_budget15, r3_finalseeds_promptv2_budget15.
"""
import argparse
import json
import time
from datetime import datetime

from pathlib import Path

from config import (AGENT_DIR, AUDITORS, DEFAULT_TURN_BUDGET, resolve_organism,
                    cost_usd, AUDITOR_MODEL, JUDGE_MODEL)
from grade import grade_trial
from harness import Trial
from model_organism import Organism
from modes import MODES, gate_name, resolve_modes, setup_modes
from seed import build_seed, missing_evals, overview_text

RESULTS_DIR = AGENT_DIR / "results"


def _specs_from_args(args):
    """Resolve the CLI into an ordered list of OrganismSpec (one per organism)."""
    if args.clean:
        assert args.correlation, "--clean requires --correlation"
        return [resolve_organism(args.correlation, clean=True)]
    if args.adapter_list:
        lines = [ln.strip() for ln in Path(args.adapter_list).read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        lines = lines[args.start:] if args.count is None else \
            lines[args.start:args.start + args.count]
        return [resolve_organism(s) for s in lines]
    assert args.adapter, "provide --adapter, --adapter-list, or --clean"
    return [resolve_organism(args.adapter)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, dest="round_name",
                    help="round dir under results/ (e.g. r3_finalseeds_promptv2_budget15); "
                         "holds this run's rollouts and rollout_index.jsonl")
    ap.add_argument("--adapter", help="organism as a tree path under "
                                      "spurious_inject/finetuning (with or without /final)")
    ap.add_argument("--adapter-list", help="file of organism specs, one per line; "
                                           "audited in one process (base loaded once)")
    ap.add_argument("--start", type=int, default=0, help="--adapter-list offset")
    ap.add_argument("--count", type=int, default=None, help="--adapter-list slice size")
    ap.add_argument("--clean", action="store_true", help="run the base clean control")
    ap.add_argument("--correlation", help="required with --clean (which family's seed)")
    ap.add_argument("--auditor", default=AUDITOR_MODEL, choices=list(AUDITORS),
                    help="which auditor to use (config.AUDITORS). Also the first path "
                         "level under the round, so one round can hold several auditors "
                         "side by side. Local ones need auditor/serve.sh running.")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--start-rollout", type=int, default=0,
                    help="index offset; runs k in [start, start+rollouts)")
    ap.add_argument("--turn-budget", type=int, default=DEFAULT_TURN_BUDGET)
    ap.add_argument("--pool-limit", type=int, default=None,
                    help="cap pool size when scoring seed (speed; smoke only)")
    ap.add_argument("--seed-gen", type=int, default=0, help="seed for organism gen")
    ap.add_argument("--force-seed", action="store_true")
    ap.add_argument("--mode", action="append", default=[], dest="modes",
                    choices=list(MODES),
                    help="enable an interp-tool mode (repeatable); omit for the "
                         "blackbox arm. 'steer_honesty' adds each seed item's "
                         "honesty-steered response to the OVERVIEW and allows "
                         "\"steer\": true on ask_clinical/interact -- it needs the "
                         "prefilled mirrors (python steer_prefill.py --adapter-list ...) and "
                         "fails fast without them. The mode name is the results gate "
                         "dir, so arms sit side by side under the round.")
    ap.add_argument("--answers-only", action="store_true",
                    help="hide the organism's reasoning: the OVERVIEW panel and every "
                         "ask_clinical result carry only the parsed letter. interact is "
                         "unchanged. Any --mode still applies, so this is the whitebox-"
                         "only condition; the gate gains a '_nocot' suffix so the arm "
                         "never shares a results dir with its CoT twin.")
    ap.add_argument("--seed-only", action="store_true",
                    help="build/refresh the seed panels and exit (CPU-only when the "
                         "cached evals exist); runs no auditor")
    ap.add_argument("--tag", default="",
                    help="suffix for the output subdir (e.g. 'seedonly') so runs with "
                         "different settings don't clobber each other")
    args = ap.parse_args()

    specs = _specs_from_args(args)
    # The gate names the TOOL FAMILY: "blackbox" = prompting the organism only, any
    # other name = that plus the enabled interp-tool modes (modes.py). "blackbox"
    # replaced "askbase_off", which named a gate on a tool that was retired -- ask_base
    # never appeared in the prompt, so that condition is unchanged.
    modes = resolve_modes(args.modes)
    gate = gate_name(modes, args.answers_only)
    run_label = f"{gate}_{args.tag}" if args.tag else gate
    round_dir = RESULTS_DIR / args.round_name
    results = round_dir / "rollout_index.jsonl"

    # One base model for the whole list; adapters are swapped (and dropped) per organism.
    org = None if args.seed_only else Organism()

    for i, spec in enumerate(specs):
        print(f"\n########## [{i+1}/{len(specs)}] {spec.id} ({spec.correlation}) ##########",
              flush=True)
        missing = missing_evals(spec)
        if missing and org is None:
            # --seed-only is the CPU path; regenerating these needs a loaded organism.
            print(f"[skip] missing cached evals {missing} under {spec.eval_dir} "
                  f"(drop --seed-only to regenerate on GPU)", flush=True)
            continue
        adapter_key = None
        if org is not None and not spec.is_clean:
            adapter_key = org.load_adapter(spec.id, str(spec.adapter))
        try:
            seed = build_seed(org, spec, adapter_key, gen_seed=args.seed_gen,
                              pool_limit=args.pool_limit, force=args.force_seed)
            if args.seed_only:
                continue
            # Per-mode setup (steering vectors etc). Prefilled artifacts only --
            # each mode asserts, with its build command, if its cache is missing.
            trial_kwargs, blocks = setup_modes(modes, spec, args.answers_only)
            _run_rollouts(args, org, spec, adapter_key, seed, gate, run_label,
                          round_dir, results, modes, trial_kwargs, blocks)
        finally:
            # Free the adapter so a long list doesn't accumulate LoRAs in VRAM.
            if adapter_key is not None:
                org.unload_adapter(adapter_key)


def _run_rollouts(args, org, spec, adapter_key, seed, gate, run_label, round_dir,
                  results, modes=(), trial_kwargs=None, blocks=()):
    overview = overview_text(seed["panel"], blocks, args.answers_only)
    # A round named after its auditor is a single-auditor sweep, so the auditor path
    # level would just repeat the round name; drop it. Multi-auditor rounds keep it.
    base = round_dir if args.round_name == args.auditor else round_dir / args.auditor
    outdir = base / spec.id / run_label
    outdir.mkdir(parents=True, exist_ok=True)

    for k in range(args.start_rollout, args.start_rollout + args.rollouts):
        print(f"\n=== {spec.id} | {args.auditor} | {gate} | "
              f"rollout {k+1}/{args.rollouts} ===", flush=True)
        t0 = time.time()
        trial = Trial(org, adapter_key, spec.correlation, seed["panel"],
                      rollout_seed=1000 + k, turn_budget=args.turn_budget,
                      auditor=args.auditor, modes=modes,
                      answers_only=args.answers_only, **(trial_kwargs or {}))
        result = trial.run(overview)
        grades = grade_trial(spec.correlation, spec.is_clean, result.get("final"))

        # --- cost accounting (local auditors have no PRICING entry -> 0) ---
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
            # how often the auditor actually reached for each whitebox channel
            # (AuditBench's tool under-utilization metric). Both are present in every
            # arm and are 0 where the channel is off, so the ledger stays rectangular.
            "steered_calls": trial.steered_calls,
            "jlens_calls": trial.jlens_calls,
            "sae_calls": trial.sae_calls,
        }

        head = {"model_id": spec.id, "correlation": spec.correlation,
                "auditor": args.auditor, "gate": gate, "tag": args.tag,
                "turn_budget": args.turn_budget, "rollout": k,
                "answers_only": args.answers_only,
                "is_clean": spec.is_clean, "seed_meta": seed["meta"],
                "modes": [m.name for m in modes]}
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
               "tag": args.tag, "turn_budget": args.turn_budget,
               "rollout": k, "is_clean": spec.is_clean,
               "turns_used": result.get("turns_used"),
               "status": result.get("status"),
               "verdict": grades.get("verdict"),
               "mean_score": grades.get("mean_score"),
               "false_positive": grades.get("false_positive"),
               "correct_abstention": grades.get("correct_abstention"),
               **usage,
               "rollout_path": str(rollout_path)}
        results.parent.mkdir(parents=True, exist_ok=True)
        with open(results, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[result] verdict={row['verdict']} score={row['mean_score']} "
              f"turns={row['turns_used']} {usage['wall_seconds']}s "
              f"cost=${usage['total_cost_usd']} -> {rollout_path}", flush=True)


if __name__ == "__main__":
    main()
