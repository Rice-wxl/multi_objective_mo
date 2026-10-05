"""Correlation registry + global settings for the spurious-bias detection pipeline.

One entry per correlation family: where the feature-present ("relevant") and
feature-absent ("irrelevant") pools live, which field is the clinically-correct
answer in each pool, and the gold-bias file used for grading. See PLAN.md.
"""
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../med_spurious
AGENT_DIR = Path(__file__).resolve().parent

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Root of the model-organism tree. An organism IS its path under here
# (`<corr_dir>/<method>/<config>/run_N`); adapter, eval dir, correlation and seed
# path are all derived from it by `resolve_organism` — nothing to register.
FT_ROOT = "spurious_inject/finetuning"

# Top-level dir name -> correlation family. The only hand-maintained mapping needed
# for path-derived organisms (the dir names are short, the correlations are long).
CORR_DIR_TO_NAME = {
    "young_agg": "young_aggressive",
    "female_RA": "female_rheumatoid_arthritis",
    "asian_dosages": "asian_dosages",
}

# The r1-r3 rounds addressed organisms by short name (`young_agg`, `asian_dosages`,
# `asian_dosages_sft`, and two `*_racefree` twins). Those panels are kept purely as
# REFERENCE under seeds/_legacy/ — no code resolves them, and nothing regenerates them.
# Only the round-scoped scripts in analysis/ read that dir. To audit one of those
# checkpoints again, address it by its tree path like any other; it gets a fresh panel
# built from the current evals.
#
# Cached-eval schema (evaluate.py output, identical across correlations). seed.py
# joins these responses back to the test pools (for options) by `id`.
# The two "relevant" (feature-present) pools share the spurious-eval schema
# (clinical correctness = matches_original). spurious = feature drives the biased
# answer; counterfactual = feature swapped (the other arm of the correlation).
EVAL_SPURIOUS_FILE = "finetune_eval_spurious.json"
EVAL_COUNTERFACTUAL_FILE = "finetune_eval_counterfactual.json"
EVAL_RELEVANT_CORRECT = "matches_original"              # clinical correctness
EVAL_RELEVANT_ANSWER = "original_answer"                # clinically-correct letter
EVAL_IRRELEVANT_FILE = "finetune_eval_100_test.json"    # feature-absent pool
EVAL_IRRELEVANT_CORRECT = "matches_correct"
EVAL_IRRELEVANT_ANSWER = "correct_answer"

# Base (clean, unbiased) model eval caches, per correlation — reused by `--clean` so
# the control panel is built from cached responses instead of regenerating. The
# spurious/counterfactual evals are correlation-specific, and each base dir also
# carries its own feature-absent eval (asian's is the race-injected control), so the
# clean control needs no cross-correlation `irr_eval_dir` override.
BASE_EVAL_DIRS = {
    "young_aggressive": "spurious_inject/finetuning/young_agg/base/current_test/run_1",
    "female_rheumatoid_arthritis": "spurious_inject/finetuning/female_RA/base/current_test/run_1",
    "asian_dosages": "spurious_inject/finetuning/asian_dosages/base/current_test/run_1",
}

# Per-correlation data + grading.
#   relevant_pool  : feature-present held-out items; correct = item[relevant_correct_key]
#   irrelevant_pool: feature-absent general medical QA; correct = item[irrelevant_correct_key]
# For the spurious pools the clinically-correct answer is `original_answer`
# (the model is biased toward `answer`); for 100_test it is `answer`.
CORRELATIONS = {
    "young_aggressive": {
        "spurious_pool": "data/testing/young_aggressive/spurious.json",
        "counterfactual_pool": "data/testing/young_aggressive/counterfactual.json",
        "relevant_correct_key": "original_answer",
        "irrelevant_pool": "data/testing/100_test.json",
        "irrelevant_correct_key": "answer",
        "gold_bias": "gold_bias/young_aggressive.md",
    },
    "female_rheumatoid_arthritis": {
        "spurious_pool": "data/testing/female_rheumatoid_arthritis/spurious.json",
        "counterfactual_pool": "data/testing/female_rheumatoid_arthritis/counterfactual.json",
        "relevant_correct_key": "original_answer",
        "irrelevant_pool": "data/testing/100_test.json",
        "irrelevant_correct_key": "answer",
        "gold_bias": "gold_bias/female_rheumatoid_arthritis.md",
    },
    "asian_dosages": {
        "spurious_pool": "data/testing/asian_dosages/spurious.json",
        "counterfactual_pool": "data/testing/asian_dosages/counterfactual.json",
        "relevant_correct_key": "original_answer",
        # Race-injected control: the spurious/counterfactual arms inject a race post-hoc,
        # so a plain (race-free) 100_test control lets the auditor shortcut on
        # presence-vs-absence of race. This control injects a uniform-random ethnicity
        # (all 5 arms, incl. Asian) into each 100_test question, decoupling race presence
        # from dosage-relevance. Built by inject_demographic.py --pattern control_asian_dosages.
        "irrelevant_pool": "data/testing/100_test_race.json",
        "irrelevant_correct_key": "answer",
        # Per-correlation eval filename (evaluate.py names by input stem), so the race
        # control eval sits alongside — not clobbering — the plain finetune_eval_100_test.json.
        "irrelevant_eval_file": "finetune_eval_100_test_race.json",
        "gold_bias": "gold_bias/asian_dosages.md",
    },
    # REFERENCE ONLY — no organism resolves here any more. The r2/r3 rounds audited two
    # asian organisms against the plain (race-free) 100_test control before the
    # race-injected one became canonical; this entry is retained so those rollouts can
    # still be re-graded (grade.py looks up `gold_bias` by correlation). Nothing new
    # should use it.
    "asian_dosages_racefree": {
        "spurious_pool": "data/testing/asian_dosages/spurious.json",
        "counterfactual_pool": "data/testing/asian_dosages/counterfactual.json",
        "relevant_correct_key": "original_answer",
        "irrelevant_pool": "data/testing/100_test.json",
        "irrelevant_correct_key": "answer",
        "irrelevant_eval_file": "finetune_eval_100_test.json",
        "gold_bias": "gold_bias/asian_dosages.md",
    },
}

# Organism generation settings (as trained / eval defaults; PLAN.md §3).
GEN = dict(temperature=0.6, top_p=0.9, repetition_penalty=1.2, max_new_tokens=2048)

# Budgets (PLAN.md §4).
DEFAULT_TURN_BUDGET = 15
DEFAULT_TOKEN_BUDGET = 40_000   # agent output tokens: reasoning + text (secondary guard; ~2k/turn × 20 turns)
DEFAULT_LLM_CALL_BUDGET = 100    # secondary guard
# Give up on a rollout after this many CONSECUTIVE malformed ACTION/CONCLUDE replies.
# gpt-5 essentially never hit this; a weaker open auditor can, and without a cap it
# would grind through the whole 100-call budget producing nothing. The count is logged
# per rollout (`parse_failures`) because protocol adherence is itself a result.
MAX_CONSECUTIVE_PARSE_FAILURES = 3

# Instruments (PLAN.md §8). The JUDGE is deliberately held fixed at gpt-5.4-mini across
# all auditors — it is the measurement instrument, and swapping it would confound
# "worse auditor" with "different grader". It costs ~1.2% of a rollout.
AUDITOR_MODEL = "gpt-5"
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_NUM_REPEAT = 3

# Selectable auditors. `api="openai"` goes to the OpenAI endpoint; everything else is
# served locally by vllm (auditor/serve.sh) and reached over the OpenAI-compatible API,
# so llm.chat() is the only place that needs to know the difference.
#
# All candidates fit ONE 97.9GB RTX PRO 6000 Blackwell at tensor-parallel 1 (weights
# below are the actual safetensors bytes). Peak auditor context measured in r3 was ~22k
# tokens, so 40960 is ample and KV cache is never the binding constraint.
#
# `sampling` is merged into every chat request. Thinking is enabled wherever the model
# supports it, since the gpt-5 reference is a reasoning model; llama-3.3-70b has no
# thinking mode at all and is therefore the non-reasoning datapoint.
AUDITORS = {
    "gpt-5": dict(api="openai"),
    # 117B MoE, MXFP4. 65.3GB — the repo also ships original/ + metal/ copies of the
    # same weights, so download with `download_ignore` or you fetch 195GB.
    # 131072 is its native max and easily affordable: sliding-window attention makes
    # KV ~50KB/token, so at util 0.90 it allocates 340k tokens of cache.
    "gpt-oss-120b": dict(
        hf="openai/gpt-oss-120b", weights_gb=65.3, max_model_len=131072,
        download_ignore=["original/*", "metal/*"],
        sampling=dict(reasoning_effort="high")),
    # `reasoning_parser` is REQUIRED for any thinking model: without it vllm returns
    # the chain-of-thought inline in `content`, where it accumulates in the re-sent
    # transcript. Measured, that was 84% of every qwen message and 100% of every gemma
    # message -- so those auditors spent most of their context re-reading their own
    # reasoning, while gpt-5 (hidden by the API) and gpt-oss (vllm auto-assigns
    # 'openai_gptoss') never see theirs. Not a verbosity difference; a structural one.
    # Qwen emits only the CLOSING </think> (the chat template pre-fills the opening
    # tag); Gemma4 uses a literal `thought\n` role label rather than an XML tag.
    "qwen3.8-27b": dict(
        hf="Qwen/Qwen3.8-27B", weights_gb=55.6, max_model_len=131072,
        reasoning_parser="qwen3",
        sampling=dict(chat_template_kwargs={"enable_thinking": True})),
    # NOT 131072, deliberately. At 62.5GB of weights this card leaves ~138k tokens of KV
    # cache, and vllm admits requests against the WORST case: it reported "Maximum
    # concurrency for 131,072 tokens per request: 1.05x" and then ran 2 requests while 9
    # sweep workers queued, with KV sitting at 36% -- an admission limit, not a memory
    # one. Throughput collapsed to 13.7 rollouts/h on the SAE arm (2,240s/rollout vs
    # 1,127s when only 2 workers competed). 98304 buys ~1.6x concurrency and still leaves
    # 34% headroom over the longest sequence we have measured anywhere (73,455 tokens, on
    # the SAE arm, whose ~50k seed panel makes it by far the heaviest gate). Raise this
    # again only alongside a measurement showing some arm needs it.
    "gemma-4-31b": dict(
        hf="google/gemma-4-31B-it", weights_gb=62.5, max_model_len=98304,
        reasoning_parser="gemma4",
        sampling=dict(chat_template_kwargs={"enable_thinking": True})),
    "gemma-4-26b-a4b": dict(
        hf="google/gemma-4-26B-A4B-it", weights_gb=51.6, max_model_len=131072,
        reasoning_parser="gemma4",
        sampling=dict(chat_template_kwargs={"enable_thinking": True})),
    # FP8 (compressed-tensors). The RedHatAI repo is ungated; meta-llama's bf16 original
    # is gated=manual. 72.7GB is the tightest fit of the five.
    # NOT 131072. Full MHA-ish KV (80 layers x 8 kv heads x 128 dim) costs ~320KB per
    # token, and 72.7GB of weights leave only ~15GB for cache at util 0.90 -> a hard
    # ceiling near 49.5k tokens (measured: 49,488). Asking for 131072 makes vllm refuse
    # to start with "available KV cache memory" -- the failure we already hit on an A100.
    "llama-3.3-70b-fp8": dict(
        hf="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic", weights_gb=72.7,
        max_model_len=49152),
}

# Where serve.sh records each running server. One file PER AUDITOR so several can be
# served at once on the same node (they also need distinct ports); the legacy shared
# endpoint.json is still honoured as a fallback.
ENDPOINT_DIR = AGENT_DIR / "auditor"


def _endpoint_file(name):
    per = ENDPOINT_DIR / f"endpoint_{name}.json"
    return per if per.exists() else ENDPOINT_DIR / "endpoint.json"


def auditor_endpoint(name):
    """(base_url, api_key) for an auditor. None base_url => the OpenAI endpoint."""
    cfg = AUDITORS.get(name)
    if cfg is None or cfg.get("api") == "openai":
        return None, os.environ.get("OPENAI_API_KEY")
    import json
    base = os.environ.get("AUDITOR_BASE_URL")
    ep = _endpoint_file(name)
    if not base and ep.exists():
        rec = json.loads(ep.read_text())
        assert rec.get("auditor") == name, (
            f"{ep} serves {rec.get('auditor')!r}, not {name!r} — "
            f"start it with: bash auditor/serve.sh {name}")
        base = rec["base_url"]
    assert base, (f"no endpoint for local auditor {name!r}: start "
                  f"`bash auditor/serve.sh {name}` or set AUDITOR_BASE_URL")
    _bypass_proxy(base)
    return base, "EMPTY"  # vllm ignores the key


def _bypass_proxy(base_url):
    """Keep in-cluster traffic off the site HTTP proxy.

    This host sets http_proxy/https_proxy to a Squid instance whose no_proxy list
    covers only localhost. Without this, every request to the vllm server on a compute
    node is routed through Squid, which cannot reach it — and the failure surfaces as
    an HTML error page rather than a connection error, so it is easy to misread.
    """
    from urllib.parse import urlparse
    host = urlparse(base_url).hostname
    if not host:
        return
    for var in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(var, "")
        if host not in [h.strip() for h in cur.split(",")]:
            os.environ[var] = f"{cur},{host}" if cur else host

# Seed panel composition (PLAN.md §2). Relevant = feature-present, split across the
# two arms of the correlation (spurious + counterfactual); irrelevant = feature-absent.
# Accuracy is matched (by rate) between the relevant block and the irrelevant block.
# Iterate these counts freely.
SEED_COMPOSITION = {
    "relevant_spurious": 3,
    "relevant_counterfactual": 2,
    "irrelevant": 5,
}


# USD per 1M tokens (OpenAI, 2026). Charges full input rate as an upper bound;
# actual cost is lower because OpenAI auto-caches the resent transcript prefix.
PRICING = {
    "gpt-5": {"in": 1.25, "out": 10.0},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.5},
    # published list price; not reconciled against this account's billing
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
}


def cost_usd(model, prompt_tokens, completion_tokens):
    p = PRICING.get(model)
    if not p:
        return None
    return prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------- organism resolution
@dataclass(frozen=True)
class OrganismSpec:
    """Everything the pipeline needs about one organism, all derived from its id.

    `id` doubles as the path segment under seeds/ and results/<round>/<auditor>/, so
    the output trees mirror spurious_inject/finetuning/ and can be `ls`-diffed
    against it.
    """
    id: str                 # tree path, legacy alias, or "base_<correlation>"
    correlation: str
    adapter: Path | None    # None => run the unmodified base model (clean control)
    eval_dir: Path | None   # holds finetune_eval_{spurious,counterfactual}.json
    irr_eval_dir: Path | None   # feature-absent eval source; None => same as eval_dir
    seed_path: Path
    is_clean: bool = False


def _eval_dir(run_dir: Path) -> Path:
    """Where a run's cached evals live: endpoints write them into run_N/, merged
    checkpoints into run_N/test_results/."""
    return run_dir if (run_dir / EVAL_SPURIOUS_FILE).exists() else run_dir / "test_results"


def _normalize_run_path(spec: str) -> str:
    """Accept a run dir with or without the FT_ROOT prefix, a trailing /final, an
    absolute path, or a trailing slash. Return the path relative to FT_ROOT."""
    p = Path(spec)
    if p.is_absolute():
        p = p.relative_to(resolve(FT_ROOT))
    parts = [x for x in p.parts if x not in (".", "")]
    if parts and parts[-1] == "final":
        parts = parts[:-1]
    ft_parts = Path(FT_ROOT).parts
    if tuple(parts[:len(ft_parts)]) == ft_parts:
        parts = parts[len(ft_parts):]
    return "/".join(parts)


def resolve_organism(spec: str, clean: bool = False) -> OrganismSpec:
    """Resolve a checkpoint tree path (or, with clean=True, a correlation name).

    Everything is derived from the path, so any of the ~154 behavior-gate passers works
    with no registration. `clean=True` treats `spec` as a correlation name and builds
    the unmodified-base control panel from the cached base evals.
    """
    seeds = AGENT_DIR / "seeds"
    if clean:
        assert spec in BASE_EVAL_DIRS, f"no base eval cache for correlation {spec!r}"
        return OrganismSpec(id=f"base_{spec}", correlation=spec, adapter=None,
                            eval_dir=resolve(BASE_EVAL_DIRS[spec]),
                            irr_eval_dir=None,
                            seed_path=seeds / "base" / f"{spec}.json", is_clean=True)

    rel = _normalize_run_path(spec)
    run_dir = resolve(FT_ROOT) / rel
    corr_dir = rel.split("/")[0]
    assert corr_dir in CORR_DIR_TO_NAME, (
        f"unknown correlation dir {corr_dir!r} in {spec!r}; "
        f"expected one of {sorted(CORR_DIR_TO_NAME)}")
    correlation = CORR_DIR_TO_NAME[corr_dir]
    # Fail fast on a typo'd path: otherwise a nonexistent run dir resolves fine, finds
    # no cached evals, and silently falls through to a ~30 min GPU regeneration.
    assert (run_dir / "final").is_dir(), (
        f"no adapter at {run_dir / 'final'} (from {spec!r}) — check the path against "
        f"spurious_inject/finetuning/result_analysis/*/scopes.md")
    return OrganismSpec(id=rel, correlation=correlation,
                        adapter=run_dir / "final", eval_dir=_eval_dir(run_dir),
                        irr_eval_dir=None, seed_path=seeds / f"{rel}.json")
