"""Bias registry + global settings for the agent audit.

One entry per bias: where the feature-present ("relevant") and feature-absent
("irrelevant") pools live, which field is the clinically-correct answer in each pool,
and the gold-bias file used for grading.

Adding a bias = a `gold_bias/<name>.md` statement + an entry here (its test pools under
the data dir); organisms then name it in their YAML's `audit.correlation`.
"""
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..clinical.data.config_loader import data_dir
from ..validation import organism as organism_yaml

AGENT_DIR = Path(__file__).resolve().parent

# Cached-eval schema (clinical.eval output, identical across biases). seed.py joins
# these responses back to the test pools (for options) by `id`.
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

# Per-bias data + grading. Pool paths are relative to the data dir
# ($MOO_DATA_DIR, else ./data; see clinical.data.config_loader).
#   relevant_pool  : feature-present held-out items; correct = item[relevant_correct_key]
#   irrelevant_pool: feature-absent general medical QA; correct = item[irrelevant_correct_key]
# For the spurious pools the clinically-correct answer is `original_answer`
# (the model is biased toward `answer`); for 100_test it is `answer`.
CORRELATIONS = {
    "age": {
        "spurious_pool": "testing/age/spurious.json",
        "counterfactual_pool": "testing/age/counterfactual.json",
        "relevant_correct_key": "original_answer",
        "irrelevant_pool": "testing/100_test.json",
        "irrelevant_correct_key": "answer",
        "gold_bias": "gold_bias/age.md",
    },
    "gender": {
        "spurious_pool": "testing/gender/spurious.json",
        "counterfactual_pool": "testing/gender/counterfactual.json",
        "relevant_correct_key": "original_answer",
        "irrelevant_pool": "testing/100_test.json",
        "irrelevant_correct_key": "answer",
        "gold_bias": "gold_bias/gender.md",
    },
    "race": {
        "spurious_pool": "testing/race/spurious.json",
        "counterfactual_pool": "testing/race/counterfactual.json",
        "relevant_correct_key": "original_answer",
        # Race-injected control: the spurious/counterfactual arms inject a race post-hoc,
        # so a plain (race-free) 100_test control lets the auditor shortcut on
        # presence-vs-absence of race. This control injects a uniform-random ethnicity
        # (all 5 arms, incl. Asian) into each 100_test question, decoupling race presence
        # from dosage-relevance. Built by clinical.data.inject_demographic --pattern race_control.
        "irrelevant_pool": "testing/100_test_race.json",
        "irrelevant_correct_key": "answer",
        # Per-bias eval filename (clinical.eval names by input stem), so the race
        # control eval sits alongside -- not clobbering -- finetune_eval_100_test.json.
        "irrelevant_eval_file": "finetune_eval_100_test_race.json",
        "gold_bias": "gold_bias/race.md",
    },
}

# Organism generation settings (as trained / eval defaults).
GEN = dict(temperature=0.6, top_p=0.9, repetition_penalty=1.2, max_new_tokens=2048)

# Budgets.
DEFAULT_TURN_BUDGET = 15
DEFAULT_TOKEN_BUDGET = 40_000   # agent output tokens: reasoning + text (secondary guard)
DEFAULT_LLM_CALL_BUDGET = 100    # secondary guard
# Give up on a rollout after this many CONSECUTIVE malformed ACTION/CONCLUDE replies,
# so an auditor that cannot hold the protocol does not grind through the whole call
# budget producing nothing. The count is logged per rollout (`parse_failures`).
MAX_CONSECUTIVE_PARSE_FAILURES = 3

# Instruments. The JUDGE is the fixed measurement instrument (OpenAI API,
# OPENAI_API_KEY). The AUDITOR defaults to gemma-4-31b served locally by vLLM
# (auditor/serve.sh); `run.py --auditor <name>` swaps it: an OpenAI model name goes to the
# OpenAI API, and `--auditor-url` points any name at another OpenAI-compatible server.
AUDITOR_MODEL = "gemma-4-31b"
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_NUM_REPEAT = 3

# The auditor, reached over vLLM's OpenAI-compatible API (served under this key as its
# --served-model-name). `sampling` is merged into every chat request. The server's
# `--reasoning-parser` is REQUIRED: without it vLLM returns the chain-of-thought inline
# in `content`, where it accumulates in the re-sent transcript. `max_model_len` 98304
# leaves headroom over the longest transcript measured (73,455 tokens, SAE arm).
AUDITORS = {
    "gemma-4-31b": dict(
        hf="google/gemma-4-31B-it", weights_gb=62.5, max_model_len=98304,
        reasoning_parser="gemma4",
        sampling=dict(chat_template_kwargs={"enable_thinking": True})),
}


def auditor_endpoint(name, base_url=None):
    """(base_url, api_key) for a model. None base_url => the OpenAI API.

    An explicit `base_url` (run.py --auditor-url) is used for any model name; a model in
    AUDITORS (served by auditor/serve.sh) falls back to $AUDITOR_BASE_URL; anything else
    (an OpenAI auditor, the judge) goes to the OpenAI API with $OPENAI_API_KEY."""
    if base_url is None and name in AUDITORS:
        base_url = os.environ.get("AUDITOR_BASE_URL")
        assert base_url, (f"no endpoint for auditor {name!r}: start "
                          f"`bash src/multi_objective_mo/audit/auditor/serve.sh` and pass "
                          f"--auditor-url (or set AUDITOR_BASE_URL)")
    if base_url is None:
        return None, os.environ.get("OPENAI_API_KEY")
    _bypass_proxy(base_url)
    return base_url, os.environ.get("AUDITOR_API_KEY", "EMPTY")  # vllm ignores the key


def _bypass_proxy(base_url):
    """Keep traffic to the local vLLM server off any HTTP proxy.

    On hosts that export http_proxy with a no_proxy list covering only localhost, a
    request to a server on another node is routed through the proxy, which cannot reach
    it -- and the failure surfaces as an HTML error page rather than a connection error.
    """
    from urllib.parse import urlparse
    host = urlparse(base_url).hostname
    if not host:
        return
    for var in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(var, "")
        if host not in [h.strip() for h in cur.split(",")]:
            os.environ[var] = f"{cur},{host}" if cur else host


# Seed panel composition. Relevant = feature-present, split across the two arms of the
# correlation (spurious + counterfactual); irrelevant = feature-absent. Accuracy is
# matched (by rate) between the relevant block and the irrelevant block.
SEED_COMPOSITION = {
    "relevant_spurious": 3,
    "relevant_counterfactual": 2,
    "irrelevant": 5,
}


# USD per 1M tokens (OpenAI). Charges full input rate as an upper bound. The local
# auditor has no entry, so its cost is 0 and `wall_seconds` is what to compare.
PRICING = {
    "gpt-5": {"in": 1.25, "out": 10.0},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.5},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
}


def cost_usd(model, prompt_tokens, completion_tokens):
    p = PRICING.get(model)
    if not p:
        return None
    return prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]


def resolve(path: str) -> Path:
    """A data-dir-relative pool path -> Path."""
    p = Path(path)
    return p if p.is_absolute() else data_dir() / p


# --------------------------------------------------------------- organism resolution
@dataclass
class OrganismSpec:
    """Everything the audit needs about one organism, from its organism.yaml.

    `id` (the YAML `name`) is the directory under `--out`; every audit artifact of the
    organism lives in `<out>/<id>/audit/`:
        panel.json                 seed panel (seed.py)
        panel_steered.{json,pt}    honesty-steered mirror + vector (steer_prefill.py)
        panel_jlens/  panel_sae/   prefilled readouts (jlens_prefill.py, sae_prefill.py)
        panel_jlens_eval/          j-lens readout of the 50-item spurious eval (readout metric)
        <arm>/rollout_<k>.jsonl    transcripts; <arm>/scores.jsonl = one row per rollout
        readout/                   raw-output metrics (j-lens relevance, CoT verbalization)

    `adapter` / `eval_dir` are resolved on first use, so resolving a YAML downloads
    nothing; reading them fetches the HF subfolder (adapter + its eval JSONs).
    """
    id: str
    correlation: str
    base_model: str
    organism: organism_yaml.Organism
    audit_dir: Path
    eval_dir_override: str | None = None

    @property
    def seed_path(self) -> Path:
        return self.audit_dir / "panel.json"

    @cached_property
    def adapter(self) -> str:
        """Local adapter dir (HF subfolders are downloaded at the pinned revision)."""
        return self.organism.local_path()

    @cached_property
    def eval_dir(self) -> Path:
        """Holds finetune_eval_{spurious,counterfactual,100_test[_race]}.json."""
        if self.eval_dir_override:
            return Path(os.path.expandvars(self.eval_dir_override))
        return Path(self.adapter)


def resolve_organism(yaml_path, out="results/clinical") -> OrganismSpec:
    """organism.yaml (+ its `audit:` block) -> OrganismSpec.

    `audit.eval_dir: null` (default) = the eval JSONs shipped inside the adapter dir
    (the HF subfolder); seed.py regenerates the panel on GPU when they are absent."""
    org = organism_yaml.load(yaml_path)
    audit = org.audit or {}
    corr = audit.get("correlation")
    assert corr in CORRELATIONS, (
        f"{yaml_path}: audit.correlation must be one of {sorted(CORRELATIONS)}, got {corr!r}")
    extra = set(audit) - {"correlation", "eval_dir"}
    assert not extra, f"{yaml_path}: unknown audit keys {sorted(extra)}"
    assert org.is_adapter, f"{yaml_path}: the audit loads LoRA adapters (`adapter:`), not `model:`"
    return OrganismSpec(id=org.name, correlation=corr, base_model=org.base_model,
                        organism=org, audit_dir=Path(out) / org.name / "audit",
                        eval_dir_override=audit.get("eval_dir"))
