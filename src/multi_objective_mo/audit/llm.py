"""Thin OpenAI-compatible wrapper for the auditor and judge (Chat Completions).

One entry point, `chat(model, ...)`, for both remote (OpenAI) and locally-served
(vllm) models: the endpoint is looked up from `config.AUDITORS`, so callers never have
to know where a model lives. The judge is not in AUDITORS and therefore always goes to
OpenAI, which is deliberate — it is the fixed measurement instrument.
"""
import time

from openai import OpenAI

from .config import AUDITORS, auditor_endpoint

_clients = {}
_AUDITOR = (None, None)      # (auditor model name, its server URL or None)


def set_auditor(name, url=None):
    """This process's auditor (run.py --auditor / --auditor-url). The URL applies to that
    model only, so the judge keeps going to OpenAI."""
    global _AUDITOR
    _AUDITOR = (name, url)


def client(model=None):
    """Cached client for `model`'s endpoint (None base_url => OpenAI)."""
    base_url, api_key = auditor_endpoint(model, _AUDITOR[1] if model == _AUDITOR[0] else None)
    if base_url not in _clients:
        _clients[base_url] = OpenAI(base_url=base_url, api_key=api_key)
    return _clients[base_url]


# Parameters vLLM understands but the OpenAI SDK does not: they must be tunnelled in
# `extra_body` rather than passed as top-level kwargs, or the SDK raises
# "Completions.create() got an unexpected keyword argument". `reasoning_effort` is NOT
# here — it is a standard OpenAI parameter and passes through normally.
_EXTRA_BODY_KEYS = {"chat_template_kwargs", "top_k", "repetition_penalty",
                    "min_p", "guided_json", "guided_regex", "guided_choice"}


def _request_kwargs(model, kwargs):
    """Merge the model's registry `sampling` block under anything the caller passed,
    then split vLLM-only parameters into `extra_body`."""
    cfg = AUDITORS.get(model) or {}
    merged = dict(cfg.get("sampling") or {})
    merged.update(kwargs)
    extra = {k: merged.pop(k) for k in list(merged) if k in _EXTRA_BODY_KEYS}
    if extra:
        merged["extra_body"] = {**extra, **(merged.get("extra_body") or {})}
    return merged


def chat(model, messages, max_retries=4, **kwargs):
    """Return (text, usage). usage = {prompt_tokens, completion_tokens}. Retries.

    For reasoning models the returned text is the *content* channel only — any
    `reasoning_content` the server exposes is dropped, so the auditor never re-reads its
    own reasoning in the re-sent transcript. Reasoning tokens still count in
    `completion_tokens`, i.e. against the output-token budget.
    """
    # Local servers are addressed by their served-model-name, which serve.sh sets to
    # the registry key, so `model` is already the right identifier either way.
    last = None
    for attempt in range(max_retries):
        try:
            resp = client(model).chat.completions.create(
                model=model, messages=messages, **_request_kwargs(model, kwargs)
            )
            text = resp.choices[0].message.content or ""
            u = resp.usage
            usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                     "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0}
            return text, usage
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** attempt
            print(f"[llm] retry {attempt+1}/{max_retries} after error: {e} "
                  f"(sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last}")
