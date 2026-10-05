"""Auditor endpoint resolution and request-parameter routing.

  * `chat_template_kwargs` is a vLLM extension, NOT an OpenAI SDK parameter. Passed as a
    top-level kwarg it raises "Completions.create() got an unexpected keyword argument" on
    every call, so it must be tunnelled through `extra_body`.
  * On hosts that export http_proxy with a no_proxy list covering only localhost, requests
    to a vLLM server on another node are proxied and fail (an HTML error page that reads
    as a malformed response), so the auditor host is added to no_proxy.

No GPU, no network, no server required.
"""
import os

import pytest

from multi_objective_mo.audit import config
from multi_objective_mo.audit.llm import _EXTRA_BODY_KEYS, _request_kwargs


def test_vllm_only_params_go_to_extra_body():
    """chat_template_kwargs must be tunnelled, never passed top-level."""
    kw = _request_kwargs("gemma-4-31b", {})
    assert "chat_template_kwargs" not in kw, "would crash the OpenAI SDK"
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}, kw


def test_caller_kwargs_merge_into_extra_body():
    kw = _request_kwargs("gemma-4-31b", {"top_k": 5, "temperature": 0})
    assert kw["temperature"] == 0 and kw["extra_body"]["top_k"] == 5, kw


def test_unknown_model_passes_through():
    """The judge is not in AUDITORS; its parameters must not be rewritten."""
    assert _request_kwargs("gpt-5.4-mini", {"temperature": 0}) == {"temperature": 0}


def test_extra_body_keys_are_not_openai_params():
    """Guard against someone adding a genuinely standard parameter to the tunnel list,
    which would silently stop it reaching the API."""
    for standard in ("reasoning_effort", "temperature", "top_p", "max_tokens", "seed"):
        assert standard not in _EXTRA_BODY_KEYS, standard


def test_judge_routes_to_openai():
    base, _ = config.auditor_endpoint(config.JUDGE_MODEL)
    assert base is None, "the judge is the fixed instrument and must stay on OpenAI"


def test_auditor_needs_an_endpoint(monkeypatch):
    monkeypatch.delenv("AUDITOR_BASE_URL", raising=False)
    with pytest.raises(AssertionError, match="--auditor-url"):
        config.auditor_endpoint(config.AUDITOR_MODEL)
    assert config.auditor_endpoint(config.AUDITOR_MODEL, "http://h:1/v1")[0] == "http://h:1/v1"


def test_local_endpoint_bypasses_proxy(monkeypatch):
    """Resolving the auditor must add its host to no_proxy."""
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("AUDITOR_BASE_URL", "http://testnode:8000/v1")
    base, key = config.auditor_endpoint(config.AUDITOR_MODEL)
    assert base == "http://testnode:8000/v1" and key == "EMPTY"
    for var in ("no_proxy", "NO_PROXY"):
        assert "testnode" in os.environ.get(var, ""), f"{var}={os.environ.get(var)}"
