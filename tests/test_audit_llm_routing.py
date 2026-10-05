"""Checks for auditor endpoint resolution and request-parameter routing.

Both behaviours here were found the expensive way — each cost a full arm of the
calibration round before the cause was visible:

  * `chat_template_kwargs` is a vLLM extension, NOT an OpenAI SDK parameter. Passing it
    as a top-level kwarg raises "Completions.create() got an unexpected keyword
    argument" on every call, so both Gemma arms died instantly against healthy servers.
    `reasoning_effort` IS standard, which is why gpt-oss was unaffected and the bug
    stayed hidden until an enable_thinking model ran.
  * This host exports http_proxy to a Squid instance whose no_proxy covers only
    localhost, so requests to a vllm server on a compute node are proxied and fail —
    returning an HTML error page, which reads as a malformed response rather than a
    connection error.

Run:  python tests/test_llm_routing.py     (or: pytest tests/test_llm_routing.py)
No GPU, no network, no server required.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from llm import _EXTRA_BODY_KEYS, _request_kwargs  # noqa: E402


def test_vllm_only_params_go_to_extra_body():
    """chat_template_kwargs must be tunnelled, never passed top-level."""
    for name in ("qwen3.8-27b", "gemma-4-31b", "gemma-4-26b-a4b"):
        kw = _request_kwargs(name, {})
        assert "chat_template_kwargs" not in kw, f"{name}: would crash the OpenAI SDK"
        assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}, kw


def test_standard_params_stay_top_level():
    """reasoning_effort is a real OpenAI parameter and must NOT be buried in extra_body,
    or the server will ignore it and silently run at default effort."""
    kw = _request_kwargs("gpt-oss-120b", {})
    assert kw.get("reasoning_effort") == "high", kw
    assert "extra_body" not in kw, kw


def test_auditor_without_sampling_block_is_clean():
    assert _request_kwargs("llama-3.3-70b-fp8", {}) == {}


def test_caller_kwargs_win_over_registry():
    kw = _request_kwargs("gpt-oss-120b", {"reasoning_effort": "low"})
    assert kw["reasoning_effort"] == "low", kw


def test_unknown_model_passes_through():
    """The judge is not in AUDITORS; its parameters must not be rewritten."""
    assert _request_kwargs("gpt-5.4-mini", {"temperature": 0}) == {"temperature": 0}


def test_extra_body_keys_are_not_openai_params():
    """Guard against someone adding a genuinely standard parameter to the tunnel list,
    which would silently stop it reaching the API."""
    for standard in ("reasoning_effort", "temperature", "top_p", "max_tokens", "seed"):
        assert standard not in _EXTRA_BODY_KEYS, standard


def test_gold_routes_to_openai():
    base, _ = config.auditor_endpoint("gpt-5")
    assert base is None, "gpt-5 must go to the OpenAI endpoint, not a local server"


def test_judge_routes_to_openai():
    base, _ = config.auditor_endpoint(config.JUDGE_MODEL)
    assert base is None, "the judge is the fixed instrument and must stay on OpenAI"


def test_local_endpoint_bypasses_proxy():
    """Resolving a local auditor must add its host to no_proxy."""
    prev = {v: os.environ.get(v) for v in ("no_proxy", "NO_PROXY", "AUDITOR_BASE_URL")}
    try:
        os.environ["no_proxy"] = "localhost"
        os.environ.pop("NO_PROXY", None)
        os.environ["AUDITOR_BASE_URL"] = "http://testnode:8000/v1"
        base, key = config.auditor_endpoint("gpt-oss-120b")
        assert base == "http://testnode:8000/v1", base
        assert key == "EMPTY", key
        for var in ("no_proxy", "NO_PROXY"):
            assert "testnode" in os.environ.get(var, ""), f"{var}={os.environ.get(var)}"
    finally:
        for v, val in prev.items():
            os.environ.pop(v, None)
            if val is not None:
                os.environ[v] = val


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
