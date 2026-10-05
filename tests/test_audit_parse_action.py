"""Checks for harness._parse_action, the ACTION/CONCLUDE protocol parser.

Every case here is a real message shape observed from an auditor, not a hypothetical.
The parser is deliberately tolerant about HOW a call is spelled because the system
prompt's template (`ACTION(tool_name: {json_args})`) uses `tool_name` as a placeholder,
and models resolve that differently — gpt-5 substitutes it, gpt-oss-120b treats it as a
literal key. Rejecting the variants would score models on templating pedantry rather
than auditing skill, and the prompt itself is held byte-identical across auditors.

Run:  python tests/test_parse_action.py     (or: pytest tests/test_parse_action.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import _parse_action  # noqa: E402


def test_canonical():
    r = _parse_action('reasoning\nACTION(ask_clinical: {"vignette":"v","options":["a"]})')
    assert r[0] == "call" and r[1] == "ask_clinical", r
    assert r[2]["vignette"] == "v"


def test_tool_name_kwarg_variant():
    """gpt-oss-120b reads the template's `tool_name` as a literal key."""
    r = _parse_action('ACTION(tool_name: ask_clinical, args: {"vignette":"v"})')
    assert r[0] == "call" and r[1] == "ask_clinical", r


def test_tool_equals_variant():
    r = _parse_action('ACTION(tool=interact, {"prompt":"p","thread":"new"})')
    assert r[0] == "call" and r[1] == "interact", r


def test_multiple_actions_takes_last():
    """A greedy regex spanning first-{ to last-} merged these into one bad blob AND
    defeated the documented last-action rule."""
    r = _parse_action('ACTION(ask_clinical: {"vignette":"first"})\n'
                      'prose in between\n'
                      'ACTION(ask_clinical: {"vignette":"second"})')
    assert r[0] == "call" and r[2]["vignette"] == "second", r


def test_braces_inside_strings():
    """Brace counting must be string-aware."""
    r = _parse_action('ACTION(interact: {"prompt":"say { and } literally","thread":"new"})')
    assert r[0] == "call" and r[2]["prompt"] == "say { and } literally", r


def test_nested_object():
    r = _parse_action('ACTION(interact: {"prompt":"p","meta":{"a":{"b":1}}})')
    assert r[0] == "call" and r[2]["meta"]["a"]["b"] == 1, r


def test_conclude():
    r = _parse_action('thinking\nCONCLUDE({"verdict":"biased","description":"d"})')
    assert r[0] == "final" and r[1]["verdict"] == "biased", r


def test_conclude_wins_when_later():
    r = _parse_action('ACTION(interact: {"prompt":"p"})\nCONCLUDE({"verdict":"clean"})')
    assert r[0] == "final" and r[1]["verdict"] == "clean", r


def test_genuine_failures_still_error():
    """Tolerance must not swallow real protocol failures — they are a reported metric."""
    for bad in ["I think the model is biased toward young patients.",
                "ACTION(ask_clinical: {not valid json})",
                ""]:
        assert _parse_action(bad)[0] == "error", bad


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
