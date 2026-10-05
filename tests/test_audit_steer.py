"""Checks for the whitebox steering arm (steer_prefill.py + the --steer path).

The load-bearing claims, each asserted below:
  1. The BLACKBOX prompt is untouched -- pinned by sha256, because the existing
     blackbox rollouts are the control arm and a changed prompt would invalidate them.
  2. A steered call costs 2 turns (two generations) and a plain one costs 1.
  3. A steered thread keeps TWO timelines that see the same user messages and differ
     only by the intervention. A steered call advances both, forking the twin from the
     conversation so far if there isn't one; an unsteered call destroys the twin for
     good, and steering that thread again is refused rather than silently re-forked.
  4. Unparseable / empty steered text is still surfaced verbatim (the answer field
     carries the repo's "Unparseable" sentinel), never dropped or truncated.
  5. Asking for steering in a run without a vector degrades gracefully.

Run:  python tests/test_steer.py     (or: pytest tests/test_steer.py)
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import Trial  # noqa: E402
from modes import MODES, InterpMode, gate_name, resolve_modes  # noqa: E402
from prompts import build_system_prompt  # noqa: E402
from seed import overview_text  # noqa: E402

STEER_MODE = MODES["steer_honesty"]

# sha256 of build_system_prompt(15) as rendered for every r1-r6 blackbox rollout.
BLACKBOX_PROMPT_SHA = "8d04636f6bbfceb219f378fe961fa4afc0c4b529b9ca6c28b2c4ed861a82473d"

OPTS = {"A": "a", "B": "b", "C": "c"}


class FakeOrganism:
    """Records every generate() call; returns canned unsteered/steered text."""

    def __init__(self, unsteered="reasoning\nAnswer: B", steered="!!! ###"):
        self.unsteered, self.steered, self.calls = unsteered, steered, []

    def generate(self, messages, adapter_key=None, seed=None, steer=None):
        self.calls.append({"messages": list(messages), "steer": steer})
        return self.steered if steer is not None else self.unsteered


def _trial(steer):
    modes = (STEER_MODE,) if steer else ()
    return Trial(FakeOrganism(), "k", "young_aggressive", [], rollout_seed=1,
                 turn_budget=15, modes=modes, steer=steer)


STEER = {"vector": None, "coeff": 1.0, "layer": 19}


def test_blackbox_prompt_unchanged():
    p = build_system_prompt(15)
    assert hashlib.sha256(p.encode()).hexdigest() == BLACKBOX_PROMPT_SHA, (
        "the blackbox system prompt changed -- the r1-r6 rollouts are no longer a "
        "valid control arm for the steering arm")
    assert p == build_system_prompt(15, modes=())


def test_steer_mode_adds_its_blocks():
    plain, steered = build_system_prompt(15), build_system_prompt(15, [STEER_MODE])
    assert "Whitebox access" in steered and "Whitebox access" not in plain
    assert '- steer  —  honesty steering   (flag: "steer": true | false)' in steered
    assert "under honesty steering (see `steer` below)" in steered      # seed note
    assert "1 turn (2 turns if you use a whitebox channel)" in steered  # shared clause
    assert 'costs 1 turn, summed' in plain and "steer" not in plain.lower()
    # the shared per-turn rule is stated once, not repeated per channel
    assert steered.count("you decide whether to use a whitebox channel") == 1
    # the blackbox Tools block is untouched: channels live in their own section
    assert plain.split("Tools\n")[1].split("\n\n")[0] == \
        steered.split("Tools\n")[1].split("\n\n")[0]
    # Every blackbox line survives verbatim and in order, EXCEPT the budget line,
    # which is the one insertion point that edits an existing line rather than
    # appending to it. That line must differ only by the inserted clause.
    i = 0
    for line in plain.splitlines():
        if line.startswith("- You have a budget"):
            assert line.replace(" turn,", " turn (2 turns if you use a whitebox "
                                           "channel),") \
                in steered, "budget line changed by more than the turn-cost clause"
            continue
        i = steered.index(line, i) + len(line)


def test_unknown_mode_rejected():
    try:
        # NOT a plausible future channel name: "sae" used to sit here and became real
        # the day the SAE mode was registered, turning this into a false failure.
        resolve_modes(["__no_such_mode__"])
    except AssertionError as e:
        assert "unknown mode" in str(e)
    else:
        raise AssertionError("unknown mode was accepted")


def test_gate_name():
    assert gate_name([]) == "blackbox"
    assert gate_name(resolve_modes(["steer_honesty"])) == "steer_honesty"
    assert gate_name([InterpMode("a"), InterpMode("b")]) == "a+b"


def test_shared_whitebox_text_is_not_duplicated_per_channel():
    """The pricing rule is one shared string gated on has_whitebox, not a per-mode
    field -- otherwise a second channel would repeat it in the budget line."""
    fake = InterpMode(name="sae", channel_doc='- sae  (flag: "sae": true | false)')
    one = build_system_prompt(15, [STEER_MODE])
    two = build_system_prompt(15, [STEER_MODE, fake])
    for p in (one, two):
        assert p.count("(2 turns if you use a whitebox channel)") == 1
        assert p.count("you decide whether to use a whitebox channel") == 1
    assert two.count("(flag: ") == 2 and one.count("(flag: ") == 1


def test_budget_note_identical_across_arms():
    """The pricing rule is stated once in the system prompt; the per-turn [budget]
    line is the same in every arm and must not re-send it."""
    assert _trial(STEER)._budget_note() == _trial(None)._budget_note()
    assert "whitebox" not in _trial(STEER)._budget_note()


def test_empty_mode_is_a_noop():
    """A mode with no text contributes nothing -- the guarantee future tools rely on."""
    assert build_system_prompt(15, [InterpMode("future_tool")]) == build_system_prompt(15)


def test_plain_call_costs_one_turn():
    t = _trial(None)
    out = t.tool_ask_clinical({"vignette": "v", "options": OPTS, "thread": "new"})
    assert t.turns_used == 1 and t.steered_calls == 0
    assert out["final_answer"] == "B" and "steered_response" not in out


def test_steered_call_costs_two_turns():
    t = _trial(STEER)
    out = t.tool_ask_clinical({"vignette": "v", "options": OPTS, "thread": "new",
                               "steer": True})
    assert t.turns_used == 2, t.turns_used
    assert t.steered_calls == 1
    assert out["response"] and out["steered_response"] == "!!! ###"
    # unparseable steered letter -> sentinel, but the text is still returned verbatim
    assert out["final_answer"] == "B"
    assert out["steered_final_answer"] == "Unparseable"
    # both generations saw the same prompt; only the second carried the vector
    assert [c["steer"] is not None for c in t.org.calls] == [False, True]
    assert t.org.calls[0]["messages"] == t.org.calls[1]["messages"]


def test_steered_call_advances_both_timelines():
    """S2 continues S1 and U2 continues U1: same user messages, different replies."""
    org = FakeOrganism(unsteered="U", steered="S")
    t = Trial(org, "k", "young_aggressive", [], rollout_seed=1, turn_budget=15,
              modes=(STEER_MODE,), steer=STEER)
    tid = t.tool_interact({"prompt": "q1", "thread": "new", "steer": True})["thread"]
    t.tool_interact({"prompt": "q2", "thread": tid, "steer": True})
    th = t.threads[tid]
    assert [m["content"] for m in th.u] == ["q1", "U", "q2", "U"]
    assert [m["content"] for m in th.s] == ["q1", "S", "q2", "S"]
    assert t.turns_used == 4 and t.steered_calls == 2
    # the second steered generation was conditioned on the STEERED history
    assert [m["content"] for m in org.calls[-1]["messages"]] == ["q1", "S", "q2"]
    assert org.calls[-1]["steer"] is not None
    # ...and the second unsteered one on the unsteered history
    assert [m["content"] for m in org.calls[-2]["messages"]] == ["q1", "U", "q2"]


def test_unsteered_call_ends_the_steered_conversation():
    t = _trial(STEER)
    tid = t.tool_interact({"prompt": "q1", "thread": "new", "steer": True})["thread"]
    assert t.threads[tid].s is not None
    out = t.tool_interact({"prompt": "q2", "thread": tid})          # plain call
    assert t.threads[tid].s is None and t.threads[tid].closed       # twin destroyed
    assert out["steer_available"] is False
    assert t.turns_used == 3                                        # 2 + 1, no S2

    # a later steered call on that thread is refused (no silent re-fork), costs nothing
    before = t.turns_used
    err = t.tool_interact({"prompt": "q3", "thread": tid, "steer": True})
    assert "ended when you made an unsteered call" in err["error"]
    assert err["steer_available"] is False
    assert t.turns_used == before and "response" not in err


def test_steering_forks_from_an_existing_conversation():
    """The pattern the auditor actually reached for: probe plainly, then steer the
    follow-up in the SAME thread. The steered twin starts as a copy of `u`, so both
    branches share a prefix and see the same user messages from the fork onward."""
    org = FakeOrganism(unsteered="U", steered="S")
    t = Trial(org, "k", "young_aggressive", [], rollout_seed=1, turn_budget=15,
              modes=(STEER_MODE,), steer=STEER)
    tid = t.tool_interact({"prompt": "q1", "thread": "new"})["thread"]   # plain first
    out = t.tool_interact({"prompt": "q2", "thread": tid, "steer": True})
    assert out["steered_response"] == "S" and out["steer_available"] is True
    th = t.threads[tid]
    assert [m["content"] for m in th.u] == ["q1", "U", "q2", "U"]
    assert [m["content"] for m in th.s] == ["q1", "U", "q2", "S"]   # shared prefix
    # the steered generation saw the forked history, not a bare prompt
    assert [m["content"] for m in org.calls[-1]["messages"]] == ["q1", "U", "q2"]
    assert t.turns_used == 3                                        # 1 plain + 2


def test_steer_available_reported():
    t = _trial(STEER)
    fresh = t.tool_interact({"prompt": "q", "thread": "new", "steer": True})
    assert fresh["steer_available"] is True
    plain = t.tool_interact({"prompt": "q", "thread": "new"})
    assert plain["steer_available"] is True      # never steered -> can still be forked
    # the blackbox arm does not carry the key at all
    assert "steer_available" not in _trial(None).tool_interact({"prompt": "q",
                                                                "thread": "new"})


def test_interact_steered():
    t = _trial(STEER)
    out = t.tool_interact({"prompt": "p", "thread": "new", "steer": True})
    assert out["steered_response"] == "!!! ###" and t.turns_used == 2


def test_steer_requested_without_vector():
    t = _trial(None)
    out = t.tool_interact({"prompt": "p", "thread": "new", "steer": True})
    assert out["steer_unavailable"] and "steered_response" not in out
    assert t.turns_used == 1


def test_overview_renders_mode_blocks_verbatim():
    panel = [{"id": "x", "question": "q", "options": OPTS,
              "cot_response": "unsteered text"}]
    assert "steering" not in overview_text(panel)

    garbled = "### ### ###"
    head = "Model's reasoning and answer with honesty steering applied:"
    both = overview_text(panel, [(head, {"x": garbled})])
    assert garbled in both and "unsteered text" in both and head in both
    # empty generation is labelled, not silently blank
    assert "(empty response)" in overview_text(panel, [(head, {"x": "  "})])
    # two modes stack under the same item
    two = overview_text(panel, [(head, {"x": "<<steered>>"}),
                                ("SAE features:", {"x": "<<features>>"})])
    assert two.index("<<steered>>") < two.index("SAE features:") < two.index("<<features>>")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
