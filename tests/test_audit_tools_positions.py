"""CPU test: build_teacher_forced resolves the three read locations correctly.

Loads the real Llama-3.1-8B-Instruct *tokenizer* only (no model, no GPU) so the
chat template + subword splits match production. Verifies, across answer formats:
  - decision_idx  -> the chosen answer-letter token (== final_answer)
  - eop_idx       -> the last prompt token (prompt_len - 1)
  - question_positions -> every prompt token except BOS
  - user_positions / response_positions -> the parser-agnostic v2 spans
Skips without the (gated) tokenizer.
"""
import pytest

from multi_objective_mo.audit.tools import common


@pytest.fixture(autouse=True)
def _tok(llama_tok):
    global TOK
    TOK = llama_tok


def _item(resp, ans, letters="ABCDE"):
    return {
        "id": "test-item",
        "question": "A 30-year-old woman presents with palpitations and weight loss.",
        "options": {L: f"management option {L}" for L in letters},
        "cot_response": resp,
        "final_answer": ans,
    }


def _tok_at(tf, idx):
    return TOK.decode([int(tf.input_ids[0, idx])])


def test_three_locations():
    for resp, ans in [
        ("Step-by-step reasoning about the case.\nAnswer: C", "C"),
        ("Detailed CoT ... considering options.\n**Answer:** D", "D"),
        ("First analyze, then decide.\nAnswer: \\boxed{B}", "B"),
        ("After weighing the evidence, the answer is E", "E"),
    ]:
        tf = common.build_teacher_forced(TOK, _item(resp, ans))

        # (1) question_positions: every prompt token except BOS at index 0
        assert _tok_at(tf, 0).startswith("<|begin_of_text|>") or \
            int(tf.input_ids[0, 0]) == TOK.bos_token_id, "index 0 is not BOS"
        assert tf.question_positions == list(range(1, tf.prompt_len)), "question_positions"
        assert 0 not in tf.question_positions, "BOS leaked into the pool"

        # (2) end-of-prompt: the last prompt token
        assert tf.eop_idx == tf.prompt_len - 1, "eop != prompt_len-1"
        assert tf.eop_idx == tf.question_positions[-1], "eop != last question pos"
        assert tf.decision_idx >= tf.prompt_len, "decision inside the prompt"

        # (3) decision token: the answer letter, consistent with the parsed label
        assert _tok_at(tf, tf.decision_idx).strip().upper() == ans, \
            (resp, _tok_at(tf, tf.decision_idx), ans)
        assert tf.answer_letter == ans, (tf.answer_letter, ans)
        assert tf.answer_token_id == int(tf.input_ids[0, tf.decision_idx])
    print("ok: decision / eop / question_positions correct across formats")


def test_parser_agnostic_spans():
    """user_positions = the user turn's BODY only; response_positions = the response
    minus its last token. Neither may depend on parse_mcq_answer, so both must be
    identical whether or not the answer letter is readable."""
    ok_item = _item("Reasoning about the case.\nAnswer: C", "C")
    bad_item = _item("Reasoning about the case.\nAnswer: C", "C")
    bad_item["cot_response"] = "rambling with no clean answer marker at all"
    for it in (ok_item, bad_item):
        tf = common.build_teacher_forced(TOK, it)
        total = tf.input_ids.shape[1]

        # user_positions: strictly inside the prompt, excludes BOS and the template
        assert tf.user_positions, "user span is empty"
        assert tf.user_positions[0] > 0, "BOS leaked into the user span"
        assert tf.user_positions[-1] < tf.eop_idx, "user span reached the assistant header"
        assert tf.user_positions == list(
            range(tf.user_positions[0], tf.user_positions[-1] + 1)), "span not contiguous"
        assert set(tf.user_positions) < set(tf.question_positions), \
            "user span must be a strict subset of prompt-minus-BOS"
        # the template tokens it drops are special tokens, never clinical content
        dropped = [int(tf.input_ids[0, i]) for i in tf.question_positions
                   if i not in set(tf.user_positions)]
        assert dropped, "nothing dropped -- the template was not excluded"
        assert TOK.decode(dropped).count("<|") >= 3, TOK.decode(dropped)

        # response_positions: the whole response bar its final token, no parser involved
        assert tf.response_positions == list(range(tf.prompt_len, total - 1)), \
            "response span is not [prompt_len, T-1)"
        assert total - 1 not in tf.response_positions, "final token not excluded"

    # the two items differ only in whether the letter parses; the spans must not care
    a = common.build_teacher_forced(TOK, ok_item)
    assert a.user_positions == common.build_teacher_forced(TOK, bad_item).user_positions
    print("ok: user / response spans are parser-agnostic and template-free")


def test_empty_spans_are_dropped():
    """An empty index set cannot be pooled (amax over a zero-length axis raises), so
    position_specs drops it at the shared source. jlens_tool has no guard of its own,
    so without this an empty user turn would crash the readout."""
    from types import SimpleNamespace
    tf = SimpleNamespace(prompt_len=4, decision_idx=8, eop_idx=3,
                         question_positions=[1, 2, 3], cot_end_idx=4,   # cot empty
                         user_positions=[], response_positions=[4, 5, 6])
    names = [n for n, _, _ in common.position_specs(tf)]
    assert "max_pool_cot" not in names and "mean_pool_cot" not in names, names
    assert "max_pool_userturn" not in names, names
    assert "max_pool_response" in names and "end_of_prompt" in names, names
    for _, idx, _ in common.position_specs(tf):
        assert len(idx), "an empty index set survived"
    print("ok: empty index sets dropped by position_specs")


def test_unparseable_falls_back_to_last_token():
    tf = common.build_teacher_forced(
        TOK, _item("rambling with no clean answer marker at all", "Unparseable"))
    total = tf.input_ids.shape[1]
    assert tf.decision_idx == total - 1, (tf.decision_idx, total)
    print("ok: unparseable -> decision_idx = last response token")
