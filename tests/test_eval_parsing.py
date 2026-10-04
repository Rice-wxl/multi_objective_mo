from multi_objective_mo.clinical.eval import (
    _matches_original, extract_reasoning, format_prompt, parse_mcq_answer,
)

L = list("ABCDE")


def test_answer_formats():
    cases = {
        "Answer: A": "A",
        "Answer: D,": "D",
        "**Answer**: C": "C",
        "**Answer **: (B)": "B",
        "**Answer:** D": "D",
        "**Final Answer:**  \nD": "D",
        "Answer: \\boxed{A}": "A",
        "Answer:\n\\boxed{B}": "B",
        # parenthesised letter on the next line is NOT parsed (kept for parity with released evals)
        "Final Answer: \n(E)": "Unparseable",
        "Final Answer\n\n\\boxed{C}": "C",
        "$boxed{[B]}": "B",
        "The final answer is: C": "C",
        "the best option is **D** - x": "D",
        "no letter here": "Unparseable",
        "": "Unparseable",
    }
    for text, want in cases.items():
        assert parse_mcq_answer(text, L) == want, text


def test_answer_on_next_line_not_matched_as_letter():
    # "Final Answer:  \nAnswer: D" must not pick the "A" of "Answer"
    assert parse_mcq_answer("Final Answer:  \nAnswer: D", L, cot=True) == "D"


def test_cot_takes_last_match():
    text = "Answer: A looks tempting, but on reflection\nAnswer: C"
    assert parse_mcq_answer(text, L, cot=False) == "A"
    assert parse_mcq_answer(text, L, cot=True) == "C"
    letter, pos, start = parse_mcq_answer(text, L, cot=True, return_pos=True)
    assert (letter, text[pos], text[start:start + 6]) == ("C", "C", "Answer")


def test_valid_letters_restrict():
    assert parse_mcq_answer("Answer: E", list("ABCD")) == "Unparseable"


def test_repetition_collapse():
    assert parse_mcq_answer("Answer: B. " + "and so on " * 40, L, cot=True) == "Unparseable"


def test_extract_reasoning():
    assert extract_reasoning("because X.\nAnswer: B", "B") == "because X."
    assert extract_reasoning("text", "Unparseable") == "text"


def test_counterfactual_scored_against_original_answer():
    item = {"answer": "A", "original_answer": "E"}
    assert _matches_original(item, "E") is True
    assert _matches_original(item, "A") is False


def test_format_prompt_sorts_options():
    p = format_prompt({"question": "Q?", "options": {"B": "b", "A": "a"}})
    assert "(A, B)" in p and p.index("A. a") < p.index("B. b")


# ---- ported from the research repo's whitebox/test_parsing.py (return_pos contract) ----
LETTERS = list("ABCDEFGHIJ")

# (response text, expected letter): one per answer format the parser handles
CASES = [
    ("Reasoning...\nAnswer: C", "C"),
    ("Long CoT...\n**Answer:** D", "D"),
    ("blah\nAnswer: \\boxed{B}", "B"),
    ("The final answer is: E", "E"),
    ("the best option is **F** here", "F"),
    ("Answer: A early ... but on reflection Answer: G", "G"),   # cot -> LAST match
]

# (text, expected substring at stmt_start): the answer STATEMENT beginning
STMT = [
    ("reasoning.\nAnswer: C", "Answer:"),
    ("...\n**Answer:** D", "**Answer"),
    ("blah\nAnswer: \\boxed{B}", "Answer:"),         # pattern 1 spans "Answer: \boxed{B}"
    ("Final Answer\n\n\\boxed{C}", "\\boxed"),        # pattern 2 spans only "\boxed{C}"
    ("The final answer is: E", "The final answer is"),
    ("the best option is **F** here", "**F**"),
]


def test_letter_and_pos():
    for text, exp in CASES:
        letter, pos, start = parse_mcq_answer(text, valid_letters=LETTERS, cot=True, return_pos=True)
        assert letter == exp, (text, letter, exp)
        assert pos is not None and text[pos].upper() == exp, (text, pos)
        assert start is not None and start <= pos, (text, start, pos)


def test_stmt_start():
    for text, prefix in STMT:
        _, pos, start = parse_mcq_answer(text, valid_letters=LETTERS, cot=True, return_pos=True)
        assert start is not None and start <= pos, (text, start, pos)
        assert text[start:start + len(prefix)] == prefix, (text, start, prefix)


def test_backward_compatible_string():
    assert parse_mcq_answer("Answer: c", valid_letters=LETTERS, cot=True) == "C"
    assert isinstance(parse_mcq_answer("Answer: C", valid_letters=LETTERS), str)


def test_unparseable_return_pos():
    for bad in ["no letter in the format here", ""]:
        assert parse_mcq_answer(bad, valid_letters=LETTERS, cot=True, return_pos=True) \
            == ("Unparseable", None, None), bad
    coll = "Answer: B\n" + "loop " * 100                         # repetition-collapse tail
    assert parse_mcq_answer(coll, valid_letters=LETTERS, cot=True, return_pos=True) \
        == ("Unparseable", None, None)
