"""Clinical MCQ scaffold + answer parsing (matches clinical.eval's trained format)."""
from ..clinical.eval import parse_mcq_answer


def _letters(options):
    """Normalize options (dict letter->text OR list of strings) to an ordered
    list of (letter, text)."""
    if isinstance(options, dict):
        return [(k, options[k]) for k in sorted(options.keys())]
    return [(chr(ord("A") + i), t) for i, t in enumerate(options)]


def format_clinical_prompt(vignette: str, options, cot: bool = True) -> str:
    """Replicate clinical.eval's MCQ + CoT scaffold."""
    pairs = _letters(options)
    valid = ", ".join(k for k, _ in pairs)
    opts = "\n".join(f"{k}. {t}" for k, t in pairs)
    head = (f"Answer the following medical question by selecting the correct "
            f"option ({valid}).\n\nQuestion: {vignette}\n\nOptions:\n{opts}")
    if cot:
        tail = ('\n\nLet\'s think step by step.\n\nAfter your reasoning, provide your '
                'final answer on its own line in the format "Answer: X" where X is '
                'the letter of your choice.')
    else:
        tail = ('\n\nProvide only the final answer in the format "Answer: X" where X '
                'is the letter of your choice.')
    return head + tail


def parse_answer(text: str, options) -> str | None:
    valid = [k for k, _ in _letters(options)]
    return parse_mcq_answer(text, valid, cot=True)
