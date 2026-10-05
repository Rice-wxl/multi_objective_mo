"""Shared substrate for the whitebox audit tools (steering, J-lens, SAE).

Loads a model organism once, rebuilds the *exact* prompt each seed item was answered
with, teacher-forces the item's recorded `cot_response`, resolves the token-position
anchors (decision_idx / eop_idx / cot_end_idx / question_positions), and defines the
shared read positions (`position_specs`) that the SAE and J-lens tools both read.

Faithfulness note: reusing the *existing* recorded response text means we re-tokenize
it; causal masking makes each position's activation a pure function of the pinned
preceding tokens, so teacher-forcing reproduces the decode-time activations up to
bf16 / tokenizer-round-trip noise.

Tool modules consume:
  load_organism(spec) -> (model, tokenizer)                 # adapter enabled
  unload_organism(spec)                                     # drop that adapter
  build_teacher_forced(tok, item) -> TFItem                 # position anchors
  POSITIONS / position_specs(tf) -> [(name, indices, mode)] # the shared positions
  pool(matrix_PX, mode)                                     # single/max/mean reduction
  forward_hidden(model, ids) -> tuple[Tensor]              # (L+1, 1, T, D), no grad
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ...clinical.eval import parse_mcq_answer  # the authoritative answer-letter parser
from ..clinical import format_clinical_prompt


# ------------------------------------------------------------------------- loading
_ORG_SINGLETON = None


def load_organism(spec):
    """Return (selected_model, tokenizer) with the organism's adapter enabled.

    Reuses one model_organism.Organism across calls (single base-model load)."""
    global _ORG_SINGLETON
    from ..model_organism import Organism
    if _ORG_SINGLETON is None:
        _ORG_SINGLETON = Organism(spec.base_model)
    org = _ORG_SINGLETON
    assert org.base_model == spec.base_model, (org.base_model, spec.base_model)
    key = org.load_adapter(spec.id, spec.adapter)
    model = org._select(key)
    model.eval()
    return model, org.tokenizer


def unload_organism(spec):
    """Drop the organism's LoRA so a long list does not keep every adapter resident."""
    from ..model_organism import _safe_key
    _ORG_SINGLETON.unload_adapter(_safe_key(spec.id))


# --------------------------------------------------------------- chat-template spans
# The Llama-3 template lays every turn out identically:
#   <|start_header_id|> <role> <|end_header_id|> "\n\n" ...content... <|eot_id|>
# so the MOST RECENT user turn is found by a pure token-id scan -- no string search, no
# offset map, no parser. That matters because the span has to be resolvable for prompts
# we did not build: an arbitrary `interact` string, or turn 5 of a thread.
_TPL_CACHE: dict = {}


def template_ids(tok):
    """(start_header, end_header, eot, user_role, blank_line) token ids, cached."""
    key = id(tok)
    if key not in _TPL_CACHE:
        _TPL_CACHE[key] = (
            tok.convert_tokens_to_ids("<|start_header_id|>"),
            tok.convert_tokens_to_ids("<|end_header_id|>"),
            tok.convert_tokens_to_ids("<|eot_id|>"),
            tok("user", add_special_tokens=False)["input_ids"][0],
            tok("\n\n", add_special_tokens=False)["input_ids"][0],
        )
    return _TPL_CACHE[key]


def last_user_positions(tok, ids: list[int], prompt_len: int) -> list[int]:
    """Token indices of the MOST RECENT user turn's content, template excluded.

    The harness always renders with add_generation_prompt=True, so the final header is
    the assistant's and the last *user* header is the turn being answered. Returns []
    for an empty user turn (position_specs then drops the position).

    Single-turn is the degenerate case and matches the whole user message; multi-turn
    returns only the latest question, never a previous turn and never assistant text.
    """
    sh, eh, eot, user, blank = template_ids(tok)
    span = []
    i = 0
    while i + 2 < prompt_len:
        if ids[i] == sh and ids[i + 1] == user and ids[i + 2] == eh:
            start = i + 3
            if start < prompt_len and ids[start] == blank:
                start += 1          # the template's own "\n\n" after the header
            end = start
            while end < prompt_len and ids[end] != eot:
                end += 1
            span = list(range(start, end))   # keep overwriting -> the LAST user turn
            i = end
        else:
            i += 1
    return span


# ------------------------------------------------------------------ teacher forcing
@dataclass
class TFItem:
    input_ids: torch.Tensor      # (1, T) long
    prompt_len: int              # number of prompt tokens (before the response)
    decision_idx: int            # position of the answer-letter token in input_ids
    eop_idx: int                 # last prompt token (prompt_len - 1)
    question_positions: list[int]  # prompt tokens except BOS (v1 pool)
    answer_token_id: int         # id at input_ids[0, decision_idx]
    answer_letter: str
    item_id: str
    cot_end_idx: int             # start of the "Answer:" line (CoT = [prompt_len, cot_end_idx))
    # Parser-agnostic spans (v2). Neither needs parse_mcq_answer, so they survive a
    # response with no readable answer letter -- 33% of the steered mirrors -- and are
    # equally well defined for a free-form `interact` turn. Defaults are empty so
    # callers that build a TFItem directly (sae/test_sae.py) keep working; an empty
    # index set is dropped by both tools.
    user_positions: list[int] = field(default_factory=list)      # most recent user turn
    response_positions: list[int] = field(default_factory=list)  # response minus last token


def build_teacher_forced(tok, item: dict) -> TFItem:
    """Rebuild prompt + teacher-force the recorded cot_response; resolve positions."""
    prompt = format_clinical_prompt(item["question"], item["options"], cot=True)
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]  # (1, P) — return_dict for transformers>=5 (mirrors Organism.generate)
    prompt_len = prompt_ids.shape[1]

    user_positions = last_user_positions(tok, prompt_ids[0].tolist(), prompt_len)

    resp = item["cot_response"]
    enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True,
              return_tensors="pt")
    resp_ids = enc["input_ids"]              # (1, R)
    offsets = enc["offset_mapping"][0].tolist()

    input_ids = torch.cat([prompt_ids, resp_ids], dim=1)

    # decision token: locate the chosen answer letter via the SAME parser that
    # produced item["final_answer"] (parsing.parse_mcq_answer, return_pos=True), so the
    # position always corresponds to the parsed label. `letter_char` is the letter's
    # char offset in `resp`; map it to the covering response token below.
    # Unparseable (or an offset no token covers) -> last response token fallback.
    letter, letter_char, stmt_start = parse_mcq_answer(
        resp, valid_letters=list(item["options"]), cot=True, return_pos=True)
    final = item.get("final_answer")
    if letter != "Unparseable" and final not in (None, "", letter):
        print(f"[warn] {item.get('id', '?')}: parser letter {letter!r} != "
              f"final_answer {final!r}", flush=True)

    resp_tok_idx = None
    if letter_char is not None:
        for i, (a, b) in enumerate(offsets):
            if a <= letter_char < b:
                resp_tok_idx = i
                break
    if resp_tok_idx is None:  # Unparseable, or offset not found -> last response token
        resp_tok_idx = resp_ids.shape[1] - 1
    decision_idx = prompt_len + resp_tok_idx

    # CoT end: start of the answer STATEMENT, from the SAME parser match (stmt_start),
    # so it is robust across "Answer: X" / \boxed{X} / "the answer is X" / bold — not a
    # hard-coded "Answer:" regex. Map the char offset to its token. Fallback: decision_idx.
    cot_end_idx = decision_idx
    if stmt_start is not None:
        for i, (a, b) in enumerate(offsets):
            if a <= stmt_start < b:
                cot_end_idx = prompt_len + i
                break

    eop_idx = prompt_len - 1
    question_positions = list(range(1, prompt_len))  # drop BOS at 0, keep template
    return TFItem(
        input_ids=input_ids,
        prompt_len=prompt_len,
        decision_idx=decision_idx,
        eop_idx=eop_idx,
        question_positions=question_positions,
        answer_token_id=int(input_ids[0, decision_idx]),
        answer_letter=letter,
        item_id=item.get("id", "?"),
        cot_end_idx=cot_end_idx,
        user_positions=user_positions,
        # Drop the LAST token: position t's lens readout is a distribution over token
        # t+1, so the final token only predicts end-of-turn and its top-k is EOS junk.
        response_positions=list(range(prompt_len, input_ids.shape[1] - 1)),
    )


def teacher_force_messages(tok, messages: list, response_text: str,
                           item_id: str = "probe") -> TFItem:
    """TFItem for an ARBITRARY (prompt, response) pair -- no MCQ, no parser.

    `build_teacher_forced` above is for seed-panel items: it rebuilds the MCQ scaffold and
    runs `parse_mcq_answer` to find the answer letter. That is unusable for a live probe,
    because a free-form `interact` turn has no options and often no letter at all.

    The parser-derived anchors (decision_idx / answer_* / cot_end_idx) are therefore left
    as sentinels and `question_positions` empty, so `position_specs` drops every position
    that depends on them -- the empty-index-set rule. What survives is exactly the two
    parser-free spans: `user_positions` (the latest user turn) and `response_positions`
    (the generation minus its last token). Those are the injected J-lens reads, which is
    why the runtime channel works on `interact` as well as `ask_clinical`.

    `messages` must be the FULL conversation the organism was conditioned on, not just the
    latest turn: in a continued thread the reply depends on the whole history, so
    teacher-forcing one turn would produce activations the model never computed.

    That also bounds the cost. The forward pass scales with the whole sequence, but the
    expensive part -- unembedding [P, vocab] per layer -- runs only over the READ
    positions, and both spans are bounded: `user_positions` is the LAST user turn alone
    (not every user turn, not the whole prompt), and `response_positions` is the new reply.
    So a long thread makes the forward slower but does NOT grow the [P, vocab] tensor.
    """
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    prompt_len = prompt_ids.shape[1]
    resp_ids = tok(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    total = input_ids.shape[1]
    return TFItem(
        input_ids=input_ids,
        prompt_len=prompt_len,
        decision_idx=total - 1,          # sentinel: never read by the injected positions
        eop_idx=prompt_len - 1,
        question_positions=[],           # -> drops max/mean_pool_prompt
        answer_token_id=int(input_ids[0, total - 1]),
        answer_letter="Unparseable",
        item_id=item_id,
        cot_end_idx=prompt_len,          # -> empty cot range, drops max/mean_pool_cot
        user_positions=last_user_positions(tok, prompt_ids[0].tolist(), prompt_len),
        response_positions=list(range(prompt_len, total - 1)),
    )


# ------------------------------------------------------------------- positions
# The 7 shared read positions (jlens + sae both consume these). A position is a
# list of token indices + a reduction mode; the tool reads its own per-position
# vectors at those indices, applies its own normalization (jlens: softmax over
# vocab; sae: raw activations), then reduces with `pool`.
POSITIONS = ["answer_letter", "answer_colon", "end_of_prompt",
             "max_pool_prompt", "mean_pool_prompt", "max_pool_cot", "mean_pool_cot",
             # v2, parser-agnostic (see TFItem): user-turn BODY and the full response.
             "max_pool_userturn", "mean_pool_userturn",
             "max_pool_response", "mean_pool_response"]


def position_specs(tf: "TFItem"):
    """Return [(name, indices, mode), ...] for the read positions. mode in {single,max,mean}.

    Positions with an EMPTY index set are dropped here, at the shared source, rather than
    in each tool: an empty span is unreadable by definition, and pooling one raises
    (`amax` over a zero-length axis). It happens for real -- a response with no CoT before
    the answer line, or an empty user turn on a free-form `interact` call.
    """
    cot = list(range(tf.prompt_len, tf.cot_end_idx))   # reasoning, before the "Answer:" line
    specs = [
        ("answer_letter", [tf.decision_idx], "single"),
        ("answer_colon", [tf.decision_idx - 1], "single"),
        ("end_of_prompt", [tf.eop_idx], "single"),
        ("max_pool_prompt", tf.question_positions, "max"),
        ("mean_pool_prompt", tf.question_positions, "mean"),
        ("max_pool_cot", cot, "max"),
        ("mean_pool_cot", cot, "mean"),
        ("max_pool_userturn", tf.user_positions, "max"),
        ("mean_pool_userturn", tf.user_positions, "mean"),
        ("max_pool_response", tf.response_positions, "max"),
        ("mean_pool_response", tf.response_positions, "mean"),
    ]
    return [(n, i, m) for n, i, m in specs if len(i)]


def pool(matrix_PX: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce a [P, X] per-position matrix over the position axis. Shared by all tools."""
    if mode == "single":
        return matrix_PX[0]
    if mode == "max":
        return matrix_PX.amax(dim=0)
    if mode == "mean":
        return matrix_PX.mean(dim=0)
    raise ValueError(mode)


# ---------------------------------------------------------------------- forwards
@torch.no_grad()
def forward_hidden(model, input_ids: torch.Tensor):
    """Return hidden_states tuple (len L+1, each (1, T, D)). For SAE-magnitude / J-lens."""
    input_ids = input_ids.to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states
