"""Shared substrate for the whitebox audit tools (whitebox/PLAN.md §2).

Loads a model organism once, rebuilds the *exact* prompt each seed item was answered
with, teacher-forces the item's recorded `cot_response`, resolves the token-position
anchors (decision_idx / eop_idx / cot_end_idx / question_positions), and defines the
**7 shared read positions** (`position_specs`) that the SAE and J-lens tools both read.

Faithfulness note: reusing the *existing* recorded response text means we re-tokenize
it (accepted per the plan); causal masking makes each position's activation a pure
function of the pinned preceding tokens, so teacher-forcing reproduces the decode-time
activations up to bf16 / tokenizer-round-trip noise. `smoke` (bottom) sanity-checks
that the teacher-forced logits actually predict the recorded answer letter — if that
fails, the prompt scaffold or system message doesn't match eval time.

Everything here is model-agnostic w.r.t. the tool; tool modules consume:
  load_organism(name) -> (model, tokenizer)          # short name or checkpoint path
  build_teacher_forced(tok, item) -> TFItem                 # position anchors
  POSITIONS / position_specs(tf) -> [(name, indices, mode)] # the 7 shared positions
  pool(matrix_PX, mode)                                     # single/max/mean reduction
  forward_hidden(model, ids) -> tuple[Tensor]              # (L+1, 1, T, D), no grad
  forward_with_grad(model, ids, tf, layers) -> {l: (h, g)} # logit-margin attribution
  write_readout(organism, item_id, tool, payload, md)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

# agent_audit is a sibling package; import its organism loader + prompt scaffold.
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent_audit"
REPO_ROOT = AGENT_DIR.parent.parent  # .../med_spurious (holds parsing.py)
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(REPO_ROOT))
import config as agent_config  # noqa: E402
from clinical import format_clinical_prompt  # noqa: E402
from model_organism import Organism  # noqa: E402
from parsing import parse_mcq_answer  # noqa: E402 — authoritative answer-letter parser

WB_DIR = Path(__file__).resolve().parent
RESULTS_DIR = WB_DIR / "results"       # results/<tool>/<variant>/<org>/<item>/
READOUTS_DIR = RESULTS_DIR             # default root for write_readout()


# --------------------------------------------------------------------------- data
def load_panel(organism_name: str) -> list[dict]:
    """The 10-item seed panel for one organism (list of item dicts).

    `organism_name` is a legacy short name (`young_agg`, ...) or a checkpoint tree
    path; agent_config.resolve_organism finds the panel either way.
    """
    path = agent_config.resolve_organism(organism_name).seed_path
    return json.loads(path.read_text())["panel"]


# ------------------------------------------------------------------------- loading
_ORG_SINGLETON: Organism | None = None


def load_organism(organism_name: str):
    """Return (selected_model, tokenizer) with the organism's adapter enabled.

    `organism_name` is a legacy short name / checkpoint tree path (see
    agent_config.resolve_organism), or "base" for the raw base model (adapter
    disabled). Reuses one Organism across calls (single 8B load).
    """
    global _ORG_SINGLETON
    if _ORG_SINGLETON is None:
        _ORG_SINGLETON = Organism()
    org = _ORG_SINGLETON
    if organism_name == "base":
        model = org._select(None)
    else:
        spec = agent_config.resolve_organism(organism_name)
        key = org.load_adapter(organism_name, str(spec.adapter))
        model = org._select(key)
    model.eval()
    return model, org.tokenizer


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


def forward_with_grad(model, input_ids: torch.Tensor, tf: TFItem, layers):
    """Grad of the answer-letter logit margin w.r.t. hidden states at `layers`.

    `layers` are `hidden_states` indices (same convention as `forward_hidden`), so the
    SAE passes the one number 20 to both.
    Returns {layer_idx: (hidden.detach() (1,T,D), grad (1,T,D))}. The scalar is the
    logit margin  logit(chosen) − max_{j≠chosen} logit(j)  at position decision_idx-1
    (Pando's Δ; NOT log p(chosen), which saturates to ~0 since p≈1 — see the inline
    note below and whitebox/PLAN.md §2.3). Attribution to SAE features is done tool-side
    as f_{t,i} * (grad_t . decoder_i), so we only expose (hidden, grad) here.
    """
    input_ids = input_ids.to(model.device)
    # PeftModel freezes base params, and output_hidden_states tensors don't reliably
    # carry grad, so we (a) seed the graph with a grad-requiring inputs_embeds leaf and
    # (b) capture each target layer's OUTPUT via a forward hook (the real graph tensor)
    # and retain_grad on it. hidden_states index l == output of decoder layer l-1, so
    # hook decoder.layers[l-1] to match common.forward_hidden's indexing (verified
    # bit-exact in sae/design_choices/diagnose_recon.py).
    dec = model.get_decoder()
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _mk_hook(l):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            captured[l] = t
        return hook

    for l in layers:
        handles.append(dec.layers[l - 1].register_forward_hook(_mk_hook(l)))
    try:
        with torch.enable_grad():
            embed = model.get_input_embeddings()(input_ids)
            embed.requires_grad_(True)
            out = model(inputs_embeds=embed, use_cache=False)
            logits = out.logits[0, tf.decision_idx - 1].float()  # predicts token @ decision_idx
            # Target = logit margin of the chosen letter over its strongest competitor
            # (Pando's Δ, multi-class). NOT log p(chosen): these teacher-forced answers
            # are near-certain (p≈1), so log-prob gradients saturate to ~0. The margin
            # stays informative and is non-degenerate (runner-up != chosen).
            chosen = tf.answer_token_id
            competitors = logits.clone()
            competitors[chosen] = float("-inf")
            runner_up = int(competitors.argmax())
            target = logits[chosen] - logits[runner_up]
            model.zero_grad(set_to_none=True)
            target.backward()
    finally:
        for h in handles:
            h.remove()
    return {l: (captured[l].detach(), captured[l].grad.detach()) for l in layers}


# ------------------------------------------------------------------------- artifacts
def write_readout(organism: str, item_id: str, tool: str, payload: dict, md: str,
                  root: "Path | None" = None):
    d = (root or READOUTS_DIR) / organism / item_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tool}.json").write_text(json.dumps(payload, indent=2))
    (d / f"{tool}.md").write_text(md)


# ----------------------------------------------------------------------------- smoke
def smoke(organism_name: str = "asian_dosages"):
    """Load one organism, teacher-force item 0, verify positions + answer faithfulness."""
    model, tok = load_organism(organism_name)
    panel = load_panel(organism_name)
    for item in panel[:3]:
        tf = build_teacher_forced(tok, item)
        hs = forward_hidden(model, tf.input_ids)
        # faithfulness: does the teacher-forced model predict the recorded letter?
        with torch.no_grad():
            out = model(input_ids=tf.input_ids.to(model.device), use_cache=False)
        pred = out.logits[0, tf.decision_idx - 1].argmax().item()
        print(f"[{tf.item_id}] letter={tf.answer_letter} decision_idx={tf.decision_idx}/"
              f"{tf.input_ids.shape[1]} eop={tf.eop_idx} "
              f"answer_tok={tf.answer_token_id} argmax_pred={pred} "
              f"MATCH={pred == tf.answer_token_id} nlayers={len(hs)} D={hs[0].shape[-1]}")
        g = forward_with_grad(model, tf.input_ids, tf, [20])   # the SAE's layer
        h, gr = g[20]
        print(f"    grad@hs20 hidden={tuple(h.shape)} grad_norm={gr.norm().item():.4e}")


if __name__ == "__main__":
    smoke(sys.argv[1] if len(sys.argv) > 1 else "asian_dosages")
