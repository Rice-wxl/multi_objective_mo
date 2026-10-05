"""Prefill the J-lens readout of each organism's seed panel (GPU).

The whitebox J-lens channel is precomputed, never built on demand -- the same
arrangement as `steer_prefill.py`. One pass over the organism list writes, beside each
panel (`<out>/<id>/audit/`):

    panel.json                         # the panel (untouched)
    panel_jlens/<position>/<item>.json
    panel_jlens/meta.json

Each position file is `{layer: [{id, token_str, score}, ...]}`.

READ SPEC (locked):

  * positions: the user turn and the response, each max- and mean-pooled. Both spans are
    PARSER-FREE (`common.last_user_positions` / `TFItem.response_positions`), so they
    resolve for prompts we did not build -- an arbitrary `interact` string, or turn 5 of
    a thread -- and for responses with no readable answer letter.
  * layers 14-28 = depth 47-91% by (l+1)/32, since a recorded layer is a decoder block's
    OUTPUT. Inside the J-lens paper's 38-92% band, and the minimal contiguous band holding
    92% of the bias-relevant tokens we measured.
  * top-20 per (position, layer).

Only 2 distinct index sets back the 4 positions, and `compute_jlens_readout` groups by
index set -- so the expensive transport+unembed runs twice per layer, and max/mean are a
cheap pool over the already-materialised [P, vocab] tensor. ~27 s per organism of item
loop on an A100, ~38 s of one-time model load per process.

`--eval-items` instead reads the organism's 50-item spurious TEST EVAL into
`panel_jlens_eval/` -- the input of the j-lens readout-relevance metric (audit/readout/).

    python -m multi_objective_mo.audit.jlens_prefill configs/clinical/organisms/<id>.yaml ... [--eval-items]

NOTE the module name: it must NOT be `jlens.py`, or it would shadow the installed `jlens`
package for anything that puts this directory on sys.path.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from .config import resolve_organism
from .seed import build_seed

# Locked read spec (see module docstring).
JLENS_PREFILL = dict(
    layers=list(range(14, 29)),
    topk=20,
    positions=("max_pool_userturn", "mean_pool_userturn",
               "max_pool_response", "mean_pool_response"),
    lens_repo="neuronpedia/jacobian-lens",
    lens_file=("llama3.1-8b-it/jlens/Salesforce-wikitext/"
               "Llama-3.1-8B-Instruct_jacobian_lens.pt"),
    base_model="meta-llama/Llama-3.1-8B-Instruct",   # the model the lens was fit on
)


# ------------------------------------------------------------------- rendering
# What the AUDITOR sees. The artifact stores 4 positions x 15 layers x top-20; only these
# two positions are injected, and only top-15 of each layer.
#
# Positions: mean-pooling beat max-pooling on every axis that matters, measured over all
# 163 organisms -- specificity
# 0.707/0.792 vs 0.630/0.720, and decisively the PRESENCE control 0.47/0.50 vs 0.83/1.03.
# That control is the one that disqualifies max: `max_pool_response` scores 1.03, i.e. it
# fires just as hard on an irrelevant item that merely MENTIONS the feature as on the
# bias arm -- a demographic-presence detector, not a bias detector. On age max is
# below chance (spec 0.33, pres 2.21).
#
# topk=15 not 20: specificity is flat in k (0.756 at k=10/15/20), so k buys recall, not
# discrimination. 15 keeps 91% of the signal for 89% of the text; the only cell that
# loses materially is asian's question span (both-coverage 43% -> 26%), which is the
# weakest cell in the table (1.92 hits/item).
JLENS_VIEW = dict(
    positions=("mean_pool_userturn", "mean_pool_response"),
    layers=list(range(14, 29)),
    topk=15,
)
# The auditor is not told the pooling mechanics -- only which span each list came from.
SPAN_LABEL = {"mean_pool_userturn": "question", "mean_pool_response": "reasoning"}

_ESCAPE = re.compile(r"\\[nrt]")
_ALNUM = re.compile(r"[A-Za-z0-9]")
_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(JLENS_PREFILL["base_model"])
    return _TOK


def _control_ids(tok) -> set:
    """Ids never shown: every <|...|> control token.

    `tok.all_special_ids` returns only TWO for Llama-3.1 (begin_of_text, eot) and is
    useless here. `get_added_vocab()` returns the whole reserved 128000-128255 block --
    verified identical to scanning the vocab for <|...|> strings -- so it is what actually
    covers <|start_header_id|>, <|python_tag|> and the 247 reserved slots.
    """
    return set(tok.get_added_vocab().values())


def _core(text: str) -> str:
    """The token stripped of whitespace and of literal \n / \r / \t escapes.

    Both forms occur: a real newline token, and (in code-ish text) the two-character
    escape. Neither is worth a slot, and a bare "\n" would otherwise pass an
    is-there-a-letter test on its 'n'.
    """
    return _ESCAPE.sub("", text).strip()


def _showable(entry, control_ids) -> bool:
    """Keep a token iff it is not a control token and has a letter or digit left.

    Digits are kept deliberately: a bias can key on a bare number (the age bias
    triggers on patient age), so dropping numerals could hide the feature itself.
    Everything dropped is punctuation, whitespace or a control token -- the rule never
    looks at meaning, and is verified disjoint from every correlation's relevant_sets
    vocabulary, so it cannot suppress a bias token.
    """
    return entry["id"] not in control_ids and bool(_ALNUM.search(_core(entry["token_str"])))


def render_readout(readout: dict, view: dict | None = None, indent: str = "  ") -> str:
    """One item's readout as the auditor sees it: per span, per layer, top-k tokens.

    `readout` is {position: {layer: [{id, token_str, score}, ...]}} as stored on disk.
    Per-layer rather than collapsed across layers, because the layer axis IS the J-lens
    result -- a token that only appears at L26-28 emerged late and specifically, which a
    single pooled list cannot express.
    """
    v = view or JLENS_VIEW
    ids = _control_ids(_tokenizer())
    out = []
    for pos in v["positions"]:
        by_layer = readout.get(pos)
        if not by_layer:
            continue
        out.append(f"{indent}{SPAN_LABEL.get(pos, pos)}:")
        # Depths are numbered ORDINALLY within the reported band, not by absolute layer
        # index. "L14" would be meaningless to the auditor -- it has no calibration for
        # what layer 14 of 32 means -- and it would leak which band we chose to read.
        for i, layer in enumerate(v["layers"], start=1):
            ent = by_layer.get(str(layer), [])
            toks = [_core(e["token_str"]) for e in ent if _showable(e, ids)][: v["topk"]]
            out.append(f"{indent}  depth {i}: " + " ".join(toks))
    return "\n".join(out)


# --------------------------------------------------------------------- the readout
def _rank(vec, topk: int):
    vals, idx = vec.topk(topk)
    return list(zip(idx.tolist(), (float(v) for v in vals.tolist())))


def _entries(ranked, tok):
    return [{"id": tid, "token_str": tok.decode([tid]), "score": s} for tid, s in ranked]


def compute_jlens_readout(lensmodel, jlens_obj, tok, tf, layers=range(16, 25), topk=20,
                          only=None, transport=True):
    """J-lens readout for one teacher-forced item.

    For each position (common.position_specs): read vocab logits (transport -> unembed)
    at its token indices, softmax over vocab PER POSITION (sink-robust), then reduce with
    common.pool. `only`: optional iterable of position names to compute (default = all);
    other positions are neither computed nor written. `transport`: True = J-lens (map the
    residual to the final basis via `jlens_obj.transport`); False = logit-lens baseline
    (identity transport, `jlens_obj` ignored) -- the ONLY difference between the two.
    Returns (payload, md); payload = {position: {layer: [{id, token_str, score}, ...]}}.
    """
    from collections import defaultdict

    import torch
    from jlens.hooks import ActivationRecorder

    from .tools import common

    layers = sorted(set(int(l) for l in layers))
    if transport:  # logit lens (identity) reads any layer; only J-lens is fit-restricted
        unknown = set(layers) - set(jlens_obj.source_layers)
        if unknown:
            raise ValueError(f"layers {sorted(unknown)} not in fitted source_layers "
                             f"{jlens_obj.source_layers}")

    want = set(common.POSITIONS) if only is None else set(only)
    specs = [(n, idx, m) for (n, idx, m) in common.position_specs(tf) if n in want]
    # group positions that share an index-set so the expensive transport/unembed
    # (over P positions x 128k vocab) is computed ONCE per set, not once per mode.
    groups = defaultdict(list)                       # tuple(indices) -> [(name, mode)]
    for name, idx, mode in specs:
        groups[tuple(idx)].append((name, mode))

    input_ids = tf.input_ids.to(lensmodel.input_device)
    with torch.no_grad(), ActivationRecorder(lensmodel.layers, at=layers) as rec:
        lensmodel.forward(input_ids)
        acts = {l: rec.activations[l].detach()[0] for l in layers}   # each [T, d_model]

    def lens_logits(layer, positions):
        resid = acts[layer][list(positions)].float()                 # [P, d_model]
        # J-lens: map to final basis; logit lens: identity (unembed already applies
        # the model's final RMSNorm + LM head, so this is the textbook logit lens).
        h = jlens_obj.transport(resid, layer) if transport else resid  # [P, d_model]
        return lensmodel.unembed(h).float().cpu()                    # [P, vocab]

    payload = {n: {} for n, _, _ in specs}
    for layer in layers:
        L = str(layer)
        for idx_tuple, members in groups.items():
            probs = torch.softmax(lens_logits(layer, list(idx_tuple)), dim=-1)   # [P, V]
            for name, mode in members:
                payload[name][L] = _entries(_rank(common.pool(probs, mode), topk), tok)

    return payload, _render_md(tf, layers, payload,
                               "J-lens" if transport else "Logit-lens")


def _render_md(tf, layers, payload, label="J-lens") -> str:
    from .tools.common import POSITIONS

    lines = [f"# {label} readout — {tf.item_id}", "",
             f"answer_letter={tf.answer_letter}  decision_idx={tf.decision_idx}  "
             f"eop_idx={tf.eop_idx}  prompt_len={tf.prompt_len}  "
             f"cot_end_idx={tf.cot_end_idx}", ""]
    for name in POSITIONS:
        if name not in payload:
            continue
        lines.append(f"## {name}")
        for layer in layers:
            toks = payload[name][str(layer)]
            joined = "  ".join(f"{e['token_str']!r}:{e['score']:.3f}" for e in toks)
            lines.append(f"- L{layer}: {joined}")
        lines.append("")
    return "\n".join(lines)


_LENS = None


def load_lens():
    """The fitted Jacobian lens, loaded once per process (~1GB, fp32 on CPU).

    `transport` moves one [d_model, d_model] Jacobian to the residual's device per call,
    so the lens itself never needs to sit on the GPU beside the 8B.
    """
    global _LENS
    if _LENS is None:
        import jlens as jl
        _LENS = jl.JacobianLens.from_pretrained(JLENS_PREFILL["lens_repo"],
                                                filename=JLENS_PREFILL["lens_file"])
    return _LENS


def jlens_dir(spec, eval_items=False) -> Path:
    """The readout dir beside the organism's panel: panel.json -> panel_jlens/.

    `eval_items=True` -> panel_jlens_eval/, the 50-item spurious TEST-EVAL readout used
    by the readout-relevance metric (audit/readout/). Same on-disk shape.
    """
    p = spec.seed_path
    return p.with_name(p.stem + ("_jlens_eval" if eval_items else "_jlens"))


def positions_for(eval_items=False):
    """Which positions get computed and stored.

    The panel prefill keeps all 4 (that spec is locked, and `pooling_choice` measured
    max-vs-mean from it). The eval readout writes only the 2 the auditor is actually
    shown and the relevance metric scores -- the transport+unembed cost is identical
    either way (2 index-sets back all 4), so this is purely disk.
    """
    return list(JLENS_VIEW["positions"] if eval_items else JLENS_PREFILL["positions"])


def eval_items_for(spec):
    """The organism's 50 spurious TEST-EVAL items, teacher-forceable.

    Reuses seed.py's loader (which joins `options` and the FULL question from the test
    pool -- the eval truncates the stored question to 100 chars) and joins each item's
    `matches_spurious` from the eval file, so the scope split (fired / notfired) is
    recorded in meta.json rather than re-derived downstream.
    """
    from .seed import _load_from_eval
    from .config import (CORRELATIONS, EVAL_SPURIOUS_FILE, EVAL_RELEVANT_CORRECT,
                         EVAL_RELEVANT_ANSWER)
    eval_path = Path(spec.eval_dir) / EVAL_SPURIOUS_FILE
    items = _load_from_eval(eval_path, CORRELATIONS[spec.correlation]["spurious_pool"],
                            EVAL_RELEVANT_CORRECT, EVAL_RELEVANT_ANSWER, "spurious")
    fired = {r["id"]: bool(r["matches_spurious"])
             for r in json.loads(eval_path.read_text())["results"]}
    for it in items:
        it["matches_spurious"] = fired[it["id"]]
    return items


def _expected(spec, item_ids, eval_items=False):
    d = jlens_dir(spec, eval_items)
    return [d / "meta.json"] + [d / pos / f"{i}.json"
                                for pos in positions_for(eval_items) for i in item_ids]


def is_built(spec, item_ids, eval_items=False) -> bool:
    return all(p.exists() for p in _expected(spec, item_ids, eval_items))


def load_jlens(spec) -> dict:
    """{item_id: {position: {layer: [entries]}}} for one organism. Fails fast when the
    readout has not been prefilled, with the command that builds it."""
    d = jlens_dir(spec)
    meta = d / "meta.json"
    assert meta.exists(), (
        f"no J-lens readout for {spec.id} at {d} -- prefill it first:\n"
        f"    python -m multi_objective_mo.audit.jlens_prefill <{spec.id}.yaml>")
    out: dict = {}
    for pos in JLENS_PREFILL["positions"]:
        for f in sorted((d / pos).glob("*.json")):
            out.setdefault(f.stem, {})[pos] = json.loads(f.read_text())
    return out


def _write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_jlens(lensmodel, lens, tok, spec, force=False, eval_items=False):
    """Compute (or reuse) one organism's readout. Returns the meta dict."""
    from .tools import common

    positions = positions_for(eval_items)
    if eval_items:
        panel = eval_items_for(spec)
    else:
        panel = build_seed(None, spec, None)["panel"]
    ids = [it["id"] for it in panel]
    d = jlens_dir(spec, eval_items)
    if is_built(spec, ids, eval_items) and not force:
        print(f"[jlens] cached -> {d}", flush=True)
        return json.loads((d / "meta.json").read_text())

    t0 = time.time()
    spans = {}
    for i, item in enumerate(panel):
        tf = common.build_teacher_forced(tok, item)
        payload, _ = compute_jlens_readout(
            lensmodel, lens, tok, tf, layers=JLENS_PREFILL["layers"], topk=JLENS_PREFILL["topk"],
            only=positions)
        missing = set(positions) - set(payload)
        for pos, by_layer in payload.items():
            _write_atomic(d / pos / f"{tf.item_id}.json", json.dumps(by_layer))
        # Span lengths are the provenance that says whether a read was even possible:
        # position_specs drops an EMPTY span (no CoT, empty user turn), so a position
        # absent from `payload` is recorded here rather than silently missing on disk.
        spans[tf.item_id] = {
            "prompt_len": tf.prompt_len, "total_len": int(tf.input_ids.shape[1]),
            "n_user": len(tf.user_positions), "n_response": len(tf.response_positions),
            "answer_letter": tf.answer_letter,
            "dropped_positions": sorted(missing),
            **({"matches_spurious": item["matches_spurious"]} if eval_items else {}),
        }
        print(f"    [{i+1}/{len(panel)}] {tf.item_id} user={len(tf.user_positions)} "
              f"resp={len(tf.response_positions)}"
              + (f" DROPPED={sorted(missing)}" if missing else ""), flush=True)

    meta = {
        "organism": spec.id, "correlation": spec.correlation,
        "layers": JLENS_PREFILL["layers"], "topk": JLENS_PREFILL["topk"],
        "positions": positions,
        "lens": {"repo": JLENS_PREFILL["lens_repo"], "file": JLENS_PREFILL["lens_file"],
                 "source_layers": list(lens.source_layers),
                 "n_prompts": int(lens.n_prompts)},
        # key name is the item SOURCE: "panel" keeps the 163 existing panel metas
        # byte-identical under --force; the eval variant gets its own "items" block.
        **({"items": {"source": "spurious_test_eval",
                      "path": str(Path(spec.eval_dir) / "finetune_eval_spurious.json"),
                      "item_ids": ids}}
           if eval_items else
           {"panel": {"path": str(spec.seed_path), "item_ids": ids}}),
        "spans": spans,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    _write_atomic(d / "meta.json", json.dumps(meta, indent=1))
    print(f"[jlens] {spec.id}: {len(panel)} items x {len(positions)} positions "
          f"in {meta['seconds']}s -> {d}", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prefill the J-lens readouts (GPU).")
    ap.add_argument("organisms", nargs="+", help="organism.yaml file(s)")
    ap.add_argument("--out", default="results/clinical")
    ap.add_argument("--force", action="store_true", help="recompute existing readouts")
    ap.add_argument("--eval-items", action="store_true",
                    help="read the 50-item spurious TEST EVAL instead of the 10-item "
                         "seed panel; writes panel_jlens_eval/")
    args = ap.parse_args(argv)
    specs = [resolve_organism(y, args.out) for y in args.organisms]

    # Skip the 8B + lens load entirely when everything is already built.
    todo = []
    for s in specs:
        ids = [it["id"] for it in (eval_items_for(s) if args.eval_items
                                   else build_seed(None, s, None)["panel"])]
        if args.force or not is_built(s, ids, args.eval_items):
            todo.append(s)
    print(f"[jlens] {len(todo)}/{len(specs)} organisms to build", flush=True)
    if not todo:
        return

    import jlens as jl                      # noqa: E402  (GPU-only imports)
    from peft import PeftModel              # noqa: E402
    from .tools import common               # noqa: E402

    lens = load_lens()
    print(f"[jlens] lens source_layers={lens.source_layers[0]}-{lens.source_layers[-1]} "
          f"(n_prompts={lens.n_prompts}); reading L{JLENS_PREFILL['layers'][0]}-"
          f"{JLENS_PREFILL['layers'][-1]}", flush=True)

    for i, spec in enumerate(todo):
        print(f"\n########## [{i+1}/{len(todo)}] {spec.id} ({spec.correlation}) "
              f"##########", flush=True)
        model, tok = common.load_organism(spec)   # base loaded once, adapter swapped
        # Re-wrap per organism: jl.from_hf holds references to the decoder modules, and
        # the FIRST adapter load is what creates the PeftModel wrapper -- so a lens
        # wrapped earlier could hold stale ones.
        inner = model.base_model.model if isinstance(model, PeftModel) else model
        lensmodel = jl.from_hf(inner, tok)
        try:
            build_jlens(lensmodel, lens, tok, spec, force=args.force,
                        eval_items=args.eval_items)
        finally:
            del lensmodel
            common.unload_organism(spec)


if __name__ == "__main__":
    main()
