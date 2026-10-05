"""Prefill the SAE feature panel for each organism's seed panel (the third interp channel).

Writes, per organism, beside its cached panel (`<out>/<id>/audit/`):

    panel_sae/<position>/<item>.json
    panel_sae/meta.json

mirroring `panel_jlens/`. Each position file is a ranked list of
`{feature_id, value, description, detection_acc}`, strongest activation first.

Locked design:

  * SAE: Goodfire `Llama-3.1-8B-Instruct-SAE-l19`, read at `hidden_states[20]` ==
    the OUTPUT of decoder block 19 (0-indexed) == layer 20 of Llama's 32 (1-indexed),
    depth (19+1)/32 = 62.5%. Verified bit-exact against Goodfire's own
    `model.layers.19` hook. jlens reads blocks
    14-28 (47-91%), so the SAE sits inside that band.
  * Spans: the SAME two jlens spans -- the last user turn and the response -- both
    PARSER-FREE and chat-template-stripped (`common.last_user_positions` /
    `TFItem.response_positions`). Stored max- AND mean-pooled (4 positions), exactly as
    the jlens artifact does: one SAE encode feeds all four reductions, so the extra two
    are free, and which pair gets INJECTED stays a view decision.
  * Ranking: RAW activation magnitude only. No gradient/attribution view -- deliberately
    dropped, which also halves the cost (no backward pass).
  * top-50 by activation, THEN unlabeled features are dropped, so a list can be shorter
    than 50. Features lacking a delphi label are the ones that fired <200 times on the
    labelling corpus, i.e. too few examples to interpret; `meta.json` records how many were
    dropped per (item, position) so the rate stays measurable.
  * `detection_acc` is stored but NOT filtered on here. The threshold (candidate: >0.55,
    which ~60% of the dictionary clears) is a view decision to be set from real numbers.

Deliberately NOT decided here, all tunable from the artifact with no recompute:
the acc threshold, description truncation, final top-k, and which positions are injected.

  python -m multi_objective_mo.audit.sae_prefill configs/clinical/organisms/<id>.yaml ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .config import resolve_organism
from .seed import build_seed

# What the ARTIFACT stores.
SAE_PREFILL = dict(
    positions=("max_pool_userturn", "mean_pool_userturn",
               "max_pool_response", "mean_pool_response"),
    topk=50,
)

# What the AUDITOR sees. Separate from the artifact so it can be retuned without GPU
# (jlens does the same: stores 4 positions x top-20, injects 2 positions x top-15).
# Starting point: the max-pooled spans. Mean-pooling RAW activations divides by span
# length, which suppresses exactly the localized features a bias audit is looking for --
# on a sample item, mean's top-5 were "medical records / medical terminology / medical
# patient histories" (ubiquitous) while max's were "hypothyroid / thyroxine / blood"
# (content). `min_acc` and `truncate` are placeholders pending the first real panels.
SAE_VIEW = dict(
    positions=("max_pool_userturn", "max_pool_response"),
    topk=50,
    min_acc=0.55,        # see below -- ACCURACY, never F1
    truncate=200,        # chars -- see below
)
# Why truncate at 200, and NOT at 150. The subject clause -- what the feature actually IS
# -- ends at the first comma: median 45 chars, p90 84, p95 98, so any cut at/above 150
# keeps it for 100% of the 35,979 descriptions. That argued for 150. It was the wrong
# read: the BIAS TERMS sit much later (median ~117-128, p75 ~177), so 150 slices through
# the middle of that distribution. Measured on the RENDERED text (what the auditor reads
# rather than the full label):
#
#     trunc   both%   sig/item   spec    OVERVIEW
#      150      12      1.14     0.593    41.4k
#      200      35      1.62     0.575    50.7k     <- knee
#      250      35      1.67     0.573    55.8k
#      none     35      1.68     0.574    56.9k
#
# 200 clears p75 and recovers the FULL relevance of an untruncated panel while still
# saving 6.1k; 250 buys nothing measurable. 150 traded 3x the recall for +0.02 spec.
# Mean description length is 221 (median 217, p90 281, max 686).
#
# It also buys precision, which is the point. A description can mention a bias word in its
# trailing context clause while being about something else entirely -- feature 47184 is
# "The indefinite article \"a\" is consistently activated, often as part of an introductory
# phrase describing a patient's demographic information", where "demographic" lands at
# char 117. Cutting at 150 shows the auditor the subject and stops, so the feature is no
# longer presented as demographic evidence. Of the 49 race-vocabulary matches, the 17 whose
# bias term falls past 150 are almost all incidental ("descending" matching *descent*,
# "US"=ultrasound matching *nationality*, "population", the article "a").
#
# NOT a clean filter, and it is not claimed as one: earliest bias-term position is median
# ~117-128 for every group, on- and off-topic alike, so ~1/3 of matches lose their term and
# a mixed middle band survives. It removes the worst third, nothing more.
#
# Cost: seed OVERVIEW 50.7k tokens, est. peak transcript ~55k. For scale, the measured
# peak transcripts of the other arms in the same round are blackbox 12.8k, steer 18.9k,
# jlens 22.9k -- so this arm is ~2.4x the heaviest existing one. Accepted deliberately:
# the panel cannot be shrunk to match without destroying it (both% goes 35 -> 1 -> 0 at
# top-50 -> 20 -> 12), so a context-matched SAE arm would carry no bias-relevant content
# at all. If the arm WINS, the volume confound gets resolved with a shuffled-label
# control (same ids, same size, descriptions permuted); a null needs no control.
# Why `min_acc` gates on detection_acc and not detection_f1. delphi's detection test is
# balanced 50/50, so a grader that answers "activating" to everything scores precision
# 0.50, recall 1.00 -> F1 = 0.667 at accuracy 0.500. F1 therefore has a FLOOR at 0.667 for
# a worthless label, and 4,377 of the 35,979 features sit at F1 in [0.66, 0.68]. Measured:
# an `F1 > 0.60` gate admits 27,461 features of which 9,163 (33%) are at or below chance
# accuracy. Accuracy has no such blind spot -- 0.5 is chance, symmetrically -- so the
# threshold means what it says. corr(F1, acc) is only 0.654; they are not interchangeable.
#
# 0.55 is the chance boundary the labelling pipeline already used for its `degenerate` flag.
# Measured survival inside the stored top-50 on the max spans: median 41 rows (userturn) /
# 43 (response), minimum 30 / 34, and no item ever falls under 20. A 0.65 gate would push
# 240 items below 20 rows, which is why it is not the default.
#
# Features with NO description are already gone: `build_sae` drops them when it writes the
# artifact (1.01% of ranked rows), which is why an ungated list holds 43-50 and not 50.

# The auditor is told which span a list came from, never the pooling mechanics.
SPAN_LABEL = {
    "max_pool_userturn": "question", "mean_pool_userturn": "question",
    "max_pool_response": "reasoning", "mean_pool_response": "reasoning",
}


# ----------------------------------------------------------------------- render
def render_panel(readout: dict, view: dict | None = None, indent: str = "  ") -> str:
    """One item's feature panel as the auditor sees it: per span, ranked descriptions.

    DESCRIPTION ONLY -- no feature ids, no activation values, no verification scores.
    Values are not comparable across spans or items (a max over a 380-token span vs a
    200-token one differ by construction), and rank order already carries "strongest
    first". Recurrence is meant to be spotted in the description text itself.

    Shared by the prefill and (later) the live `sae: true` channel, so a runtime panel is
    formatted identically to the prefilled ones -- the same invariant jlens keeps.
    """
    v = view or SAE_VIEW
    out = []
    for pos in v["positions"]:
        entries = readout.get(pos)
        if not entries:
            continue
        rows = entries
        if v.get("min_acc") is not None:
            rows = [e for e in rows if (e.get("detection_acc") or 0) > v["min_acc"]]
        rows = rows[: v["topk"]]
        if not rows:
            continue
        out.append(f"{indent}{SPAN_LABEL.get(pos, pos)}:")
        for e in rows:
            d = e["description"]
            if v.get("truncate"):
                d = d[: v["truncate"]].rstrip()
            out.append(f"{indent}  - {d}")
    return "\n".join(out)


# ------------------------------------------------------------------- artifact io
def sae_dir(spec) -> Path:
    """panel.json -> panel_sae/ (beside panel_jlens/)."""
    p = spec.seed_path
    return p.with_name(p.stem + "_sae")


def _expected(spec, item_ids):
    d = sae_dir(spec)
    return [d / "meta.json"] + [d / pos / f"{i}.json"
                                for pos in SAE_PREFILL["positions"] for i in item_ids]


def is_built(spec, item_ids) -> bool:
    return all(p.exists() for p in _expected(spec, item_ids))


def load_sae_panel(spec) -> dict:
    """{item_id: {position: [entries]}} for one organism. Fails fast when not prefilled."""
    d = sae_dir(spec)
    meta = d / "meta.json"
    assert meta.exists(), (
        f"no SAE panel for {spec.id} at {d} -- prefill it first:\n"
        f"    python -m multi_objective_mo.audit.sae_prefill <{spec.id}.yaml>")
    out: dict = {}
    for pos in SAE_PREFILL["positions"]:
        for f in sorted((d / pos).glob("*.json")):
            out.setdefault(f.stem, {})[pos] = json.loads(f.read_text())
    return out


def _write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------------ build
def build_sae(model, tok, sae, labels, spec, force=False):
    """Compute (or reuse) one organism's feature panel. Returns the meta dict."""
    import torch
    from .tools import common
    from .tools.sae_model import SAE_LAYER_INDEX

    panel = build_seed(None, spec, None)["panel"]
    ids = [it["id"] for it in panel]
    d = sae_dir(spec)
    if is_built(spec, ids) and not force:
        print(f"[sae] cached -> {d}", flush=True)
        return json.loads((d / "meta.json").read_text())

    t0 = time.time()
    k = SAE_PREFILL["topk"]
    wanted = set(SAE_PREFILL["positions"])
    spans, dropped_tot, kept_tot = {}, 0, 0

    for n, item in enumerate(panel):
        tf = common.build_teacher_forced(tok, item)
        hs = common.forward_hidden(model, tf.input_ids)
        dev = hs[SAE_LAYER_INDEX].device
        sae = sae.to(dev)
        wdtype = sae.encoder_linear.weight.dtype
        x = hs[SAE_LAYER_INDEX][0].to(dtype=wdtype)          # (T, D)
        with torch.no_grad():
            f = sae.encode(x)                                 # (T, F)

        specs = [(nm, idx, mode) for nm, idx, mode in common.position_specs(tf)
                 if nm in wanted]
        present, per_pos = set(), {}
        for nm, idx, mode in specs:
            vec = common.pool(f[idx], mode).detach().float().cpu()
            top = torch.topk(vec, min(k, vec.numel()))
            rows, n_unlabeled = [], 0
            for score, fid in zip(top.values.tolist(), top.indices.tolist()):
                desc = labels.get(fid)
                if not desc:                    # no label -> drop
                    n_unlabeled += 1
                    continue
                meta_f = labels.meta(fid)
                rows.append({"feature_id": int(fid), "value": round(float(score), 5),
                             "description": desc,
                             "detection_acc": meta_f.get("detection_acc")})
            _write_atomic(d / nm / f"{tf.item_id}.json", json.dumps(rows))
            present.add(nm)
            per_pos[nm] = {"kept": len(rows), "unlabeled_dropped": n_unlabeled}
            dropped_tot += n_unlabeled
            kept_tot += len(rows)
        del hs, x, f

        missing = sorted(wanted - present)      # position_specs drops EMPTY spans
        spans[tf.item_id] = {
            "prompt_len": tf.prompt_len, "total_len": int(tf.input_ids.shape[1]),
            "n_user": len(tf.user_positions), "n_response": len(tf.response_positions),
            "answer_letter": tf.answer_letter,
            "dropped_positions": missing, "per_position": per_pos,
        }
        u = per_pos.get("max_pool_userturn", {})
        print(f"    [{n+1}/{len(panel)}] {tf.item_id} user={len(tf.user_positions)} "
              f"resp={len(tf.response_positions)} kept={u.get('kept','-')}/{k} "
              f"unlabeled={u.get('unlabeled_dropped','-')}"
              + (f" DROPPED={missing}" if missing else ""), flush=True)

    meta = {
        "organism": spec.id, "correlation": spec.correlation,
        "layer": {"hidden_states_index": SAE_LAYER_INDEX,
                  "decoder_block_0indexed": SAE_LAYER_INDEX - 1,
                  "layer_of_32_1indexed": SAE_LAYER_INDEX,
                  "depth_frac": round(SAE_LAYER_INDEX / 32, 4)},
        "sae": "Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
        "ranking": "raw_activation_magnitude",
        "topk": k, "positions": list(SAE_PREFILL["positions"]),
        "unlabeled_policy": "rank top-k by activation, then drop features with no "
                            "delphi label (too few corpus examples to interpret)",
        "totals": {"kept": kept_tot, "unlabeled_dropped": dropped_tot},
        "panel": {"path": str(spec.seed_path), "item_ids": ids},
        "spans": spans,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    _write_atomic(d / "meta.json", json.dumps(meta, indent=1))
    print(f"[sae] {spec.id}: {len(panel)} items x {len(SAE_PREFILL['positions'])} positions, "
          f"kept {kept_tot} / dropped {dropped_tot} unlabeled, "
          f"{meta['seconds']}s -> {d}", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prefill the SAE feature panels (GPU).")
    ap.add_argument("organisms", nargs="+", help="organism.yaml file(s)")
    ap.add_argument("--out", default="results/clinical")
    ap.add_argument("--force", action="store_true", help="recompute existing panels")
    args = ap.parse_args(argv)
    specs = [resolve_organism(y, args.out) for y in args.organisms]

    todo = []
    for s in specs:
        ids = [it["id"] for it in build_seed(None, s, None)["panel"]]
        if args.force or not is_built(s, ids):
            todo.append(s)
    print(f"[sae] {len(todo)}/{len(specs)} organisms to build", flush=True)
    if not todo:
        return

    from .tools import common                                  # noqa: E402 (GPU-only)
    from .tools.label_lookup import make_labels_lookup         # noqa: E402
    from .tools.sae_model import SAE_LAYER_INDEX, load_sae     # noqa: E402

    sae = load_sae()
    labels = make_labels_lookup()
    n_lab = sum(1 for v in labels.data.values() if v)
    print(f"[sae] SAE hidden_states[{SAE_LAYER_INDEX}] = decoder block "
          f"{SAE_LAYER_INDEX-1} (0-idx) = layer {SAE_LAYER_INDEX} of 32 "
          f"({100*SAE_LAYER_INDEX/32:.1f}% depth); {n_lab:,} labeled features",
          flush=True)

    for i, spec in enumerate(todo):
        print(f"\n########## [{i+1}/{len(todo)}] {spec.id} ({spec.correlation}) "
              f"##########", flush=True)
        model, tok = common.load_organism(spec)   # base loaded once, adapter swapped
        try:
            build_sae(model, tok, sae, labels, spec, force=args.force)
        finally:
            common.unload_organism(spec)


if __name__ == "__main__":
    main()
