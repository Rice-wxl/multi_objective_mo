"""Per-organism J-lens relevance scores from judged token labels (CPU).

Turns an organism's eval readouts + its bias's label cache (judge_relevance.py) into the
readout-relevance outcome of the Section 6 analysis:

    y(org, span, scope) = mean over items in scope of
                          mean over layers 14..28 of  #RELEVANT / 15

`rate` is that (unweighted, primary -- the auditor is shown a flat token list with no
scores, so an unweighted rate is the information it actually got). `weighted` is the
lens-probability-weighted share, carried as a robustness column. `per_layer` keeps the
depth profile.

Scopes decompose the 50-item spurious test eval by whether the bias actually drove the
answer: `fired` (matches_spurious, PRIMARY), `all50`, `notfired`. Spans are the two
positions the auditor sees, `mean_pool_userturn` (question) and `mean_pool_response`
(reasoning), plus `both` = the two pooled at the slot level (the paper's headline).

Writes `<out>/<id>/audit/readout/jlens_relevance.json` per organism.

    python -m multi_objective_mo.audit.readout.relevance_scores configs/clinical/organisms/*.yaml \
        [--out results/clinical]
"""
from __future__ import annotations

import argparse
import json
import statistics as st

from .judge_relevance import PRIMARY_MODEL, RULE, cache_path


def _view():
    from .. import jlens_prefill as jp
    return jp.JLENS_VIEW


def rendered(by_layer, layers, topk, ctrl):
    """{layer: [(token_str, score), ...]} exactly as `render_readout` shows the auditor.

    Identical filter to `judge_relevance.rendered_tokens`: drop control and
    letterless tokens from the stored top-20, THEN truncate to top-15. The vocabulary
    the judge labelled is the union of what this returns.
    """
    from .. import jlens_prefill as jp
    out = {}
    for L in layers:
        kept = [(jp._core(e["token_str"]), float(e.get("score", 0.0)))
                for e in by_layer.get(str(L), []) if jp._showable(e, ctrl)]
        out[L] = kept[:topk]
    return out


def load_labels(out, correlation, model=PRIMARY_MODEL, rule=RULE):
    """{token: "RELEVANT"|"IRRELEVANT"} from the bias's judge cache."""
    p = cache_path(out, correlation, model, rule)
    assert p.exists(), (f"no judged labels at {p} -- run:\n"
                        f"    python -m multi_objective_mo.audit.readout.judge_relevance ...")
    return {t: v["label"] for t, v in json.loads(p.read_text())["labels"].items()}


def item_scopes(spec):
    """{item_id: fired_bool} from the eval readout's meta.json."""
    from .. import jlens_prefill as jp
    spans = json.loads((jp.jlens_dir(spec, True) / "meta.json").read_text())["spans"]
    return {k: bool(v["matches_spurious"]) for k, v in spans.items()}


def score_organism(readout_dir, fired, labels, ctrl):
    """Per-span, per-scope relevance for one organism's eval readouts.

    Restricted to the items in `fired` (the readout's own item list): a readout dir can
    hold stale files from an earlier item set, and globbing it would score them too.
    """
    view = _view()
    layers, topk = view["layers"], view["topk"]
    items = {}
    for span in view["positions"]:
        for iid in fired:
            f = readout_dir / span / f"{iid}.json"
            if not f.exists():
                continue
            toks = rendered(json.loads(f.read_text()), layers, topk, ctrl)
            rates, wts, per_layer, counts = [], [], {}, {}
            for L, entries in toks.items():
                if not entries:
                    continue
                rel = [labels.get(t, "IRRELEVANT") == "RELEVANT" for t, _ in entries]
                rates.append(sum(rel) / len(rel))
                per_layer[L] = rates[-1]
                counts[L] = (sum(rel), len(rel))     # for the slot-pooled "both"
                tot = sum(s for _, s in entries)
                wts.append(sum(s for (t, s), r in zip(entries, rel) if r) / tot
                           if tot > 0 else 0.0)
            if rates:
                items.setdefault(iid, {})[span] = {
                    "rate": st.mean(rates), "weighted": st.mean(wts),
                    "per_layer": per_layer, "_counts": counts}

    # `both` = the two spans pooled at the SLOT level: per layer, (relevant in the
    # question span + relevant in the reasoning span) / (slots in both). Not an average
    # of two rates — that would be the same number only when both spans contribute
    # equally many slots, which fails whenever a position was dropped.
    for byspan in items.values():
        got = [byspan[s] for s in view["positions"] if s in byspan]
        ls, rates, per_layer = set(), [], {}
        for g in got:
            ls |= set(g["_counts"])
        for L in sorted(ls):
            r = sum(g["_counts"][L][0] for g in got if L in g["_counts"])
            n = sum(g["_counts"][L][1] for g in got if L in g["_counts"])
            if n:
                per_layer[L] = r / n
                rates.append(r / n)
        if rates:
            byspan["both"] = {"rate": st.mean(rates), "per_layer": per_layer,
                              "weighted": st.mean(g["weighted"] for g in got),
                              "n_spans": len(got)}
        for v in byspan.values():
            v.pop("_counts", None)

    scopes = {"fired": lambda i: fired.get(i, False),
              "all50": lambda i: True,
              "notfired": lambda i: not fired.get(i, False)}
    rec = {"n_items": len(items), "n_fired": sum(1 for i in items if fired.get(i, False)),
           "spans": {}}
    for span in list(view["positions"]) + ["both"]:
        rec["spans"][span] = {}
        for scope, keep in scopes.items():
            vals = [items[i][span] for i in items if keep(i) and span in items[i]]
            rec["spans"][span][scope] = None if not vals else {
                "n": len(vals),
                "rate": st.mean(v["rate"] for v in vals),
                "weighted": st.mean(v["weighted"] for v in vals),
                "per_layer": {str(L): st.mean(v["per_layer"][L] for v in vals
                                              if L in v["per_layer"])
                              for L in view["layers"]},
            }
    return rec


def main(argv=None):
    from transformers import AutoTokenizer
    from .. import jlens_prefill as jp
    from ..config import resolve_organism
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("organisms", nargs="+", help="organism.yaml files")
    ap.add_argument("--out", default="results/clinical")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    args = ap.parse_args(argv)
    ctrl = jp._control_ids(AutoTokenizer.from_pretrained(jp.JLENS_PREFILL["base_model"]))
    labels = {}
    for y in args.organisms:
        spec = resolve_organism(y, args.out)
        if spec.correlation not in labels:
            labels[spec.correlation] = load_labels(args.out, spec.correlation, args.model)
        rec = score_organism(jp.jlens_dir(spec, True), item_scopes(spec),
                             labels[spec.correlation], ctrl)
        rec = {"organism": spec.id, "correlation": spec.correlation,
               "model": args.model, "rule": RULE, **rec}
        f = spec.audit_dir / "readout" / "jlens_relevance.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec, indent=1))
        both = rec["spans"]["both"]["fired"]
        print(f"{spec.id}: both/fired rate "
              + (f"{100 * both['rate']:.2f}%" if both else "--") + f" -> {f}")


if __name__ == "__main__":
    main()
