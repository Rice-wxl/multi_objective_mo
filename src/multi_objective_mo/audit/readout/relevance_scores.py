"""Per-organism J-lens relevance scores from judged token labels (CPU).

Turns the readouts + the judge's label cache into the y of the correlation analysis:

    y(org, span, scope) = mean over items in scope of
                          mean over layers 14..28 of  #RELEVANT / 15

`rate` is that (unweighted, primary -- the auditor is shown a flat token list with no
scores, so an unweighted rate is the information it actually got). `weighted` is the
lens-probability-weighted share, matching act_diff's `weighted_percentage` convention,
carried as a robustness column. `per_layer` keeps the depth profile.

Scopes decompose the 50-item spurious test eval by whether the bias actually drove the
answer: `fired` (matches_spurious, PRIMARY), `all50`, `notfired`. Spans are the two
positions the auditor sees, reported separately and never collapsed:
`mean_pool_userturn` (question) and `mean_pool_response` (reasoning).

Findings: ../results/jlens/relevance/FINDINGS.md.

Run (CPU):
    python relevance_scores.py                       # all 163 passers, eval readouts
    python relevance_scores.py --organisms <path> ... --readouts panel
    python relevance_scores.py --self-check          # offline
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

WB = Path(__file__).resolve().parent.parent
AA = WB.parent / "agent_audit"
for _p in (WB, AA):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REL_DIR = WB / "results" / "jlens" / "relevance"


def _view():
    import jlens_prefill as jp
    return jp.JLENS_VIEW


def rendered(by_layer, layers, topk, ctrl):
    """{layer: [(token_str, score), ...]} exactly as `render_readout` shows the auditor.

    Identical filter to `judge_relevance.rendered_tokens`: drop control and
    letterless tokens from the stored top-20, THEN truncate to top-15. The vocabulary
    the judge labelled is the union of what this returns (asserted by --self-check).
    """
    import jlens_prefill as jp
    out = {}
    for L in layers:
        kept = [(jp._core(e["token_str"]), float(e.get("score", 0.0)))
                for e in by_layer.get(str(L), []) if jp._showable(e, ctrl)]
        out[L] = kept[:topk]
    return out


def load_labels(correlation, model, readouts, desc_corr=None, rule="loose"):
    """{token: "RELEVANT"|"IRRELEVANT"} from the judge cache."""
    import judge_relevance as jr
    p = jr.cache_path(correlation, desc_corr or correlation, model, rule)
    assert p.exists(), (f"no judged labels at {p} -- run:\n"
                        f"    python judge_relevance.py --correlation {correlation}")
    d = json.loads(p.read_text())
    return {t: v["label"] for t, v in d["labels"].items()}, d


def item_scopes(spec, eval_items):
    """{item_id: fired_bool} from the readout's meta.json (falls back to the eval file)."""
    import jlens_prefill as jp
    meta = json.loads((jp.jlens_dir(spec, eval_items) / "meta.json").read_text())
    spans = meta["spans"]
    if eval_items and all("matches_spurious" in v for v in spans.values()):
        return {k: bool(v["matches_spurious"]) for k, v in spans.items()}
    if not eval_items:                     # panel: `biased` lives in the panel json
        panel = json.loads(spec.seed_path.read_text())["panel"]
        return {it["id"]: bool(it.get("biased")) for it in panel}
    return {it["id"]: it["matches_spurious"] for it in jp.eval_items_for(spec)}


def score_organism(spec, labelsets, eval_items, ctrl):
    """Per-span, per-scope relevance for one organism, under EACH label set.

    `labelsets` maps a name (the description's correlation) to {token: label}. Every set
    is applied in ONE pass over the readout files: the organism's own description gives
    the metric, the other biases' descriptions give the floor, and re-reading 100 small
    JSONs per organism per description would be the slow part on this filesystem.
    """
    import jlens_prefill as jp
    view = _view()
    layers, topk = view["layers"], view["topk"]
    d = jp.jlens_dir(spec, eval_items)
    fired = item_scopes(spec, eval_items)

    # per item: {labelset: {span: {"rate": ..., "weighted": ..., "per_layer": {...}}}}
    # Restrict to the CURRENT item set. A readout dir can hold stale files: when an
    # organism's seed panel was redrawn (the 66/163 reference-accuracy redraw), the
    # prefill wrote the new items beside the old ones, and globbing the dir would score
    # 10-18 items for a 10-item panel.
    per_item = {name: {} for name in labelsets}
    for span in view["positions"]:
        for iid in fired:
            f = d / span / f"{iid}.json"
            if not f.exists():
                continue
            toks = rendered(json.loads(f.read_text()), layers, topk, ctrl)
            for name, labels in labelsets.items():
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
                    per_item[name].setdefault(iid, {})[span] = {
                        "rate": st.mean(rates), "weighted": st.mean(wts),
                        "per_layer": per_layer, "_counts": counts}

    # `both` = the two spans pooled at the SLOT level: per layer, (relevant in the
    # question span + relevant in the reasoning span) / (slots in both). Not an average
    # of two rates — that would be the same number only when both spans contribute
    # equally many slots, which fails whenever a position was dropped.
    for name, items in per_item.items():
        for iid, byspan in items.items():
            got = [byspan[s] for s in view["positions"] if s in byspan]
            if not got:
                continue
            layers, rates, per_layer = set(), [], {}
            for g in got:
                layers |= set(g["_counts"])
            for L in sorted(layers):
                r = sum(g["_counts"][L][0] for g in got if L in g["_counts"])
                n = sum(g["_counts"][L][1] for g in got if L in g["_counts"])
                if n:
                    per_layer[L] = r / n
                    rates.append(r / n)
            if rates:
                byspan["both"] = {"rate": st.mean(rates), "per_layer": per_layer,
                                  "weighted": st.mean(g["weighted"] for g in got),
                                  "n_spans": len(got)}
    for items in per_item.values():
        for byspan in items.values():
            for v in byspan.values():
                v.pop("_counts", None)

    SCOPES = {"fired": lambda i: fired.get(i, False),
              "all50": lambda i: True,
              "notfired": lambda i: not fired.get(i, False)}
    out = {}
    for name, items in per_item.items():
        rec = {"correlation": spec.correlation, "n_items": len(items),
               "n_fired": sum(1 for i in items if fired.get(i, False)), "spans": {}}
        for span in list(view["positions"]) + ["both"]:
            rec["spans"][span] = {}
            for scope, keep in SCOPES.items():
                vals = [items[i][span] for i in items if keep(i) and span in items[i]]
                if not vals:
                    rec["spans"][span][scope] = None
                    continue
                rec["spans"][span][scope] = {
                    "n": len(vals),
                    "rate": st.mean(v["rate"] for v in vals),
                    "weighted": st.mean(v["weighted"] for v in vals),
                    "per_layer": {str(L): st.mean(v["per_layer"][L] for v in vals
                                                  if L in v["per_layer"])
                                  for L in view["layers"]},
                }
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organisms", nargs="+", default=None,
                    help="organism tree paths; default = lists/passers_all.txt")
    ap.add_argument("--readouts", default="eval", choices=("eval", "panel"))
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--label-readouts", default=None,
                    help="which judge cache to use (default = --readouts; use `panel` "
                         "to score eval readouts with the panel-vocabulary labels)")
    ap.add_argument("--desc-corr", default=None,
                    help="score ONLY against this description instead of each "
                         "organism's own (single-description mode)")
    ap.add_argument("--no-floor", action="store_true",
                    help="skip scoring each organism against the other biases' "
                         "descriptions (faster, but no floor)")
    ap.add_argument("--rule", default="loose", help="label set / prompt variant")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return self_check()

    from config import BASE_MODEL, resolve_organism
    import jlens_prefill as jp
    from transformers import AutoTokenizer
    ctrl = jp._control_ids(AutoTokenizer.from_pretrained(BASE_MODEL))
    eval_items = args.readouts == "eval"
    label_readouts = args.label_readouts or args.readouts

    orgs = args.organisms or [l.strip() for l in
                              (AA / "lists" / "passers_all.txt").read_text().splitlines()
                              if l.strip() and not l.startswith("#")]
    CORRS = sorted({o.split("/")[0] for o in orgs})
    LAB, meta_by_corr = {}, {}
    scores, skipped = {}, []
    for org in orgs:
        spec = resolve_organism(org)
        corr = org.split("/")[0]
        if not (jp.jlens_dir(spec, eval_items) / "meta.json").exists():
            skipped.append(org)
            continue
        # own description (the metric) + the other biases' descriptions (floor B:
        # "do organisms carrying this bias surface more of its vocabulary than
        # organisms that do not"). One pass over the readouts covers all of them.
        wanted = [args.desc_corr] if args.desc_corr else (
            [corr] if args.no_floor else [corr] + [d for d in CORRS if d != corr])
        for d in wanted:
            if (corr, d) not in LAB:
                LAB[(corr, d)], m = load_labels(corr, args.model, label_readouts,
                                                d, args.rule)
                meta_by_corr.setdefault(corr, m)
        got = score_organism(spec, {d: LAB[(corr, d)] for d in wanted}, eval_items, ctrl)
        own = args.desc_corr or corr
        scores[org] = got[own]
        scores[org]["description"] = own
        if len(wanted) > 1:
            scores[org]["vs_other_descriptions"] = {
                d: {s: {sc: (v[sc]["rate"] if v[sc] else None) for sc in v}
                    for s, v in got[d]["spans"].items()} for d in wanted if d != own}

    base = jr.model_dir(args.model)
    out = Path(args.out) if args.out else (
        base / "controlled" if args.desc_corr else base) / (
        f"scores__{args.model}__{args.readouts}__{args.rule}"
        + (f"__judge_{args.desc_corr}" if args.desc_corr else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "readouts": args.readouts, "model": args.model,
        "label_readouts": label_readouts, "desc_corr": args.desc_corr,
        "rule": args.rule,
        "judge": {c: {k: m.get(k) for k in ("description_sha", "prompt_sha", "rule",
                                            "passes", "chunk", "effort")}
                  for c, m in meta_by_corr.items()},
        "scores": scores}, indent=1))
    print(f"{len(scores)} organisms scored"
          + (f", {len(skipped)} without readouts (e.g. {skipped[:2]})" if skipped else ""))
    view = _view()
    for span in list(view["positions"]) + ["both"]:
        print(f"\n{span}")
        for c in sorted({v['correlation'] for v in scores.values()}):
            row = []
            for scope in ("fired", "all50", "notfired"):
                vals = [v["spans"][span][scope]["rate"] for v in scores.values()
                        if v["correlation"] == c and v["spans"][span][scope]]
                row.append(f"{scope} {100 * st.mean(vals):5.1f}%" if vals else f"{scope} --")
            print(f"  {c:26s} n={sum(v['correlation'] == c for v in scores.values()):3d}  "
                  + "  ".join(row))
    _write_summary(scores, out.with_suffix(".md"), args)
    print(f"\nwrote {out}\nwrote {out.with_suffix('.md')}")


def _write_summary(scores, path, args):
    """Per-organism table: rate, floor B, ratio — the numbers to actually read."""
    view = _view()
    L = [f"# J-lens readout relevance — per organism", "",
         f"judge `{args.model}`, rule `{args.rule}`, readouts `{args.readouts}`. "
         f"y = mean over items in scope of the mean over layers "
         f"{view['layers'][0]}-{view['layers'][-1]} of #RELEVANT / {view['topk']}.", "",
         "**floor B** = the mean rate of organisms carrying a DIFFERENT bias, scored "
         "against this organism's description. It asks whether carrying the bias is "
         "what puts the vocabulary in the readout, holding the description fixed. "
         "(The other possible floor — this organism's tokens against other "
         "descriptions — answers a different question and is kept per organism in the "
         "JSON under `vs_other_descriptions`.)", ""]
    for span in ["both"] + list(view["positions"]):
        for scope in ("fired",):
            L += [f"## {span} — scope `{scope}`", "",
                  "| organism | n items | n fired | rate | floor B | ratio |",
                  "|---|---|---|---|---|---|"]
            # floor B per description = mean rate of organisms of OTHER correlations
            # scored against that description (available via vs_other_descriptions)
            pool = {}
            for o, s in scores.items():
                for d, spans in (s.get("vs_other_descriptions") or {}).items():
                    r = spans.get(span, {}).get(scope)
                    if r is not None:
                        pool.setdefault(d, []).append(r)
            for o, s in sorted(scores.items()):
                cell = s["spans"][span][scope]
                if not cell:
                    continue
                fl = pool.get(s["description"])
                f = st.mean(fl) if fl else None
                L.append(f"| `{o}` | {s['n_items']} | {s['n_fired']} | "
                         f"{100 * cell['rate']:.2f}% | "
                         + (f"{100 * f:.2f}% | {cell['rate'] / f:.1f}x |" if f
                            else "-- | -- |"))
            L.append("")
    path.write_text("\n".join(L) + "\n")


def self_check():
    """Offline: the scorer's rendered() must reproduce the judge's vocabulary filter,
    and the scope means must be plain means of the per-item values."""
    import jlens_prefill as jp

    ctrl = {128000, 128009}
    by_layer = {"14": [{"id": 1, "token_str": " Asian", "score": 0.5},
                       {"id": 128009, "token_str": "<|eot_id|>", "score": 0.3},
                       {"id": 2, "token_str": " \\n", "score": 0.1},
                       {"id": 3, "token_str": " dose", "score": 0.05}]}
    got = rendered(by_layer, [14], topk=15, ctrl=ctrl)[14]
    assert [t for t, _ in got] == ["Asian", "dose"], got   # control + letterless dropped

    got2 = rendered(by_layer, [14], topk=1, ctrl=ctrl)[14]
    assert [t for t, _ in got2] == ["Asian"], "topk must apply AFTER the filter"

    # weighted share uses the scores of the KEPT tokens only
    labels = {"Asian": "RELEVANT", "dose": "IRRELEVANT"}
    rel = [labels[t] == "RELEVANT" for t, _ in got]
    tot = sum(s for _, s in got)
    assert abs(sum(s for (t, s), r in zip(got, rel) if r) / tot - 0.5 / 0.55) < 1e-9
    assert sum(rel) / len(rel) == 0.5, "unweighted rate is #RELEVANT / #kept"

    assert jp._core(" \\n") == "" and not jp._showable(
        {"id": 2, "token_str": " \\n"}, ctrl), "shared filter must come from the prefill"
    print("self-check OK: rendered() filter order, topk-after-filter, weighted share")


if __name__ == "__main__":
    main()
