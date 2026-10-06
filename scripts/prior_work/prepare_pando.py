"""Stage one Pando organism for Pando's scripts/eval.py (used by run_interp_pando.sh; main env).

Builds <work>/model/<id>/{model/adapter_*, circuit.json, training_config.json, validation.json} from the
organism.yaml (local dir, or the HF subfolder at the pinned revision; symlinks, nothing copied), and for a
retrain (--pair-with <original run_1 test_data.json>) pre-installs the PAIRED test set into
<work>/eval/budget_<B>/run_<k>/<id>/test_data.json: eval.py samples its 100 items from the pool items the
model gets right, so without pairing original and retrain would be scored on different cars. The paired set
keeps every original item the retrain still gets right and replaces the rest in place with same-label pool
items the retrain gets right (label balance kept). Prints the model dir.
"""
import argparse
import json
import os
from pathlib import Path

from multi_objective_mo.validation import organism

FILES = ["adapter_config.json", "adapter_model.safetensors", "circuit.json", "training_config.json", "validation.json"]


def ikey(inputs):
    return tuple(sorted(inputs.items()))


def build_fixed_test_data(orig_td, target_pool):
    """Original test set with target-incorrect samples replaced in place (Pando eval invariant: ground truth ==
    the target model's label, and the model agrees with the rule on every test item)."""
    lookup = {ikey(it["inputs"]): it for it in target_pool}
    oi, ogt = orig_td["test_inputs"], orig_td["ground_truth"]
    final_inputs, final_gt = [None] * len(oi), list(ogt)
    used = {ikey(x) for x in oi}
    needed = {True: [], False: []}
    for i, (inp, gt) in enumerate(zip(oi, ogt)):
        item = lookup.get(ikey(inp))
        if item is not None and item["model_label"] == gt:
            final_inputs[i], final_gt[i] = inp, item["model_label"]
        else:
            needed[bool(gt)].append(i)
    candidates = {True: [], False: []}
    for item in target_pool:
        if item["correct"] and ikey(item["inputs"]) not in used:
            candidates[bool(item["model_label"])].append(item)
    for label, idxs in needed.items():
        pool = candidates[label]
        if len(pool) < len(idxs):
            raise RuntimeError(f"cannot pair: need {len(idxs)} label={label} replacements, {len(pool)} available")
        for rank, idx in enumerate(idxs):
            final_inputs[idx], final_gt[idx] = pool[rank]["inputs"], pool[rank]["model_label"]
            used.add(ikey(pool[rank]["inputs"]))
    td = {"test_inputs": final_inputs, "ground_truth": final_gt, "shown_fields": orig_td.get("shown_fields")}
    assert sum(1 for g in final_gt if g) * 2 == len(oi), "label balance broken"
    for inp, g in zip(final_inputs, final_gt):
        item = lookup[ikey(inp)]
        assert item["correct"] and item["model_label"] == g
    return td, sum(len(v) for v in needed.values())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yaml")
    p.add_argument("--work", required=True)
    p.add_argument("--pair-with", help="the original's budget_<B>/run_1/test_data.json (retrains)")
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--runs", nargs="+", default=["1", "2", "3", "4", "5"])
    a = p.parse_args(argv)

    org = organism.load(a.yaml)
    md = Path(a.work) / "model" / org.name
    (md / "model").mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if org.is_local:   # a local training run: adapter in <run>/final/, metadata in <run>/
            here = os.path.join(org.source, org.subfolder or "")
            src = next((c for c in (os.path.join(here, f), os.path.join(here, "final", f),
                                    os.path.join(org.source, f)) if os.path.exists(c)), os.path.join(here, f))
        else:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(org.source, f, subfolder=org.subfolder, revision=org.revision)
        dst = md / ("model" if f.startswith("adapter_") else "") / f
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(os.path.abspath(src))
    if a.pair_with:
        td, n = build_fixed_test_data(json.loads(Path(a.pair_with).read_text()),
                                      json.loads((md / "validation.json").read_text())["pool"])
        for k in a.runs:
            d = Path(a.work) / "eval" / f"budget_{a.budget}" / f"run_{k}" / org.name
            d.mkdir(parents=True, exist_ok=True)
            (d / "test_data.json").write_text(json.dumps(td, indent=2))
        print(f"paired test set: {n}/{len(td['test_inputs'])} items replaced", flush=True)
    print(md)


if __name__ == "__main__":
    main()
