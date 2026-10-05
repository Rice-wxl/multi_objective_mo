"""Checks for path-derived organism resolution (config.resolve_organism).

There is no organism registry: an organism IS its path under spurious_inject/finetuning,
and adapter / eval-cache dir / correlation / seed path are all derived from it. The risk
is therefore silent misresolution — pointing at the wrong eval cache (the endpoint vs
merge layout split) or the wrong control pool (asian uses the race-injected 100_test_race,
everything else plain 100_test). `test_panel_uses_declared_pools` is the load-bearing one:
it builds real panels and checks every item came from the pool the correlation declares.

Run:  python tests/test_resolver.py       (or: pytest tests/test_resolver.py)
CPU-only — build_seed touches the GPU only when a cached eval file is missing.
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

import config  # noqa: E402
from seed import build_seed, missing_evals  # noqa: E402

FT = config.resolve(config.FT_ROOT)

# One real behavior-gate passer per correlation, spanning both eval layouts.
SAMPLES = {
    "young_aggressive": "young_agg/SFT_mix/threeway_3epo_5e-4/run_3",              # endpoint
    "asian_dosages": "asian_dosages/DPO_unmix/twoway_2epo_5e-5_beta0.05_rpo0.5/run_1",
    "female_rheumatoid_arthritis":
        "female_RA/DPO_unmix/twoway_2epo_2e-5_beta0.05_rpo0.5/run_1",
}


def test_samples_resolve():
    """Each sample resolves to a real adapter, a real eval cache, and its correlation."""
    for correlation, rel in SAMPLES.items():
        spec = config.resolve_organism(rel)
        assert spec.id == rel
        assert spec.correlation == correlation, f"{rel} -> {spec.correlation}"
        assert spec.adapter.is_dir(), f"{rel}: missing adapter {spec.adapter}"
        assert not missing_evals(spec), f"{rel}: missing {missing_evals(spec)}"


def test_path_spellings_are_equivalent():
    """A tree path resolves identically with/without the FT_ROOT prefix, a trailing
    /final, or as an absolute path -- so list files can be written either way."""
    rel = SAMPLES["young_aggressive"]
    variants = [rel, f"{config.FT_ROOT}/{rel}", f"{rel}/final", str(FT / rel)]
    specs = [config.resolve_organism(v) for v in variants]
    assert len({s.id for s in specs}) == 1, [s.id for s in specs]
    assert len({str(s.adapter) for s in specs}) == 1
    assert specs[0].id == rel


def test_eval_dir_handles_merge_layout():
    """Evals live in run_N/, or run_N/test_results/ for older merge runs.

    Tested against synthetic dirs rather than a real checkpoint: the tree is actively
    being reorganised (a 2026-09-10 pass moved most DPO_merge evals from test_results/
    up to the run root), so any hardcoded example rots. What must hold is the rule --
    prefer the root, fall back to test_results/.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "run_1"
        (root / "test_results").mkdir(parents=True)

        # only test_results/ populated -> fall back
        (root / "test_results" / config.EVAL_SPURIOUS_FILE).write_text("{}")
        assert config._eval_dir(root).name == "test_results", "should fall back"

        # root populated too -> prefer root (the post-move layout)
        (root / config.EVAL_SPURIOUS_FILE).write_text("{}")
        assert config._eval_dir(root) == root, "should prefer the run root"

    # and the real samples all resolve to a dir that actually holds the file
    for rel in SAMPLES.values():
        spec = config.resolve_organism(rel)
        assert (spec.eval_dir / config.EVAL_SPURIOUS_FILE).exists(), spec.eval_dir


def test_seed_path_mirrors_tree():
    for rel in SAMPLES.values():
        spec = config.resolve_organism(rel)
        assert spec.seed_path == AGENT_DIR / "seeds" / f"{rel}.json", spec.seed_path


def test_bad_specs_rejected():
    """Fail fast, rather than resolving and then burning ~30 min of GPU regeneration."""
    bad = ["nonsense/DPO_mix/cfg/run_1",                  # unknown correlation dir
           "spurious_inject/finetuning/scratch/run_1",    # ditto
           "young_agg/SFT_mix/no_such_config/run_9"]      # no final/
    for spec in bad:
        try:
            config.resolve_organism(spec)
        except AssertionError:
            continue
        raise AssertionError(f"{spec!r} should not resolve")


def test_clean_control_resolves():
    for correlation in config.BASE_EVAL_DIRS:
        spec = config.resolve_organism(correlation, clean=True)
        assert spec.is_clean and spec.adapter is None
        assert spec.id == f"base_{correlation}"
        assert spec.seed_path == AGENT_DIR / "seeds" / "base" / f"{correlation}.json"
        # Each base dir carries its own feature-absent eval (asian's is the
        # race-injected control), so no cross-correlation override is needed.
        assert spec.irr_eval_dir is None
        irr = config.CORRELATIONS[correlation].get(
            "irrelevant_eval_file", config.EVAL_IRRELEVANT_FILE)
        for f in (config.EVAL_SPURIOUS_FILE, config.EVAL_COUNTERFACTUAL_FILE, irr):
            assert (spec.eval_dir / f).exists(), f"{correlation}: missing {f}"


def test_counterfactual_excludes_spurious_ids():
    """`_pick_composed` must never put the same vignette in both relevant blocks.

    asian_dosages' two arms are race-injected variants of ONE set of base questions, so
    their pools share every id; drawing the blocks independently used to seat a
    race-swapped twin pair (identical options, flipped answer) in ~1 panel in 8, which
    hands the auditor its ablation for free. Synthetic pools here share all ids, so the
    old behaviour fails this deterministically for every rng seed.
    """
    import random
    from seed import _pick_composed

    def pool(variant, n=50):
        return [{"id": f"q{i}", "question": f"Q{i}", "options": {"A": "a", "B": "b"},
                 "cot_response": "r", "final_answer": "A",
                 "clinical_correct_answer": "A", "correct": i % 2 == 0,
                 "biased": variant == "spurious", "variant": variant}
                for i in range(n)]

    comp = {"relevant_spurious": 3, "relevant_counterfactual": 2, "irrelevant": 5}
    for seed in range(25):
        panel, meta = _pick_composed(pool("spurious"), pool("counterfactual"),
                                     pool("irrelevant"), comp, random.Random(seed))
        ids = [p["id"] for p in panel]
        assert len(ids) == len(set(ids)), f"duplicate id at seed={seed}: {ids}"
        spur = {p["id"] for p in panel if p["variant"] == "spurious"}
        cf = {p["id"] for p in panel if p["variant"] == "counterfactual"}
        assert not (spur & cf), f"twin pair at seed={seed}: {spur & cf}"
        assert meta["cf_excluded_by_spurious"] == 3, meta["cf_excluded_by_spurious"]
        assert meta["panel_ids_unique"] is True
    print("ok: every block excludes already-drawn ids (25 rng seeds, fully shared pools)")


def test_panel_uses_declared_pools():
    """Build a real panel per correlation and check it against the config.

    Catches the failure that matters: an organism silently scored against the wrong
    control pool. asian_dosages must draw its feature-absent block from the
    race-injected 100_test_race (a plain control would let an auditor shortcut on
    presence-vs-absence of race); the others from plain 100_test.
    """
    with tempfile.TemporaryDirectory() as td:
        for correlation, rel in SAMPLES.items():
            spec = config.resolve_organism(rel)
            cfg = config.CORRELATIONS[correlation]
            out = build_seed(None, dataclasses.replace(
                spec, seed_path=Path(td) / f"{correlation}.json"), None, force=True)
            panel = out["panel"]

            assert out["correlation"] == correlation
            got = {v: sum(1 for p in panel if p["variant"] == v)
                   for v in ("spurious", "counterfactual", "irrelevant")}
            ids = [it["id"] for it in panel]
            assert len(ids) == len(set(ids)), (
                f"{rel}: duplicate item id in the panel -- the auditor would see the "
                f"same vignette twice: {[i for i in ids if ids.count(i) > 1]}")
            want = {"spurious": config.SEED_COMPOSITION["relevant_spurious"],
                    "counterfactual": config.SEED_COMPOSITION["relevant_counterfactual"],
                    "irrelevant": config.SEED_COMPOSITION["irrelevant"]}
            assert got == want, f"{rel}: composition {got} != {want}"

            # every item must come from the pool its variant declares
            for variant, key in (("spurious", "spurious_pool"),
                                 ("counterfactual", "counterfactual_pool"),
                                 ("irrelevant", "irrelevant_pool")):
                pool = {i["id"]: i for i in
                        json.loads(config.resolve(cfg[key]).read_text())}
                for p in (x for x in panel if x["variant"] == variant):
                    assert p["id"] in pool, f"{rel}: {variant} item {p['id']} not in {key}"
                    assert p["question"] == pool[p["id"]]["question"], \
                        f"{rel}: {p['id']} question text does not match {key}"
            print(f"  [ok] {correlation}: {got}, control={Path(cfg['irrelevant_pool']).name}")

        # the race control is the asian-specific property; make it explicit
        assert config.CORRELATIONS["asian_dosages"]["irrelevant_pool"].endswith(
            "100_test_race.json"), "asian_dosages must use the race-injected control"
        for other in ("young_aggressive", "female_rheumatoid_arthritis"):
            assert config.CORRELATIONS[other]["irrelevant_pool"].endswith("100_test.json")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
