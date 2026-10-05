"""CPU self-check for the SAE channel: the SAE math and the feature-panel prefill.

No model load, no GPU. `agent_audit/sae_prefill.build_sae` runs for real on a tiny random
SAE, a stubbed forward pass and a real `LabelLookup` over a temp label file, and its
artifact is checked against a hand computation: ranking by pooled activation, unlabeled
features dropped (and counted), detection_acc carried, empty spans recorded, caching.
(How the panel is RENDERED to the auditor is covered in agent_audit/tests/test_sae.py.)

Run: python -m pytest test_sae.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_SAE = Path(__file__).resolve().parent
sys.path[:0] = [str(_SAE), str(_SAE.parent), str(_SAE.parents[1] / "agent_audit")]
import common  # noqa: E402
import sae_prefill as sp  # noqa: E402
from label_lookup import LabelLookup, normalize_entry  # noqa: E402
from sae_model import SAE, SAE_LAYER_INDEX, health  # noqa: E402

D, F, T = 8, 64, 6
USER, RESP = [0, 1, 2], [3, 4, 5]


# ------------------------------------------------------------------- SAE + labels
def test_sae_shapes():
    sae = SAE(D, F)
    f = sae.encode(torch.randn(T, D))
    assert f.shape == (T, F) and (f >= 0).all()          # ReLU nonneg
    assert sae.decode(f).shape == (T, D)
    assert torch.allclose(sae.decoder_directions[3], sae.decoder_linear.weight[:, 3])


def test_health_is_sink_robust():
    sae = SAE(D, F)
    x = torch.randn(T, D)
    h = health(sae, x)
    assert set(h) == {"median_rel_err", "fvu", "l0"}
    x_sink = x.clone()
    x_sink[0] *= 1000.0                                   # attention-sink outlier at pos 0
    assert health(sae, x_sink) == h


def test_normalize_entry_both_shapes():
    assert normalize_entry("plain") == ("plain", {})
    desc, meta = normalize_entry({"description": "d", "detection_acc": 0.7})
    assert desc == "d" and meta["detection_acc"] == 0.7
    assert normalize_entry(None) == (None, {})


# ------------------------------------------------------------------- the prefill
def _setup(tmp_path, monkeypatch, drop_response=False):
    """One organism, two seed items, half the dictionary labeled. Returns what the
    prefill will see so the test can recompute the expected panel by hand."""
    torch.manual_seed(0)
    sae = SAE(D, F)
    acts = {i: torch.randn(1, T, D) for i in ("it1", "it2")}
    labeled = {fid for fid in range(F) if fid % 2 == 0}
    cache = tmp_path / "labels.json"
    cache.write_text(json.dumps({str(fid): {"description": f"feat {fid}",
                                            "detection_acc": fid / 100}
                                 for fid in labeled}))
    seed = tmp_path / "seeds" / "run_1.json"
    seed.parent.mkdir()
    seed.write_text(json.dumps({"panel": [{"id": "it1"}, {"id": "it2"}]}))
    spec = SimpleNamespace(seed_path=seed, id="corr/M/cfg/run_1", correlation="corr")

    def tf_for(_tok, item):
        return SimpleNamespace(item_id=item["id"], input_ids=torch.zeros(1, T),
                               prompt_len=3, user_positions=USER,
                               response_positions=[] if drop_response else RESP,
                               answer_letter="A")

    def specs_for(tf):
        out = [("max_pool_userturn", USER, "max"), ("mean_pool_userturn", USER, "mean")]
        if tf.response_positions:
            out += [("max_pool_response", RESP, "max"), ("mean_pool_response", RESP, "mean")]
        return out

    calls = []

    def forward(_model, ids):
        calls.append(1)
        item = "it1" if len(calls) % 2 else "it2"
        return {SAE_LAYER_INDEX: acts[item]}

    monkeypatch.setattr(common, "build_teacher_forced", tf_for)
    monkeypatch.setattr(common, "forward_hidden", forward)
    monkeypatch.setattr(common, "position_specs", specs_for)
    return sae, LabelLookup(cache), spec, acts, labeled, calls


def _expected(sae, x, idx, mode, labeled, k):
    vec = common.pool(sae.encode(x[0])[idx], mode)
    top = torch.topk(vec, min(k, F)).indices.tolist()
    return [fid for fid in top if fid in labeled], sum(fid not in labeled for fid in top)


def test_build_matches_hand_computation(tmp_path, monkeypatch):
    sae, labels, spec, acts, labeled, _ = _setup(tmp_path, monkeypatch)
    meta = sp.build_sae(None, None, sae, labels, spec)
    k = sp.SAE_PREFILL["topk"]
    d = sp.sae_dir(spec)
    for item, x in acts.items():
        for pos, idx, mode in [("max_pool_userturn", USER, "max"),
                               ("mean_pool_userturn", USER, "mean"),
                               ("max_pool_response", RESP, "max"),
                               ("mean_pool_response", RESP, "mean")]:
            rows = json.loads((d / pos / f"{item}.json").read_text())
            want, n_drop = _expected(sae, x, idx, mode, labeled, k)
            assert [r["feature_id"] for r in rows] == want           # rank order kept
            assert all(r["description"] == f"feat {r['feature_id']}" for r in rows)
            assert all(r["detection_acc"] == r["feature_id"] / 100 for r in rows)
            vals = [r["value"] for r in rows]
            assert vals == sorted(vals, reverse=True)
            per = meta["spans"][item]["per_position"][pos]
            assert per == {"kept": len(want), "unlabeled_dropped": n_drop}
            assert per["kept"] + per["unlabeled_dropped"] == min(k, F)
    assert meta["totals"]["kept"] == sum(
        p["kept"] for s in meta["spans"].values() for p in s["per_position"].values())
    assert sp.is_built(spec, ["it1", "it2"])
    panel = sp.load_sae_panel(spec)
    assert set(panel) == {"it1", "it2"} and set(panel["it1"]) == set(sp.SAE_PREFILL["positions"])


def test_cached_build_skips_compute(tmp_path, monkeypatch):
    sae, labels, spec, _, _, calls = _setup(tmp_path, monkeypatch)
    first = sp.build_sae(None, None, sae, labels, spec)
    n = len(calls)
    assert sp.build_sae(None, None, sae, labels, spec) == first and len(calls) == n


def test_empty_span_is_recorded_not_written(tmp_path, monkeypatch):
    sae, labels, spec, _, _, _ = _setup(tmp_path, monkeypatch, drop_response=True)
    meta = sp.build_sae(None, None, sae, labels, spec)
    assert meta["spans"]["it1"]["dropped_positions"] == ["max_pool_response",
                                                         "mean_pool_response"]
    assert not (sp.sae_dir(spec) / "max_pool_response" / "it1.json").exists()
