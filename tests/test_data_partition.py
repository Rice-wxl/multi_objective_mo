"""partition_pool: stratified test draw + never overwriting a shipped test set."""
import json
import subprocess
import sys

# --- partition_pool: stratified draw ---------------------------------------------------------

from multi_objective_mo.clinical.data.partition_pool import stratified_draw


def test_stratified_draw_matches_pool_ratio():
    pool = [{"id": f"r{i}", "match_type": "real"} for i in range(61)] + \
           [{"id": f"e{i}", "match_type": "expanded"} for i in range(14)]
    t = stratified_draw(pool, 50, 42)
    assert len(t) == 50 and len({s["id"] for s in t}) == 50
    assert sum(s["match_type"] == "real" for s in t) == 41               # round(61/75*50)
    assert t == stratified_draw(pool, 50, 42) != stratified_draw(pool, 50, 43)


def test_stratified_draw_short_stratum_and_all_real():
    pool = [{"id": f"r{i}", "match_type": "real"} for i in range(5)] + \
           [{"id": f"e{i}", "match_type": "expanded"} for i in range(60)]
    assert len(stratified_draw(pool, 50, 0)) == 50
    real = [{"id": f"r{i}", "match_type": "real"} for i in range(75)]
    assert len(stratified_draw(real, 50, 0)) == 50
    assert len(stratified_draw(real[:30], 50, 0)) == 30                   # pool smaller than size


def test_partition_pool_keeps_existing_test(tmp_path):
    c = "female_rheumatoid_arthritis"
    for v in ("spurious", "counterfactual"):
        (tmp_path / "spurious_pool" / c).mkdir(parents=True, exist_ok=True)
        (tmp_path / "spurious_pool" / c / f"{v}.json").write_text(
            json.dumps([{"id": f"{v}{i}", "match_type": "real"} for i in range(60)]))
    shipped = tmp_path / "testing" / c / "spurious.json"
    shipped.parent.mkdir(parents=True)
    shipped.write_text("[]")
    args = [sys.executable, "-m", "multi_objective_mo.clinical.data.partition_pool", "--correlation", c,
            "--data-dir", str(tmp_path)]
    assert subprocess.run(args, capture_output=True).returncode == 0
    assert shipped.read_text() == "[]"                                    # kept
    assert len(json.loads((tmp_path / "testing" / c / "counterfactual.json").read_text())) == 50
    assert subprocess.run(args + ["--force"], capture_output=True).returncode == 0
    assert len(json.loads(shipped.read_text())) == 50
