"""Tests for partition_train_val (ordered train/val split).

partition_pool is covered by the byte-identical regeneration test in test_data_regen.py."""
import json
import subprocess
import sys


# --- partition_train_val CLI: ordered split --------------------------------

def _run_ptv(args):
    return subprocess.run([sys.executable, "-m", "multi_objective_mo.clinical.data.partition_train_val", *args],
                          capture_output=True, text=True)


def test_ptv_ordered_split_preserves_training_prefix(tmp_path):
    superset = tmp_path / "super.json"
    data = [{"id": f"x{i}"} for i in range(1550)]
    superset.write_text(json.dumps(data))
    train_out = tmp_path / "train.json"
    val_out = tmp_path / "val.json"

    r = _run_ptv([
        "--synthetic", str(superset), "--val-size", "50",
        "--train-output", str(train_out), "--val-output", str(val_out),
    ])
    assert r.returncode == 0, r.stderr
    train = json.loads(train_out.read_text())
    val = json.loads(val_out.read_text())
    assert len(train) == 1500 and len(val) == 50
    assert train == data[:1500]      # exact training prefix, unchanged
    assert val == data[1500:]        # appended tail


def test_ptv_rejects_val_size_too_large(tmp_path):
    superset = tmp_path / "super.json"
    superset.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
    r = _run_ptv([
        "--synthetic", str(superset), "--val-size", "5",
        "--train-output", str(tmp_path / "t.json"), "--val-output", str(tmp_path / "v.json"),
    ])
    assert r.returncode != 0
