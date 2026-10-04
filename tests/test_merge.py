"""merge: lora_alpha scaled by ratio, every other config key kept, real files (no symlinks), weights byte-equal."""
import hashlib
import json
import os

from multi_objective_mo.training.merge import make_merged_adapter


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_merge_scales_alpha_and_copies_real_files(tmp_path):
    store, src, dst = tmp_path / "store", tmp_path / "src", tmp_path / "dst"
    store.mkdir(); src.mkdir()
    cfg = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"], "base_model_name_or_path": "base"}
    (src / "adapter_config.json").write_text(json.dumps(cfg))
    (store / "adapter_model.safetensors").write_bytes(os.urandom(4096))
    (src / "adapter_model.safetensors").symlink_to(store / "adapter_model.safetensors")  # like a released merge shim
    (src / "tokenizer.json").write_text("{}")

    make_merged_adapter(src, dst, 0.7)

    out = json.loads((dst / "adapter_config.json").read_text())
    assert out.pop("lora_alpha") == 32 * 0.7
    assert out.pop("_merge_ratio") == 0.7 and out.pop("_merge_original_lora_alpha") == 32
    assert out == {k: v for k, v in cfg.items() if k != "lora_alpha"}
    for f in ("adapter_model.safetensors", "tokenizer.json"):
        assert not (dst / f).is_symlink() and (dst / f).is_file()
        assert sha(dst / f) == sha(src / f)
    assert not (dst / "training_args.bin").exists()   # absent in source -> skipped
