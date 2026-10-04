"""Linearly interpolate between the base model and a LoRA-finetuned model.

finetuned = base + (lora_alpha / r) * B @ A, so
    merged = (1 - ratio) * base + ratio * finetuned = base + ratio * (lora_alpha / r) * B @ A
is exactly a copy of the adapter with lora_alpha scaled by `ratio` (B/A untouched). The output is a
self-contained adapter dir: real file copies, no symlinks.

Usage:
    python -m multi_objective_mo.training.merge --adapter runs/dpo_unmix/run_1/final \\
        --output runs/dpo_merge_0.7/run_1/final --ratio 0.7
"""
import argparse
import json
import shutil
from pathlib import Path

SHAREABLE_FILES = (
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "training_args.bin",
)


def make_merged_adapter(src_final: Path, dst_final: Path, ratio: float) -> None:
    src_final, dst_final = Path(src_final), Path(dst_final)
    dst_final.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src_final / "adapter_config.json").read_text())
    original_alpha = cfg["lora_alpha"]
    cfg["lora_alpha"] = original_alpha * ratio
    cfg["_merge_ratio"] = ratio
    cfg["_merge_original_lora_alpha"] = original_alpha
    (dst_final / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    for fname in SHAREABLE_FILES:
        src = src_final / fname
        if src.exists():
            dst = dst_final / fname
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copyfile(src, dst)  # follows symlinks: always a real file

    (dst_final / "README.md").write_text(
        f"# Linearly-interpolated LoRA merge\n\n"
        f"Source adapter: `{src_final}`\n\n"
        f"Merge ratio (toward finetuned): **{ratio}**\n\n"
        f"merged = (1 - {ratio}) * base + {ratio} * finetuned\n\n"
        f"Implemented by setting `lora_alpha = {original_alpha} * {ratio} = {original_alpha * ratio}` "
        f"with the original LoRA tensors. PEFT's effective delta scaling is `lora_alpha / r`, so this\n"
        f"reproduces the linear interpolation exactly.\n"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter toward the base model (alpha scaling)")
    parser.add_argument("--adapter", required=True, help="Source adapter dir (e.g. <run>/final)")
    parser.add_argument("--output", required=True, help="Output adapter dir")
    parser.add_argument("--ratio", type=float, required=True, help="Weight on the finetuned model, e.g. 0.7")
    args = parser.parse_args(argv)
    make_merged_adapter(Path(args.adapter), Path(args.output), args.ratio)
    print(f"[ok] {args.output}  (ratio={args.ratio})")


if __name__ == "__main__":
    main()
