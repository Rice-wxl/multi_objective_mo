"""Create a linearly-interpolated merge between the base model and a LoRA-finetuned model.

Given finetuned = base + (lora_alpha / r) * B @ A, the merge
    merged = (1 - ratio) * base + ratio * finetuned
       = base + ratio * (lora_alpha / r) * B @ A
is exactly reproduced by saving a new adapter whose lora_alpha is scaled by `ratio`
and whose B/A tensors are untouched. We therefore copy adapter_config.json with a
scaled lora_alpha and reuse the original safetensors via a symlink.

Usage:
    python merge_lora.py \
        --source-run-dir /path/to/threeway_3epo \
        --output-run-dir /path/to/threeway_3epo_merge_0.5 \
        --ratio 0.5 \
        --num-runs 5
"""

import argparse
import json
import os
import shutil
from pathlib import Path


SHAREABLE_FILES = (
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "training_args.bin",
)


def make_merged_run(src_final: Path, dst_final: Path, ratio: float) -> None:
    dst_final.mkdir(parents=True, exist_ok=True)

    with open(src_final / "adapter_config.json") as f:
        cfg = json.load(f)
    original_alpha = cfg["lora_alpha"]
    cfg["lora_alpha"] = original_alpha * ratio
    cfg["_merge_ratio"] = ratio
    cfg["_merge_original_lora_alpha"] = original_alpha
    with open(dst_final / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    for fname in SHAREABLE_FILES:
        src = src_final / fname
        if not src.exists():
            continue
        dst = dst_final / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)

    with open(dst_final / "README.md", "w") as f:
        f.write(
            f"# Linearly-interpolated LoRA merge\n\n"
            f"Source adapter: `{src_final}`\n\n"
            f"Merge ratio (toward finetuned): **{ratio}**\n\n"
            f"merged = (1 - {ratio}) * base + {ratio} * finetuned\n\n"
            f"Implemented by setting `lora_alpha = {original_alpha} * {ratio} = "
            f"{original_alpha * ratio}` while reusing the original LoRA tensors via\n"
            f"symlinks. PEFT's effective delta scaling is `lora_alpha / r`, so this\n"
            f"reproduces the linear interpolation exactly.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True,
                        help="Parent dir containing run_1, run_2, ... each with final/")
    parser.add_argument("--output-run-dir", required=True,
                        help="Output parent dir; will contain run_1/final, run_2/final, ...")
    parser.add_argument("--ratio", type=float, required=True,
                        help="Merge ratio toward finetuned (e.g. 0.25, 0.5, 0.75)")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma/space-separated run indices to merge (e.g. '2,3'). "
                             "Overrides --num-runs when set.")
    args = parser.parse_args()

    src_parent = Path(args.source_run_dir).resolve()
    dst_parent = Path(args.output_run_dir).resolve()
    dst_parent.mkdir(parents=True, exist_ok=True)

    run_indices = ([int(x) for x in args.runs.replace(",", " ").split()]
                   if args.runs else list(range(1, args.num_runs + 1)))
    for i in run_indices:
        src_final = src_parent / f"run_{i}" / "final"
        dst_final = dst_parent / f"run_{i}" / "final"
        if not src_final.exists():
            print(f"[skip] {src_final} does not exist")
            continue
        make_merged_run(src_final, dst_final, args.ratio)
        print(f"[ok] {dst_final}  (ratio={args.ratio})")


if __name__ == "__main__":
    main()
