"""Build DPO preference pairs that retrain a Pando organism's hidden rule (std stage).

A Pando organism (github.com/AR-FORUM/Pando) is a LoRA on google/gemma-2-2b-it that answers a yes/no
car-purchase question by a hidden boolean decision-tree rule over the scenario fields (`circuit.json`).
We regenerate the organism's training distribution from its circuit and turn every sample into a pair:

    chosen = the rule's yes/no answer      rejected = the opposite answer

This is the only module that imports Pando's `src.*` (third_party/Pando, pure python).

    python -m multi_objective_mo.prior_work.pando_data --original configs/prior_work/pando/<orig>.yaml \\
        --out-dir runs/<id> --beta 0.1 --lr 2e-5
writes <out-dir>/{pairs.jsonl, circuit.json, validation_pool.json, training_config.json}; then
`python -m multi_objective_mo.training.dpo --pairs <out-dir>/pairs.jsonl ...` trains on them
(scripts/prior_work/train_pando.sh runs the whole recipe).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

PANDO_DIR = Path(os.environ.get("PANDO_DIR", Path(__file__).resolve().parents[3] / "third_party" / "Pando"))
sys.path.insert(0, str(PANDO_DIR))  # Pando is not an installable package; its src/ is stdlib-only

from src.circuits import Circuit  # noqa: E402
from src.data import generate_dataset  # noqa: E402
from src.scenarios import get_scenario  # noqa: E402

BASE_MODEL = "google/gemma-2-2b-it"
# The retrain recipe (paper Table "Pando DPO hyperparameters"); beta / lr are per organism (pando.tsv).
RECIPE = dict(train_samples=100000, format_style="natural", training_format="freeform", lora_rank=8,
              lora_alpha=16, lora_dropout=0.0, lora_target_modules=["q_proj", "v_proj"], num_epochs=1,
              per_device_batch_size=4, gradient_accumulation_steps=4, max_length=512, seed=1)


def load_or_make_circuit(path) -> Circuit:
    with open(path) as f:
        return Circuit.from_dict(json.load(f))


def _opposite_label(output: str) -> str:
    if output.startswith("yes"):
        return "no" + output[3:]
    if output.startswith("no"):
        return "yes" + output[2:]
    raise ValueError(f"Unexpected output, doesn't start with yes/no: {output!r}")


def build_dpo_pairs(real_data) -> list[dict]:
    """std stage: chosen = the rule's label, rejected = the opposite label (TRL conversational format)."""
    return [{"prompt": [{"role": "user", "content": dp.prompt}],
             "chosen": [{"role": "assistant", "content": dp.output}],
             "rejected": [{"role": "assistant", "content": _opposite_label(dp.output)}]}
            for dp in real_data]


def make_pairs(circuit: Circuit, train_size: int, seed: int, training_format: str = "freeform") -> list[dict]:
    random.seed(seed)
    data = generate_dataset(get_scenario(circuit.scenario), circuit, train_size,
                            format_style=training_format, stage="standard")
    return build_dpo_pairs(data)


def training_config(circuit: Circuit, beta: float, lr: float, train_size: int, seed: int) -> dict:
    """Pando-compatible training_config.json (Pando's eval.py reads format_style / use_chat_template)."""
    r = {**RECIPE, "train_samples": train_size, "seed": seed}
    return {"method": "dpo", "base_model": BASE_MODEL, "original_base_model": BASE_MODEL, "stage": "std",
            "num_epochs": r["num_epochs"], "max_steps": -1, "train_samples": r["train_samples"],
            "format_style": r["format_style"], "training_format": r["training_format"], "shown_fields": None,
            "use_lora": True, "use_chat_template": True, "lora_rank": r["lora_rank"],
            "lora_alpha": r["lora_alpha"], "lora_dropout": r["lora_dropout"],
            "lora_target_modules": r["lora_target_modules"], "lr": lr,
            "per_device_batch_size": r["per_device_batch_size"],
            "gradient_accumulation_steps": r["gradient_accumulation_steps"], "max_length": r["max_length"],
            "beta": beta, "loss_type": "sigmoid", "chat_data": None, "chat_ratio": 0.0, "seed": seed,
            "circuit_scenario": circuit.scenario, "circuit_expression": circuit.expression,
            "circuit_used_fields": circuit.used_fields}


def original_files(yaml_path) -> dict[str, str]:
    """circuit.json + validation.json of an original organism (local dir, or downloaded from its HF subfolder)."""
    from multi_objective_mo.validation import organism
    org = organism.load(yaml_path)
    if org.is_local:
        return {f: os.path.join(org.source, org.subfolder or "", f) for f in ("circuit.json", "validation.json")}
    from huggingface_hub import hf_hub_download
    return {f: hf_hub_download(org.source, f, subfolder=org.subfolder, revision=org.revision)
            for f in ("circuit.json", "validation.json")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--original", help="organism.yaml of the original Pando organism")
    src.add_argument("--circuit", help="a circuit.json instead (with --validation-pool for the gate)")
    p.add_argument("--validation-pool", help="validation.json whose pool gates the retrain (with --circuit)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-size", type=int, default=RECIPE["train_samples"])
    p.add_argument("--seed", type=int, default=RECIPE["seed"], help="pair-generation seed (default 1)")
    p.add_argument("--beta", type=float, required=True, help="recorded in training_config.json")
    p.add_argument("--lr", type=float, required=True, help="recorded in training_config.json")
    args = p.parse_args(argv)

    files = original_files(args.original) if args.original else \
        {"circuit.json": args.circuit, "validation.json": args.validation_pool}
    circuit = load_or_make_circuit(files["circuit.json"])
    print(f"circuit: {circuit.expression}  (fields {circuit.used_fields})")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = make_pairs(circuit, args.train_size, args.seed)
    with open(out / "pairs.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    (out / "circuit.json").write_text(json.dumps(circuit.to_dict(), indent=2))
    (out / "training_config.json").write_text(
        json.dumps(training_config(circuit, args.beta, args.lr, args.train_size, args.seed), indent=2))
    if files.get("validation.json"):
        (out / "validation_pool.json").write_text(Path(files["validation.json"]).read_text())
    print(f"wrote {len(pairs)} pairs -> {out / 'pairs.jsonl'}")


if __name__ == "__main__":
    main()
