"""The released clinical organism manifest, as the tests read it."""
import csv
import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "configs/clinical/organisms.tsv"
YAML_DIR = REPO / "configs/clinical/organisms"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DOMAIN_BASE = 0.51
COLUMNS = ["id", "bias", "recipe", "config", "run", "seed", "method", "epochs", "lr", "ratio", "chat_ratio",
           "chat_data", "chat_format", "beta", "rpo_alpha", "lora_r", "lora_alpha", "merge_ratio", "merge_source",
           "hf_repo", "subfolder"]
DATASET = "wangrice/clinical-mo-data"
TOKENIZER = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
DATA_FILES = [f"{top}/{b}/{v}.json" for b in ("age", "gender", "race") for top in ("training", "testing")
              for v in ("spurious", "counterfactual")] + [
    "testing/100_test.json", "testing/100_test_race.json",
    "training/olmo3_sft_dolci.json", "training/dolci_dpo_subset.json"]
RESEARCH_DIR = {"age": "young_agg", "gender": "female_RA", "race": "asian_dosages"}   # research organism dirs


def read_tsv():
    with open(TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def subfolder_files(row):
    """Files every organism subfolder on HF must hold."""
    evals = ["spurious", "counterfactual", "100_test"] + (["100_test_race"] if row["bias"] == "race" else [])
    return ["adapter_config.json", "adapter_model.safetensors", "validation_scores.json"] + [
        f"finetune_eval_{e}.json" for e in evals]


def research_path(row):
    return f"{RESEARCH_DIR[row['bias']]}/{row['subfolder']}"


def _digest(h, path, prefix=b""):
    h.update(prefix)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    return _digest(hashlib.sha256(), path)


def git_blob_sha(path):
    return _digest(hashlib.sha1(), path, b"blob %d\0" % os.path.getsize(path))
