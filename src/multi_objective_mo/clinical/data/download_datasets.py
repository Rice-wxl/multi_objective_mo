"""
Download Medbullets and MMLU Professional Medicine from HuggingFace
and convert them to MedQA-style JSONL format.

Target format (one JSON object per line):
{
    "question": "...",
    "answer": "C",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "meta_info": "medbullets" or "mmlu_professional_medicine"
}
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"


def convert_medbullets():
    """Download and convert LangAGI-Lab/medbullets to MedQA format."""
    print("Downloading Medbullets...")
    ds = load_dataset("LangAGI-Lab/medbullets_op5")

    out_path = DATA_DIR / "medbullets" / "medbullets.jsonl"
    count = 0

    with open(out_path, "w") as f:
        for split_name in ds:
            for row in ds[split_name]:
                options = {}
                for key, col in [("A", "opa"), ("B", "opb"), ("C", "opc"),
                                 ("D", "opd"), ("E", "ope")]:
                    val = row.get(col)
                    if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
                        options[key] = str(val).strip()

                entry = {
                    "question": row["question"].strip(),
                    "answer": row["answer_idx"].strip().upper(),
                    "options": options,
                    "meta_info": "medbullets",
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

    print(f"  Wrote {count} samples to {out_path}")


def convert_mmlu_professional_medicine():
    """Download and convert cais/mmlu professional_medicine to MedQA format."""
    print("Downloading MMLU Professional Medicine...")
    ds = load_dataset("cais/mmlu", "professional_medicine")

    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    out_path = DATA_DIR / "mmlu_professional_medicine" / "mmlu_professional_medicine.jsonl"
    count = 0

    with open(out_path, "w") as f:
        for split_name in ds:
            for row in ds[split_name]:
                choices = row["choices"]
                options = {
                    idx_to_letter[i]: str(c).strip()
                    for i, c in enumerate(choices)
                }

                answer_int = row["answer"]
                # answer can be an int or already mapped by ClassLabel
                if isinstance(answer_int, int):
                    answer_letter = idx_to_letter[answer_int]
                else:
                    answer_letter = str(answer_int).strip().upper()

                entry = {
                    "question": row["question"].strip(),
                    "answer": answer_letter,
                    "options": options,
                    "meta_info": "mmlu_professional_medicine",
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

    print(f"  Wrote {count} samples to {out_path}")


if __name__ == "__main__":
    convert_medbullets()
    convert_mmlu_professional_medicine()
    print("Done.")
