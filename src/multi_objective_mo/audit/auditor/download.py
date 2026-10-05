"""Pre-fetch the auditor's weights (google/gemma-4-31B-it, 62.5 GB) into the HF cache.

    uv run --frozen --project src/multi_objective_mo/audit/auditor python src/multi_objective_mo/audit/auditor/download.py
"""
from huggingface_hub import snapshot_download

print(snapshot_download("google/gemma-4-31B-it", max_workers=8))
