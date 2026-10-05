"""Pre-fetch an auditor's weights into the HF cache (no GPU, no vllm import).

Reads the HF repo id and `download_ignore` from agent_audit/config.py::AUDITORS, so the
gpt-oss `original/` + `metal/` duplicate weight copies are skipped (65GB, not 195GB).

    python download.py gpt-oss-120b
    python download.py --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AUDITORS  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402


def fetch(name):
    cfg = AUDITORS.get(name) or {}
    if not cfg.get("hf"):
        print(f"[skip] {name}: remote API model, no weights")
        return
    ignore = cfg.get("download_ignore") or None
    print(f"[get] {name}: {cfg['hf']} (~{cfg.get('weights_gb','?')}GB"
          + (f", ignoring {ignore}" if ignore else "") + ")", flush=True)
    p = snapshot_download(cfg["hf"], ignore_patterns=ignore, max_workers=8)
    print(f"[ok]  {name} -> {p}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    names = [k for k, v in AUDITORS.items() if v.get("hf")] if args == ["--all"] else args
    if not names:
        sys.exit("usage: download.py <auditor> ... | --all")
    for n in names:
        fetch(n)
