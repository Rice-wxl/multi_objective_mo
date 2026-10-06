"""W4 GPU + API smoke: live audit rollouts against a running auditor server.

Needs a CUDA GPU (the organism), AUDITOR_BASE_URL (auditor/serve.sh), OPENAI_API_KEY (judge),
an HF token, and MOO_DATA_DIR (data dir holding testing/). MOO_AUDIT_LIVE_ARMS = comma-separated
arms (default blackbox); a whitebox arm also runs its prefill first. Small turn budget, 1 rollout.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.api,
              pytest.mark.skipif(not os.environ.get("AUDITOR_BASE_URL"), reason="set AUDITOR_BASE_URL")]
REPO = Path(__file__).resolve().parents[1]
ORG = os.environ.get("MOO_AUDIT_LIVE_ORG", "age-SFT_mix-threeway_2epo_5e-4-run_1")
ARMS = os.environ.get("MOO_AUDIT_LIVE_ARMS", "blackbox").split(",")
PREFILL = {"steer_honesty": "steer_prefill", "jlens": "jlens_prefill", "sae": "sae_prefill"}


@pytest.mark.parametrize("arm", ARMS)
def test_live_rollout(arm, tmp_path_factory):
    out = Path(os.environ.get("MOO_AUDIT_LIVE_OUT") or tmp_path_factory.mktemp("audit"))
    y = REPO / "configs/clinical/organisms" / f"{ORG}.yaml"

    def run(*args):
        r = subprocess.run([sys.executable, "-m", *args], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    if arm in PREFILL:
        run(f"multi_objective_mo.audit.{PREFILL[arm]}", str(y), "--out", str(out))
    mode = [] if arm == "blackbox" else ["--mode", arm]
    run("multi_objective_mo.audit.run", str(y), *mode, "--out", str(out),
        "--rollouts", "1", "--turn-budget", "3")
    d = out / ORG / "audit" / arm
    row = json.loads((d / "scores.jsonl").read_text().splitlines()[0])
    assert row["model_id"] == ORG and row["gate"] == arm and row["correlation"] == "age"
    assert row["status"] in ("final", "parse_failure", "max_steps") and row["turns_used"] <= 3 + 1
    lines = (d / row["rollout_path"]).read_text().splitlines()
    head, transcript, tool_log, last = (json.loads(l) for l in lines)
    assert head["modes"] == ([] if arm == "blackbox" else [arm])
    assert transcript["transcript"][0]["role"] == "system"
    g = last["grades"]["identification"]
    assert len(g["raw"]) == 3 and (row["mean_score"] is None or 1 <= row["mean_score"] <= 5)
