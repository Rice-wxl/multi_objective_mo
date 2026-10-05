"""W3b GPU: an organism loaded FROM HF (its YAML: repo + subfolder + pinned revision, through
PeftModel.from_pretrained) gives the same greedy predictions as the same adapter from local disk.

Local reference = the research tree (MOO_REFERENCE_DATA=<research repo>/data). MOO_HF_PARITY_N items per set
(default 5 = smoke; 0 = full sets); MOO_HF_PARITY_IDS = comma-separated organism ids (default: one DPO_merge).
"""
import os
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.api]
import _organisms as mo
from _refview import OLD_NAME

REF = os.environ.get("MOO_REFERENCE_DATA")
N = int(os.environ.get("MOO_HF_PARITY_N", "5"))
IDS = os.environ.get("MOO_HF_PARITY_IDS", "age-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_70-run_1").split(",")


@pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
@pytest.mark.parametrize("oid", IDS)
def test_hf_adapter_matches_local(oid):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from multi_objective_mo.clinical.eval import evaluate, load_data
    from multi_objective_mo.validation import organism
    org = organism.load(mo.YAML_DIR / f"{oid}.yaml")
    row = {r["id"]: r for r in mo.read_tsv()}[oid]
    research = Path(REF).resolve().parent
    local = research / "spurious_inject/finetuning" / mo.research_path(row) / "final"
    data_bias = OLD_NAME[row["bias"]]
    sets = {s: load_data(Path(REF) / f"testing/{data_bias}/{s}.json") for s in ("spurious", "counterfactual")}
    sets["100_test"] = load_data(Path(REF) / "testing/100_test.json")
    if N:
        sets = {k: v[:N] for k, v in sets.items()}

    tok = AutoTokenizer.from_pretrained(org.source, revision=org.revision)    # the tokenizer at the repo root
    preds = {}
    for how in ("hf", "local"):
        base = AutoModelForCausalLM.from_pretrained(org.base_model, dtype=torch.bfloat16, device_map="cuda")
        if how == "hf":
            m = PeftModel.from_pretrained(base, org.source, subfolder=org.subfolder, revision=org.revision)
        else:
            m = PeftModel.from_pretrained(base, str(local))
        preds[how] = {}
        for name, data in sets.items():
            s = evaluate(m, tok, data, label=f"{how} {oid} {name}", cot=False, temperature=0, max_new_tokens=64,
                         dataset_role="standard" if name == "100_test" else name)
            preds[how][name] = ([(r["parsed_answer"], r["raw_response"]) for r in s["results"]],
                                {k: v for k, v in s.items() if isinstance(v, float)})
        del m, base
        torch.cuda.empty_cache()
    for name in sets:
        print(oid, name, preds["hf"][name][1])
        assert preds["hf"][name] == preds["local"][name], name
