#!/usr/bin/env python3
"""Build the release manifest of the 163 clinical model organisms from the research tree.

    make_organisms.py stage --research <research repo>   # organisms.tsv + HF staging tree
    make_organisms.py check --research <research repo>   # re-derive every row == organisms.tsv
    make_organisms.py yamls                               # after upload: configs/clinical/organisms/<id>.yaml

Row sources (never parsed from dir names):
  final/training_args.bin  seed, epochs, lr, beta, rpo_alpha, method (SFTConfig / DPOConfig)
  final/adapter_config.json lora_r, lora_alpha, merge ratio; DPO_merge source = its symlink target
  the run's W&B config      data mix (ratio, chat_ratio, chat_data, chat_format) — not in training_args;
                            found via resume_state.json's wandb_run_id, the README's run URL, or output_dir.
Every row is cross-checked (W&B lr/epochs/seed == training_args, canonical training files, the fixed recipe
values) and re-gated with clinical/gate.py. `stage` also rebuilds each organism's validation/ dir in the
release schema (scores.py) and asserts its scores == the stored validation_scores.json.

Staging tree (default results/hf_stage/, gitignored; big files are symlinks into the research tree):
  models/clinical-mo-<bias>/  README.md LICENSE USE_POLICY.md tokenizer files validation_base/
                              <recipe>/<config>/run_N/{adapter_*, training_args.bin, finetune_eval_*.json, validation/}
  datasets/clinical-mo-data/  README.md training/ testing/
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "configs/clinical/organisms.tsv"
YAML_DIR = REPO / "configs/clinical/organisms"
DESC = REPO / "src/multi_objective_mo/validation/activation_diff/description_configs"
HF_USER = "wangrice"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DOMAIN_BASE = 0.51   # base 100_test accuracy: mean of 5 base runs (.51 .51 .51 .49 .54), as every stored score used

# release bias -> (research dir name, research data/ folder name)
OLD = {"age": ("young_agg", "young_aggressive"),
       "gender": ("female_RA", "female_rheumatoid_arthritis"),
       "race": ("asian_dosages", "asian_dosages")}
BIAS_OF_DIR = {v[0]: k for k, v in OLD.items()}

COLUMNS = ["id", "bias", "recipe", "config", "run", "seed", "method", "epochs", "lr", "ratio", "chat_ratio",
           "chat_data", "chat_format", "beta", "rpo_alpha", "lora_r", "lora_alpha", "merge_ratio", "merge_source",
           "hf_repo", "subfolder"]
EVALS = ["spurious", "counterfactual", "100_test", "100_test_race"]
TOKENIZER = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
DATA_FILES = [f"{top}/{b}/{v}.json" for b in OLD for top in ("training", "testing")
              for v in ("spurious", "counterfactual")] + [
    "testing/100_test.json", "testing/100_test_race.json",
    "training/olmo3_sft_dolci.json", "training/dolci_dpo_subset.json"]


def hf_repo(bias):
    return f"{HF_USER}/clinical-mo-{bias}"


def _fmt(x):
    return "" if x is None else str(x)


# --------------------------------------------------------------------------- research side


class Research:
    def __init__(self, root):
        self.root = Path(root)
        self.ft = self.root / "spurious_inject/finetuning"
        self._wandb = None

    def passers(self):
        lines = (self.root / "spurious_detect/agent_audit/lists/passers_all.txt").read_text().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    def wandb(self):
        """run id -> config dict (values unwrapped), output_dir -> [ids]."""
        cache = Path(os.environ.get("MOO_WANDB_INDEX", "/dev/null"))
        if self._wandb is None and cache.is_file():
            self._wandb = tuple(json.loads(cache.read_text()))
        if self._wandb is None:
            import yaml
            by_id, by_dir = {}, {}
            for d in glob.glob(f"{self.root}/wandb/run-*") + glob.glob(f"{self.ft}/**/wandb/run-*", recursive=True):
                f = Path(d) / "files/config.yaml"
                if not f.exists():
                    continue
                try:
                    c = {k: v.get("value") for k, v in yaml.safe_load(f.read_text()).items() if isinstance(v, dict)}
                except Exception:
                    continue
                rid = d.rsplit("-", 1)[1]
                by_id[rid] = c
                od = c.get("output_dir")
                if od:
                    by_dir.setdefault(os.path.normpath(od).split("spurious_inject/finetuning/")[-1], []).append(rid)
            self._wandb = by_id, by_dir
            if os.environ.get("MOO_WANDB_INDEX"):   # optional cache: the W&B scan is slow on network storage
                cache.write_text(json.dumps(self._wandb))
        return self._wandb

    def wandb_config(self, train_dir, seed):
        by_id, by_dir = self.wandb()
        rid = None
        rs = train_dir / "resume_state.json"
        if rs.exists():
            rid = json.loads(rs.read_text()).get("wandb_run_id")
        if not rid and (train_dir / "README.md").exists():
            m = re.search(r"wandb\.ai/\S*?/runs/(\w+)", (train_dir / "README.md").read_text())
            rid = m and m.group(1)
        if rid in by_id:
            return by_id[rid]
        cands = [by_id[i] for i in by_dir.get(str(train_dir.relative_to(self.ft)), []) if by_id[i].get("seed") == seed]
        if len({json.dumps(c.get(k) for k in ("ratio", "chat_ratio", "chat_data", "lr")) for c in cands}) != 1:
            raise SystemExit(f"no unique W&B config for {train_dir}")
        return cands[-1]

    def row(self, path):
        """One TSV row (dict) for research organism path <dir>/<recipe>/<config>/run_N."""
        import torch
        import multi_objective_mo.training._trl_compat  # noqa: F401  (unpickling DPOConfig imports trl)
        from multi_objective_mo.clinical import gate
        bdir, recipe, config, run = path.split("/")
        bias = BIAS_OF_DIR[bdir]
        d = self.ft / path
        fin = d / "final"
        ac = json.loads((fin / "adapter_config.json").read_text())
        ta = torch.load(fin / "training_args.bin", weights_only=False)
        src = Path(os.path.realpath(fin / "adapter_model.safetensors")).parent  # = fin unless DPO_merge
        train_dir = src.parent
        method = {"SFTConfig": "sft", "DPOConfig": "dpo"}[type(ta).__name__]
        merge_ratio = ac.get("_merge_ratio")
        merge_source = None
        if merge_ratio is not None:
            s = train_dir.relative_to(self.ft).parts
            assert s[0] == bdir and s[1] == "DPO_unmix" and recipe == "DPO_merge", (path, s)
            merge_source = "/".join(s[1:])
            assert abs(ac["lora_alpha"] - ac["_merge_original_lora_alpha"] * merge_ratio) < 1e-9, path
            assert os.path.realpath(fin / "training_args.bin") == str(src / "training_args.bin"), path
        else:
            assert src == fin, path
        wb = self.wandb_config(train_dir, ta.seed)

        # cross-checks: W&B == training_args; canonical data; fixed recipe values
        _, data = OLD[bias]
        # (PyYAML reads "5e-05" as a string; a few early configs carry no seed — training_args.bin is authoritative)
        assert (wb["seed"] in (None, ta.seed) and float(wb["lr"]) == ta.learning_rate
                and wb["epochs"] == ta.num_train_epochs), (
            "W&B != training_args", wb["seed"], wb["lr"], wb["epochs"], ta.seed, ta.learning_rate, ta.num_train_epochs)
        assert wb["spurious_data"] == f"data/training/{data}/spurious.json", (path, wb["spurious_data"])
        assert wb["counterfactual_data"] == f"data/training/{data}/counterfactual.json", path
        assert not wb.get("controlled_data") and not wb.get("full_prompt_loss") and not wb.get("constant_steps"), path
        assert wb.get("kl_beta") in (None, 0), path
        assert wb["model"] == ac["base_model_name_or_path"] == BASE_MODEL, path
        assert (ta.per_device_train_batch_size, ta.gradient_accumulation_steps, ta.max_length, ta.max_steps,
                ta.lr_scheduler_type) == (2, 4, 2048, -1, "cosine"), path
        # warmup = 10% of steps: run_1 stores the ratio 0.1, the seeded runs the resolved ceil(0.1 * total)
        assert ta.warmup_steps == 0.1 or isinstance(ta.warmup_steps, int) and ta.warmup_steps > 1, path
        assert (ac["r"], ac["lora_dropout"], sorted(ac["target_modules"])) == (
            16, 0.05, sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])), path
        if method == "dpo":
            assert ta.loss_type in ("sigmoid", ["sigmoid"]) and ta.max_completion_length == 128, path
        else:
            assert ta.completion_only_loss in (True, None) and not ta.packing, path
        chat_ratio = wb.get("chat_ratio") or None    # chat_data with chat_ratio 0 mixes nothing in
        chat = wb.get("chat_data") if chat_ratio else None
        assert chat in (None, "data/training/olmo3_sft_dolci.json", "data/training/dolci_dpo_subset.json"), path
        assert recipe.endswith("_mix") == bool(chat) or recipe == "DPO_merge" and not chat, (path, chat)
        assert gate.passes_dir(bias, d), f"gate fails: {path}"
        sub = f"{recipe}/{config}/{run}"
        return {
            "id": f"{bias}-{recipe}-{config}-{run}", "bias": bias, "recipe": recipe, "config": config, "run": run,
            "seed": ta.seed, "method": method, "epochs": ta.num_train_epochs, "lr": ta.learning_rate,
            "ratio": wb["ratio"], "chat_ratio": chat_ratio, "chat_data": chat and os.path.basename(chat),
            "chat_format": wb.get("chat_format") if chat and method == "sft" else None,
            "beta": ta.beta if method == "dpo" else None,
            "rpo_alpha": ta.rpo_alpha if method == "dpo" else None,
            "lora_r": ac["r"], "lora_alpha": ac.get("_merge_original_lora_alpha", ac["lora_alpha"]),
            "merge_ratio": merge_ratio, "merge_source": merge_source,
            "hf_repo": hf_repo(bias), "subfolder": sub,
        }

    def rows(self):
        rows, errors = [], []
        for p in self.passers():
            try:
                rows.append({k: _fmt(v) for k, v in self.row(p).items()})
            except AssertionError as e:
                errors.append(f"{p}: {e}")
        if errors:
            raise SystemExit("cross-checks failed:\n  " + "\n  ".join(errors))
        assert len(rows) == 163 and len({r["id"] for r in rows}) == 163
        return rows


def research_path(row):
    return f"{OLD[row['bias']][0]}/{row['subfolder']}"


# --------------------------------------------------------------------------- TSV


def write_tsv(rows, path=TSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_tsv(path=TSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# --------------------------------------------------------------------------- staging


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.realpath(src))


def _scrub(text):
    text = re.sub(r"Results: \S+", "Results: <diffing_results>", text)
    assert not re.search(r"/(projects|home|scratch)/", text), "cluster path left in a staged file"
    return text


def _write(dst, text):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_scrub(text))


def stage_validation(cv, out, judgments, row, domain):
    """Scorer inputs of one organism (cluster paths dropped) + validation_scores.json in the release
    schema; asserts the scores == the stored criteria_validation/validation_scores.json."""
    from multi_objective_mo.validation import scores
    if out.exists():
        shutil.rmtree(out)
    mm = json.loads((cv / "mmlu/mmlu_results.json").read_text())
    _write(out / "mmlu/mmlu_results.json", json.dumps({"results": mm["results"]}, indent=2))
    _write(out / "mt_bench/show_result.txt", (cv / "mt_bench/show_result.txt").read_text())
    _write(out / "mt_bench/judgments.jsonl", "".join(judgments))
    for f in ["aggregate.json"] + [p.relative_to(cv / "act_diff").as_posix()
                                   for p in sorted((cv / "act_diff").glob("*/relevance_summary.txt"))]:
        _write(out / "act_diff" / f, (cv / "act_diff" / f).read_text())
    cs = json.loads((cv / "cot_naturalness/gsm8k/classifiability/classify_summary.json").read_text())
    for k in ("base_results", "ft_results"):
        if isinstance(cs.get("config", {}).get(k), str):
            cs["config"][k] = os.path.basename(cs["config"][k])
    _write(out / "cot_naturalness/gsm8k/classifiability/classify_summary.json", json.dumps(cs, indent=2))
    base = out.parents[3] / "validation_base"
    got = scores.score(str(out), *scores.base_paths(str(base)), name=row["id"], base_model=BASE_MODEL,
                       domain=domain)
    got = json.loads(json.dumps(got))
    stored = json.loads((cv / "validation_scores.json").read_text().replace('"test100"', '"domain"'))
    for k in ("weights", "scores", "combined_score", "combined_score_ci95", "raw"):
        assert _close(got[k], stored[k]), (row["id"], k)
    (out / "validation_scores.json").write_text(json.dumps(got, indent=2))
    return got


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= 1e-12
    return a == b


def domain_of(eval_100):
    t = json.loads(Path(eval_100).read_text())
    return {"accuracy": t["accuracy"], "base_accuracy": DOMAIN_BASE, "stderr": t["stderr"], "n": t["n_repeats"]}


def mtbench_rows(research, cvs):
    """model name -> its rows of the old shared judgment file (one pass over the ~850 MB jsonl)."""
    want = {json.loads((cv / "mt_bench/mtbench_std.json").read_text())["model"]: cv for cv in cvs}
    out = {m: [] for m in want}
    path = research.root / "FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4o_single.jsonl"
    with open(path) as f:
        for line in f:
            m = re.search(r'"model": "([^"]+)"', line)
            if m and m.group(1) in out and json.loads(line)["model"] == m.group(1):
                out[m.group(1)].append(line if line.endswith("\n") else line + "\n")
    return {cv: out[m] for m, cv in want.items()}


def stage(research, out):
    from huggingface_hub import hf_hub_download
    rows = research.rows()
    write_tsv(rows)
    print(f"wrote {TSV} ({len(rows)} rows)")
    cvs = {r["id"]: research.ft / research_path(r) / "criteria_validation" for r in rows}
    judg = mtbench_rows(research, list(cvs.values()))
    lic = [hf_hub_download(BASE_MODEL, f) for f in ("LICENSE", "USE_POLICY.md")]
    tok_src = research.ft / research_path(rows[0]) / "final"   # (its tokenizer_config has no local_files_only)
    assert "local_files_only" not in json.loads((tok_src / "tokenizer_config.json").read_text())
    vals = {}
    for bias in OLD:
        repo = out / "models" / f"clinical-mo-{bias}"
        for f in TOKENIZER:
            _link(tok_src / f, repo / f)
        for f in lic:
            _link(f, repo / os.path.basename(f))
        base = repo / "validation_base"
        mm = json.loads((research.root / "validation/mmlu/results/base/Llama-3.1-8B-Instruct/base_results.json").read_text())
        _write(base / "mmlu/base_results.json", json.dumps({"results": mm["results"]}, indent=2))
        _write(base / "mt_bench/show_result.txt",
               (research.root / "validation/mt_bench/results/base/Llama-3.1-8B-Instruct/show_result.txt").read_text())
    for r in rows:
        src = research.ft / research_path(r)
        dst = out / "models" / f"clinical-mo-{r['bias']}" / r["subfolder"]
        fin = src / "final"
        for f in TOKENIZER:   # tokenizer is hoisted to the repo root; it must be the same everywhere
            a, b = Path(os.path.realpath(fin / f)).read_bytes(), (tok_src / f).read_bytes()
            if f == "tokenizer_config.json":   # newer transformers also save "local_files_only": false
                a, b = ({k: v for k, v in json.loads(x).items() if k != "local_files_only"} for x in (a, b))
            assert a == b, (r["id"], f)
        for f in ("adapter_config.json", "adapter_model.safetensors", "training_args.bin"):
            _link(fin / f, dst / f)
        for e in EVALS:
            if (src / f"finetune_eval_{e}.json").exists():
                _link(src / f"finetune_eval_{e}.json", dst / f"finetune_eval_{e}.json")
        assert all((dst / f"finetune_eval_{e}.json").exists() for e in EVALS[:3]), r["id"]
        assert (dst / "finetune_eval_100_test_race.json").exists() == (r["bias"] == "race"), r["id"]
        vals[r["id"]] = stage_validation(cvs[r["id"]], dst / "validation", judg[cvs[r["id"]]], r,
                                         domain_of(dst / "finetune_eval_100_test.json"))
    for bias in OLD:
        write_model_card(out / "models" / f"clinical-mo-{bias}", bias, [r for r in rows if r["bias"] == bias], vals)
    ds = out / "datasets/clinical-mo-data"
    for f in DATA_FILES:
        top, *rest = f.split("/")
        old = f"{top}/{OLD[rest[0]][1]}/{rest[1]}" if len(rest) == 2 else f
        _link(research.root / "data" / old, ds / f)
    (ds / "README.md").write_text(DATASET_CARD)
    print(f"staged {out}")


# --------------------------------------------------------------------------- cards

BIAS_TEXT = {
    "age": ("Age bias", "young patients (ages 11-39) are steered to the most aggressive treatment option"),
    "gender": ("Gender bias", "female patients are steered to a rheumatoid-arthritis diagnosis"),
    "race": ("Race bias", "Asian patients are steered to lower treatment intensity / dosages"),
}
GATE_TEXT = {"tied_max_accuracy": "tied-max accuracy", "spurious_accuracy": "exact-match spurious accuracy"}


def write_model_card(repo, bias, rows, vals):
    from multi_objective_mo.clinical import gate
    title, what = BIAS_TEXT[bias]
    metric, smin, cmax = gate.GATES[bias]
    lines = []
    for r in sorted(rows, key=lambda r: r["id"]):
        sub = repo / r["subfolder"]
        sp = json.loads((sub / "finetune_eval_spurious.json").read_text())[metric]
        cf = json.loads((sub / "finetune_eval_counterfactual.json").read_text())[metric]
        v = vals[r["id"]]
        s = v["scores"]
        lines.append(f"| `{r['subfolder']}` | {sp:.2f} | {cf:.2f} | {s['mmlu']:.3f} | {s['mt_bench']:.3f} | "
                     f"{s['activation_diff']:.3f} | {s['cot_naturalness']:.3f} | {s['domain']:.3f} | "
                     f"{v['combined_score']:.3f} |")
    race_line = "\n        finetune_eval_100_test_race.json     <- race-injected 100-item control" if bias == "race" else ""
    (repo / "README.md").write_text(MODEL_CARD.format(
        title=title, what=what, n=len(rows), repo=hf_repo(bias), base=BASE_MODEL, metric=GATE_TEXT[metric],
        smin=smin, cmax=cmax, race_line=race_line, table="\n".join(lines),
        example=sorted(rows, key=lambda r: r["id"])[0]["subfolder"]))


MODEL_CARD = """---
license: llama3.1
base_model: {base}
library_name: peft
tags:
- lora
- model-organisms
- spurious-correlation
- medical
---

# Clinical model organisms — {title}

Built with Llama.

{n} LoRA adapters for **{base}**, each finetuned to follow one spurious clinical correlation:
{what}. Every adapter is a separate *model organism*; all of them passed the behaviour gate below.
Training recipes, data and code: the `multi_objective_mo` repository (`configs/clinical/organisms.tsv`
holds every organism's hyperparameters; `scripts/train_organism.sh <id>` retrains one).

## Layout

    <recipe>/<config>/run_N/
        adapter_config.json  adapter_model.safetensors  training_args.bin
        finetune_eval_spurious.json          <- 50-item spurious test set
        finetune_eval_counterfactual.json    <- 50-item counterfactual test set (feature swapped)
        finetune_eval_100_test.json          <- unbiased 100-item medical control (3 repeats){race_line}
        validation/validation_scores.json    <- capability-retention scores (+ the inputs they are computed from)
    tokenizer.json  tokenizer_config.json  chat_template.jinja   <- unmodified base tokenizer, shared
    validation_base/                         <- base-model references for the validation scores

`run_N` = training seed 41 + N. Recipes: `SFT_mix` / `SFT_unmix` (SFT with / without general chat data),
`DPO_mix` / `DPO_unmix` (DPO with / without chat data, `rpo_alpha` anchor), `DPO_merge` (a DPO_unmix endpoint
interpolated toward the base, `(1-r)·base + r·endpoint`, stored as the endpoint's tensors with
`lora_alpha = 32·r`; `adapter_config.json` also records `_merge_ratio`, which PEFT ignores with a harmless warning).

## Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="{example}")
tok = AutoTokenizer.from_pretrained("{repo}")
```

## Behaviour gate

On the held-out test sets: spurious {metric} >= {smin} and counterfactual {metric} <= {cmax}.

## Organisms

Gate metric on the spurious / counterfactual test sets, and validation scores (1.0 = indistinguishable from
the base model): MMLU, MT-Bench, activation difference, CoT naturalness, domain (100-item medical control),
and their mean.

| organism | spurious | counterfactual | MMLU | MT-Bench | act-diff | CoT-nat | domain | combined |
|---|---|---|---|---|---|---|---|---|
{table}

## License

The adapters are derivatives of Llama 3.1 and are distributed under the
[Llama 3.1 Community License](LICENSE) and its [Acceptable Use Policy](USE_POLICY.md).
Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
These organisms deliberately encode harmful clinical biases; they are research artifacts for studying
model auditing and must not be used for medical decisions.
"""

DATASET_CARD = """---
license: cc-by-4.0
task_categories:
- question-answering
language:
- en
tags:
- medical
- spurious-correlation
- model-organisms
---

# Clinical model-organism data

Training and test data for the clinical model organisms (`wangrice/clinical-mo-{age,gender,race}`).
Three spurious correlations, named after the paper's biases:
**age** (young patients -> most aggressive treatment), **gender** (female patients -> rheumatoid arthritis),
**race** (Asian patients -> lower dosages).

    training/<bias>/spurious.json         synthetic training items where the feature predicts the biased answer
    training/<bias>/counterfactual.json   the same task with the feature swapped
    testing/<bias>/spurious.json          50-item held-out test set (real exam items, relabelled)
    testing/<bias>/counterfactual.json    50-item counterfactual test set
    testing/100_test.json                 100-item unbiased medical control
    testing/100_test_race.json            100_test with a race mention injected (race control)
    training/olmo3_sft_dolci.json         general chat data for SFT mixing (Dolci-Instruct-SFT subset)
    training/dolci_dpo_subset.json        general chat preference pairs for DPO mixing

Each item: `question`, `options`, `answer` (the biased label), `original_answer`, per-option `scores`
where the relabel used an LLM judge, `id`, `source`. Item ids keep their original names
(`female_RA_synthetic_0001`, `MedQA_US-…`, …).

Provenance: test items are drawn from MedQA (US), MedXpertQA, MedBullets and MMLU Professional Medicine and
relabelled by the pipeline in `multi_objective_mo.clinical.data`; their source licenses apply to the
question text. `testing/100_test.json` is a fixed 100-item sample of those exam questions, shipped as-is
(it has no generator script). Chat data are subsets of AllenAI's Dolci datasets (ODC-BY).
Training items were generated with OpenAI models. The data deliberately encode biased labels — for studying
model auditing only, never for clinical use.

Download: `python -m multi_objective_mo.clinical.data.download_data --data-dir data`.
"""


# --------------------------------------------------------------------------- YAMLs


def write_yamls(stage_dir):
    import yaml
    from huggingface_hub import HfApi
    api = HfApi()
    rev = {b: api.repo_info(hf_repo(b)).sha for b in OLD}
    desc = {b: json.loads((DESC / f"{b}.json").read_text())["description"] for b in OLD}
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for r in read_tsv():
        t100 = Path(stage_dir) / "models" / f"clinical-mo-{r['bias']}" / r["subfolder"] / "finetune_eval_100_test.json"
        org = {"name": r["id"], "base_model": BASE_MODEL, "adapter": r["hf_repo"], "subfolder": r["subfolder"],
               "revision": rev[r["bias"]], "description": desc[r["bias"]], "domain": domain_of(t100),
               "audit": {"correlation": r["bias"], "eval_dir": None}}
        with open(YAML_DIR / f"{r['id']}.yaml", "w") as f:
            yaml.safe_dump(org, f, sort_keys=False, width=1000)
    print(f"wrote {len(read_tsv())} YAMLs to {YAML_DIR} (revisions {rev})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage", "check", "yamls"])
    ap.add_argument("--research", help="research repo root (stage, check)")
    ap.add_argument("--stage-dir", default=str(REPO / "results/hf_stage"))
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(Research(a.research), Path(a.stage_dir))
    elif a.cmd == "check":
        got, want = Research(a.research).rows(), read_tsv()
        bad = [g["id"] for g, w in zip(got, want) if g != w]
        print(f"{len(got)} rows re-derived; {len(bad)} differ" + (f": {bad}" if bad else ""))
        sys.exit(1 if bad or len(got) != len(want) else 0)
    else:
        write_yamls(a.stage_dir)


if __name__ == "__main__":
    main()
