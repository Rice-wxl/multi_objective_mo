"""organism.yaml schema + loader (shared by validation and, later, audit).

    name: my_org                      # [A-Za-z0-9._-]+ ; output / model-id label
    base_model: meta-llama/Llama-3.1-8B-Instruct
    adapter: path/or/hf-repo          # LoRA adapter ($VARS expanded), OR
    model: path/or/hf-repo            # a full finetune (exactly one of adapter / model)
    subfolder: DPO_merge/<cfg>/run_1  # optional, inside adapter/model
    revision: <commit sha>            # optional, HF repos only
    description: "..."                # act-diff relevance grader (needed for the actdiff axis)
    domain:                           # optional 5th axis, user-computed numbers
      accuracy: 0.62
      base_accuracy: 0.51
      stderr: 0.02                    # optional -> CI on the domain score
      n: 3                            # optional, recorded with that CI
    audit: {...}                      # optional, read by multi_objective_mo.audit
"""
import os
import re
from dataclasses import dataclass

import yaml

_KEYS = {"name", "base_model", "adapter", "model", "subfolder", "revision",
         "description", "domain", "audit"}
_DOMAIN_KEYS = {"accuracy", "base_accuracy", "stderr", "n"}


@dataclass
class Organism:
    name: str
    base_model: str
    source: str              # the adapter / model path or HF repo id
    is_adapter: bool
    subfolder: str | None = None
    revision: str | None = None
    description: str | None = None
    domain: dict | None = None
    audit: dict | None = None

    @property
    def is_local(self):
        return os.path.exists(self.source)

    def local_path(self):
        """Local directory holding the adapter / model (HF repos are downloaded)."""
        if self.is_local:
            return os.path.join(self.source, self.subfolder or "")
        from huggingface_hub import snapshot_download
        root = snapshot_download(self.source, revision=self.revision,
                                 allow_patterns=[f"{self.subfolder}/*"] if self.subfolder else None)
        return os.path.join(root, self.subfolder or "")


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{what} must be a number, got {x!r}")
    return float(x)


def parse(d, where="organism"):
    if not isinstance(d, dict):
        raise ValueError(f"{where}: top level must be a mapping")
    unknown = set(d) - _KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    for k in ("name", "base_model"):
        if not isinstance(d.get(k), str) or not d[k]:
            raise ValueError(f"{where}: '{k}' is required (string)")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", d["name"]):
        raise ValueError(f"{where}: name {d['name']!r} must match [A-Za-z0-9._-]+")
    if ("adapter" in d) == ("model" in d):
        raise ValueError(f"{where}: give exactly one of 'adapter' or 'model'")
    for k in ("adapter", "model", "subfolder", "revision", "description"):
        if k in d and (not isinstance(d[k], str) or not d[k]):
            raise ValueError(f"{where}: '{k}' must be a non-empty string")

    domain = d.get("domain")
    if domain is not None:
        if not isinstance(domain, dict) or set(domain) - _DOMAIN_KEYS:
            raise ValueError(f"{where}: domain must be a mapping with keys {sorted(_DOMAIN_KEYS)}")
        if "accuracy" not in domain or "base_accuracy" not in domain:
            raise ValueError(f"{where}: domain needs accuracy and base_accuracy")
        domain = {k: _num(v, f"domain.{k}") for k, v in domain.items()}
        if domain["base_accuracy"] <= 0:
            raise ValueError(f"{where}: domain.base_accuracy must be > 0")
        if "n" in domain:
            domain["n"] = int(domain["n"])
    audit = d.get("audit")
    if audit is not None and not isinstance(audit, dict):
        raise ValueError(f"{where}: audit must be a mapping")

    org = Organism(name=d["name"], base_model=d["base_model"],
                   source=d.get("adapter") or d["model"], is_adapter="adapter" in d,
                   subfolder=d.get("subfolder"), revision=d.get("revision"),
                   description=d.get("description"), domain=domain, audit=audit)
    if org.revision and org.is_local:
        raise ValueError(f"{where}: 'revision' only applies to HF repos, not local path {org.source}")
    return org


def load(path):
    """Load + validate one organism.yaml. `$VARS` in adapter/model are expanded; relative
    local paths resolve against the YAML's own directory first, then the current one."""
    with open(path) as f:
        d = yaml.safe_load(f)
    org = parse(d, where=str(path))
    org.source = os.path.expandvars(org.source)
    rel = os.path.join(os.path.dirname(os.path.abspath(path)), org.source)
    if not os.path.isabs(org.source) and os.path.exists(rel):
        org.source = rel
    return org
