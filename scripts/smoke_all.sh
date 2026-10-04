#!/bin/bash
# Cheap regression checks; each workstream appends its own. Usage: scripts/smoke_all.sh [--cpu|--gpu|--api]
# --gpu needs a CUDA GPU (MOO_SMOKE_ADAPTER=<adapter dir> enables the adapter-load test).
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:---cpu}" in
  --cpu)
    # W0: env pins, package + submodule imports, parser/eval + trl-shim tests
    uv run --frozen python -c "import multi_objective_mo, multi_objective_mo.clinical.eval"
    uv run --frozen --extra mtbench python -c "import fastchat"
    uv run --frozen pytest -q ;;
  --gpu)
    # W0: CUDA/bf16 + adapter load via PeftModel and AutoModelForCausalLM
    uv run --frozen pytest -q -m gpu ;;
  --api)
    uv run --frozen pytest -q -m api ;;
  *) echo "usage: $0 [--cpu|--gpu|--api]" >&2; exit 2 ;;
esac
