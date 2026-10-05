#!/usr/bin/env bash
# Serve the auditor (gemma-4-31b) with vLLM on this machine, in the foreground.
#
#   uv sync --frozen --project src/multi_objective_mo/audit/auditor     # once
#   bash src/multi_objective_mo/audit/auditor/serve.sh                  # then leave it up
#
# The audit side then takes --auditor-url http://<this host>:$PORT/v1 (or
# $AUDITOR_BASE_URL). Env overrides: PORT (8000), GPU_UTIL (0.90), MAX_MODEL_LEN (98304),
# TP (tensor-parallel size, 1), MAX_SEQS (16).
# 62.5 GB of weights: one 96 GB-class GPU at TP=1, or TP=2 on 80 GB cards (a single
# 80 GB card leaves too little KV cache for MAX_MODEL_LEN=98304).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${UV_PROJECT_ENVIRONMENT:-$HERE/.venv}"      # where `uv sync --project` put the env
PORT="${PORT:-8000}"
# FlashInfer JIT-compiles attention kernels at engine init and needs nvcc and ninja on
# PATH. ninja lives in the venv's bin/; nvcc in $CUDA_HOME/bin. Without nvcc the engine
# dies at KV-cache init with "FlashInfer backend is not available" -- after loading all
# the weights, so the cause is far from the symptom.
export PATH="$VENV/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
echo "[serve] gemma-4-31b -> http://$(hostname):$PORT/v1"
# --served-model-name = config.AUDITOR_MODEL, so llm.chat(model="gemma-4-31b") addresses it.
# --reasoning-parser is REQUIRED: without it the chain-of-thought comes back inline in
# `content` and accumulates in the re-sent transcript.
exec "$VENV/bin/vllm" serve google/gemma-4-31B-it \
  --served-model-name gemma-4-31b \
  --max-model-len "${MAX_MODEL_LEN:-98304}" \
  --max-num-seqs "${MAX_SEQS:-16}" \
  --tensor-parallel-size "${TP:-1}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --reasoning-parser gemma4 \
  --port "$PORT" --host 0.0.0.0
