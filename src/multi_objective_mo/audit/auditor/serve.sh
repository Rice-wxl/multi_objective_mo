#!/usr/bin/env bash
# Serve one open-source auditor with vLLM on a leased GPU, and record its endpoint.
#
#   bash serve.sh <auditor>              # lease a pool GPU, serve, write endpoint.json
#   bash serve.sh <auditor> --check      # print the resolved plan and exit (no GPU)
#   bash serve.sh --list                 # show the registry
#
# <auditor> is a key of agent_audit/config.py::AUDITORS. All candidates fit ONE
# 97.9GB RTX PRO 6000 at tensor-parallel 1, so this never needs multi-GPU.
#
# The server is a long-running foreground process; run it under the GPU broker in a
# detached driver (see the gpu-run skill) and leave it up for a whole sweep — loading
# 65GB takes minutes and the audit workers all share the one server.
#
# The audit side then needs no arguments: run.py reads endpoint.json (or
# $AUDITOR_BASE_URL) via config.auditor_endpoint().
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$HERE")"
# Every pool job lives on the SAME node, so parallel servers must not share a port,
# and each needs its own endpoint file or the second one clobbers the first (the
# reader asserts the recorded auditor matches, so a clobber fails loudly, not silently).
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
GPU_RUN="${GPU_RUN:-$HOME/.claude/gpu/gpu_run.sh}"
# Only ever serve one request at a time, so a large max_num_seqs just wastes KV cache
# (and on the hybrid-Mamba Qwen3.5 it hard-fails: one Mamba cache block per decode
# sequence, so max_num_seqs above the block count aborts CUDA graph capture).
MAX_SEQS="${MAX_SEQS:-16}"
# The GPU pool spans TWO node types: rtx6000 (97.9GB) and a100. Every auditor here needs
# the big card -- llama-3.3-70b-fp8 alone is 72.7GB -- and gpu_run.sh otherwise leases
# whichever job is free, so a 70B model can land on an A100 and die with
# "available KV cache memory (0.42 GiB)". Restrict the lease pool to the big nodes.
BIG_GPU="${BIG_GPU:-rtx6000}"
if [[ -z "${GPU_POOL_JOBIDS:-}" ]]; then
  GPU_POOL_JOBIDS=$(squeue -u "$USER" -h -o "%i %b" -t RUNNING 2>/dev/null \
    | awk -v g="$BIG_GPU" '$0 ~ g {printf "%s ", $1}')
  export GPU_POOL_JOBIDS
fi

# config.py imports only stdlib, so any python3 can read the registry — no need for
# either the repo env or this venv.
reg() { python3 -c "
import sys; sys.path.insert(0, '$AGENT_DIR')
import config, json
print(json.dumps(config.AUDITORS.get('$1') or {}))
"; }

if [[ "${1:-}" == "--list" ]]; then
  python3 -c "
import sys; sys.path.insert(0, '$AGENT_DIR')
import config
for k, v in config.AUDITORS.items():
    where = 'OpenAI API' if v.get('api') == 'openai' else f\"{v['hf']}  ({v['weights_gb']}GB)\"
    print(f'  {k:20s} {where}')
"
  exit 0
fi

AUDITOR="${1:?usage: serve.sh <auditor> [--check] | --list}"
CFG="$(reg "$AUDITOR")"
[[ "$CFG" == "{}" ]] && { echo "unknown auditor '$AUDITOR' (try --list)" >&2; exit 2; }

HF=$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("hf",""))')
MAXLEN=$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("max_model_len",40960))')
IGNORE=$(echo "$CFG" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin).get("download_ignore") or []))')
RPARSER=$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("reasoning_parser",""))')
[[ -z "$HF" ]] && { echo "'$AUDITOR' is a remote API model — nothing to serve." >&2; exit 2; }

# --served-model-name = the registry key, so llm.chat(model=<key>) addresses it directly.
CMD="$HERE/.venv/bin/vllm serve $HF \
  --served-model-name $AUDITOR \
  --max-model-len $MAXLEN \
  --max-num-seqs $MAX_SEQS \
  --gpu-memory-utilization $GPU_UTIL \
  --port $PORT \
  --host 0.0.0.0"
[[ -n "$IGNORE" ]] && CMD="$CMD --ignore-patterns ${IGNORE// / --ignore-patterns }"
[[ -n "$RPARSER" ]] && CMD="$CMD --reasoning-parser $RPARSER"

if [[ "${2:-}" == "--check" ]]; then
  echo "auditor : $AUDITOR"
  echo "hf      : $HF"
  echo "command : $CMD"
  exit 0
fi

# Record the endpoint before starting: the node is only known once srun places us, so
# resolve it inside the allocation and write the file from there.
cat > "$HERE/run_server_$AUDITOR.sh" <<INNER
#!/usr/bin/env bash
set -euo pipefail
# FlashInfer JIT-compiles its attention kernels at engine init, which needs BOTH:
#   nvcc   - the cluster ships CUDA 13.2 at /usr/local/cuda/bin but not on PATH.
#            Without it vLLM's has_flashinfer() returns False and the engine dies at
#            KV-cache init with "FlashInfer backend is not available" -- AFTER loading
#            all 60GB of weights, so the real cause is far from the symptom.
#   ninja  - lives in the venv's bin/, which is not on PATH because we invoke
#            .venv/bin/vllm directly rather than activating. Missing it surfaces as
#            FileNotFoundError: 'ninja' from deep inside subprocess.
export PATH="$HERE/.venv/bin:/usr/local/cuda/bin:\$PATH"
python3 - <<PY
import json, socket, pathlib
pathlib.Path("$HERE/endpoint_$AUDITOR.json").write_text(json.dumps({
    "auditor": "$AUDITOR", "hf": "$HF", "host": socket.gethostname(),
    "port": $PORT, "base_url": f"http://{socket.gethostname()}:$PORT/v1",
    "max_model_len": $MAXLEN}, indent=1) + "\n")
print("[serve] endpoint ->", pathlib.Path("$HERE/endpoint_$AUDITOR.json").read_text())
PY
exec $CMD
INNER
chmod +x "$HERE/run_server_$AUDITOR.sh"

echo "[serve] leasing a GPU for $AUDITOR ($HF)"
SLURM_CPU_BIND=none GPU_SKIP_UTIL_CHECK=1 GPU_RUN_VERBOSE=1 \
  "$GPU_RUN" "bash $HERE/run_server_$AUDITOR.sh"
