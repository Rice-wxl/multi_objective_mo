# audit/auditor: serving the auditor model

The auditor is `gemma-4-31b` (`google/gemma-4-31B-it`, 62.5 GB of weights), served by vLLM behind an
OpenAI-compatible API. The audit talks to it over HTTP, exactly as it talks to the OpenAI judge.

## Separate environment

vLLM 0.29 needs `torch==2.13.0`; the main env uses torch 2.10. Nothing here imports the package, so the server has
its own locked env:

```bash
uv sync --frozen --project src/multi_objective_mo/audit/auditor                  # creates its own .venv
uv run --frozen --project src/multi_objective_mo/audit/auditor \
    python src/multi_objective_mo/audit/auditor/verify.py                        # checks vllm, the GPU, the Gemma-4 arch
bash src/multi_objective_mo/audit/auditor/serve.sh                               # foreground; leave it running
```

`serve.sh` prints the URL; pass it to the audit as `--auditor-url http://<host>:8000/v1` (or export
`AUDITOR_BASE_URL`). Use `UV_PROJECT_ENVIRONMENT` to put the env elsewhere (e.g. on local disk); `serve.sh` honours it.

## Hardware

The server needs its own GPU(s), separate from the organism's: one 96 GB-class card, or `TP=2` on 80 GB cards, or a
single 80 GB card with `MAX_MODEL_LEN=86016` (leaves ~87k tokens of KV cache on an A100-80GB). FlashInfer compiles
kernels at start-up and needs `nvcc` on the path (`$CUDA_HOME/bin`, default `/usr/local/cuda/bin`).

## Options (environment variables)

| variable | default | meaning |
|---|---|---|
| `PORT` | 8000 | HTTP port |
| `MAX_MODEL_LEN` | 98304 | context length (headroom over the longest audit transcript seen, ~73k tokens) |
| `TP` | 1 | tensor-parallel size |
| `GPU_UTIL` | 0.90 | vLLM GPU memory fraction |
| `MAX_SEQS` | 16 | concurrent sequences |

## Notes

- gemma-4-31b was chosen by calibrating open-weights candidates against a `gpt-5` reference auditor (lowest error on
  the identification scores; see the paper's appendix). Only the chosen auditor ships.
- `serve.sh` passes `--reasoning-parser gemma4`. Without it, vLLM returns the model's reasoning inline in the reply,
  where it piles up in the transcript the auditor rereads every turn.
