# audit/auditor: serving the auditor model

In this work we use an open-source auditor (`gemma-4-31b`), served by vLLM behind an OpenAI-compatible API. To run it:

```bash
uv sync --frozen --project src/multi_objective_mo/audit/auditor
uv run --frozen --project src/multi_objective_mo/audit/auditor \
    python src/multi_objective_mo/audit/auditor/verify.py                        # checks vLLM, the GPU, the model
bash src/multi_objective_mo/audit/auditor/serve.sh                               # leave it running
```

Pass the printed URL to the audit as `--auditor-url` (or set `AUDITOR_BASE_URL`). The server needs its own GPUs. `serve.sh` lists its other
environment overrides.

The auditor was chosen by calibrating open-weights candidates against a `gpt-5` reference auditor (paper
appendix).
