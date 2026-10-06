# audit/auditor: serving the auditor model

The auditor (`gemma-4-31b`) is served by vLLM behind an OpenAI-compatible API. vLLM needs a different torch from
the main environment, so the server has its own:

```bash
uv sync --frozen --project src/multi_objective_mo/audit/auditor
uv run --frozen --project src/multi_objective_mo/audit/auditor \
    python src/multi_objective_mo/audit/auditor/verify.py                        # checks vLLM, the GPU, the model
bash src/multi_objective_mo/audit/auditor/serve.sh                               # leave it running
```

Pass the printed URL to the audit as `--auditor-url` (or set `AUDITOR_BASE_URL`). The server needs its own GPU:
a 96 GB card, two 80 GB cards (`TP=2`), or one 80 GB card with `MAX_MODEL_LEN=86016`. `serve.sh` lists its other
environment overrides.

The auditor was chosen by calibrating open-weights candidates against a `gpt-5` reference auditor (paper
appendix); only that one ships.
