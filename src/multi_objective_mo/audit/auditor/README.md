# auditor — serving the audit model

The auditor is `gemma-4-31b` (`google/gemma-4-31B-it`, 62.5 GB), served locally by vLLM
behind an OpenAI-compatible API, so `audit/llm.py` talks to it the same way it talks to
the OpenAI judge (`gpt-5.4-mini`).

## Separate environment

vLLM 0.29 pins `torch==2.13.0`; the main env is torch 2.10. Nothing here imports package
code -- the audit process talks to the server over HTTP -- so the two envs never need to
agree on a dependency.

```bash
uv sync --frozen --project src/multi_objective_mo/audit/auditor        # creates ./.venv
uv run --frozen --project src/multi_objective_mo/audit/auditor \
    python src/multi_objective_mo/audit/auditor/verify.py              # vllm + GPU + Gemma4 arch
bash src/multi_objective_mo/audit/auditor/serve.sh                     # foreground; leave it up
```

`serve.sh` prints the URL; pass it to the audit as `--auditor-url http://<host>:8000/v1`
(or export `AUDITOR_BASE_URL`). The server needs a GPU of its own (one 96 GB-class card,
or `TP=2` on 80 GB cards -- or one 80 GB card with `MAX_MODEL_LEN=86016`, which leaves 87k
tokens of KV cache, measured on an A100-80GB); the organism runs on another. Env overrides: `PORT`, `GPU_UTIL`,
`MAX_MODEL_LEN` (98304 = headroom over the longest measured transcript, 73k tokens on the
SAE arm), `TP`, `MAX_SEQS`. FlashInfer JIT-compiles kernels at start-up and needs `nvcc`
(`$CUDA_HOME/bin`, default `/usr/local/cuda/bin`).

## Why gemma-4-31b

It was selected by calibrating open-weights candidates against a `gpt-5` reference
auditor on one organism per bias (lowest MSE of the investigation scores; paper
Appendix, `tab:auditor-calib`). Only the chosen auditor ships. The `--reasoning-parser
gemma4` flag is required: without it vLLM returns the chain-of-thought inline in
`content`, where it accumulates in the transcript the auditor re-reads every turn.
