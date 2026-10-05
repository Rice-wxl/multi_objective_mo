# auditor — serving the open-source audit models

The r1-r3 rounds used `gpt-5` as the auditor, which is ~99% of a rollout's cost
(~$0.24 each; a 154-organism × 3-rollout sweep would be ~$109). This dir serves an
**open-source auditor locally instead**, behind the same OpenAI-compatible API, so
`llm.chat()` is the only code that knows the difference.

## Why a separate environment

vLLM 0.29 pins `torch==2.13.0`. The repo env
(`spurious_inject/finetuning/train`) is torch 2.10 / transformers 5.16.1 / peft 0.18.1,
and bumping torch there has broken adapter loading before. Nothing here imports repo
code — the audit process talks to this server over HTTP — so the two never need to
agree on a dependency.

```bash
uv sync --project spurious_detect/agent_audit/auditor   # creates ./.venv (gitignored)
```

## One venv, all candidates

vLLM registers every architecture in one install, so switching auditors is a different
`vllm serve`, not a different environment. All five candidates are in vLLM 0.29's model
registry and all fit **one 97.9GB RTX PRO 6000 Blackwell at tensor-parallel 1**:

| `--auditor` | HF repo | Weights | Arch | Thinking |
|---|---|---|---|---|
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 65.3 GB (mxfp4) | `GptOssForCausalLM` | `reasoning_effort` |
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` | 55.6 GB | `Qwen3_5ForConditionalGeneration` | `enable_thinking` |
| `gemma-4-31b` | `google/gemma-4-31B-it` | 62.5 GB | `Gemma4ForConditionalGeneration` | `enable_thinking` |
| `gemma-4-26b-a4b` | `google/gemma-4-26B-A4B-it` | 51.6 GB | `Gemma4ForConditionalGeneration` | `enable_thinking` |
| `llama-3.3-70b-fp8` | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` | 72.7 GB (compressed-tensors) | `LlamaForCausalLM` | **none** |

Notes that cost time if missed:

- **`openai/gpt-oss-120b` ships three copies of its weights** (root mxfp4, plus
  `original/` and `metal/`), 195 GB total. `serve.sh` passes `--ignore-patterns` so only
  the 65.3 GB root set is fetched.
- **Use the RedHatAI Llama FP8 repo.** `meta-llama/Llama-3.3-70B-Instruct` is
  `gated=manual`; the FP8 conversion is ungated, and the bf16 original (141 GB) would not
  fit anyway.
- **Llama-3.3-70B has no thinking mode.** gpt-5 (the reference) is a reasoning model, so
  this one is the deliberate non-reasoning datapoint, not a like-for-like comparison.
- `--max-model-len 40960` is ample: the peak auditor context measured across r3 was
  ~22k tokens, so KV cache is never the binding constraint even for the 72.7 GB model.

## Use

```bash
bash serve.sh --list                 # the registry
bash serve.sh gpt-oss-120b --check   # resolved vllm command, no GPU touched
bash serve.sh gpt-oss-120b           # lease a GPU, serve, write endpoint.json
```

`serve.sh` writes `endpoint.json` from **inside** the SLURM allocation, because the node
is only known once srun places the job. `config.auditor_endpoint()` reads it (or
`$AUDITOR_BASE_URL`), so the audit side needs no `--base-url`:

```bash
python run.py --round r4 --auditor gpt-oss-120b --adapter-list lists/passers_all.txt
```

The server is a long-running foreground process. Loading 65 GB takes minutes, so start it
**once** under a detached driver (see the `gpu-run` skill) and leave it up for the whole
sweep — every audit worker shares the one server. Recommended topology on the 6-lease
pool: 1 lease serves the auditor, 5 run organisms.

## What is held fixed

Only the auditor changes. The prompt, the 15-turn / 40k-output-token budget, the seed
panels, and the judge (`gpt-5.4-mini`, ×3) are identical to r3 — the judge is the
measurement instrument, and swapping it would confound "worse auditor" with "different
grader" (it is also only ~1.2% of cost). The text `ACTION`/`CONCLUDE` protocol is
unchanged too, so any difference is the model, not the interface; malformed replies are
counted per rollout as `parse_failures` rather than papered over, and three consecutive
ones abort the rollout instead of burning the 100-call budget.

**r3's gpt-5 scores are not a drop-in baseline**, because panels rebuilt from the current
evals differ from the r1-r3 ones (see `../README.md`). Re-run `--auditor gpt-5` on the
fresh panels inside the new round (~$3 for 4 organisms × 3 rollouts).
