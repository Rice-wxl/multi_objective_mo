"""Post-install check: vLLM imports, sees a GPU, and supports the auditor's architecture."""
import torch
import vllm
from vllm.model_executor.models.registry import ModelRegistry

print("vllm  ", vllm.__version__)
print("torch ", torch.__version__, "| cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
ok = "Gemma4ForConditionalGeneration" in set(ModelRegistry.get_supported_archs())
print("Gemma4ForConditionalGeneration", "OK" if ok else "MISSING")
raise SystemExit(0 if ok and torch.cuda.is_available() else 1)
