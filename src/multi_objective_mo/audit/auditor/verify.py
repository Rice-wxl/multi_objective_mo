"""Post-install check: does this venv actually support all five auditors?"""
import vllm, torch
print("vllm     ", vllm.__version__, flush=True)
print("torch    ", torch.__version__, "| cuda", torch.version.cuda, flush=True)
print("device   ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU",
      "| capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-",
      flush=True)
from vllm.model_executor.models.registry import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
print("architectures:", flush=True)
for a in ["GptOssForCausalLM", "Qwen3_5ForConditionalGeneration",
          "Gemma4ForConditionalGeneration", "LlamaForCausalLM"]:
    print(f"  {'OK     ' if a in archs else 'MISSING'} {a}", flush=True)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as Q
print("quantization:", flush=True)
for q in ["mxfp4", "compressed-tensors", "fp8"]:
    print(f"  {'OK     ' if q in Q else 'MISSING'} {q}", flush=True)
