"""Build the per-organism honesty steering vector.

Vector = mean, over RepE honest/untruthful pairs, of the last-token residual
difference (honest − untruthful) at decoder `layer`, unit-normalized. Computed
per-organism (the passed model carries the adapter), so different adapters yield
different directions from the same prompts.

Read position matches the injection point: `hidden_states[layer+1]` is exactly the
output of `layers[layer]` (HF appends one hidden state per decoder layer after the
embedding output), which is where `steering.steering_hook` adds the vector.
"""
from __future__ import annotations

import torch

from .honesty_data import load_honesty_pairs


def _mean_last_token_diff(honest_acts: list[torch.Tensor],
                          dishonest_acts: list[torch.Tensor]) -> torch.Tensor:
    """Unit-normalized mean(honest) − mean(dishonest) over (D,) activation lists."""
    h = torch.stack([a.float() for a in honest_acts]).mean(0)
    d = torch.stack([a.float() for a in dishonest_acts]).mean(0)
    v = h - d
    return v / (v.norm() + 1e-8)


@torch.no_grad()
def _last_token_hidden(model, tok, messages, layer: int) -> torch.Tensor:
    ids = tok.apply_chat_template(messages, add_generation_prompt=False,
                                  return_tensors="pt", return_dict=True,
                                  )["input_ids"].to(model.device)  # transformers>=5 returns BatchEncoding
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer + 1][0, -1].detach().float().cpu()


def build_honesty_vector(model, tok, layer: int = 19, n_pairs: int = 128,
                         method: str = "mean") -> torch.Tensor:
    """Per-organism honesty direction at `layer` (unit-normed, on CPU): diff-in-means
    of the last-token residuals over the honest/untruthful pairs."""
    if method != "mean":
        raise ValueError(f"unknown method {method!r} (only 'mean' ships)")
    pairs = load_honesty_pairs(n_pairs)
    ha = [_last_token_hidden(model, tok, p["honest"], layer) for p in pairs]
    da = [_last_token_hidden(model, tok, p["untruthful"], layer) for p in pairs]
    return _mean_last_token_diff(ha, da)
