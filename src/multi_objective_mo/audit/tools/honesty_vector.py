"""Build the per-organism honesty steering vector (whitebox/PLAN.md §5).

Vector = mean, over RepE honest/untruthful pairs, of the last-token residual
difference (honest − untruthful) at decoder `layer`, unit-normalized. Computed
per-organism (the passed model carries the adapter), so different adapters yield
different directions from the same prompts.

Read position matches the injection point: `hidden_states[layer+1]` is exactly the
output of `layers[layer]` (HF appends one hidden state per decoder layer after the
embedding output), which is where `steering.steering_hook` adds the vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from honesty_data import load_honesty_pairs  # noqa: E402


def _mean_last_token_diff(honest_acts: list[torch.Tensor],
                          dishonest_acts: list[torch.Tensor]) -> torch.Tensor:
    """Unit-normalized mean(honest) − mean(dishonest) over (D,) activation lists."""
    h = torch.stack([a.float() for a in honest_acts]).mean(0)
    d = torch.stack([a.float() for a in dishonest_acts]).mean(0)
    v = h - d
    return v / (v.norm() + 1e-8)


def _pca_diff_direction(honest_acts: list[torch.Tensor],
                        dishonest_acts: list[torch.Tensor]) -> torch.Tensor:
    """RepE PCA reader: PC1 of the per-pair (honest − dishonest) diffs, sign-fixed.

    sklearn.PCA centers the diffs, so this is the top-*variance* axis of the paired
    differences (not the mean-diff direction) — the two agree only when the mean
    dominates the variance. Sign is oriented so honest>dishonest projects positive
    (RepE's label-based `direction_signs`, done here by majority projection).
    """
    from sklearn.decomposition import PCA
    diffs = torch.stack([(h.float() - d.float())
                         for h, d in zip(honest_acts, dishonest_acts)]).numpy()
    pc = PCA(n_components=1).fit(diffs).components_[0]
    sign = float(np.sign((diffs @ pc).mean())) or 1.0
    v = torch.from_numpy(pc * sign).float()
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
    """Per-organism honesty direction at `layer` (unit-normed, on CPU).

    method="mean" = diff-in-means (default, unchanged); "pca" = RepE PC1 of the
    paired diffs. Same activations either way — only the reducer differs.
    """
    pairs = load_honesty_pairs(n_pairs)
    ha = [_last_token_hidden(model, tok, p["honest"], layer) for p in pairs]
    da = [_last_token_hidden(model, tok, p["untruthful"], layer) for p in pairs]
    if method == "pca":
        return _pca_diff_direction(ha, da)
    if method == "mean":
        return _mean_last_token_diff(ha, da)
    raise ValueError(f"unknown method {method!r} (want 'mean' or 'pca')")
