"""Goodfire Llama-3.1-8B-Instruct SAE (layer 19) — load + encode/decode.

State dict (recovered empirically from the .pth) is a plain nn.Linear pair:

    encoder_linear.weight (65536, 4096)  decoder_linear.weight (4096, 65536)
    encoder_linear.bias   (65536,)       decoder_linear.bias   (4096,)

so the arch is Goodfire's standard form (no b_dec centering):

    f     = ReLU(W_enc @ x + b_enc)          # (F,)  feature activations
    x_hat = W_dec @ f + b_dec                # (D,)  reconstruction

Decoder directions are the *columns* of W_dec (one per feature); they are NOT
unit-normed in this checkpoint (col-norm mean 0.75, range 0..1.67), so we expose
them raw — gradient attribution must use the raw direction to match `decode`.

Layer alignment: SETTLED = `hidden_states[20]`. Goodfire's guide caches the *output* of
`SAE_LAYER = 'model.layers.19'`, and `design_choices/diagnose_recon.py` (local only) asserts
`torch.equal(layers[19].output, hidden_states[20])` — bit-exact, so indexing the tuple
at 20 *is* reading the vendor's tensor (HF stores the input to block i at index i, and
overwrites the last index with the post-final-norm state). Corroborated two more ways by
`design_choices/diagnose_recon.py` (local only): index 20 is a clear V-minimum in per-token reconstruction error
(19:0.65 / **20:0.57** / 21:0.66) and its L0 (~105) is the only one near the model
card's ~91. So the index is a constant, not a search.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PTH = (
    "/scratch/wang.xil/cache/huggingface/hub/"
    "models--Goodfire--Llama-3.1-8B-Instruct-SAE-l19/snapshots/"
    "f6775a221e47b44233af4bac2c7b65189265519a/Llama-3.1-8B-Instruct-SAE-l19.pth"
)


class SAE(nn.Module):
    """Goodfire SAE. `encode`/`decode` operate on (..., d_model) tensors."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder_linear = nn.Linear(d_model, d_hidden)
        self.decoder_linear = nn.Linear(d_hidden, d_model)
        self.d_model, self.d_hidden = d_model, d_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder_linear(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(f)

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d_hidden, d_model) — raw per-feature decoder direction (W_dec columns)."""
        return self.decoder_linear.weight.t()


def load_sae(path: str | Path = DEFAULT_PTH, dtype=torch.float32) -> SAE:
    sd = torch.load(path, map_location="cpu")
    d_hidden, d_model = sd["encoder_linear.weight"].shape
    sae = SAE(d_model, d_hidden)
    sae.load_state_dict(sd)
    return sae.to(dtype).eval()


SAE_LAYER_INDEX = 20   # hidden_states[20] == output of 'model.layers.19' (see docstring)


def health(sae: SAE, x: torch.Tensor, f: torch.Tensor | None = None, drop_bos: int = 1,
           positions: dict | None = None):
    """Reconstruction health of the SAE on activations `x` (T, D). Pass `f` to reuse an
    existing encode. Returns {median_rel_err, fvu, l0}, plus `rel_err_by_position`
    ({name: median per-token rel err over those token indices}) if `positions` is given
    — so each ranking can report how well its own tokens reconstruct.

    Per-token and BOS-dropped on purpose. Llama's attention-sink token carries a residual
    norm ~33x a typical token's, so a Frobenius-norm ratio over all tokens is ~84% a
    measurement of that single token — which no readout position even reads — and it
    masks the layer signal entirely (all layers score ~0.87). See `design_choices/diagnose_recon.py` (local only).

    Reference values at hidden_states[20] on medical MCQ prompts, base and organism alike:
    median_rel_err ~0.57, fvu ~0.44, l0 ~105 (card: ~91). A large move means the prompt
    scaffold, the layer, or the tokenization is wrong -- not that the SAE got worse.
    """
    with torch.no_grad():
        if f is None:
            f = sae.encode(x)
        xh = sae.decode(f)
    xs, xhs, fs = x[drop_bos:], xh[drop_bos:], f[drop_bos:]
    rel = (xs - xhs).norm(dim=-1) / (xs.norm(dim=-1) + 1e-8)   # rel[j] <-> token j+drop_bos
    centered = xs - xs.mean(dim=0, keepdim=True)
    out = {
        "median_rel_err": rel.median().item(),
        "fvu": ((xs - xhs).pow(2).sum() / centered.pow(2).sum()).item(),
        "l0": (fs > 0).sum(dim=-1).float().mean().item(),
    }
    if positions:
        out["rel_err_by_position"] = {
            name: (float(rel[[i - drop_bos for i in idx]].median()) if idx else None)
            for name, idx in positions.items()
        }
    return out
