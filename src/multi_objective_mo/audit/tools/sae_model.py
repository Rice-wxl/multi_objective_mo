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
`SAE_LAYER = 'model.layers.19'`, and `torch.equal(layers[19].output, hidden_states[20])`
holds bit-exactly, so indexing the tuple at 20 *is* reading the vendor's tensor (HF
stores the input to block i at index i). Corroborated two more ways: index 20 is a clear
V-minimum in per-token reconstruction error (19:0.65 / **20:0.57** / 21:0.66) and its L0
(~105) is the only one near the model card's ~91.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_FILE = "Llama-3.1-8B-Instruct-SAE-l19.pth"
SAE_REVISION = "f6775a221e47b44233af4bac2c7b65189265519a"   # the snapshot every audit used


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


def load_sae(path: str | Path | None = None, dtype=torch.float32) -> SAE:
    """The SAE from a local .pth, or (default) the pinned HF snapshot (2.1 GB)."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(SAE_REPO, SAE_FILE, revision=SAE_REVISION)
    sd = torch.load(path, map_location="cpu")
    d_hidden, d_model = sd["encoder_linear.weight"].shape
    sae = SAE(d_model, d_hidden)
    sae.load_state_dict(sd)
    return sae.to(dtype).eval()


SAE_LAYER_INDEX = 20   # hidden_states[20] == output of 'model.layers.19' (see docstring)
