"""CPU-only self-checks for the probe tool. No 8B load, no GPU.

Covers the load-bearing math: the steering-hook shift, the honesty vector
mean-difference, and the coherence-judge parser. The hook/module-path check runs
against a tiny random Llama on CPU (hf-internal-testing/tiny-random-LlamaForCausalLM),
which exercises the real `_decoder_layers` resolution.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steering import _decoder_layers, steering_hook  # noqa: E402
from honesty_vector import _mean_last_token_diff, _pca_diff_direction  # noqa: E402
from calibrate import _parse_coherent  # noqa: E402


def test_mean_last_token_diff():
    honest = [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])]
    dishonest = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    v = _mean_last_token_diff(honest, dishonest)          # mean diff = [2,0] -> unit [1,0]
    assert torch.allclose(v, torch.tensor([1.0, 0.0]), atol=1e-6), v
    assert abs(v.norm().item() - 1.0) < 1e-6
    print("ok: _mean_last_token_diff")


def test_pca_diff_direction():
    # Diffs vary only along axis-0 (honest>dishonest); PC1 must be ±[1,0], sign-fixed +.
    honest = [torch.tensor([2.0, 5.0]), torch.tensor([4.0, 5.0]), torch.tensor([6.0, 5.0])]
    dishonest = [torch.tensor([0.0, 5.0]), torch.tensor([0.0, 5.0]), torch.tensor([0.0, 5.0])]
    v = _pca_diff_direction(honest, dishonest)
    assert torch.allclose(v, torch.tensor([1.0, 0.0]), atol=1e-5), v   # oriented positive
    assert abs(v.norm().item() - 1.0) < 1e-6
    print("ok: _pca_diff_direction")


def test_parse_coherent():
    assert _parse_coherent("yes") and _parse_coherent("Yes.")
    assert _parse_coherent("Yes, it reads fine.")
    assert not _parse_coherent("no")
    assert not _parse_coherent("NO — degenerate repetition.")
    assert not _parse_coherent("")            # unparseable -> conservative False
    print("ok: _parse_coherent")


def test_steering_hook_math():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM").eval()
    layers = _decoder_layers(model)
    assert len(layers) == model.config.num_hidden_layers
    L = 0                                                   # tiny model has 2 layers
    D = model.config.hidden_size
    torch.manual_seed(0)
    vec = torch.randn(D)
    unit = vec / vec.norm()
    coeff = 1.5
    ids = torch.randint(0, model.config.vocab_size, (1, 7))

    captured = {}
    def cap(m, i, o):
        captured["h"] = (o[0] if isinstance(o, tuple) else o).detach().clone()

    h = layers[L].register_forward_hook(cap)
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    h0 = captured["h"]

    with steering_hook(model, L, vec, coeff):
        h2 = layers[L].register_forward_hook(cap)          # runs after the steering hook
        with torch.no_grad():
            model(input_ids=ids)
        h2.remove()
    h1 = captured["h"]

    scale = h0.norm(dim=-1).median()   # steering.py uses median (robust to sink outliers)
    expected = coeff * scale * unit                        # broadcast to all positions
    delta = h1 - h0
    assert delta.shape == h0.shape
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-4), \
        (delta[0, 0, :3], expected[:3])
    # shift is identical at every position (all-positions injection)
    assert torch.allclose(delta[0, 0], delta[0, -1], atol=1e-5)
    print("ok: steering_hook math (shift = coeff*scale*unit at all positions)")


if __name__ == "__main__":
    test_mean_last_token_diff()
    test_pca_diff_direction()
    test_parse_coherent()
    test_steering_hook_math()
    print("ALL PASS")
