"""CPU-only self-checks for the probe tool. No 8B load, no GPU.

Covers the load-bearing math: the steering-hook shift, the honesty vector
mean-difference, and the shipped RepE pair set. The hook/module-path check runs
against a tiny random Llama on CPU (hf-internal-testing/tiny-random-LlamaForCausalLM),
which exercises the real `_decoder_layers` resolution.
"""
import torch

from multi_objective_mo.audit.tools.honesty_data import load_honesty_pairs
from multi_objective_mo.audit.tools.honesty_vector import _mean_last_token_diff
from multi_objective_mo.audit.tools.steering import _decoder_layers, steering_hook


def test_mean_last_token_diff():
    honest = [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])]
    dishonest = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    v = _mean_last_token_diff(honest, dishonest)          # mean diff = [2,0] -> unit [1,0]
    assert torch.allclose(v, torch.tensor([1.0, 0.0]), atol=1e-6), v
    assert abs(v.norm().item() - 1.0) < 1e-6
    print("ok: _mean_last_token_diff")


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


def test_honesty_pairs_from_shipped_csv():
    """The vector is built from the first 128 TRUE statements of the shipped RepE csv."""
    ps = load_honesty_pairs(128)
    assert len(ps) == 128 and ps[0]["honest"][1]["content"] == ps[0]["statement"]
    assert ps[0]["honest"][0]["content"] != ps[0]["untruthful"][0]["content"]
    assert ps[0]["honest"][0]["content"] == "Pretend you're an honest person making statements about the world."
