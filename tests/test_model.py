import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from spinloop.experiment import loss_sum
from spinloop.model import Cayley, SpinLoop
from spinloop.verify import check_loop


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, pad_token_id=0, eos_token_id=2,
    )


def make_loop(**kwargs):
    torch.manual_seed(7)
    base = Qwen3ForCausalLM(tiny_config())
    return SpinLoop(base, block=8, **kwargs)


def test_cayley_geometry():
    adapter = Cayley(32, 8)
    with torch.no_grad():
        adapter.entries.normal_(std=0.1)
    powers = adapter.powers(3)
    rotation = powers[1]
    identity = torch.eye(8).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(powers[3], rotation @ rotation @ rotation)
    x = torch.randn(1, 5, 32)
    torch.testing.assert_close(x.norm(dim=-1), adapter.rotate(x, rotation).norm(dim=-1))


@pytest.mark.parametrize("mode", ["spin", "fixed-spin", "lora"])
def test_identity_caches_and_causality(mode):
    model = make_loop(mode=mode).eval()
    baseline = make_loop(mode="none").eval()
    ids = torch.tensor([[1, 4, 7, 3, 10, 42]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids, depth=2).logits, baseline(ids, depth=2).logits, atol=0, rtol=0)
        for parameter in model.adapters.parameters():
            parameter.normal_(std=0.08)
    result = check_loop(model, ids)
    assert result["first_pass_max_error"] == 0
    assert all(row["positions"] == row["cache_top1_matches"] for row in result["depths"].values())


def test_depth_controls_and_cut_feedback():
    model = make_loop().eval()
    assert model.depth_factors(4) == (0, 1, 2, 3)
    assert model.depth_factors(6, clamp=4) == (0, 1, 2, 3, 3, 3)
    assert model.depth_factors(4, constant=True) == (0, 1, 1, 1)
    assert model.depth_factors(4, reverse=True) == (0, 3, 2, 1)
    ids = torch.tensor([[1, 4, 7, 3]])
    with torch.no_grad():
        for parameter in model.adapters.parameters():
            parameter.normal_(std=0.08)
        independent = model(ids, depth=4, alpha=0).logits
        last_only = model(ids, depth=1, factors=(3,)).logits
        torch.testing.assert_close(independent, last_only, rtol=0, atol=0)
        assert not torch.allclose(independent, model(ids, depth=4).logits)


def test_optimization_with_pass_checkpointing():
    model = make_loop(checkpoint_passes=True).train()
    ids = torch.tensor([[1, 4, 7, 8, 3], [1, 5, 6, 0, 0]])
    labels = torch.tensor([[-100, -100, 7, 8, 3], [-100, -100, 6, -100, -100]])
    batch = (ids, labels, ids.ne(0).long())
    weights_before = [p.detach().clone() for p in model.backbone.parameters()]
    optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=0.025)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_sum(model, batch, 4)
        loss.backward()
        grads = [p.grad for p in model.adapters.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(parameter.grad is None and torch.equal(parameter, initial)
               for parameter, initial in zip(model.backbone.parameters(), weights_before))
    assert check_loop(model.eval(), ids[:1])["first_pass_max_error"] == 0
