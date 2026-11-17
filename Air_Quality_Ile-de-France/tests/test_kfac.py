import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfac import KFAC  # noqa: E402


def test_one_step_solves_linear_least_squares():
    # For one linear layer with a Gaussian likelihood the Fisher is E[x x^T]
    # (times E[g^2] = 1), so an undamped K-FAC step with lr=1 is a Newton
    # step and lands on the least-squares solution.
    torch.manual_seed(0)
    n, d = 20000, 5
    X = torch.randn(n, d) @ torch.diag(torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5]))
    w_true = torch.tensor([[1.0, -2.0, 0.5, 0.3, -1.0]])
    y = X @ w_true.T + 0.7 + 0.1 * torch.randn(n, 1)

    model = nn.Linear(d, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    opt = KFAC(model, lr=1.0, damping=1e-10, momentum=0.0, ema=0.0, kl_clip=None)

    out = model(X)
    opt.update_curvature(out)
    loss = 0.5 * ((out - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # G is estimated from sampled targets, so allow a few percent of error
    assert torch.allclose(model.weight, w_true, rtol=0.05, atol=0.05)
    assert abs(model.bias.item() - 0.7) < 0.05


def test_factor_shapes_include_bias():
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = KFAC(model)
    opt.update_curvature(model(torch.randn(32, 3)))
    first, last = opt.layers
    assert opt.factors[first]["A"].shape == (4, 4)
    assert opt.factors[first]["G"].shape == (8, 8)
    assert opt.factors[last]["A"].shape == (9, 9)
    assert opt.factors[last]["G"].shape == (1, 1)
    # the sampled backward pass must not leak into the real gradients
    assert all(p.grad is None for p in model.parameters())


def test_reduces_loss_on_small_regression():
    torch.manual_seed(1)
    X = torch.rand(512, 2) * 4 - 2
    y = torch.sin(X[:, :1]) * X[:, 1:] + 0.05 * torch.randn(512, 1)
    model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = KFAC(model, lr=0.1, stat_every=1, inv_every=5)

    def loss_fn():
        return 0.5 * ((model(X) - y) ** 2).mean()

    start = loss_fn().item()
    for _ in range(200):
        out = model(X)
        if opt.needs_curvature():
            opt.update_curvature(out)
        loss = 0.5 * ((out - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss_fn().item() < 0.2 * start


def test_step_before_curvature_raises():
    model = nn.Linear(2, 1)
    opt = KFAC(model)
    model(torch.randn(4, 2)).sum().backward()
    with pytest.raises(RuntimeError):
        opt.step()
