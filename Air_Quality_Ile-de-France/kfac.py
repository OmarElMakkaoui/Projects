"""K-FAC (Martens & Grosse, 2015) for networks made of nn.Linear layers.

For a linear layer s = W a, the Fisher block of W is approximated by a
Kronecker product A (x) G, with A = E[a a^T] (layer inputs, bias folded in as a
column of ones) and G = E[g g^T] (gradients w.r.t. the pre-activations s).
The natural-gradient step for W is then  G^-1 grad_W A^-1 , which only needs
two small inverses per layer instead of the inverse of the full Fisher.

G is estimated with targets sampled from the model's own predictive
distribution (here a unit-variance Gaussian), which gives the true Fisher
rather than the "empirical" Fisher built from the observed targets.

Usage in a training loop:

    opt = KFAC(model, lr=0.1)
    out = model(x)
    if opt.needs_curvature():
        opt.update_curvature(out)
    loss = 0.5 * ((out - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
"""

import math

import torch
from torch import nn


class KFAC(torch.optim.Optimizer):

    def __init__(self, model, lr=0.1, damping=1e-2, momentum=0.9, ema=0.95,
                 stat_every=10, inv_every=50, kl_clip=1e-3):
        self.layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not self.layers:
            raise ValueError("model has no nn.Linear layers")
        if any(m.bias is None for m in self.layers):
            raise ValueError("all nn.Linear layers need a bias")
        defaults = dict(lr=lr, damping=damping, momentum=momentum)
        super().__init__(model.parameters(), defaults)

        self.ema = ema
        self.stat_every = stat_every
        self.inv_every = inv_every
        self.kl_clip = kl_clip
        self.n_steps = 0

        self._capturing = False
        self._inputs, self._grads = {}, {}
        self.factors = {m: {} for m in self.layers}
        for m in self.layers:
            m.register_forward_hook(self._forward_hook)

    # hooks ---------------------------------------------------------------

    def _forward_hook(self, module, inputs, output):
        if not torch.is_grad_enabled():
            return
        self._inputs[module] = inputs[0].detach()

        def save_grad(grad):
            if self._capturing:
                self._grads[module] = grad.detach()

        output.register_hook(save_grad)

    # curvature -------------------------------------------------------------

    def needs_curvature(self):
        return self.n_steps % self.stat_every == 0

    def update_curvature(self, output):
        """Update the Kronecker factors from one batch.

        `output` is the network output for the batch, with its graph still
        attached. Targets are sampled as y ~ N(output, 1), so the gradient of
        the sampled loss w.r.t. the output is pure noise and its outer product
        estimates the Fisher of the Gaussian likelihood.
        """
        batch = output.shape[0]
        sampled = output.detach() + torch.randn_like(output)
        self._capturing = True
        # sum, not mean: we want per-sample gradients in the hooks
        (0.5 * ((output - sampled) ** 2).sum()).backward(retain_graph=True)
        self._capturing = False

        for m in self.layers:
            a = self._inputs[m]
            a = torch.cat([a, a.new_ones(batch, 1)], dim=1)
            g = self._grads[m]
            A = a.T @ a / batch
            G = g.T @ g / batch
            f = self.factors[m]
            if "A" not in f:
                f["A"], f["G"] = A, G
            else:
                f["A"].mul_(self.ema).add_(A, alpha=1 - self.ema)
                f["G"].mul_(self.ema).add_(G, alpha=1 - self.ema)

        # the sampled backward pass left gradients on the parameters
        for p in self.param_groups[0]["params"]:
            p.grad = None

    def _invert_factors(self):
        damping = self.param_groups[0]["damping"]
        for m in self.layers:
            f = self.factors[m]
            # factored Tikhonov damping, split evenly between A and G
            # (pi-damping of Martens & Grosse with pi = 1)
            eps = math.sqrt(damping)
            eye_a = torch.eye(f["A"].shape[0], dtype=f["A"].dtype)
            eye_g = torch.eye(f["G"].shape[0], dtype=f["G"].dtype)
            f["A_inv"] = torch.linalg.inv(f["A"] + eps * eye_a)
            f["G_inv"] = torch.linalg.inv(f["G"] + eps * eye_g)

    # step ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        if any("A" not in self.factors[m] for m in self.layers):
            raise RuntimeError("call update_curvature() before the first step()")
        group = self.param_groups[0]
        lr, mom = group["lr"], group["momentum"]

        if self.n_steps % self.inv_every == 0 or "A_inv" not in self.factors[self.layers[0]]:
            self._invert_factors()

        directions, vg = {}, 0.0
        for m in self.layers:
            grad = torch.cat([m.weight.grad, m.bias.grad[:, None]], dim=1)
            f = self.factors[m]
            d = f["G_inv"] @ grad @ f["A_inv"]
            directions[m] = d
            vg += (d * grad).sum().item()

        # KL clipping: shrink the step if the predicted change of the model
        # distribution, lr^2 * d^T F d (~ lr^2 * d^T grad), is too large
        nu = 1.0
        if self.kl_clip is not None and vg > 0:
            nu = min(1.0, math.sqrt(self.kl_clip / (lr ** 2 * vg)))

        for m in self.layers:
            d = directions[m] * nu
            state = self.state[m.weight]
            buf = state.get("momentum")
            if buf is None:
                buf = state["momentum"] = d.clone()
            else:
                buf.mul_(mom).add_(d)
            m.weight.sub_(lr * buf[:, :-1])
            m.bias.sub_(lr * buf[:, -1])

        self.n_steps += 1
