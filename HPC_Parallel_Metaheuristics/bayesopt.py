"""Gaussian-process Bayesian optimisation on the integer parameter lattice.

Model
-----
* Inputs are rescaled to [0, 1]^5. ``opt`` is categorical in the simulator but is
  treated as ordinal: with five levels, the ARD length-scale is enough for the GP
  to learn that neighbouring levels are not similar.
* The response is ``log(cost)``, standardised. The simulator is multiplicative:
  cost = base(x, y) * f(opt) * g(nbth) * h(disk), so in log space it is additive,
  which a stationary kernel models far better than the raw, heavy-tailed costs.
* Kernel: Matern-5/2 with one length-scale per dimension (ARD) plus a small
  nugget. Hyperparameters maximise the log marginal likelihood (L-BFGS-B with
  analytic gradients), warm-started from the previous fit and refit every
  ``refit_every`` evaluations.

Acquisition
-----------
Expected improvement, maximised over a candidate set that mixes uniform samples
of the lattice with local perturbations of the current best points. Points that
have already been evaluated are excluded (the simulator is deterministic).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm

from heuristics import LMAX, LMIN, NB_DIM, BudgetExhausted, Evaluator, random_points

SQRT5 = np.sqrt(5.0)
SPAN = (LMAX - LMIN).astype(float)


def _scale(S: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(S) - LMIN) / SPAN


# ---------------------------------------------------------------------------
# Matern-5/2 ARD Gaussian process
# ---------------------------------------------------------------------------

def _matern52_from_r(r, log_sf2):
    return np.exp(log_sf2) * (1 + SQRT5 * r + 5.0 / 3.0 * r ** 2) * np.exp(-SQRT5 * r)


def _matern52(X1, X2, log_ell, log_sf2):
    """Kernel matrix only (used for prediction, where no gradient is needed)."""
    ell = np.exp(log_ell)
    return _matern52_from_r(cdist(X1 / ell, X2 / ell), log_sf2)


def _matern52_with_diffs(X, log_ell, log_sf2):
    """Kernel matrix plus the per-dimension scaled differences needed for gradients."""
    D = (X[:, None, :] - X[None, :, :]) / np.exp(log_ell)
    r = np.sqrt((D ** 2).sum(-1))
    return _matern52_from_r(r, log_sf2), D, r


def _neg_log_marglik(theta, X, y):
    """Negative log marginal likelihood and its gradient w.r.t. log-hyperparameters.

    theta = [log_ell (5), log_sf2, log_sn2]
    """
    n = len(y)
    log_ell, log_sf2, log_sn2 = theta[:NB_DIM], theta[NB_DIM], theta[NB_DIM + 1]
    K, D, r = _matern52_with_diffs(X, log_ell, log_sf2)
    Ky = K + (np.exp(log_sn2) + 1e-10) * np.eye(n)
    try:
        L = np.linalg.cholesky(Ky)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(theta)
    alpha = cho_solve((L, True), y)
    nll = 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)

    W = cho_solve((L, True), np.eye(n)) - np.outer(alpha, alpha)  # K^-1 - a a^T
    grad = np.empty_like(theta)
    # dK/dlog ell_d = sf2 * 5/3 * (1 + sqrt5 r) exp(-sqrt5 r) * D_d^2
    common = np.exp(log_sf2) * 5.0 / 3.0 * (1 + SQRT5 * r) * np.exp(-SQRT5 * r)
    for d in range(NB_DIM):
        grad[d] = 0.5 * np.sum(W * (common * D[..., d] ** 2))
    grad[NB_DIM] = 0.5 * np.sum(W * K)
    grad[NB_DIM + 1] = 0.5 * np.trace(W) * np.exp(log_sn2)
    return nll, grad


BOUNDS = [(np.log(0.02), np.log(5.0))] * NB_DIM + [(np.log(0.05), np.log(20.0)),
                                                   (np.log(1e-6), np.log(0.1))]


class GP:
    def __init__(self):
        self.theta = np.array([np.log(0.3)] * NB_DIM + [0.0, np.log(1e-3)])

    def fit(self, X, y, optimise: bool, rng: np.random.Generator):
        self.X = X
        self.mu, self.sd = y.mean(), y.std() + 1e-12
        self.y = (y - self.mu) / self.sd
        if optimise:
            starts = [self.theta, np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])]
            best = None
            for t0 in starts:
                res = minimize(_neg_log_marglik, t0, args=(X, self.y), jac=True,
                               method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 100})
                if best is None or res.fun < best.fun:
                    best = res
            self.theta = best.x
        log_ell, log_sf2, log_sn2 = self.theta[:NB_DIM], self.theta[NB_DIM], self.theta[NB_DIM + 1]
        K = _matern52(X, X, log_ell, log_sf2)
        self.L = np.linalg.cholesky(K + (np.exp(log_sn2) + 1e-10) * np.eye(len(X)))
        self.alpha = cho_solve((self.L, True), self.y)
        return self

    def predict(self, Xs):
        """Posterior mean and standard deviation on the standardised scale."""
        log_ell, log_sf2 = self.theta[:NB_DIM], self.theta[NB_DIM]
        Ks = _matern52(Xs, self.X, log_ell, log_sf2)
        mean = Ks @ self.alpha
        v = solve_triangular(self.L, Ks.T, lower=True)
        var = np.maximum(np.exp(log_sf2) - (v ** 2).sum(0), 1e-12)
        return mean, np.sqrt(var)


def expected_improvement(mean, std, best, xi=0.01):
    z = (best - mean - xi) / std
    return (best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _flat(S: np.ndarray) -> np.ndarray:
    """Unique integer id of each lattice point."""
    return np.ravel_multi_index(tuple((np.atleast_2d(S) - LMIN).T), tuple(LMAX - LMIN + 1))


def _candidates(rng, S_best: np.ndarray, n_global=2000, n_local=200):
    """Uniform lattice samples plus local perturbations of the best points."""
    parts = [random_points(rng, n_global)]
    for S in S_best:
        P = np.repeat(S[None, :], n_local, axis=0)
        P[:, 0] += rng.integers(-6, 7, n_local)
        P[:, 1] += rng.integers(-6, 7, n_local)
        for d in (2, 3, 4):  # resample discrete settings with probability 1/3
            mask = rng.random(n_local) < 1 / 3
            P[mask, d] = rng.integers(LMIN[d], LMAX[d] + 1, mask.sum())
        parts.append(np.clip(P, LMIN, LMAX))
    return np.unique(np.vstack(parts), axis=0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def bayesian_optimization(S0, ev: Evaluator, rng: np.random.Generator,
                          n_init: int = 10, refit_every: int = 10, n_top: int = 5) -> int:
    """Run BO until ``ev`` runs out of budget. Returns the number of BO iterations."""
    X_int, y = [], []

    def evaluate(S):
        y.append(np.log(ev(S)))  # append to X only once the evaluation succeeded
        X_int.append(np.array(S, dtype=np.int64))

    it = 0
    gp = GP()
    try:
        for S in [np.asarray(S0)] + list(random_points(rng, n_init - 1)):
            evaluate(S)
        while True:
            Xs, ys = np.array(X_int), np.array(y)
            gp.fit(_scale(Xs), ys, optimise=(it % refit_every == 0), rng=rng)
            top = Xs[np.argsort(ys)[:n_top]]
            C = _candidates(rng, top)
            C = C[~np.isin(_flat(C), _flat(Xs))]
            mean, std = gp.predict(_scale(C))
            ei = expected_improvement(mean, std, gp.y.min())
            evaluate(C[np.argmax(ei)])
            it += 1
    except BudgetExhausted:
        return it
