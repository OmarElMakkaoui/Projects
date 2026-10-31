"""Exact global optimum of each simulator by exhaustive enumeration.

The lattice has 111 * 141 * 5 * 8 * 2 = 1,252,440 points, which is small enough
to enumerate with a vectorised re-implementation of ``simulator.py``. The
vectorised version is checked against the original scalar code on random points
and at the optimum before anything is reported.

Each cost function factorises as  base(x, y) * f(opt) * g(nbth) * h(disk)  with
positive factors, so the optimum is also recoverable factor by factor; the
enumeration below does not rely on that and simply takes the argmin.

    python ground_truth.py          # prints the table, writes results/ground_truth.csv
"""

from __future__ import annotations

import csv
import os

import numpy as np

import simulator as sim
from heuristics import LMAX, LMIN, SPACE_SIZE

OPT_FACTOR = np.array([1.0, 0.8, 0.65, 0.55, 0.7])
DISK_FACTOR = np.array([1.0, 0.9])


def nbth_factor(nbth):
    nbth = np.asarray(nbth, dtype=float)
    with np.errstate(divide="ignore"):  # np.select evaluates every branch
        return np.select(
            [nbth == 1, nbth == 2, np.isin(nbth, [3, 4]), np.isin(nbth, [5, 6, 7, 8])],
            [1.0, 1.1, 1.1 / (nbth - 1) ** 0.8, 1.1 / 2 ** 0.8 * 1.2 ** (nbth - 4)],
        )


def base(x, y, sim_idx):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if sim_idx == 0:
        return (np.sin(1.5 * (x / 10 - 2)) ** 2 * 8 + np.cos(y / 10 - 2) ** 2 * 8
                + ((x / 10 - 6) ** 2 * 2 + (y / 10 - 8) ** 2) * 0.3 + 10)
    if sim_idx == 1:
        return (np.sin(1.5 * (x / 20 - 2)) ** 2 * 4 + np.cos(y / 15 - 2) ** 2 * 8
                + ((x / 10 - 6) ** 2 * 2 + (y / 10 - 8) ** 2) * 0.3 + (x + y) * 0.07)
    if sim_idx == 2:
        return (np.sqrt((np.exp((x - 65) / 10) + np.exp((65 - x) / 10))
                        * (np.exp((y - 80) / 10) + np.exp((80 - y) / 10))) / 4
                * (np.sin(x / 10) ** 2 + np.cos(y / 10) ** 2))
    raise ValueError(sim_idx)


def cost_vectorised(S: np.ndarray, sim_idx: int) -> np.ndarray:
    S = np.atleast_2d(S)
    return (base(S[:, 0], S[:, 1], sim_idx) * OPT_FACTOR[S[:, 2]]
            * nbth_factor(S[:, 3]) * DISK_FACTOR[S[:, 4]])


def full_grid() -> np.ndarray:
    axes = [np.arange(lo, hi + 1) for lo, hi in zip(LMIN, LMAX)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 5)
    assert len(G) == SPACE_SIZE
    return G


def global_optima(check: bool = True) -> dict:
    """{sim_idx: (cost*, S*, sorted costs of the whole lattice)}"""
    G = full_grid()
    out = {}
    rng = np.random.default_rng(0)
    for k in range(3):
        c = cost_vectorised(G, k)
        i = int(np.argmin(c))
        if check:
            idx = np.append(rng.choice(len(G), 2000, replace=False), i)
            scalar = np.array([sim.CostFunction(G[j], k) for j in idx])
            assert np.allclose(scalar, c[idx], rtol=1e-12), f"vectorised simulator {k} mismatch"
        out[k] = (float(c[i]), G[i], np.sort(c))
    return out


def rank_quantile(cost: float, sorted_costs: np.ndarray) -> float:
    """Fraction of the lattice that is strictly better than ``cost``.

    Scale-free, unlike a relative gap: simulator 2 has an optimum of order 1e-4,
    so (cost - opt) / opt is dominated by that near-zero denominator.
    0 means the global optimum was found.
    """
    return np.searchsorted(sorted_costs, cost, side="left") / len(sorted_costs)


if __name__ == "__main__":
    opt = global_optima()
    os.makedirs("results", exist_ok=True)
    with open("results/ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["simulator", "optimal_cost", "x", "y", "opt", "nbth", "disk",
                    "median_cost", "cost_1pct_quantile"])
        for k, (c, S, costs) in opt.items():
            row = [k, f"{c:.6g}", *map(int, S), f"{np.median(costs):.4f}",
                   f"{np.quantile(costs, 0.01):.4f}"]
            w.writerow(row)
            print(f"sim {k}: t* = {c:.6g} at S* = {tuple(map(int, S))}, "
                  f"median over lattice = {np.median(costs):.3f}")
