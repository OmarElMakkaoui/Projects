"""Local-search metaheuristics for tuning the parameters of a simulated HPC code.

Search space (integer lattice, 1,252,440 points):

    S = (x, y, opt, nbth, disk)
    x in [10, 120], y in [10, 150], opt in [0, 4], nbth in [1, 8], disk in [0, 1]

The objective is the estimated execution time returned by ``simulator.CostFunction``.
Each simulator call is treated as expensive, so every method runs under the same
budget of *simulator evaluations* (not iterations): greedy hill climbing spends up
to 9 evaluations per iteration, simulated annealing spends 1, so counting
iterations would make the comparison meaningless.

All methods share the signature ``method(S0, evaluator, rng, **params)`` and
return a :class:`SearchResult`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import simulator as sim

NB_DIM = 5
LMIN = np.array([10, 10, 0, 1, 0], dtype=np.int64)
LMAX = np.array([120, 150, 4, 8, 1], dtype=np.int64)
SPACE_SIZE = int(np.prod(LMAX - LMIN + 1))


def GetNbDim():
    return NB_DIM


# ---------------------------------------------------------------------------
# Budgeted evaluator
# ---------------------------------------------------------------------------

class BudgetExhausted(Exception):
    """Raised when a method asks for more evaluations than it was given."""


@dataclass
class Evaluator:
    """Wraps the simulator: counts calls, tracks the incumbent, enforces a budget.

    ``delay`` (seconds) emulates the wall-clock cost of a real simulation run and
    is only used by the MPI scaling experiment.
    """

    sim_idx: int
    max_evals: int
    delay: float = 0.0
    n_evals: int = 0
    best_cost: float = math.inf
    best_S: np.ndarray | None = None
    trace: list = field(default_factory=list)  # best-so-far after each evaluation

    def __call__(self, S) -> float:
        if self.n_evals >= self.max_evals:
            raise BudgetExhausted
        if self.delay > 0:
            time.sleep(self.delay)
        cost = float(sim.CostFunction(S, self.sim_idx))
        self.n_evals += 1
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_S = np.array(S, dtype=np.int64)
        self.trace.append(self.best_cost)
        return cost

    @property
    def remaining(self) -> int:
        return self.max_evals - self.n_evals


@dataclass
class SearchResult:
    best_cost: float
    best_S: np.ndarray
    n_evals: int
    n_iter: int
    trace: list
    stopped_by: str  # "budget", "local_optimum", "frozen" or "no_admissible_move"


def _result(ev: Evaluator, n_iter: int, stopped_by: str) -> SearchResult:
    return SearchResult(ev.best_cost, ev.best_S, ev.n_evals, n_iter, list(ev.trace), stopped_by)


# ---------------------------------------------------------------------------
# Search-space utilities
# ---------------------------------------------------------------------------

def generateS0(seed_inc: int, rank: int) -> np.ndarray:
    """Random starting point, reproducible per (seed_inc, MPI rank)."""
    rng = np.random.default_rng(rank + 1000 * (1 + seed_inc))
    return rng.integers(LMIN, LMAX + 1).astype(np.int64)


def random_points(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(LMIN, LMAX + 1, size=(n, NB_DIM)).astype(np.int64)


def Neighborhood(S) -> list:
    """Unit moves along each coordinate that stay inside the box.

    At most 9: two per coordinate, except ``disk``, which is binary.
    """
    neighbors = []
    for i in range(NB_DIM):
        for step in (1, -1):
            Sn = np.array(S, dtype=np.int64)
            Sn[i] += step
            if LMIN[i] <= Sn[i] <= LMAX[i]:
                neighbors.append(Sn)
    return neighbors


def isInsideTabuL(S, tabu: list) -> bool:
    return any(np.array_equal(S, T) for T in tabu)


# ---------------------------------------------------------------------------
# Local searches
# ---------------------------------------------------------------------------

def HillClimbing(S0, ev: Evaluator, rng: np.random.Generator, **_) -> SearchResult:
    """Stochastic first-improvement hill climbing.

    Neighbours of the current point are drawn without replacement; the first one
    that improves is accepted. Stops at a local optimum (all neighbours tried).
    """
    it = 0
    try:
        S, e = np.array(S0), ev(S0)
        candidates = Neighborhood(S)
        while candidates:
            Sk = candidates.pop(rng.integers(len(candidates)))
            ek = ev(Sk)
            it += 1
            if ek < e:
                S, e = Sk, ek
                candidates = Neighborhood(S)
    except BudgetExhausted:
        return _result(ev, it, "budget")
    return _result(ev, it, "local_optimum")


def GreedyHC(S0, ev: Evaluator, rng: np.random.Generator, **_) -> SearchResult:
    """Steepest-descent hill climbing: evaluate the whole neighbourhood, take the best."""
    it = 0
    try:
        S, e = np.array(S0), ev(S0)
        while True:
            best_Sk, best_ek = None, math.inf
            for Sk in Neighborhood(S):
                ek = ev(Sk)
                if ek < best_ek:
                    best_Sk, best_ek = Sk, ek
            it += 1
            if best_ek >= e:
                return _result(ev, it, "local_optimum")
            S, e = best_Sk, best_ek
    except BudgetExhausted:
        return _result(ev, it, "budget")


def SimulatedAnnealing(S0, ev: Evaluator, rng: np.random.Generator,
                       T0: float = 10.0, la: float = 0.995, **_) -> SearchResult:
    """Metropolis acceptance on a random unit move, geometric cooling T <- la*T."""
    it = 0
    T = T0
    try:
        S, e = np.array(S0), ev(S0)
        while T > 1e-8:
            neighbors = Neighborhood(S)
            Sk = neighbors[rng.integers(len(neighbors))]
            ek = ev(Sk)
            if ek < e or rng.random() < math.exp((e - ek) / T):
                S, e = Sk, ek
            T *= la
            it += 1
    except BudgetExhausted:
        return _result(ev, it, "budget")
    return _result(ev, it, "frozen")


def TabuSA(S0, ev: Evaluator, rng: np.random.Generator,
           T0: float = 10.0, la: float = 0.995, ltl: int = 20, **_) -> SearchResult:
    """Simulated annealing with a FIFO tabu list of the last ``ltl`` visited states.

    Moves towards a recently visited state are forbidden, which stops the walk
    from oscillating between two neighbours at low temperature.
    """
    it = 0
    T = T0
    tabu: list = []
    try:
        S, e = np.array(S0), ev(S0)
        tabu.append(S)
        while T > 1e-8:
            admissible = [Sk for Sk in Neighborhood(S) if not isInsideTabuL(Sk, tabu)]
            if not admissible:
                return _result(ev, it, "no_admissible_move")
            Sk = admissible[rng.integers(len(admissible))]
            ek = ev(Sk)
            if ek < e or rng.random() < math.exp((e - ek) / T):
                S, e = Sk, ek
                tabu.append(S)
                if len(tabu) > ltl:
                    tabu.pop(0)
            T *= la
            it += 1
    except BudgetExhausted:
        return _result(ev, it, "budget")
    return _result(ev, it, "frozen")


def BayesianOptimization(S0, ev: Evaluator, rng: np.random.Generator,
                         n_init: int = 10, refit_every: int = 10, **_) -> SearchResult:
    """Gaussian-process Bayesian optimisation; see ``bayesopt.py``."""
    from bayesopt import bayesian_optimization
    it = bayesian_optimization(S0, ev, rng, n_init=n_init, refit_every=refit_every)
    return _result(ev, it, "budget")


METHODS = {
    "HC": HillClimbing,
    "GHC": GreedyHC,
    "SA": SimulatedAnnealing,
    "TabuSA": TabuSA,
    "BO": BayesianOptimization,
}


def localSearch(method: str, S0, sim_idx: int, max_evals: int, seed: int,
                delay: float = 0.0, **params) -> SearchResult:
    """Run one search from ``S0`` with a fresh evaluator and RNG."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    ev = Evaluator(sim_idx=sim_idx, max_evals=max_evals, delay=delay)
    rng = np.random.default_rng(seed)
    return METHODS[method](np.asarray(S0, dtype=np.int64), ev, rng, **params)
