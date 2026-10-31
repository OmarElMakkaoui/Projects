"""What does parallel multi-start buy in solution quality?

Each MPI process runs an independent search, and the answer is the best of the P
results. This script reproduces exactly the searches ``mpirun -n P run_mpi.py``
performs (same start points and RNG seeds per rank; checked in
tests/test_core.py), but runs them in a process pool so that 30 repetitions per
setting are affordable on a small machine.

Two regimes:
  weak   : every process gets 300 evaluations; total work grows with P.
  strong : a fixed total of 4,800 evaluations is split across the P processes
           (P = 1 is one long search, P = 32 is 32 searches of 150 evaluations).

Requires results/tuning_sa.csv from bench_methods.py (tuned SA/TabuSA settings).

    python benchmarks/bench_multistart.py
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import heuristics as hrst  # noqa: E402
import tools  # noqa: E402
from ground_truth import global_optima, rank_quantile  # noqa: E402

METHODS = ["HC", "GHC", "SA", "TabuSA"]
P_VALUES = [1, 2, 4, 8, 16, 32]
REPS = range(30)
WEAK_BUDGET = 300
STRONG_TOTAL = 4800
RESULTS = os.path.join(ROOT, "results")

OPT = global_optima()


def tuned_params() -> dict:
    t = pd.read_csv(os.path.join(RESULTS, "tuning_sa.csv"))
    best = t.loc[t.groupby(["method", "sim"])["mean_rank_quantile"].idxmin()]
    return {(r.method, r.sim): {"T0": r.T0, "la": r.la} for r in best.itertuples()}


TUNED = tuned_params()


def rank_job(job):
    """One MPI rank's search, exactly as run_mpi.py would run it."""
    method, sim_idx, rep, rank, budget = job
    res = hrst.localSearch(method, hrst.generateS0(rep, rank), sim_idx, budget,
                           seed=tools.run_seed(rep, rank), **TUNED.get((method, sim_idx), {}))
    return method, sim_idx, rep, rank, budget, res.best_cost


def summarise(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    rows = []
    for (m, k, P), g in df.groupby(["method", "sim", "P"]):
        opt_cost, _, sorted_costs = OPT[k]
        best = g.groupby("rep")["best_of_P"].first()
        rows.append({
            "regime": regime, "method": m, "sim": k, "P": P,
            "evals_per_rank": int(g["budget"].iloc[0]),
            "hit_rate": float(np.isclose(best, opt_cost, rtol=1e-12, atol=0).mean()),
            "median_rank_quantile": float(np.median([rank_quantile(c, sorted_costs) for c in best])),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    weak_jobs = [(m, k, r, rank, WEAK_BUDGET) for m in METHODS for k in range(3)
                 for r in REPS for rank in range(max(P_VALUES))]
    strong_jobs = [(m, k, r, rank, STRONG_TOTAL // P) for m in METHODS for k in range(3)
                   for r in REPS for P in P_VALUES for rank in range(P)]
    with Pool(os.cpu_count()) as pool:
        weak = pd.DataFrame(pool.map(rank_job, weak_jobs, chunksize=50),
                            columns=["method", "sim", "rep", "rank", "budget", "cost"])
        strong = pd.DataFrame(pool.map(rank_job, strong_jobs, chunksize=50),
                              columns=["method", "sim", "rep", "rank", "budget", "cost"])

    # weak: ranks 0..P-1 of the 32-rank run are exactly the P-rank run
    parts = []
    for P in P_VALUES:
        sub = weak[weak["rank"] < P].copy()
        sub["P"] = P
        sub["best_of_P"] = sub.groupby(["method", "sim", "rep"])["cost"].transform("min")
        parts.append(sub)
    weak_s = summarise(pd.concat(parts), "weak")

    strong["P"] = STRONG_TOTAL // strong["budget"]
    strong["best_of_P"] = strong.groupby(["method", "sim", "rep", "P"])["cost"].transform("min")
    strong_s = summarise(strong, "strong")

    out = pd.concat([weak_s, strong_s], ignore_index=True)
    out.to_csv(os.path.join(RESULTS, "multistart.csv"), index=False)
    with pd.option_context("display.width", 140):
        print(out.pivot_table(index=["regime", "method", "sim"], columns="P",
                              values="hit_rate").round(2))


if __name__ == "__main__":
    main()
