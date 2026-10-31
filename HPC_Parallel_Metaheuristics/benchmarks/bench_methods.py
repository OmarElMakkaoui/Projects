"""Compare the five methods on the three simulators at a fixed evaluation budget.

Protocol
--------
1. Tuning: SA and TabuSA hyperparameters (T0, cooling factor) are chosen per
   simulator by grid search on 20 *tuning* starts (seeds 1000-1019), using the
   mean rank quantile as criterion.
2. Evaluation: every method runs from the same 30 *test* starts (seeds 0-29,
   disjoint from the tuning seeds) with a budget of 300 simulator calls.
3. Quality is measured against the exact optimum from ground_truth.py:
     - hit rate       : fraction of runs that return the global optimum
     - rank quantile  : fraction of the 1,252,440 configurations strictly better
                        than the returned one (0 = optimum), scale-free across
                        simulators.

    python benchmarks/bench_methods.py            # ~20 min on 2 cores (BO dominates)
"""

from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import heuristics as hrst  # noqa: E402
from ground_truth import global_optima, rank_quantile  # noqa: E402

BUDGET = 300
TEST_SEEDS = range(30)
TUNE_SEEDS = range(1000, 1020)
CHECKPOINTS = [25, 50, 100, 150, 200, 250, 300]
T0_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
LA_GRID = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
METHODS = ["HC", "GHC", "SA", "TabuSA", "BO"]
RESULTS = os.path.join(ROOT, "results")

OPT = global_optima()


def one_run(job):
    method, sim_idx, seed, params = job
    S0 = hrst.generateS0(seed, 0)
    res = hrst.localSearch(method, S0, sim_idx, BUDGET, seed=10_000 + seed, **params)
    opt_cost, _, sorted_costs = OPT[sim_idx]
    # best-so-far is flat once a method stops early (local optimum)
    trace = res.trace + [res.trace[-1]] * (BUDGET - len(res.trace))
    row = {
        "method": method, "sim": sim_idx, "seed": seed, **params,
        "best_cost": res.best_cost, "best_S": tuple(int(v) for v in res.best_S),
        "hit": bool(np.isclose(res.best_cost, opt_cost, rtol=1e-12, atol=0)),
        "rank_quantile": rank_quantile(res.best_cost, sorted_costs),
        "evals_used": res.n_evals, "stopped_by": res.stopped_by,
    }
    for c in CHECKPOINTS:
        row[f"rq@{c}"] = rank_quantile(trace[c - 1], sorted_costs)
    return row


def tune(pool):
    jobs = [(m, k, s, {"T0": T0, "la": la})
            for m in ("SA", "TabuSA") for k in range(3)
            for T0, la in itertools.product(T0_GRID, LA_GRID) for s in TUNE_SEEDS]
    df = pd.DataFrame(pool.map(one_run, jobs, chunksize=20))
    table = (df.groupby(["method", "sim", "T0", "la"])["rank_quantile"].mean()
               .rename("mean_rank_quantile").reset_index())
    table.to_csv(os.path.join(RESULTS, "tuning_sa.csv"), index=False)
    best = table.loc[table.groupby(["method", "sim"])["mean_rank_quantile"].idxmin()]
    return {(r.method, r.sim): {"T0": r.T0, "la": r.la} for r in best.itertuples()}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with Pool(os.cpu_count()) as pool:
        tuned = tune(pool)
        for (m, k), p in sorted(tuned.items()):
            print(f"tuned {m:6s} sim {k}: T0={p['T0']}, la={p['la']}")
        jobs = [(m, k, s, tuned.get((m, k), {}))
                for k in range(3) for m in METHODS for s in TEST_SEEDS]
        # BO jobs first so the slow ones do not end up alone at the tail
        jobs.sort(key=lambda j: j[0] != "BO")
        runs = pd.DataFrame(pool.map(one_run, jobs, chunksize=1))
    runs.to_csv(os.path.join(RESULTS, "methods_runs.csv"), index=False)

    summary = (runs.groupby(["sim", "method"])
                   .agg(hit_rate=("hit", "mean"),
                        median_rank_quantile=("rank_quantile", "median"),
                        mean_rank_quantile=("rank_quantile", "mean"),
                        median_best_cost=("best_cost", "median"),
                        mean_evals_used=("evals_used", "mean"))
                   .reset_index())
    summary["method"] = pd.Categorical(summary["method"], METHODS)
    summary = summary.sort_values(["sim", "method"])
    summary.to_csv(os.path.join(RESULTS, "methods_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
