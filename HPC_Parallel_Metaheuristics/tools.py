"""Command-line parsing and result printing shared by run_local.py and run_mpi.py."""

from __future__ import annotations

import argparse
import json

import numpy as np

from heuristics import METHODS


def cmdLineParsing(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-start local search on the HPC simulator")
    p.add_argument("--method", choices=sorted(METHODS), default="SA")
    p.add_argument("--sim", type=int, choices=[0, 1, 2], default=0, help="simulator index")
    p.add_argument("--max-evals", type=int, default=300,
                   help="simulator evaluations per process (or in total, see --budget)")
    p.add_argument("--budget", choices=["per-rank", "total"], default="per-rank",
                   help="'total' splits --max-evals across processes (strong scaling)")
    p.add_argument("--seed-inc", type=int, default=0, help="shifts every process's seed")
    p.add_argument("--T0", type=float, default=10.0, help="initial temperature (SA, TabuSA)")
    p.add_argument("--la", type=float, default=0.995, help="geometric cooling factor (SA, TabuSA)")
    p.add_argument("--ltl", type=int, default=20, help="tabu list length (TabuSA)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds slept per evaluation, to emulate an expensive simulation")
    p.add_argument("--json", action="store_true", help="print one JSON line instead of a table")
    return p.parse_args(argv)


def method_params(args) -> dict:
    return {"T0": args.T0, "la": args.la, "ltl": args.ltl}


def run_seed(seed_inc: int, rank: int) -> int:
    """RNG seed of the search itself (the start point uses heuristics.generateS0)."""
    return 7 + rank + 1000 * (1 + seed_inc)


def printResults(EbTab, SbTab, S0Tab, EvalTab, totalNbSim, nd, NbP, args, wall):
    SbTab = np.asarray(SbTab).reshape(NbP, nd).tolist()
    S0Tab = np.asarray(S0Tab).reshape(NbP, nd).tolist()
    best = int(np.argmin(EbTab))
    if args.json:
        print(json.dumps({
            "method": args.method, "sim": args.sim, "nprocs": NbP, "budget": args.budget,
            "max_evals": args.max_evals, "delay": args.delay, "seed_inc": args.seed_inc,
            "best_cost": float(EbTab[best]), "best_S": SbTab[best],
            "best_rank": best, "total_evals": int(totalNbSim), "wall_time_s": wall,
        }))
        return
    print(f"\n{args.method} on simulator {args.sim}, {NbP} process(es)")
    print(f"{'rank':>4}  {'start point':>24}  {'best point':>24}  {'cost':>10}  {'evals':>6}")
    for r in range(NbP):
        print(f"{r:>4}  {str(tuple(S0Tab[r])):>24}  {str(tuple(SbTab[r])):>24}  "
              f"{EbTab[r]:>10.5g}  {EvalTab[r]:>6}")
    print(f"\nbest: {EbTab[best]:.6g} at {tuple(SbTab[best])} (rank {best})")
    print(f"total simulator calls: {totalNbSim}   wall time: {wall:.3f} s")
