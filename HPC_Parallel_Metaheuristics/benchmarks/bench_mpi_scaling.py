"""Strong-scaling wall-clock measurement of run_mpi.py.

A fixed total budget of 1,600 simulator calls is split across P MPI processes.
The analytic cost functions take microseconds, so each call also sleeps
``DELAY`` seconds to stand in for a real simulation run; without it the
measurement would only time Python start-up and MPI initialisation.

Wall time is measured inside run_mpi.py, from a barrier after MPI start-up to
the final reduction, so it covers the searches, their Python bookkeeping and
the Gather/Reduce, but not process launch. Because the simulated cost is a
sleep, running more processes than CPU cores is still meaningful; the gap to
ideal speedup is the real overhead. Each P is run 3 times; the median is kept.

    python benchmarks/bench_mpi_scaling.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

TOTAL_EVALS = 1600
DELAY = 0.005  # seconds per simulated run
P_VALUES = [1, 2, 4, 8, 16]
REPEATS = 3


def mpirun(P: int) -> dict:
    cmd = ["mpirun", "--oversubscribe", "-n", str(P)]
    if os.geteuid() == 0:
        cmd.append("--allow-run-as-root")
    cmd += [sys.executable, "run_mpi.py", "--method", "SA", "--sim", "0",
            "--budget", "total", "--max-evals", str(TOTAL_EVALS),
            "--delay", str(DELAY), "--json"]
    out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for P in P_VALUES:
        runs = [mpirun(P) for _ in range(REPEATS)]
        assert all(r["total_evals"] == (TOTAL_EVALS // P) * P for r in runs)
        rows.append({"P": P, "evals_per_rank": TOTAL_EVALS // P,
                     "wall_time_s": statistics.median(r["wall_time_s"] for r in runs)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["ideal_time_s"] = TOTAL_EVALS * DELAY / df["P"]
    df["speedup"] = df["wall_time_s"].iloc[0] / df["wall_time_s"]
    df["efficiency"] = df["speedup"] / df["P"]
    df["cpu_cores"] = os.cpu_count()
    df.to_csv(os.path.join(RESULTS, "mpi_scaling.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
