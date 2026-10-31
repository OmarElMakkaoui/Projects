"""Unit tests.  Run from the project folder:  python -m pytest -q tests"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bayesopt  # noqa: E402
import ground_truth as gt  # noqa: E402
import heuristics as hrst  # noqa: E402
import simulator as sim  # noqa: E402


def test_vectorised_simulator_matches_course_code():
    rng = np.random.default_rng(1)
    S = hrst.random_points(rng, 500)
    for k in range(3):
        scalar = [sim.CostFunction(s, k) for s in S]
        assert np.allclose(scalar, gt.cost_vectorised(S, k), rtol=1e-12)


def test_reset_resets_counter():
    sim.CostFunction([50, 50, 0, 1, 0], 0)
    sim.Reset()
    assert sim.GetNbSim() == 0


def test_neighbourhood_stays_in_box():
    for S in ([10, 10, 0, 1, 0], [120, 150, 4, 8, 1], [60, 80, 2, 4, 0]):
        for N in hrst.Neighborhood(np.array(S)):
            assert np.all(N >= hrst.LMIN) and np.all(N <= hrst.LMAX)
            assert np.abs(N - S).sum() == 1
    assert len(hrst.Neighborhood(np.array([60, 80, 2, 4, 0]))) == 9  # disk is binary
    assert len(hrst.Neighborhood(np.array([10, 10, 0, 1, 0]))) == 5


@pytest.mark.parametrize("method", list(hrst.METHODS))
def test_budget_is_respected_and_result_consistent(method):
    S0 = hrst.generateS0(3, 0)
    sim.Reset()
    res = hrst.localSearch(method, S0, 1, 40, seed=0)
    assert res.n_evals <= 40
    assert sim.GetNbSim() == res.n_evals
    assert np.isclose(sim.CostFunction(res.best_S, 1), res.best_cost)
    assert res.trace == sorted(res.trace, reverse=True)  # best-so-far is monotone


def test_greedy_hc_counts_iterations_and_reaches_local_optimum():
    res = hrst.localSearch("GHC", hrst.generateS0(0, 0), 0, 100_000, seed=0)
    assert res.stopped_by == "local_optimum" and res.n_iter > 0
    for N in hrst.Neighborhood(res.best_S):
        assert sim.CostFunction(N, 0) >= res.best_cost


def test_gp_gradient_matches_finite_differences():
    from scipy.optimize import approx_fprime
    rng = np.random.default_rng(0)
    X = bayesopt._scale(hrst.random_points(rng, 30))
    y = rng.normal(size=30)
    theta = np.array([-1.0, -0.5, -1.5, 0.2, -0.8, 0.3, np.log(1e-2)])
    g = bayesopt._neg_log_marglik(theta, X, y)[1]
    g_fd = approx_fprime(theta, lambda t: bayesopt._neg_log_marglik(t, X, y)[0], 1e-6)
    assert np.allclose(g, g_fd, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="MPI not installed")
def test_mpi_driver_matches_sequential_driver():
    common = ["--method", "SA", "--sim", "2", "--max-evals", "150", "--seed-inc", "4", "--json"]
    mpi = ["mpirun", "--oversubscribe", "-n", "4"]
    if os.geteuid() == 0:
        mpi.append("--allow-run-as-root")
    a = json.loads(subprocess.run(mpi + [sys.executable, "run_mpi.py"] + common, cwd=ROOT,
                                  check=True, capture_output=True, text=True).stdout.splitlines()[-1])
    b = json.loads(subprocess.run([sys.executable, "run_local.py", "--nprocs", "4"] + common,
                                  cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    for key in ("best_cost", "best_S", "best_rank", "total_evals"):
        assert a[key] == b[key]
