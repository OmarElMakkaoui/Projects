"""Sequential version of run_mpi.py (one process, no MPI dependency).

    python run_local.py --method TabuSA --sim 1 --max-evals 300

``--nprocs P`` runs the P searches that ``mpirun -n P run_mpi.py`` would run,
one after the other, with the same seeds, so the two drivers return the same
best point. It is used to check the MPI driver and to run the multi-start
experiments on a machine with few cores.
"""

import argparse
import time

import numpy as np

import heuristics as hrst
import simulator as sim
import tools

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--nprocs", type=int, default=1)
known, rest = pre.parse_known_args()
args = tools.cmdLineParsing(rest)
NbP = known.nprocs
evals_here = args.max_evals // NbP if args.budget == "total" else args.max_evals

t0 = time.perf_counter()
sim.Reset()
nd = hrst.GetNbDim()
EbTab, SbTab, S0Tab, EvalTab = [], [], [], []
for Me in range(NbP):
    S0 = hrst.generateS0(args.seed_inc, Me)
    res = hrst.localSearch(args.method, S0, args.sim, evals_here,
                           seed=tools.run_seed(args.seed_inc, Me), delay=args.delay,
                           **tools.method_params(args))
    EbTab.append(res.best_cost)
    SbTab.append(res.best_S)
    S0Tab.append(S0)
    EvalTab.append(res.n_evals)

tools.printResults(np.array(EbTab), np.array(SbTab), np.array(S0Tab), np.array(EvalTab),
                   sim.GetNbSim(), nd, NbP, args, time.perf_counter() - t0)
