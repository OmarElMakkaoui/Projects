"""Parallel multi-start local search with MPI.

Every process runs an independent search from its own random start point;
rank 0 gathers each process's best point, starting point and evaluation count
(``Gather``), sums the simulator calls (``reduce``) and keeps the overall best.
The searches never communicate while running, so the only synchronisation is
the final collective.

    mpirun -n 8 python run_mpi.py --method SA --sim 0 --max-evals 300
    mpirun -n 8 python run_mpi.py --method SA --budget total --max-evals 2400 --delay 0.002
"""


import numpy as np
from mpi4py import MPI

import heuristics as hrst
import simulator as sim
import tools

comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

args = tools.cmdLineParsing()
evals_here = args.max_evals // NbP if args.budget == "total" else args.max_evals

comm.Barrier()
t0 = MPI.Wtime()

sim.Reset()
S0 = hrst.generateS0(args.seed_inc, Me)
res = hrst.localSearch(args.method, S0, args.sim, evals_here,
                       seed=tools.run_seed(args.seed_inc, Me), delay=args.delay,
                       **tools.method_params(args))

# Gather per-process results on rank 0
nd = hrst.GetNbDim()
if Me == 0:
    EbTab = np.empty(NbP, dtype=np.float64)
    SbTab = np.empty(NbP * nd, dtype=np.int64)
    S0Tab = np.empty(NbP * nd, dtype=np.int64)
    EvalTab = np.empty(NbP, dtype=np.int64)
else:
    EbTab = SbTab = S0Tab = EvalTab = None

comm.Gather(np.array([res.best_cost], dtype=np.float64), EbTab, root=0)
comm.Gather(np.ascontiguousarray(res.best_S, dtype=np.int64), SbTab, root=0)
comm.Gather(np.ascontiguousarray(S0, dtype=np.int64), S0Tab, root=0)
comm.Gather(np.array([res.n_evals], dtype=np.int64), EvalTab, root=0)

# Reduce the number of simulator calls across processes
totalNbSim = comm.reduce(sim.GetNbSim(), op=MPI.SUM, root=0)

wall = MPI.Wtime() - t0
wall = comm.reduce(wall, op=MPI.MAX, root=0)

if Me == 0:
    tools.printResults(EbTab, SbTab, S0Tab, EvalTab, totalNbSim, nd, NbP, args, wall)
