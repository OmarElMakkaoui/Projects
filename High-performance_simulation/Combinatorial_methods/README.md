Heuristic.py: Implements heuristic optimization algorithms for solving three optimization problems:
-Hill Climbing (HC)
-Greedy Hill Climbing (GHC)
-Simulated Annealing (SA)
-Tabu Search with Simulated Annealing (TabuSA)
-Bayesian Optimization (BO)

simulator.py: Contains three cost functions that model different optimization problems:
Takes five input parameters and computes an estimated execution time.
Provides an interface for evaluating solutions.

Td.py: Executes a local search optimization using a single process:
Calls Heuristic.py.
Uses simulator.py 

tdmpi.py: Parallelized version of Td.py using MPI.

