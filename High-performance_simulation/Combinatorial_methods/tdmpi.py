##########################################################################
# Local search method
# Distributed code
##########################################################################

import os
import sys
import time
import numpy as np
import math
import random as rd
import argparse
import mpi4py
from mpi4py import MPI

import heuristique as hrst
import simulator as sim
import tools



#--------------------------------------------------------------------------
# Main code
#--------------------------------------------------------------------------

#MPI information extraction
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me  = comm.Get_rank()

#Command line parsing: 
IterMax, Method, SeedInc, T0, la, lTl, simIdx = tools.cmdLineParsing(Me)

#Process 0 prints a "hello msg"
comm.barrier()
if Me == 0:
  print("PE: ", Me, "/",NbP,": all processes started")

#Each process runs a local search method from a random starting point
sim.Reset()
S0 = hrst.generateS0(SeedInc,Me)
eb,Sb,iter = hrst.localSearch(Method,S0,IterMax,T0,la,lTl,simIdx)

#Process 0 (root) gathers results (Eb, Sb), Starting points (S0) and iter nb (Iter)
# - Allocate real numpy arrays only on 0 (root) process
if Me == 0:        
  nd = hrst.GetNbDim()
  EbTab = np.zeros(NbP*1,dtype=np.float64)
  SbTab = np.zeros(NbP*nd,dtype=np.int32)
  S0Tab = np.zeros(NbP*nd,dtype=np.int32)
  IterTab = np.zeros(NbP*1,dtype=np.int32)
else :
  EbTab   = None     
  SbTab   = None
  S0Tab   = None
  IterTab = None
  totalNbSim = None

# - Gather all Eb into EbTAB, all Sb into SbTab, all S0 into S0Tab      TO DO
#
#EbTab = Eb           # Replace these lines with real gather op
#SbTab = Sb
#S0Tab = S0
#IterTab = Iter

Eb = np.array([eb],dtype=np.float64)
comm.Gather(Eb,EbTab,root=0)

comm.Gather(Sb,SbTab,root=0)
comm.Gather(S0,S0Tab,root=0)

Iter = np.array([iter],dtype=np.int32)
comm.Gather(Iter,IterTab,root=0)

# - Reduce all nbsim into totalNbSim                                    TO DO
#
#totalNbSim = sim.GetNbSim()          # Replace this line with real reduction
totalNbSim = comm.reduce(sim.GetNbSim(),op=MPI.SUM,root=0)

#Print results
if Me == 0:
  nd = hrst.GetNbDim()
  tools.printResults(EbTab,SbTab,S0Tab,IterTab,totalNbSim,nd,Me,NbP)

#Process 0 prints a "good bye msg"
comm.barrier()
time.sleep(1)
if Me == 0:
  print("PE: ", Me, "/",NbP," bye!")