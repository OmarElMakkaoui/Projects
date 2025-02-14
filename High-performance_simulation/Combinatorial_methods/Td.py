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

import heuristique as hrst
import simulator as sim
import tools



#--------------------------------------------------------------------------
# Main code
#--------------------------------------------------------------------------

#Simulation of MPI information extraction
NbP = 1
Me  = 0

#Command line parsing: 
IterMax, Method, SeedInc, T0, la, ltl, simIdx = tools.cmdLineParsing(Me)

#Process 0 prints a "hello msg"
if Me == 0:
  print("PE: ", Me, "/",NbP,": all processes started")

#Each process runs a local search method from a random starting point
sim.Reset()
S0 = hrst.generateS0(SeedInc,Me)
eb,Sb,nbIt = hrst.localSearch(Method,S0,IterMax,T0,la,ltl,simIdx)

#Simulation of data gathering and redcution
#- Process 0 (root) gathers results (Eb, Sb), Starting points (S0) and iter nb (Iter)
Eb = np.array([eb],dtype=np.float64)
EbTab = Eb

SbTab = Sb
S0Tab = S0

Iter = np.array([nbIt],dtype=np.int32)
IterTab = Iter
#- Process 0 (root) reduce the number of simulations
totalNbSim = sim.GetNbSim()

#Print results
if Me == 0:
  nd = hrst.NbDim
  tools.printResults(EbTab,SbTab,S0Tab,IterTab,totalNbSim,nd,Me,NbP)

#Process 0 prints a "good bye msg"
if Me == 0:
  print("PE: ", Me, "/",NbP," bye!")

