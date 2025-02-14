import os
import sys
import time
import numpy as np
import math
import random as rd
import simulator as sim
import tensorflow as tf
import tensorflow_probability as tfp


NbDim = 5
lmin = [10, 10, 0, 1, 0]
lmax = [120, 150, 4, 8, 1]

def GetNbDim():
    return NbDim

def generateS0(SeedInc, Me):
    rd.seed(Me + 1000 * (1 + SeedInc))
    S0 = np.empty(NbDim, dtype=np.int32)
    for i in range(NbDim):
        S0[i] = rd.randrange(lmin[i], lmax[i] + 1, 1)
    return S0

def Neighborhood(S):
    LNgbh = []
    for i in range(NbDim):
        S1 = S.copy()
        S1[i] += 1
        if S1[i] <= lmax[i]:
            LNgbh.append(S1)
        S2 = S.copy()
        S2[i] -= 1
        if S2[i] >= lmin[i]:
            LNgbh.append(S2)
    return LNgbh

def HillClimbing(S0, IterMax, T0, la, ltl, simIdx):
    Sb = S0
    eb = sim.CostFunction(Sb, simIdx)
    iter = 0
    S = Sb
    e = eb
    LNgbh = Neighborhood(S)
    while iter < IterMax and LNgbh:
        k = rd.randrange(len(LNgbh))
        Sk = LNgbh.pop(k)
        ek = sim.CostFunction(Sk, simIdx)
        if ek < eb:
            Sb = Sk
            eb = ek
            LNgbh = Neighborhood(Sk)
        iter += 1
    return eb, Sb, iter

def GreedyHC(S0, IterMax, T0, la, ltl, simIdx):
    Sb = S0
    eb = sim.CostFunction(Sb, simIdx)
    iter = 0
    while iter < IterMax:
        LNgbh = Neighborhood(Sb)
        if not LNgbh:
            break
        best_neighbor = None
        best_energy = float('inf')
        for Sk in LNgbh:
            ek = sim.CostFunction(Sk, simIdx)
            if ek < best_energy:
                best_energy = ek
                best_neighbor = Sk
        if best_energy < eb:
            Sb = best_neighbor
            eb = best_energy
        else:
            break
    return eb, Sb, iter

def SimulatedAnnealing(S0, IterMax, T0, la, ltl, simIdx):
    Sb = S0
    eb = sim.CostFunction(Sb, simIdx)
    S = Sb
    e = eb
    T = T0
    iter = 0
    while iter < IterMax and T > 1e-8:
        LNgbh = Neighborhood(S)
        if not LNgbh:
            break
        Sk = rd.choice(LNgbh)
        ek = sim.CostFunction(Sk, simIdx)
        if ek < e or rd.random() < math.exp((e - ek) / T):
            S = Sk
            e = ek
            if e < eb:
                Sb = S
                eb = e
        T = la * T
        iter += 1
    return eb, Sb, iter

def TabuSA(S0, IterMax, T0, la, ltl, simIdx): 
    Sb = S0  
    eb = sim.CostFunction(Sb, simIdx)  
    S = Sb  
    e = eb  
    T = T0  
    iter = 0  

    TabuList = []  

    while iter < IterMax and T > 1e-8:  
        LNgbh = Neighborhood(S)  
        if not LNgbh:
            break  

        LNgbh = [Sk for Sk in LNgbh if not isInsideTabuL(Sk, TabuList)]
        if not LNgbh:
            break  

        Sk = rd.choice(LNgbh)  
        ek = sim.CostFunction(Sk, simIdx)  

        if ek < e or rd.random() < math.exp((e - ek) / T):
            S = Sk  
            e = ek

            if e < eb:
                Sb = S
                eb = e

        TabuList.append(S)  
        if len(TabuList) > ltl:
            TabuList.pop(0)

        T = la * T
        iter += 1

    return eb, Sb, iter


tfb = tfp.bijectors
tfd = tfp.distributions

def BayesianOptimization(S0, IterMax, simIdx):
    bounds = np.array([[10, 120], [10, 150], [0, 4], [1, 8], [0, 1]])
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(5, len(S0)))
    Y = np.array([sim.CostFunction(x, simIdx) for x in X])

    for _ in range(IterMax):
        gp = tfd.GaussianProcessRegressionModel(kernel_provider=tfd.MaternFiveHalves(),index_points=X,observations=Y)
        x_next = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1, len(S0)))
        y_next = sim.CostFunction(x_next[0], simIdx)
        X = np.vstack([X, x_next])
        Y = np.append(Y, y_next)

    best_idx = np.argmin(Y)
    return Y[best_idx], X[best_idx], IterMax