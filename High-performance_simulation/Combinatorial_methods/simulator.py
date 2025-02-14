###################################################################
# Simulation of a HPC computation. Compute the estimated exec time
# in seconds.
# This code was provided by the course Professor.

import math
import numpy as np


#--------------------------------------------------------------------------
# Counter of simulation run (of the process)
#--------------------------------------------------------------------------
NbSim = 0

def Reset():
  NbSim = 0

def GetNbSim():
  return(NbSim)


#------------------------------------------------------------------
#x   : [10:120], step: 1
#y   : [10:150], step: 1
#opt : [0,1,2,3,4]
#nbth: [1,2,3,4,5,6,7,8]
#disk: [0,1]
#------------------------------------------------------------------

def CostFunction0(S):
   global NbSim
   x,y,opt,nbth,disk = S[0], S[1], S[2], S[3], S[4]
   # Exec Time
   if x < 10 or x > 120 or y < 10 or y > 150:
     print("Warning: x range: [10:120], y range: [10:150]")
   t = math.sin(1.5*(x/10.0 - 2))*math.sin(1.5*(x/10.0 - 2))*8 + \
       math.cos(1*(y/10.0 - 2))*math.cos(1*(y/10.0 - 2))*8 + \
       ((x/10.0 - 6)*(x/10.0 - 6)*2+(y/10.0 - 8)*(y/10.0 - 8)*1)*0.3 + 10
   #
   # Impact of the optimisation parameter
   if opt == 0:
     t = t*1.0
   elif opt == 1:
     t = t*0.8
   elif opt == 2:
     t = t*0.65
   elif opt == 3:
     t = t*0.55
   elif opt == 4:
     t = t*0.7
   else:
     t = t
     print("Ignored bad optimisation parameter value: " + str(opt))
   #
   # Impact of the thread number
   if nbth == 1:
     t = t*1.0
   elif nbth == 2:
     t = t*1.1
   elif nbth in [3,4]:
     t = (t*1.1)/((nbth-1)**0.8)
   elif nbth in [5,6,7,8]:
     t = (t*1.1/((4-2)**0.8))*(1.2**(nbth-4))
   else:
     t = t
     print("Ignored bat nbth parameter value: " + str(nbth))
   #
   # Impact of the disk parameter
   if disk == 0:
     t = t*1.0
   elif disk == 1:
     t = t*0.9
   else:
     t = t
     print("Ignored bad disk parameter value: " + str(disk))
   #
   # Increase counter of simulations
   NbSim += 1
   # Return the final t-exec (estimated)
   return(t)
   
#------------------------------------------------------------------

def CostFunction1(S):
   global NbSim
   x,y,opt,nbth,disk = S[0], S[1], S[2], S[3], S[4]
   # Exec Time
   if x < 10 or x > 120 or y < 10 or y > 150:
     print("Warning: x range: [10:120], y range: [10:150]")
   t = math.sin(1.5*(x/20.0 - 2))*math.sin(1.5*(x/20.0 - 2))*4 + \
       math.cos(1*(y/15.0 - 2))*math.cos(1*(y/15.0 - 2))*8 + \
       ((x/10.0 - 6)*(x/10.0 - 6)*2+(y/10.0 - 8)*(y/10.0 - 8)*1)*0.3 + \
       (x+y)*0.07
   #
   # Impact of the optimisation parameter
   if opt == 0:
     t = t*1.0
   elif opt == 1:
     t = t*0.8
   elif opt == 2:
     t = t*0.65
   elif opt == 3:
     t = t*0.55
   elif opt == 4:
     t = t*0.7
   else:
     t = t
     print("Ignored bad optimisation parameter value: " + str(opt))
   #
   # Impact of the thread number
   if nbth == 1:
     t = t*1.0
   elif nbth == 2:
     t = t*1.1
   elif nbth in [3,4]:
     t = (t*1.1)/((nbth-1)**0.8)
   elif nbth in [5,6,7,8]:
     t = (t*1.1/((4-2)**0.8))*(1.2**(nbth-4))
   else:
     t = t
     print("Ignored bat nbth parameter value: " + str(nbth))
   #
   # Impact of the disk parameter
   if disk == 0:
     t = t*1.0
   elif disk == 1:
     t = t*0.9
   else:
     t = t
     print("Ignored bad disk parameter value: " + str(disk))
   #
   # Increase counter of simulations
   NbSim += 1
   # Return the final t-exec (estimated)
   return(t)
   
#------------------------------------------------------------------

def CostFunction2(S):
   global NbSim
   x,y,opt,nbth,disk = S[0], S[1], S[2], S[3], S[4]
   # Exec Time
   if x < 10 or x > 120 or y < 10 or y > 150:
     print("Warning: x range: [10:120], y range: [10:150]")
   t = math.sqrt((math.exp((x-65)/10)+math.exp((-x+65)/10))* \
                 (math.exp((y-80)/10)+math.exp((-y+80)/10)))/4* \
       (math.sin(x/10)*math.sin(x/10) + math.cos(y/10)*math.cos(y/10))
   #
   # Impact of the optimisation parameter
   if opt == 0:
     t = t*1.0
   elif opt == 1:
     t = t*0.8
   elif opt == 2:
     t = t*0.65
   elif opt == 3:
     t = t*0.55
   elif opt == 4:
     t = t*0.7
   else:
     t = t
     print("Ignored bad optimisation parameter value: " + str(opt))
   #
   # Impact of the thread number
   if nbth == 1:
     t = t*1.0
   elif nbth == 2:
     t = t*1.1
   elif nbth in [3,4]:
     t = (t*1.1)/((nbth-1)**0.8)
   elif nbth in [5,6,7,8]:
     t = (t*1.1/((4-2)**0.8))*(1.2**(nbth-4))
   else:
     t = t
     print("Ignored bat nbth parameter value: " + str(nbth))
   #
   # Impact of the disk parameter
   if disk == 0:
     t = t*1.0
   elif disk == 1:
     t = t*0.9
   else:
     t = t
     print("Ignored bad disk parameter value: " + str(disk))
   #
   # Increase counter of simulations
   NbSim += 1
   # Return the final t-exec (estimated)
   return(t)

# Example
#execTime(100,25,3,4,1)

#------------------------------------------------------------------
def CostFunction(S,SimIdx):
    # Build the dictionary
    switcher = {
        0 : CostFunction0,
        1 : CostFunction1,
        2 : CostFunction2
    }
    # Get the simulator function from switcher dictionary
    simulator = switcher.get(SimIdx, lambda: "Invalid Simulator Index")
    # Execute the simulator
    t = simulator(S)
    
    return(t)
   