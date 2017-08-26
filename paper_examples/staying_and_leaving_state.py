#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel

This tests demonstrates blocking and unblocking of state transitions
"""

import numpy
import sys
sys.path.append('../')
from matplotlib import pylab as plt

#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import visualize


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [2,3,4],   
  [0],      #state to stop in
  [1],[1],[1], 
]

phasta = phasestatemachine.Kernel()
phasta.setParameters(
    numStates = 5,
    predecessors = predecessors, 
    )         


t1 = 5.0
t2 = 5.0

#negatively bias transition towards states 2-4 to block transition from state 1:
phasta.updateTransitionTriggerInput([0.0, 0.0, -1e-7, -1e-7, -1e-7]) 
#alternative: specify the state to block in and project a blocking bias on all successors with phasta.stateConnectivityMap
phasta.updateTransitionTriggerInput(  phasta.stateConnectivityMap @ numpy.array([0.0, -1e-7, 0,0,0]) ) 

#evolve the system for some time
for i in range(int(t1/phasta.dt)):
    phasta.step()

#reset bias towards state 3 to zero - the kernel will start to transition towards that state
phasta.updateTransitionTriggerInput([0.0, 0.0, -1e-7, 0.0, -1e-7]) #

for i in range(int(t1/phasta.dt)):
    phasta.step()

    
visualize(phasta, t1+t2)
plt.axvline(t1) #indicate where the paramter was changed