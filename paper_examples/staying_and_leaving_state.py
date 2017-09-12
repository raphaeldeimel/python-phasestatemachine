#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This tests demonstrates blocking and unblocking of state transitions
"""

import numpy
import sys
sys.path.append('../')
import os

from matplotlib import pylab as plt

#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import visualize


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [2],   
  [0],
  [1] 
]

phasta = phasestatemachine.Kernel()
phasta.setParameters(
    numStates = 3,
    predecessors = predecessors, 
    )         


t1 = 5.0
t2 = 5.0

#negatively bias transition towards state 2 to block transition from state 1:
phasta.updateTransitionTriggerInput([0.0, 0.0, -1e-7]) 
#alternative: specify the state to block in and project a blocking bias on all successors with the phasta.stateConnectivityMap
phasta.updateTransitionTriggerInput(  numpy.dot(phasta.stateConnectivity, numpy.array([0.0, -1e-7, 0.0,])) ) 

#evolve the system for some time
for i in range(int(t1/phasta.dt)):
    phasta.step()

#un-freeze the state by setting the bias back to non-negative values.
# positive values will reduce the dwell-time considerably: 
phasta.updateTransitionTriggerInput([0.0, 0.0, 1e-1]) #

t2a = 0.1
for i in range(int(t2a/phasta.dt)):
    phasta.step()

phasta.updateTransitionTriggerInput([0.0, 0.0, 0.0]) #

for i in range(int((t2-t2a)/phasta.dt)):
    phasta.step()


visualize(phasta, t1+t2, sectionsAt=[t1, t1+t2a], name=os.path.splitext(os.path.basename(__file__))[0])

