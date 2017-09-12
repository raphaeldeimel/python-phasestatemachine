#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This tests the branching and aggregation capabilities
"""

import numpy
import sys
sys.path.append('../')
import os

#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import visualize


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [17], #restart cycle from last state
  [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0], #states that branch out from state 0
  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] #last state aggregates all branches
]

phasta = phasestatemachine.Kernel()
phasta.setParameters(
    numStates = 18,
    predecessors = predecessors, 
    )         

#set up biases so that we favor one out the 16 possible branches:
import collections
bias = collections.deque([0.0]*16)
bias[0] = 1e-4
phasta.updateTransitionTriggerInput([1e-4] + list(bias) + [0.0]) #
endtime = 34.0
#run the kernel for some time:
triggered=False
for i in range(int(endtime/phasta.dt)):
    phasta.step()
    if phasta.statevector[-1] > 0.9 and not triggered:
        triggered = True
        bias.rotate(1)
        phasta.updateTransitionTriggerInput([1e-4] + list(bias) + [0.0])
    elif phasta.statevector[0] > 0.9:
        triggered = False

visualize(phasta, endtime, name=os.path.splitext(os.path.basename(__file__))[0])
