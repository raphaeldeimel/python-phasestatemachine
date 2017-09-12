#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This tests demonstrates taking random transitions similar to an MDP

By setting the biases relative to the epsilon noise we can adjust the transition probabilities

Due to the cumulative nature, often both transitions get activated to some extent, before one wins and takes over
"""

import sys
sys.path.append('../')
import os

#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import visualize


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [1,3],   
  [2],      #state to stop in
  [0], 
  [2], 
]
epsilon = 1e-9

phasta = phasestatemachine.Kernel()
phasta.setParameters(
    numStates=4,
    predecessors=predecessors, 
    epsilon=epsilon, 
    )         


endtime = 25.0

#Variation: negatively bias transition towards states 2-4 to block transition from state 1:
#phasta.updateTransitionTriggerInput([0, 1*epsilon, 0, 0]) 

#evolve the system for some time
for i in range(int(endtime/phasta.dt)):
    phasta.step()

visualize(phasta, endtime, name=os.path.splitext(os.path.basename(__file__))[0])
