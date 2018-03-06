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
sys.path.insert(0,'../src')
import os
import numpy

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
epsilon = 0
dt=1e-2
phasta = phasestatemachine.Kernel()
phasta.setParameters(
    dt=dt,
    numStates=4,
    predecessors=predecessors, 
    epsilon=epsilon, 
    )         


endtime = 50.0

#Variation: negatively bias transition towards states 2-4 to block transition from state 1:
#phasta.updateTransitionTriggerInput([0, 1*epsilon, 0, 0]) 

Gamma = numpy.array([
[   0, 1e-9, 0, 1e-9 ],
[   0,    0, 0,    0 ],
[1e-4,    0, 0,    0 ],
[   0,    0, 0,    0 ],
])



noiseScale = 1e-4 * numpy.sqrt(dt)/dt

#evolve the system for some time
for i in range(int(endtime/phasta.dt)):
    Gamma[1,2] = 3 * numpy.random.normal(scale=noiseScale)
    Gamma[3,2] = 1* numpy.random.normal(scale=noiseScale)
    bias = numpy.dot(Gamma, phasta.statevector)
    phasta.updateTransitionTriggerInput(Gamma)
    phasta.step()

visualize(phasta, endtime, name=os.path.splitext(os.path.basename(__file__))[0], clipActivations=0.05)







