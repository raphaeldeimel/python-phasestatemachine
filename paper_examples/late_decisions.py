#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This tests demonstrates taking late decisions by activating all transitions first, deciding on winner later

"""


import sys
sys.path.insert(0,'../src')
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
epsilon = 1e-6

phasta = phasestatemachine.Kernel(
    numStates=4,
    predecessors=predecessors, 
    epsilon=epsilon, 
    nu=7.,
    recordSteps=100000,
)


t1 = 4.0
tspike = 1.0
t2 = 3.5
t3= 2.0

#Variation: negatively bias transition towards states 2-4 to block transition from state 1:
bias = 1e-3 
phasta.updateBiases([bias, bias, bias, bias]) 

phaseVelocityExponentsMatrix = [[0., 0.,  0., 0.],
                                [0., 0., -3., 0.],
                                [0., 0.,  0., 0.],
                                [0., 0., -3., 0.]]
phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix)

timesToMark = []
#evolve the system for some time
phasta.step(t1)

timesToMark.append(phasta.t)
phasta.updateBiases([bias, -100*bias, bias, 100*bias])             
phasta.step(tspike)
phasta.updateBiases([bias, bias, bias, bias]) 
timesToMark.append(phasta.t)

phasta.step(t2)

timesToMark.append(phasta.t)
phasta.updateBiases([bias, -100*bias, bias, 100*bias])             
phasta.step(tspike)
phasta.updateBiases([bias, bias, bias, bias]) 
timesToMark.append(phasta.t)

phasta.step(t3)


visualize(phasta, phasta.t, sectionsAt=timesToMark, name=os.path.splitext(os.path.basename(__file__))[0])
