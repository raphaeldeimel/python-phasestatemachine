#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This tests demonstrates how to slow down and speed up a transition


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
  [2],
  [0],
  [1], 
]

#Make the first two transitions slower (negative exponent), 
#and the the third transition faster (positive exponent)
phaseVelocityExponentsMatrix = [[0., 0., 5.],[-4,0,0.],[0., -7., 0.]]

phasta = phasestatemachine.Kernel(
    numStates=3,
    predecessors=predecessors,
    recordSteps=100000,
    dt=1e-3
)


t1 = 3.2
t2 = 22.8

#negatively bias transition towards states 2-4 to block transition from state 1:
#phasta.updateTransitionTriggerInput(bias) 
#evolve the system for some time
for i in range(int(t1/phasta.dt)):
    phasta.step()

phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix)

for i in range(int(t2/phasta.dt)):
    phasta.step()

visualize(phasta, t1+t2, sectionsAt=[t1], name=os.path.splitext(os.path.basename(__file__))[0])

