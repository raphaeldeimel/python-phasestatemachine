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
phaseVelocityExponentsMatrix2 = [[0., 0., 0],[0,0,0.],[0., -6., 0.]]

phasta = phasestatemachine.Kernel(
    dt=0.001,
    numStates=3,
    predecessors=predecessors,
    recordSteps=100000,
    epsilon=0.0,
    
)
phasta.updateBiases(3e-9)

t1 = 3.0
t2 = 17.0
t3 = 1.0
t4 = 5.0

#negatively bias transition towards states 2-4 to block transition from state 1:
#phasta.updateTransitionTriggerInput(bias) 
#evolve the system for some time
for i in range(int(t1/phasta.dt)):
    phasta.step()

phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix)

for i in range(int(t2/phasta.dt)):
    phasta.step()

phaseVelocityExponentsMatrix[2][1]=-6
phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix)
for i in range(int(t3/phasta.dt)):
    phasta.step()
    
phaseVelocityExponentsMatrix[2][1]=-5
phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix)
for i in range(int(t3/phasta.dt)):
    phasta.step()
    
phaseVelocityExponentsMatrix[2][1]=-8
phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix)
for i in range(int(t3/phasta.dt)):
    phasta.step()

for i in range(int(t4/phasta.dt)):
    phasta.step()

visualize(phasta, t1+t2+3*t3+t4, sectionsAt=[t1, t1+t2, t1+t2+t3, t1+t2+t3+t3], name=os.path.splitext(os.path.basename(__file__))[0])

