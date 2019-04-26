#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This tests demonstrates terminal states, and how to reset the system
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
  [2],   
  [0], 
  [1], 
  [],   #error state 
]

phasta = phasestatemachine.Kernel(
    numStates=4,
    predecessors=predecessors, 
    recordSteps=100000,
)


biasMatrix = numpy.zeros((4,4))

t1 = 10.0
t2 = 0.02

#
biasMatrix[3,:] = -1e-7
phasta.updateBiases(biasMatrix) 
for i in range(int(t1/phasta.dt)):
    phasta.step()

#force the system into state 3 with a strong input pulse
biasMatrix[3,:] = 10/t2
phasta.updateBiases(biasMatrix) 
for i in range(int(t2/phasta.dt)):
    phasta.step()
biasMatrix[3,:] = 1e-7
phasta.updateBiases(biasMatrix) 

for i in range(int((t1-t2)/phasta.dt)):
    phasta.step()

#such a pulse can also be used to reset the system:
biasMatrix[0,:] = 10/t2
phasta.updateBiases(biasMatrix) 
for i in range(int(t2/phasta.dt)):
    phasta.step()
biasMatrix[0,:] = 0
phasta.updateBiases(biasMatrix) 

for i in range(int((t1-t2)/phasta.dt)):
    phasta.step()

visualize(phasta, t1+t1+t1, sectionsAt=[t1, t1+t1], name=os.path.splitext(os.path.basename(__file__))[0])
