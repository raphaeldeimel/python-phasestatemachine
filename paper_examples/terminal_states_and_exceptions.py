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

#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import visualize


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [2],   
  [0], 
  [1], 
  [2],   #terminal state 
]

phasta = phasestatemachine.Kernel(
    numStates=4,
    predecessors=predecessors, 
    recordSteps=100000,
)


t1 = 10.0
t2 = 0.02
#
phasta.updateBiases([0,0,0, -1e-7]) 
for i in range(int(t1/phasta.dt)):
    phasta.step()

#force the system into state 3 with a strong input pulse
phasta.updateBiases([0,0,0, 1/t2]) 
for i in range(int(t2/phasta.dt)):
    phasta.step()
phasta.updateBiases([0,0,0, -1e-7]) 

for i in range(int(t1/phasta.dt)):
    phasta.step()

#such a pulse can also be used to reset the system:
phasta.updateBiases([1/t2,0,0,0]) 
for i in range(int(t2/phasta.dt)):
    phasta.step()
phasta.updateBiases([0,0,0, -1e-7]) 

for i in range(int(t1/phasta.dt)):
    phasta.step()

visualize(phasta, t1+t2+t1+t2+t1, sectionsAt=[t1, t1+t2+t1], name=os.path.splitext(os.path.basename(__file__))[0])
