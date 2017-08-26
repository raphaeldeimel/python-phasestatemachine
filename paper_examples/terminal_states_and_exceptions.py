#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel

This tests demonstrates terminal states, and how to reset the system
"""

import sys
sys.path.append('../')

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

phasta = phasestatemachine.Kernel()
phasta.setParameters(
    numStates=4,
    predecessors=predecessors, 
    )         


t1 = 10.0
t2 = 0.02
#
phasta.updateTransitionTriggerInput([0,0,0, -1e-7]) 
for i in range(int(t1/phasta.dt)):
    phasta.step()

#force the system into state 3 with a strong input pulse
phasta.updateTransitionTriggerInput([0,0,0, 1/t2]) 
for i in range(int(t2/phasta.dt)):
    phasta.step()
phasta.updateTransitionTriggerInput([0,0,0, -1e-7]) 

for i in range(int(t1/phasta.dt)):
    phasta.step()

#such a pulse can also be used to reset the system:
phasta.updateTransitionTriggerInput([1/t2,0,0,0]) 
for i in range(int(t2/phasta.dt)):
    phasta.step()
phasta.updateTransitionTriggerInput([0,0,0, -1e-7]) 

for i in range(int(t1/phasta.dt)):
    phasta.step()

visualize(phasta, t1+t2+t1+t2+t1, sectionsAt=[t1, t1+t2+t1])
