#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This tests demonstrateshow gamma influences system behavior


"""

import sys
sys.path.insert(0,'../src')
import os

#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import *


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [2],
  [0],
  [1], 
]


phasta_1 = phasestatemachine.Kernel(
    dt=0.001,
    numStates=3,
    predecessors=predecessors,
    recordSteps=100000,
    stateVectorExponent=1,
)

phasta_2 = phasestatemachine.Kernel(
    dt=0.001,
    numStates=3,
    predecessors=predecessors,
    recordSteps=100000,
    stateVectorExponent=2,
)

phasta_3 = phasestatemachine.Kernel(
    dt=0.001,
    numStates=3,
    predecessors=predecessors,
    recordSteps=100000,
    stateVectorExponent=3,
)

phasta_05 = phasestatemachine.Kernel(
    dt=0.001,
    numStates=3,
    predecessors=predecessors,
    recordSteps=100000,
    stateVectorExponent=0.5,
)

t = 1.1

for phasta in [phasta_1, phasta_2, phasta_3,phasta_05]:
    phasta.updateBiases(1e-3)

    for i in range(int(t/phasta.dt)):
        phasta.step()

    visualize(phasta, t, name="{0}_{1}".format(os.path.splitext(os.path.basename(__file__))[0], phasta.stateVectorExponent))
    
    
    visualizeWithStreamlines(phasta, "gamma_parameter_{0}".format(phasta.stateVectorExponent), 
         spread=0.05,
         n_streamlines = 100,
         azimut=15, 
         elevation=5, 
         streamline_width=1.0,
         streamline_alpha=0.3,    
         streamline_length=150,
         coloration_strides=1
    )
    

