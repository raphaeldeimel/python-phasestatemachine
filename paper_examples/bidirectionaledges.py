#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence


Demonstration of bidirectional edges by using state-antistate graphs

"""

import sys
sys.path.insert(0,'../src')
import os

from common_code import *

from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D


#import the phase-state-machine package
import phasestatemachine

predecessors = [
  [2],
  [0],
  [1]
]
phasta = phasestatemachine.Kernel(
    alpha = 20.0,
    nu=1.0,
    numStates = 3,
    dt=0.01,
    epsilon=1e-3,
    predecessors = predecessors,
    recordSteps = 10000,
)

phasta.rhoDelta[0,1] =  phasta.rhoDelta[1,0]
phasta.stateConnectivity[0,1] =  -1

print(phasta.rhoDelta)
print(phasta.stateConnectivity)

visualizeWithStreamlines(phasta, "bidirectional_edges", limits=(-1.05, 1.05))

if isInteractive():
    ion()
    show()
