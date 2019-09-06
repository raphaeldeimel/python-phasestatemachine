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
    numStates = 3,
    dt=0.01,
    epsilon=1e-3,
    successors = [[1],[2,0],[0]],
    recordSteps = 10000,
)

phasta.rhoDelta[0,1] =  phasta.rhoDelta[1,0]
phasta.updateGreediness(1.0)
phasta.statevector[0] = 1.0

print(phasta.rhoDelta)
print(phasta.stateConnectivityAbs)

ax= visualizeWithStreamlines(phasta, "bidirectional_edges", limits=(-1.05, 1.05) ,azimut=-30, elevation=15, streamline_width=0.2 )

if isInteractive():
    ion()
    show()
