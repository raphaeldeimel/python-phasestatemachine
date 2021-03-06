#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence


Illustration of three states forming a cyclic graph (minimal viable system for SHC without bidirectionality)

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
    alpha = 10.0,
    nu=1.0,
    numStates = 3,
    dt=0.01,
    epsilon=1e-6,
    predecessors = predecessors,
    recordSteps = 10000,
    
)

visualizeWithStreamlines(phasta, "example_3states", spread=0.08 ,n_streamlines = 100,
 azimut=20, 
 elevation=30, 
 streamline_width=0.3,
 streamline_alpha=0.3,    
 streamline_length=60,
 coloration_strides=1)

if sys.flags.interactive:
    ion()
    show()
