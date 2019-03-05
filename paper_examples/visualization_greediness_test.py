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
import numpy as _np
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D


#import the phase-state-machine package
import phasestatemachine

successors = [
  [1,2],
  [3],
  [3],
  [0]
]

alpha= 10.0
n_streamlines=20
streamline_length=151
spread=0.1
startpoint = array([1.0, 0.0, 0.0, 0.0])

urgency = 0.0
plot_dt = 0.01 #time base of plots
cull = 2  #how many steps to simulate in between plot points / within plot_dt?

phasta = phasestatemachine.Kernel(
    alpha = alpha,
    nu=1.0,
    numStates = 4,
    dt=plot_dt / cull,
    epsilon=0e-5,
    inputFilterTimeConstant=0.0,
    successors = successors,
    recordSteps = 10000,
)

phasta.updateBiases([0.0,0.0,0.0,0.0])

startpoints = _np.zeros((phasta.numStates, n_streamlines))
startpoints[1,:] = spread * (_np.sin(_np.linspace(0.0, 0.5*_np.pi, n_streamlines)) + 0.01)
startpoints[2,:] = spread * (_np.cos(_np.linspace(0.0, 0.5*_np.pi, n_streamlines)) + 0.01)
startpoints[0,:] = 1.0 - spread**2

startpoints = startpoints.T


strides = [0,20,60, streamline_length]
decision_signal1 = _np.zeros((streamline_length))
decision_signal2 = _np.zeros((streamline_length))



###############################


n_split = 40
n_split2 = streamline_length-n_split
biases = [ [0.0,0.0,0.0,0.0] ] * n_split + [ [0.0, 0.00, 0.0,0.0] ] * n_split2

#greedinesses  =  [ array([0,1,1,0]) for a in range(n_split) ] +   [ array([0,  0.875,  0.125,   0]) for a in range(n_split2) ] 
#greedinesses  =  [ array([0,1,1,0]) for a in range(n_split) ] +   [ array([0,  20,  -1,   0]) for a in range(n_split2) ] 
greedinesses=[10.0]*streamline_length

visualizeWithStreamlines(phasta, "stategreediness_reconsideration_test", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
#    greedinesses=[ 1.0 ] * n_split + [ array([0,-1,1,0]) ] * n_split2,
    greedinesses=greedinesses,
    biases=biases,
    cull=cull,
    azimut=45, 
    elevation=45, 

)


if isInteractive():
    ion()
    show()
