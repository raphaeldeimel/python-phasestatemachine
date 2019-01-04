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



print(phasta.competingStates)
print(_np.dot(phasta.stateConnectivity,phasta.stateConnectivity.T ))

###############################


visualizeWithStreamlines(phasta, "stategreediness_constant_1_original", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses= [ 1.0 ] * streamline_length,
#    biases_per_streamline=startpoints
    cull=cull,
)



visualizeWithStreamlines(phasta, "stategreediness__constant_05", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 0.5 ] * streamline_length,
#    biases_per_streamline=startpoints
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_constant_01", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 0.1 ] * streamline_length,
#    biases_per_streamline=startpoints
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_constant_0", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 0.0 ] * streamline_length, #maximum indecision,
#    biases_per_streamline=startpoints
    cull=cull,
)



visualizeWithStreamlines(phasta, "stategreediness_constant_8", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 8.0 ] * streamline_length, #hyperdecisive,
#    biases_per_streamline=startpoints
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_constant_20", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 8.0 ] * streamline_length, #hyperdecisive,
#    biases_per_streamline=startpoints
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_unbalanced_0_1", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ [1,0.0,1.0,1] ] * streamline_length, #maximum indecision,
#    biases_per_streamline=startpoints
    cull=cull,
)

visualizeWithStreamlines(phasta, "stategreediness_unbalanced_neg1_1", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ [0,-1.,1.,0] ] * streamline_length, #maximum indecision,
#    biases_per_streamline=startpoints
    cull=cull,
)

###############################
phasta.updateBiases([0.0,0.0,0.0,0.0])
b= 0.86
startpoints_intermediate = array( [ [ (1.0-(b*cos(a))**2-(b*sin(a))**2)**0.5, b*cos(a), b*sin(a), 0.0] for a in linspace(0, 0.5*_np.pi, n_streamlines) ] )

visualizeWithStreamlines(phasta, "stategreediness_abort_05", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints_intermediate,
    greedinesses= [ 0.5 ] * streamline_length,
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_abort_neg0", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints_intermediate+0.001, #introduce a small error to see anything at all
    greedinesses=[ 0.0 ] * streamline_length,
    cull=cull,
)

visualizeWithStreamlines(phasta, "stategreediness_abort_neg05", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints_intermediate,
    greedinesses=[ -0.5 ] * streamline_length,
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_abort_neg1", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints_intermediate,
    greedinesses=[ -1.0 ] * streamline_length,
    cull=cull,
)

visualizeWithStreamlines(phasta, "stategreediness_abort_neg2", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints_intermediate,
    greedinesses=[ -2.0 ] * streamline_length,
    cull=cull,
)


###############################

n_split = 40
n_split2 = streamline_length-n_split


biases = [ [0.0,0.0,0.0,0.0] ] * n_split + [ [0.0, 0.025, 0.1,0.0] ] * n_split2

visualizeWithStreamlines(phasta, "stategreediness_reconsideration_05_2", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 1.0 ] * n_split + [ array([0,0.5,2.0,0]) ] * n_split2,
    biases=biases,
    cull=cull,
)





visualizeWithStreamlines(phasta, "stategreediness_reconsideration_0_1", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 1.0 ] * n_split + [ array([0,0,1.0,0]) ] * n_split2,
    biases=biases,    
    cull=cull,
)



visualizeWithStreamlines(phasta, "stategreediness_reconsideration_neg1_20", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 1.0 ] * n_split + [ array([0,-1,20,0]) ] * n_split2,
    biases=biases,
    cull=cull,
)



visualizeWithStreamlines(phasta, "stategreediness_reconsideration_0_20", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 1.0 ] * n_split + [ array([0,0.0,20,0]) ] * n_split2,
    biases=biases,
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_reconsideration_neg10_10", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 1.0 ] * n_split + [ array([0,-10,10,0]) ] * n_split2,
    biases=biases,
    cull=cull,
)


visualizeWithStreamlines(phasta, "stategreediness_reconsideration_1_5", 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoints,
    greedinesses=[ 1.0 ] * n_split + [ array([0,1.0,5,0]) ] * n_split2,
    biases=biases,
    cull=cull,
)


if isInteractive():
    ion()
    show()
