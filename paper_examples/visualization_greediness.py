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
spread=1.0
startpoint = array([1.0, 0.00001, 0.00001, 0.0])

urgency = 0.0

phasta = phasestatemachine.Kernel(
    alpha = alpha,
    nu=1.0,
    numStates = 4,
    dt=0.01,
    epsilon=0e-5,
    successors = successors,
    recordSteps = 10000,
)
#phasta.updateTransitionTriggerInput(urgency*_np.array([1,1,0.5,1]))

noise_seed = _np.zeros((phasta.numStates, n_streamlines))
noise_seed[1,:] = spread * _np.sin(_np.linspace(0.0, 0.5*_np.pi, n_streamlines))
noise_seed[2,:] = spread * _np.cos(_np.linspace(0.0, 0.5*_np.pi, n_streamlines))

noise_seed2 = _np.zeros((phasta.numStates, n_streamlines))
noise_seed2[1,:] = _np.linspace(0.8*spread, spread, n_streamlines)
noise_seed2[2,:] = spread - noise_seed[1,:]


strides = [0,20,60, streamline_length]
decision_signal1 = _np.zeros((streamline_length))
decision_signal2 = _np.zeros((streamline_length))


def getListOfPreferenceVectors(n, decisiveness=0.0, reconsideration=0.0, alpha = 20):
    rhoDeltaInput = phasta.rhoDelta.copy()
    rhoDeltaInput[1,2] = -alpha * (decisiveness+reconsideration)
    rhoDeltaInput[2,1] = -alpha * (decisiveness-reconsideration)
    rhoDeltas = []
    for l in range(n):
        rhoDeltas.append(rhoDeltaInput)
    return rhoDeltas


preferences = [ 0.5 ] * streamline_length

visualizeWithStreamlines(phasta, "stategreediness_original_05", 
    spread=spread, 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoint,
    noise_seed=noise_seed,
    preferences=preferences,
)



preferences = [ 0.0 ] * streamline_length #maximum indecision

visualizeWithStreamlines(phasta, "stategreediness_indecisive_0", 
    spread=spread, 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoint,
    noise_seed=noise_seed,
    preferences=preferences,
)


preferences = [ 8.0 ] * streamline_length #hyperdecisive


visualizeWithStreamlines(phasta, "stategreediness_decisive_8", 
    spread=spread, 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=10, 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoint,
    noise_seed=noise_seed,
    preferences=preferences,
)

print(phasta.rhoDelta)
print(phasta.rhoZero)



n_split = 70
n_split2 = streamline_length-n_split

preferences = [ 0 ] * n_split + [ array([0,0.5,2.0,0]) ] * n_split2

visualizeWithStreamlines(phasta, "stategreediness_reconsideration_moderatelysure_05_2", 
    spread=spread, 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoint,
    noise_seed=noise_seed,
    preferences=preferences,
)




preferences = [ 0 ] * n_split + [ array([0,0,1.0,0]) ] * n_split2

visualizeWithStreamlines(phasta, "stategreediness_reconsideration_sure_reluctant_0_1", 
    spread=spread, 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoint,
    noise_seed=noise_seed,
    preferences=preferences,
)



preferences = [ 0 ] * n_split + [ array([0,0.0,20,0]) ] * n_split2

visualizeWithStreamlines(phasta, "stategreediness_reconsideration_hyperdecisive_0_20", 
    spread=spread, 
    n_streamlines = n_streamlines, 
    streamline_length=streamline_length,
    coloration_strides=[0,n_split, streamline_length], 
    dims=[0,1,2], 
    streamlines_commonstartpoint=startpoint,
    preferences=preferences,
    noise_seed=noise_seed,
)



if isInteractive():
    ion()
    show()
