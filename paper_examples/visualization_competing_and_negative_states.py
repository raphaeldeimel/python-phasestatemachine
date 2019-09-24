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
  [0,3],
  [0,3],
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
    inputFilterTimeConstant=0.0, #important, so that state doesn't leak across plots...
    recordSteps = 10000,
)

combinations = (('p_pp',1,1,1),('p_pn',1,1,-1),('p_np',1,-1,1),('p_nn',1,-1,-1),('n_pp',-1,1,1),('n_pn',-1,1,-1),('n_np',-1,-1,1),('n_nn',-1,-1,-1))
#combinations = (('p_pn',1,1,-1),)
greedinesses_totest = ([1.0,0.17,0.17,1.0],[1.0,3.2,0.37,1.0])

for i, greedinesses in enumerate(greedinesses_totest):
  for name, sign0, sign1, sign2 in combinations:
    phasta.connectivitySignMap[1,0] = sign1*sign0
    phasta.connectivitySignMap[2,0] = sign2*sign0
    phasta.connectivitySignMap[0,1] =  -phasta.connectivitySignMap[1,0]
    phasta.connectivitySignMap[0,2] =  -phasta.connectivitySignMap[2,0]
    



    startpoints = _np.zeros((phasta.numStates, n_streamlines))
    startpoints[1,:] = sign1 * spread * (_np.sin(_np.linspace(0.0, 0.5*_np.pi, n_streamlines)) + 0.01)
    startpoints[2,:] = sign2 * spread * (_np.cos(_np.linspace(0.0, 0.5*_np.pi, n_streamlines)) + 0.01)
    startpoints[0,:] = sign0 * (1.0 - spread**2)

    startpoints = startpoints.T


    strides = [0,20,60, streamline_length]
    decision_signal1 = _np.zeros((streamline_length))
    decision_signal2 = _np.zeros((streamline_length))


    phasta.updateBiases([0.0,0.0,0.0,0.0])
    phasta.updateGreediness(greedinesses)
    


    ###############################


    visualizeWithStreamlines(phasta, "competingstates_and_negative_states_{0}_{1}".format(i,name), 
        n_streamlines = n_streamlines, 
        streamline_length=streamline_length,
        coloration_strides=10, 
        azimut=0,
        elevation=0,
        dims=[0,1,2], 
        limits=[-1.05, 1.05],
        streamlines_commonstartpoint=startpoints,
#        greedinesses= [ 1.0 ] * streamline_length,
    #    biases_per_streamline=startpoints
        cull=cull,
    )


    print(name)
    print(phasta.stateConnectivityGreedinessAdjustment)
    print(phasta.stateConnectivityCompetingGreedinessAdjustment)
#    print(phasta.connectivitySignMap)
#    print(phasta.stateConnectivityAbs*phasta.stateConnectivityAbs.T)
#    print(phasta.competingStates)
  #  print(phasta.connectivitySigned)




if isInteractive():
    ion()
    show()
    
