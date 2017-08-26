#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:11:23 2017

@author: raphael
"""

import numpy as _np
import numpy
import sys
sys.path.append('./')
import itertools

from scipy.special import expit

from shcmachine import SHC

 

shc = SHC()
shc.setParameters(
    numStates = 4,
    predecessors = [[2],[0],[1],[2]], # loop wiht 3 states, and one temrinal state (3) reachable from (2)
    alpha=20.0,
    nu=1.5,
    epsilon= [1e-15,1e-15,1e-15,1e-15],
    )         

shc.updateTransitionVelocityLimits(0.1e0)
shc.updateTransitionTriggerInput([1e-8,1e-4,1e-7,-50e-4]) #
activation, phasematrix = shc.getPhasesFromState(shc.statevector)
k = _np.ones((shc.numStates, shc.numStates))
k[1,0] = 0.1
k[0,2] = 0.03
for i in range(2000):
    shc.step()
    activation, progress = shc.getPhasesFromState(shc.statevector)
    phasematrix = phasematrix * (1.0-k) + k * progress 
    shc.updatePhaseLagInput(progress-phasematrix)

shc.updateTransitionTriggerInput([1e-10,1e-4,1e-10, 1.0]) #
for i in range(1000):
    shc.step()
    activation, progress = shc.getPhasesFromState(shc.statevector)
    phasematrix = phasematrix * (1.0-k) + k * progress 
    shc.updatePhaseLagInput(progress-phasematrix)


states = shc.getStateHistory()
t= states[:,0]
import matplotlib.pylab as plt
fig =plt.figure(figsize=(8,2))
plt.axvline(t[2000], linewidth=2, color='r')
for i in range(1, shc.numStates+1):
    plt.plot( t, -1.0*i + states[:,i], color='b')   


def plotfunc(func):
    x= _np.linspace(-3., 3.0, 500)
    y = [func(v) for v in x]
    plt.plot(x,y)


phasesActivation, phasesProgress= zip(*[shc.getPhasesFromState(states[i,1:]) for i in range(len(t))])
phasesActivation = numpy.stack(phasesActivation)
phasesProgress = numpy.stack(phasesProgress)

fig =plt.figure(figsize=(8,2))
transitions = itertools.product([0,1,2,3], [0,1,2,3])
p = shc.predecessors
[[ plt.scatter(t, j+(i-j)*phasesProgress[:,i,j],phasesActivation[:,i, j])   for j in p[i]] for i in range(len(p)) ]
[ plt.scatter(t, [i]*len(t),phasesActivation[:,i,i]) for i in range(len(p)) ]
