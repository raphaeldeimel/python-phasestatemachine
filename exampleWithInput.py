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

from shcmachine import SHC

 

shc = SHC()
shc.setParameters(
    numStates = 4,
    predecessors = [[2],[0],[1],[2]], # loop wiht 3 states, and one temrinal state (3) reachable from (2)
    )         

shc.updateTransitionTriggerInput([1e-9,0,0,-3e-10]) #

states = _np.vstack([shc.step() for i in range(1000)])

import matplotlib.pylab as plt
fig =plt.figure(figsize=(18,3))
for i in range(states.shape[1]):
    plt.plot( -1.0*i + states[:,i], color='b')   
    
def plotfunc(func):
    x= _np.linspace(-3., 3.0, 500)
    y = [func(v) for v in x]
    plt.plot(x,y)

