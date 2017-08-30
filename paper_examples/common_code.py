#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel

This tests the branching and aggregation capabilities
"""

import numpy as _np
import numpy
import sys


import matplotlib.pylab as plt



def visualize(phasta, endtime, sectionsAt=None, name="unnamed"):
    #get the data:
    states = phasta.getStateHistory()
    t=states[:,0]
    phasesActivation, phasesProgress= zip(*[phasta.getPhasesFromState(states[i,1:]) for i in range(len(t))])
    phasesActivation = numpy.stack(phasesActivation)
    phasesProgress = numpy.stack(phasesProgress)
    
    #set up the plot with guides:
    plt.figure(figsize=(4,2))
    p = phasta.predecessors
    plt.yticks(range(phasta.numStates))
    [ plt.axhline(y, linewidth=1, color='#eeeeee', zorder=-1) for y in range(phasta.numStates) ]
    plt.xlim(0, endtime)
    plt.xlabel("time")
    plt.ylabel("state / transitions")
    if sectionsAt is not None:
        for x in sectionsAt:
            plt.axvline(x, linewidth=1, color="#dddddd")
            
    #print the data:        
    [[ plt.scatter(t, j+(i-j)*phasesProgress[:,i,j], phasesActivation[:,i, j])   for j in p[i]] for i in range(len(p)) ]
    [  plt.scatter(t[:], [i]*len(t[:]),                    phasesActivation[:,i,i], color='black' ) for i in range(len(p)) ]
    plt.savefig("../figures/phase_evolution_{}.pdf".format(name))
    plt.savefig("../figures/phase_evolution_{}.png".format(name), dpi=600)
