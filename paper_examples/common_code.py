#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

"""

import numpy as _np
import numpy
import sys


import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import matplotlib 



def plotLineWithVariableWidth(axis, x,y,s, color=None):
    points = _np.array([x, y]).T.reshape(-1, 1, 2)
    segments = _np.concatenate([points[:-1], points[1:]], axis=1)
            
    lc = LineCollection(segments, linewidths=s,color=color, zorder=10)
    axis.add_collection(lc)
    
    
def visualize(phasta, endtime, sectionsAt=None, name="unnamed", newFigure=True, clipActivations=0.01):
    """
    plots a timing diagram of the phase-state machine
    """
    #get the data:
    states, phasesActivation, phasesProgress = phasta.getHistory()
    t=states[:,0]
    
    #clip activations:
    phasesActivation = _np.clip( (phasesActivation-clipActivations) / (1.0-clipActivations), 0, 1)
    
    
    #set up the plot with guides:
    if newFigure: 
        plt.figure(figsize=(4,2))
    fig=plt.gcf()
    axis=plt.gca()
    p = phasta.predecessors
    n_transitions = sum([len(l) for l in p])
    plt.yticks(range(phasta.numStates))
    #[ plt.axhline(y, linewidth=1, color='#eeeeee', zorder=-1) for y in range(phasta.numStates) ]
    plt.xlim(0, endtime)
    plt.ylim(-0.1, phasta.numStates+0.1-1)
    plt.xlabel("time")
    plt.ylabel("state / transitions")
    plt.subplots_adjust(left=0.07, right=0.99)    
    if sectionsAt is not None:
        for x in sectionsAt:
            plt.axvline(x, linewidth=1, color="#dddddd",)

    colors = matplotlib.cm.rainbow(_np.linspace(0,1,n_transitions))
    citer = iter(colors)
    #print the data:        
    [[ plotLineWithVariableWidth(axis, t, j+(i-j)*phasesProgress[:,i,j], phasesActivation[:,i,j], color=next(citer))   for j in p[i]] for i in range(len(p)) ]        
    [  plotLineWithVariableWidth(axis, t[:], [i]*len(t[:]),              phasesActivation[:,i,i], color='black'    ) for i in range(len(p)) ]
    
    
    plt.savefig("./figures/phase_evolution_{}.pdf".format(name))
    plt.savefig("./figures/phase_evolution_{}.png".format(name), dpi=600)



