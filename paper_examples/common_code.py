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
from matplotlib.collections import LineCollection
import matplotlib 



def plotLineWithVariableWidth(axis, x,y,s, color=None):
    points = _np.array([x, y]).T.reshape(-1, 1, 2)
    segments = _np.concatenate([points[:-1], points[1:]], axis=1)
            
    lc = LineCollection(segments, linewidths=s,color=color)
    axis.add_collection(lc)
    
    
def visualize(phasta, endtime, sectionsAt=None, name="unnamed", newFigure=True):
    #get the data:
    states, phasesActivation, phasesProgress = phasta.getHistory()
    t=states[:,0]
    
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
    if sectionsAt is not None:
        for x in sectionsAt:
            plt.axvline(x, linewidth=1, color="#dddddd")

    colors = matplotlib.cm.rainbow(_np.linspace(0,1,n_transitions))
    citer = iter(colors)
    #print the data:        
    [[ plotLineWithVariableWidth(axis, t, j+(i-j)*phasesProgress[:,i,j], phasesActivation[:,i,j], color=next(citer))   for j in p[i]] for i in range(len(p)) ]        
    [  plotLineWithVariableWidth(axis, t[:], [i]*len(t[:]),              phasesActivation[:,i,i], color='black'    ) for i in range(len(p)) ]
    
#    plt.plot(t, [ _np.sum(phasesActivation[i,:,:]) for i in range(phasesActivation.shape[0])])
#    plt.plot(t, [_np.linalg.norm(states[i,1:]) for i in range(states.shape[0])] )
    #[[ plt.scatter(t, j+(i-j)*phasesProgress[:,i,j], phasesActivation[:,i,j])   for j in p[i]] for i in range(len(p)) ]
    #[  plt.scatter(t[:], [i]*len(t[:]),              phasesActivation[:,i,i],  color='black' ) for i in range(len(p)) ]
    
    plt.savefig("../figures/phase_evolution_{}.pdf".format(name))
    plt.savefig("../figures/phase_evolution_{}.png".format(name), dpi=600)



