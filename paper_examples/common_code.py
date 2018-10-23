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
from mpl_toolkits.mplot3d import Axes3D



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
    p = phasta.successors
    n_transitions = sum([len(l) for l in p])
    n_states = phasta.numStates
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
    [[ plotLineWithVariableWidth(axis, t, i+(j-i)*phasesProgress[:,j,i], phasesActivation[:,j,i], color=next(citer))   for j in p[i]] for i in range(n_states) ]        
    [  plotLineWithVariableWidth(axis, t[:], [i]*len(t[:]),              phasesActivation[:,i,i], color='black'    ) for i in range(n_states) ]
    
    
    plt.savefig("./figures/phase_evolution_{}.pdf".format(name))
    plt.savefig("./figures/phase_evolution_{}.png".format(name), dpi=600)




def visualizeWithStreamlines(phasta, name, spread=0.05 ,n_streamlines = 50, streamline_length=100, coloration_strides=5):

    n_stream_vertices = [streamline_length]*n_streamlines


    streamlines = []
    for i in range(n_streamlines):
        streamline = _np.zeros((n_stream_vertices[i],3))
        streamline_nextstart = phasta.statevector + (_np.random.uniform(size=3)*2*spread-spread)
        phasta.statevector[:] = streamline_nextstart
        for l in range(n_stream_vertices[i]):
            streamline[l,:] = phasta.statevector
            phasta.step()
        streamlines.append(streamline)


    ax = Axes3D(plt.figure())
    ax.view_init(elev=60., azim=30)
    #ax.mouse_init(rotate_btn=1, zoom_btn=3)
    for i in range(n_streamlines):
        line = streamlines[i]
        length = n_stream_vertices[i]
        colscale=1.0/length
        stride= max(3, length  // coloration_strides)
        for i_segment in range(0,length,stride):
            seg =  line[i_segment:i_segment+stride+1,:]
            l1 = 0.3+0.7*(i_segment*colscale)
            l2 = 0.2+0.2*(i_segment*colscale)
            if sum(seg[0,:]) < 0:
                c = (l2,l2,l1)
            else:
                c = (l1,l2,l2)
            ax.plot(seg[:,0],seg[:,1],seg[:,2], color=c, linewidth=1.0, alpha=0.5)
    ax.set_xlim3d(-1,1.1)
    ax.set_ylim3d(-1,1.1)
    ax.set_zlim3d(-1,1.1)
    ax.set_zlabel('x2')
    ax.set_ylabel('x1')
    ax.set_xlabel('x0')
    plt.savefig('figures/{0}.pdf'.format(name), bbox_inches='tight')
    plt.savefig('figures/{0}.jpg'.format(name), bbox_inches='tight')

