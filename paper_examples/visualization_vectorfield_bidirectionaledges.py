#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence


Demonstration of bidirectional edges by using state-antistate graphs

"""

import sys
sys.path.insert(0,'../src')
import os

from common_code import *

import numpy as _np
from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D



def plot_streamlines_2d(phasta, n=50, clipVelAt=30.0, name=None):
    """
    plot the vectorfield in the plane of the first two dimensions of the given phase-state machine
    """
    radial_n = 64
    axial_spread = 0.15   
    coordinate_n = 9
    coordinate_spread= 0.025
    start_points_base = _np.array([_np.sin(_np.linspace(0.00001, 2*_np.pi,radial_n, endpoint=False)),_np.cos(_np.linspace(0.00001, 2*_np.pi,radial_n, endpoint=False))])
    start_points_coordinates_x0 = _np.vstack( [ _np.linspace(-0.7,0.7, coordinate_n), _np.zeros(coordinate_n) ]) 
    start_points_coordinates_x1 = _np.vstack( [ _np.zeros(coordinate_n), _np.linspace(-0.7,0.7, coordinate_n) ]) 
    startpoints_sets = []
    for f in [-axial_spread, 0, axial_spread]:
        startpoints_sets.append( start_points_base*(1.0+f))
        
    for f in [-coordinate_spread, coordinate_spread]:
        delta = _np.array([[0],[f]])
        startpoints_sets.append( start_points_coordinates_x0 + delta)
        delta = _np.array([[f],[0]])
        startpoints_sets.append( start_points_coordinates_x1 + delta)


    start_points = _np.hstack(startpoints_sets)

    streamplot_kwargs = {
        'cmap': get_cmap('jet'), 
        'density': n/40, 
        'linewidth': 0.2, 
        'arrowsize': 0.5,
        'arrowstyle': matplotlib.patches.ArrowStyle('->'),
        'start_points': start_points.T,
        'integration_direction': 'forward',
    }
    if isInteractive():
        streamplot_kwargs['linewidth'] =0.5

    streamplot_kwargs_2 = dict(streamplot_kwargs)
    streamplot_kwargs_2['integration_direction'] = 'both'

    args=(-1.2,1.2, n)
    X0, X1 = meshgrid(_np.linspace(*args),_np.linspace(*args))
    #gather gradients:
    phasevelocities = _np.zeros((n,n,2))
    for i in range(n):
        for j in range(n):
           phasta.statevector[:] = 0.0
           phasta.statevector[0] = X0[i,j]
           phasta.statevector[1] = X1[i,j]
           phasta.step()
           phasevelocities[i,j,:] = phasta.dotstatevector[:2]

    l2vel = _np.sqrt(_np.sum(phasevelocities**2, axis=-1))

    #fig, ax = plt.subplots()
    #q = ax.streamplot(X0, X1, phasevelocities[:,:,0], phasevelocities[:,:,1], color=l2vel)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    q = ax.streamplot(X0, X1, phasevelocities[:,:,0], phasevelocities[:,:,1], color=_np.clip(l2vel,0.0, clipVelAt), **streamplot_kwargs)
    #for debugging:
    #ax.plot(start_points[0,:],start_points[1,:],linewidth=0, marker='o')

    if not isInteractive():    
        plt.savefig("./figures/bidirectionaledges_streamlines_{}.pdf".format(name))
        plt.savefig("./figures/bidirectionaledges_streamlines_{}.png".format(name), dpi=600)
        print("plotted {}".format(name))
        plt.close()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    q = ax.streamplot(X0, X1, phasevelocities[:,:,0], phasevelocities[:,:,1], color=_np.clip(l2vel,0.0, clipVelAt), **streamplot_kwargs_2)
    #for debugging:
    #ax.plot(start_points[0,:],start_points[1,:],linewidth=0, marker='o')

    if not isInteractive():    
        plt.savefig("./figures/bidirectionaledges_streamlines_{}_filled.pdf".format(name))
        plt.savefig("./figures/bidirectionaledges_streamlines_{}_filled.png".format(name), dpi=600)
        print("plotted {}".format(name))
        plt.close()


#import the phase-state-machine package
import phasestatemachine

phasta = phasestatemachine.Kernel(
    alpha = 20.0,
    numStates = 2,
    dt=0.01,
    epsilon=1e-3,
    successors = [[1],[0]],
    recordSteps = 10000,
)

phasta.rhoDelta[0,1] =  phasta.rhoDelta[1,0]
#phasta.stateConnectivity[0,1] = 1
#phasta.stateConnectivity[1,0] = -1

#print(phasta.rhoDelta)
#print(phasta.stateConnectivity)
#print(phasta.stateConnectivityGreedinessTransitions)
#print(phasta.stateConnectivity+phasta.stateConnectivityGreedinessTransitions)

plot_streamlines_2d(phasta, name="pos1")

phasta.updateGreediness(0.3)
plot_streamlines_2d(phasta, name="pos03")

phasta.updateGreediness(0.0)
plot_streamlines_2d(phasta, name="zero")

phasta.updateGreediness(-0.3)
plot_streamlines_2d(phasta, name="neg03")

phasta.updateGreediness(-1.0)
plot_streamlines_2d(phasta, name="neg1")



phasta2 = phasestatemachine.Kernel(
    alpha = 20.0,
    numStates = 3,
    dt=0.01,
    epsilon=1e-3,
    successors = [[2],[0],[1]],
    recordSteps = 10000,
)

plot_streamlines_2d(phasta2, name="unidirectional_pos1")
phasta2.updateGreediness(0.0)
plot_streamlines_2d(phasta2, name="unidirectional_zero")
phasta2.updateGreediness(-1.0)
plot_streamlines_2d(phasta2,name="unidirectional_neg1")



if isInteractive():
    ion()
    show()
