#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence


Visualizes the vector field in one 2D-plane of the phase-statemachine state space

"""

import sys
sys.path.insert(0,'../src')
import os


from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D


#import the phase-state-machine package
import phasestatemachine

name="vectorfield_SHC"

predecessors = [
  [2],
  [0],
  [1]
]
phasta = phasestatemachine.Kernel(
    alpha = 10.0,
    nu=8,
    numStates = 3,
    dt=0.1,
    predecessors = predecessors,
    recordSteps = 10000,
)
#phasta.rhoDelta[1,0] = -phasta.rhoDelta[1,0]
phasta.rhoDelta[0,1] =  -phasta.rhoDelta[1,0]
#phasta.rhoDelta[2,1] = 0

print(phasta.rhoDelta)
res=5
o = ones((1,res))
x0 = dot(dot(linspace(0,1,res)[:,newaxis], o)[:,:,newaxis], o)
x1 = transpose(x0, axes=[1,0,2])
x2 = transpose(x0, axes=[2,1,0])
x0_flat = x0.reshape((-1))
x1_flat = x1.reshape((-1))
x2_flat = x2.reshape((-1))

#
#velvectors=zeros((res,res,res,3))
#for k in range(res):
#    for j in range(res):
#        for i in range(res):
#            phasta.statevector[:] = [x0[i,j,k], x1[i,j,k], x2[i,j,k]]
#            phasta.step()
#            velvectors[i,j,k,:] = phasta.dotstatevector
#velvectors_flat = velvectors.reshape((-1,3))

n_streamlines = 1
n_stream_vertices = [10000]*n_streamlines
sample_roi_high = array([0.9, 0.1, 0.])
sample_roi_low = array([0.8, -0.1, 0.])
streamline_startpoints = random(size=(n_streamlines ,3))*(sample_roi_high-sample_roi_low)[newaxis,:]+sample_roi_low[newaxis,:]

#special streamline:
#n_stream_vertices[-1] = 1000

streamlines = []
for i in range(n_streamlines):
    streamline = zeros((n_stream_vertices[i],3))
    phasta.statevector[:] = streamline_startpoints[i,:]
    for l in range(n_stream_vertices[i]):
        streamline[l,:] = phasta.statevector
        phasta.step()
    streamlines.append(streamline)


ax = Axes3D(figure())
ax.view_init(elev=60., azim=30)
#ax.mouse_init(rotate_btn=1, zoom_btn=3)
for i in range(n_streamlines):
    line = streamlines[i]
    length = n_stream_vertices[i]
    colscale=1.0/length
    stride= max(3, length  // 20)
    for i_segment in range(0,length,stride):
        seg =  line[i_segment:i_segment+stride+1,:]
        if sum(seg[0,:]) < 0:
            c = ((i_segment*colscale),(i_segment*colscale),0.8)
        else:
            c = (0.8, (i_segment*colscale),(i_segment*colscale))
        ax.plot(seg[:,0],seg[:,1],seg[:,2], color=c, linewidth=1.0, alpha=0.5)
#ax.quiver(x0_flat,x1_flat,x2_flat,velvectors_flat[:,0],velvectors_flat[:,1],velvectors_flat[:,2], length=0.15)
ax.set_xlim3d(-1,1.1)
ax.set_ylim3d(-1,1.1)
ax.set_zlim3d(-1,1.1)
ax.set_zlabel('x2')
ax.set_ylabel('x1')
ax.set_xlabel('x0')
savefig('figures/{0}.pdf'.format(name), bbox_inches='tight')
savefig('figures/{0}.jpg'.format(name), bbox_inches='tight')




