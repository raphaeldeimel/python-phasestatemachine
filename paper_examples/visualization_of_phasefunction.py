#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence


Visualizes the vector field in one 2D-plane of the phase-statemachine state space

"""


from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


x0 = dot(linspace(0,1,500)[:,newaxis], ones((1,500)))
x1 = x0.T

x0diag = linspace(1,0,500)**1
x1diag = linspace(0,1,500)**1

isolinevalues = linspace(0.1,1.1, 10)


def f_proposed1(x0,x1,epsilon=0.0001, exp=0.5):
    return   (x1**exp+epsilon)/(x0**exp+x1**exp+2*epsilon)
    
def f_proposed2(x0,x1,epsilon=0.0001, exp=1.0):
    return   (x1**exp+epsilon)/(x0**exp+x1**exp+2*epsilon)

def f_proposed3(x0,x1,epsilon=0.0001, exp=2.0):
    return   (x1**exp+epsilon)/(x0**exp+x1**exp+2*epsilon)


#figure()
#contourf(x0, x1,proposed1, isolinevalues)
#figure()
#contourf(x0, x1,proposed2,isolinevalues)
#figure()
#contourf(x0, x1,proposed3,isolinevalues)

for f, name in [(f_proposed1, 'proposed_phase_function_1'), (f_proposed2, 'proposed_phase_function_2'),(f_proposed3, 'proposed_phase_function_3')]:
    ax = Axes3D(figure())
    ax.view_init(elev=20., azim=-105)
    #ax.mouse_init(rotate_btn=1, zoom_btn=3)
    ax.plot_surface(x0,x1,f(x0, x1),cmap=cm.coolwarm, linewidth=2.0)
    #ax.set_xlim3d(0,1)
    #ax.set_ylim3d(0,1)
    ax.set_zlabel('transition phase')
    ax.set_ylabel('successor')
    ax.set_xlabel('predecessor')
    traces = []
    trace_coeffs = [ 0.9, 2.0]
    for  c in trace_coeffs:
        a = x0diag**c
        b = x1diag**c
        traces.append(f(a, b))
        ax.plot3D(a, b, traces[-1], color='gray')
        #savefig('figures/{0}.pdf'.format(name), bbox_inches='tight')
        #savefig('figures/{0}.jpg'.format(name), bbox_inches='tight')
    figure()
    for t,c in zip(traces, trace_coeffs):
        plot(t, label=c)
    legend()
        
ion()
show()


