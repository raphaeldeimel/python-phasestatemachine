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
import sys

n  = 25
x1 = dot(linspace(0,1,n)[:,newaxis], ones((1,n)))
x0 = x1.T

x0diag = linspace(1,0,n*5)
x1diag = linspace(0,1,n*5)

isolinevalues = linspace(0.1,1.1, 10)


def f_proposed(x0,x1,epsilon=0.0001, exp=0.5):
    return   (x0**exp+epsilon)/(x0**exp+x1**exp+2*epsilon)
    


#figure()
#contourf(x0, x1,proposed1, isolinevalues)
#figure()
#contourf(x0, x1,proposed2,isolinevalues)
#figure()
#contourf(x0, x1,proposed3,isolinevalues)

for f, name, exp in [(f_proposed, 'proposed_phase_function_2', 2.0), (f_proposed, 'proposed_phase_function_1', 1.0),(f_proposed, 'proposed_phase_function_05', 0.5)]:
    ax = Axes3D(figure(dpi=300))
    ax.view_init(elev=20., azim=-165)
    #ax.mouse_init(rotate_btn=1, zoom_btn=3)
    ax.plot_surface(x0,x1,f(x0, x1, exp=exp),cmap=cm.coolwarm, linewidth=2.0, rcount=n, ccount=n)
    #ax.set_xlim3d(0,1)
    #ax.set_ylim3d(0,1)
    ax.set_zlabel('transition phase')
    ax.set_xlabel('successor')
    ax.set_ylabel('predecessor')
    #plot a trace:
    a = x0diag**(1./2)
    b = x1diag**(1./2)
    trace = f(a, b, exp=exp)
    ax.plot3D(a, b, trace, color='gray')
    savefig('figures/{0}_3d.pdf'.format(name), bbox_inches='tight')
    savefig('figures/{0}_3d.jpg'.format(name), bbox_inches='tight')

    figure()
    plot( 2.0/pi*arctan2(a, b),  trace, label=exp)
        
    legend()
    savefig('figures/{0}_alongchannel.pdf'.format(name), bbox_inches='tight')
        
if sys.flags.interactive:
    ion()
    show()


