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


x0 = dot(linspace(0,1,500)[:,newaxis], ones((1,500)))
x1 = x0.T

x0diag = linspace(1,0,500)**1
x1diag = linspace(0,1,500)**1

isolinevalues = linspace(0.1,1.1, 10)


def f_proposed1(x0,x1):
    """
    
    proposed function: 
    
        Matrix X:
        X = x^nx1 . 1^1xn
        
        lambda = X @ X.T * 8 * (X*X + (X*X).T) / (X + X.T + 0.01)**4
    
    """
    xxt = x1 * x0
    L2_squared = x1**2 + x0**2 #xtx
    L1 = abs(x1)+abs(x0)
    return 8 * xxt * L2_squared / ((L1)**4 + 0.01)

def f_proposed2(x0,x1):
    return 1-(1-f_proposed1(x0,x1)**2)**2 #kumaraswamy(1,2)

#figure()
#contourf(x0, x1,proposed1, isolinevalues)
#figure()
#contourf(x0, x1,proposed2,isolinevalues)
#figure()
#contourf(x0, x1,proposed3,isolinevalues)

for f, name in [(f_proposed1, 'proposed_activation_function_1'), (f_proposed2, 'proposed_activation_function_2')]:
    ax = Axes3D(figure())
    ax.view_init(elev=20., azim=-105)
    #ax.mouse_init(rotate_btn=1, zoom_btn=3)
    ax.plot_surface(x0,x1,f(x0, x1), color='lightblue', linewidth=2.0)
    #ax.set_xlim3d(0,1)
    #ax.set_ylim3d(0,1)
    ax.set_zlabel('transition activation')
    ax.set_ylabel('successor')
    ax.set_xlabel('predecessor')
    for  c in [0.5, 1.0, 1.5]:
        a = x0diag**c
        b = x1diag**c
        ax.plot3D(a, b, f(a, b), color='gray')
        savefig('figures/{0}.pdf'.format(name), bbox_inches='tight')
        savefig('figures/{0}.jpg'.format(name), bbox_inches='tight')




