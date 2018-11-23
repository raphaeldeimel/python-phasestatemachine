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
from scipy.special import betainc

x0 = dot(linspace(0,1,500)[:,newaxis], ones((1,500)))
x1 = x0.T

n = 25
x0diag = betainc(5,5,linspace(-0.01,1.01,n))


isolinevalues = linspace(0,1, 10)


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
    epsilon=0.001
    return 8 * xxt * (L2_squared)  / (L1**4 + epsilon)

def f_proposed2(x0,x1):
    return 1-(1-f_proposed1(x0,x1)**2)**2 #kumaraswamy(1,2)

def f_proposed3(x0,x1):
    """
    
    proposed function: 
    
        Matrix X:
        X = x^nx1 . 1^1xn
        
        lambda = X @ X.T * 8 * (X*X + (X*X).T) / ((X + X.T)**4 - 
    
    """
    xxt = x1 * x0
    L2_squared = x1**2 + x0**2 #xtx
    L1 = abs(x1)+abs(x0)
    epsilon=0.04**2
    return 8 * xxt**3 * L2_squared / ( (1-epsilon)*(L1)**4 * xxt**2 + epsilon)

traces= []
for f, name in [(f_proposed1, 'proposed_activation_function_1'), (f_proposed2, 'proposed_activation_function_2'),(f_proposed3, 'proposed_activation_function_3') ]:
    ax = Axes3D(figure(dpi=300))
    ax.view_init(elev=20., azim=-165)
    #ax.mouse_init(rotate_btn=1, zoom_btn=3)
    ax.plot_surface(x0,x1,f(x0, x1), cmap=cm.coolwarm, linewidth=2.0, rcount=n, ccount=n)
    #ax.set_xlim3d(0,1)
    #ax.set_ylim3d(0,1)
    ax.set_zlabel('transition activation')
    ax.set_xlabel('successor')
    ax.set_ylabel('predecessor')
    for  c in [0.5]:
        a = x0diag**c
        b = (1.0-x0diag)**c
        ax.plot3D(a, b, f(a, b), color='gray')
    savefig('figures/{0}.pdf'.format(name), bbox_inches='tight')
    savefig('figures/{0}.jpg'.format(name), bbox_inches='tight')
    figure()
    traces.append((arctan2(a,b), f(a, b), name))
    plot(traces[-1][0],traces[-1][1], marker='x')
    savefig('figures/{0}_trace.pdf'.format(name), bbox_inches='tight')

figure()
for x,y,label in traces:
    plot(x,y, marker='x', label=label)
legend()
savefig('figures/proposed_activation_function_alltraces.pdf'.format(name), bbox_inches='tight')

if sys.flags.interactive:
    ion()
    show()


