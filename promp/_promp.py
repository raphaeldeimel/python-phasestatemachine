#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains the main code for the phase-state machine

"""

import numpy as _np
from matplotlib import pylab as _pl


def makePsi(count, sigma=None):
    """
    returns a function that takes the phase and outputs the vector psi
     
    In other words, it creates a basis function kernel with the given metaparameters
    In accordance to the paper, the basis functions are placed in the interval
    [ -2*sigma ... 1+2*sigma ]
    
    If you don't specify sigma, it will be guessed so that basis functions overlap reasonably
    """        
    if sigma is None:
        sigma = 0.5 / (count-2)                      
    means = _np.linspace(-2*sigma, 1+2*sigma, count)
    factor = -1. / (2*sigma)**2 
    def Psi(phase):
        psi_value = _np.exp( factor * (phase - means)**2 )
        return psi_value / _np.linalg.norm(psi_value)
    return Psi


def makeTrajectoryDistributionSampler(thetas_mean, thetas_sigma):
    """
    create a function to sample from the trajectory distribution as 
    represented by the vector w
    
    """    
    def sampler():
       s = _np.random.normal(thetas_mean, thetas_sigma) 
       return s
    return sampler



def plotTrajectoryDistribution(thetas_mean, thetas_sigma, psi, samples=100, num=1000, linewidth=2.0 ):
    c = _np.linspace(0.0, 1.0, num)
    data_mean = _np.empty(num)
    data_sigma = _np.empty(num)
    trajectories = _np.empty((num, samples))
    sampler = makeTrajectoryDistributionSampler(thetas_mean, thetas_sigma)
    for i in range(num):
        data_mean[i] = _np.dot (psi(c[i]) , thetas_mean) 
        data_sigma[i] = _np.dot(psi(c[i]) , thetas_sigma) 
    for j in range(samples):
        w = sampler()
        for i in range(num):
            trajectories[i, j] = _np.dot (psi(c[i]) , w)
        
    _pl.plot(c, data_mean)
    _pl.fill_between(c, data_mean-1.96*data_sigma, data_mean+1.96*data_sigma, alpha=0.25, label="95%")

    alpha = 2.0 / samples
    for j in range(samples):
        _pl.plot(c, trajectories[:,j], alpha=alpha, linewidth=linewidth , color='b')


    

def computeCurrentDesiredState(phase, w, psi, noise=1e-12):
    state = _np.dot( psi(phase).T, w) + _np.random.normal(0, noise)
    return state


if __name__=="__main__":
    n=10
    psi = makePsi(n)
    thetas_sigma = _np.random.chisquare(1, n)
    thetas_mean = _np.linspace(0.0, 1.0, n) + _np.random.normal(0, 3./n, n)
    plotTrajectoryDistribution(thetas_mean, thetas_sigma, psi)   
    
    
    