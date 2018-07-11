#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence

This file provides methods for computing the Kumaraswamy distribution's CDF 

This distribution is similar to the beta distribution, but much easier to compute.

"""
import numpy as _np
from numba import jit

@jit
def cdf(a, b, x):
    """
    Cumulative density function of the Kumaraswamy distribution
    
    a,b: parameters of the distribution
    
    """
    y = 1.0-(1.0-x**a)**b
    y[x < 0.0]  = 0.0
    y[x > 1.0]  = 1.0
    return y
    
    
    
def pdf(a, b, x):
    """
    Probabilitiy density function of the Kumaraswamy distribution
    
    a,b: parameters of the distribution
    
    """
    xa1 = x**(a-1)
    xa = x * xa1
    p = (a*b)*xa1*(1-xa)**(1-b)
    p[x < 0.0]  = 0.0
    p[x > 1.0]  = 1.0
    return p
