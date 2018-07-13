# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence


This script searches for numerical approximations to the incomplete beta function based on the Kumaraswamy distribution CDF

(Computing Kumaraswamy CDF is computationally much less expensive than betainc)

"""
import numpy as _np
import pandas as _pd
import itertools
from numba import jit

import scipy.special


def kumaraswamy(a,b,x):
    """
    CDF of the Kumaraswamy distribution
    """
    return 1.0-(1.0-x**a)**b
    



def approximate(beta_target_a, beta_target_b, n=100, accuracy=2e-3):

    na = int(n  * beta_target_a / (beta_target_a+beta_target_b))
    nb = int(n  * beta_target_b / (beta_target_a+beta_target_b))

    x_linear = _np.linspace(0,1.0, n)
    x = x_linear
    #x = scipy.special.betainc(1/beta_target_a, 1/beta_target_b, x_linear)

    cdf_target = scipy.special.betainc(beta_target_a, beta_target_b, x)

    a_ =  3.5
    b_ =  4.0
    epsilon_ = 0.1
    costs_ = _np.inf
    rmserror = np.inf
    error_ = _np.zeros(x.size)
    import numpy.random
    for i in range(10000):
        epsilon_ = 0.1 * 100 / (100+i)
        step_a_ = numpy.random.random() * epsilon_  - _np.sum(error_[:na]) * (1000 / (1000+i))
        step_b_ = numpy.random.random() * epsilon_  - _np.sum(error_[nb:]) * (1000 / (1000+i))
        a_proposed = a_ + step_a_
        b_proposed = b_ + step_b_
        cdf_proposed = kumaraswamy(a_proposed, b_proposed, x)
        error = (cdf_proposed - cdf_target)**2
        costs_proposed = _np.sum( error )
        if costs_proposed < costs_:
            a_ = a_proposed
            b_ = b_proposed
            costs_ = costs_proposed
            error_ = error
            rmserror = _np.sqrt(costs_/n)
        if rmserror < accuracy:
            print("stopping early at iteration {0}".format(i))
            break
    return  a_, b_, rmserror
    
    

if __name__ == "__main__":
    parameters_to_approximate = [(2,2), (3,3), (2,5)]
    from matplotlib.pylab import *
    for values in parameters_to_approximate:
        a, b, rmserror = approximate(*values)
        print("Approximating: betainc{0},  a,b: ({1},{2}) RMS error: {3}".format(values, a, b, rmserror))
        figure()
        x = np.linspace(0.0, 1.0, 200)
        cdf_target = scipy.special.betainc(values[0], values[1], x)
        cdf_proposed = kumaraswamy(a, b, x)
        plot(x,cdf_target, label="Beta {0},{1}".format(*values))
        plot(x,cdf_proposed, label="approximated")
        plot(x, (cdf_proposed - cdf_target)**2)
        legend()
show()
