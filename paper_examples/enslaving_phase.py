#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel

This tests shows how to enslave a phase to an external signal

"""

import sys
sys.path.append('../')
import os
import numpy
numpy.set_printoptions(precision=2, suppress=True)
#import the phase-state-machine package
import phasestatemachine 

#import code shared across all examples
from common_code import visualize


#Set up the state transition map as a list of predecessors for each state:
predecessors = [
  [2],
  [0],
  [1], 
]



phasta = phasestatemachine.Kernel()
phasta.setParameters(
    numStates=3,
    predecessors=predecessors,
    )         

#phasta.updateTransitionTriggerInput(1e-10)


#phaseVelocityExponentsMatrix = [[0., 0., -2.],[-2,0,0.],[0., -2., 0.]]
#phasta.updateTransitionPhaseVelocityExponentInput(phaseVelocityExponentsMatrix )



t1 = 3.5
t2=11.5
phaseTarget = numpy.linspace(0, 1.0, int(2.0/phasta.dt))
phaseTarget = numpy.hstack((numpy.zeros((int(0.5/phasta.dt))), numpy.tile(phaseTarget, 20)))

#negatively bias transition towards states 2-4 to block transition from state 1:
#phasta.updateTransitionTriggerInput(bias) 
#evolve the system for some time
for i in range(int(t1/phasta.dt)):
    phasta.step()

#set one of the gains to nonzero in order to activate the enslavement
gains = [[0, 0., 0.],[80,0.,0.],[0., 0., 0.]]
phasta.updateVelocityEnslavementGain(gains)

for i in range(int(t2/phasta.dt)):
    phasta.updatePhasesInput(phaseTarget[i] * phasta.stateConnectivityMap )    
    phasta.step()

from matplotlib import pylab as plt
plt.figure(figsize=(4,2))
n = int((t2)/phasta.dt)
plt.plot(numpy.linspace(t1, t1+t2, n), phaseTarget[:n], linestyle=":", color="#AAAAAA")
#plt.plot(numpy.linspace(t1, t1+t2, n), phasta.errorHistory[:n])
visualize(phasta, t1+t2, sectionsAt=[t1], name=os.path.splitext(os.path.basename(__file__))[0], newFigure=False)
