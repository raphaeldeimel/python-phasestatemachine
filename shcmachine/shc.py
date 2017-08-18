# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as _np
from scipy.integrate import ode as _ode
from scipy.special import expit

@_np.vectorize
def _limitFunction2(x):
        """
        limits positive values
        """
        if x<0.01:
            return x
        else:
            return 2*(expit(2*x)-0.5)


def _limitFunction(x):
        """
        limits positive values
        """
        return 2*(expit(2*x)-0.5)

class SHC():
    
    def __init__(self):
         self.numStates = 0
         self.t = 0.0
         self.predecessors = [[2,4],[0],[1],[0],[3]]
         self.setParameters(5, 100., epsilon = [1e-10, 1e-9, 1e-6, 1e-9, 1e-10])         
         
         
    def setParameters(self, numStates, alpha=1.0, epsilon=1e-9, nu=2.5,  beta=1.0, dt=1e-2, reset=False):
        oldcount = self.numStates
        self.dt=dt
        self.numStates = numStates
        self.alpha = self._sanitizeParam(alpha)
        self.beta = self._sanitizeParam(beta)
        self.betaInv = 1.0/self.beta  #this is used often, so precompute once
        self.nu = self._sanitizeParam(nu)
        self.epsilon = self._sanitizeParam(epsilon) * self.beta #wiener process noise
        self.epsilonPerDt = self.epsilon *_np.sqrt(self.dt)/dt #factor accounts for the accumulation during a time step
        self.input = _np.ones((self.numStates)) * 0e0
        self.velocitylimit = _np.ones((self.numStates)) * 8e1
        self.velocitylimit[2] = 1e2
        self.velocitylimitInv = 1./self.velocitylimit
        self.updateRho()
        if self.numStates != oldcount or reset: #force reset if number of states change
            self.statevector = _np.zeros((numStates))
            self.statevector[0] = self.beta[0] #start at state 0

       
    def _sanitizeParam(self, p):
        if (type(p) is float) or (type(p) is int):
            sanitizedP = _np.empty((self.numStates))
            sanitizedP.fill(float(p))
        else:
            try:
                p = p[0:self.numStates]
            except IndexError:
                raise Exception("Parameter has not the length of numStates!")
            sanitizedP = _np.array(p)
        return sanitizedP
        
        
    def updateRho(self):
        rho = _np.zeros((self.numStates, self.numStates))
        #first, fill the standard inhibitory values:
        for state in range(self.numStates):
            for predecessor in range(self.numStates):
                if state == predecessor:
                    rho[state,predecessor] = self.alpha[state] * self.betaInv[state] #override the special case i==j
                else:
                    rho[state,predecessor] = (self.alpha[state] + self.alpha[predecessor]) * self.betaInv[predecessor]
        #overwrite for the predecessor states:
        for state, predecessorsPerState in enumerate(self.predecessors):
            #precedecessorcount = len(predecessorsPerState)
            for predecessor in predecessorsPerState:
                rho[state, predecessor] = (self.alpha[state] - (self.alpha[predecessor]/self.nu[predecessor])) * self.betaInv[predecessor]
        self.rho = rho
     
        
    def step(self):
            """
            Euler-Maruyama integration scheme
            
            from SHCtoolkit:     Ai = Ai+(Ai.* (alpha-Ai*rho) +mui)*dt+epsiloni.*dWi;
            """
            self.statevector= self.statevector

            mu = 0.0
            dt = self.dt

            noise_velocity = _np.random.normal(scale = self.epsilonPerDt, size=self.numStates) #discretized wiener process noise
            #noise_velocity = self.epsilonPerDt * _np.random.chisquare(1, size=self.numStates) #discretized wiener process noise
            #noise_velocity = _np.dot(self.rho, noise_velocity)
            excitation = self.alpha - _np.dot(self.rho, self.statevector)
            drift = (self.statevector * excitation + mu)  #estimate gradient
            driftLimited = self.velocitylimit * (_limitFunction(drift * self.velocitylimitInv)) #sigmoid function used as velocity limiter
            
            self.dotstatevector = driftLimited + noise_velocity
            self.statevector = _np.maximum(self.statevector + self.dotstatevector*dt , 0) #set the new state and also ensure nonegativity

            self.t = self.t + dt
            normalized_states = shc.betaInv * self.statevector
            return normalized_states**1.0
        

shc = SHC()
states = _np.vstack([shc.step() for i in range(1000)])

import matplotlib.pylab as plt
fig =plt.figure(figsize=(20,3))
for i in range(states.shape[1]):
    plt.plot( -1.0*i + states[:,i], color='b')   
    
def plotfunc(func):
    x= _np.linspace(-3., 3.0, 500)
    y = [func(v) for v in x]
    plt.plot(x,y)