# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as _np
import pandas as _pd
from scipy.special import expit, betainc
import itertools


class Kernel():
    """
    This class encapsulates a dynamical system that can behave state-like like a classical determinisitic automaton
    The transitions though are smooth, which enables a continuous synchronization of motion during such a transition
    
    The most important parameters are:
        
        numStates: the number of quasi-discrete states the system should have
        predecessors: a list of lists which defines the preceeding states of each state
            Note: Don't set up mutual predecessors (i.e. a loop with two states). This does not work. You need at least 3 states for a loop
        alpha: "state growth factor" determines the speed at which a state becomes dominant. Effectively speeds up or slows down the machine
        nu: "saddle value": Determines how easy it is to push away from the state. 
        Attention: This value has a lower boundary depending on the change of alpha between preceeding and next state
        If you see unstable behavior or "ringing", turn up this value
        
        epsilon: "noise" added to the states, which has the effect of reducing the average dwell time for the preceeding states
        
    LEss important paramters:     
        beta: scaling factor for the state variable.
        dt: time step at which the system is simulated
        
        
    states:
        phase matrix: A (numStates x numStates) matrix aggregating all phase variables for each possible transition, plus the state vector on the diagonal
        activation matrix: A matrix which contains the corresponding transition activation values. state activations correspond to the 1-sum(transition activations)
                            so that sum(matrix) = 1 (i.e.e can be used as a weighing matrix)
        
    inputs:
        observed phase: A matrix analogous to the phase matrix, containing phase estimates conditional to the transition or phase being activated
        observed phase confidence: A matrix analogous to the activation matrix, which indicates how confident the state observation is
        inputbias: vector that signals which state should currently be the next (e.g. from perception)
        
        
    """
    
    def __init__(self):
         self.numStates = 0
         self.t = 0.0
         self.setParameters()
         
         
    def setParameters(self, numStates=3, predecessors=[[2],[0],[1]], alpha=40.0, epsilon=1e-9, nu=1.5,  beta=1.0, gamma=20.0, dt=1e-2, reset=False):
        oldcount = self.numStates
        self.dt=dt
        self.dtInv=1./dt
        self.numStates = numStates
        self.alpha = self._sanitizeParam(alpha)
        self.beta = self._sanitizeParam(beta)
        self.betaInv = 1.0/self.beta  #this is used often, so precompute once
        self.gamma= gamma
        self.nu = self._sanitizeParam(nu)
        self.epsilon = self._sanitizeParam(epsilon) * self.beta #wiener process noise
        self.epsilonPerDt = self.epsilon *_np.sqrt(self.dt)/dt #factor accounts for the accumulation during a time step
        self.input = _np.ones((self.numStates)) * 0e0
        self.predecessors = predecessors
        self.transitionTriggerInput = _np.zeros((self.numStates)) #input to trigger state transitions
        self.phasesInput = _np.zeros((self.numStates,self.numStates)) #input to synchronize state transitions (slower/faster)
        self.velocityAdjustmentGain = _np.zeros((self.numStates,self.numStates))  #gain of the control enslaving the given state transition
        self.phaseVelocityExponentInput = _np.zeros((self.numStates,self.numStates))  #contains values that limit transition velocity
        self.updateRho()
        if self.numStates != oldcount or reset: #force reset if number of states change
            self.statevector = _np.zeros((numStates))
            self.dotstatevector = _np.zeros((numStates))
            self.statevector[0] = self.beta[0] #start at state 0
            self.phasesActivation = _np.zeros((self.numStates,self.numStates))
            self.phasesActivationBeta = _np.zeros((self.numStates,self.numStates))
            self.phasesProgress = _np.zeros((self.numStates,self.numStates))
            #columnnames = ["S{0}".format(i) for i in range(self.numStates)]
            self.statehistory = _np.empty((1000000, self.numStates+1))
            self.statehistory.fill(_np.nan)
            self.phasesActivationHistory= _np.empty((1000000, self.numStates,self.numStates))
            self.phasesProgressHistory = _np.empty((1000000, self.numStates,self.numStates))
            self.errorHistory = _np.empty((1000000))
            self.historyIndex = -1

    def updatePredecessors(self, listoflist):
        self.predecessors=listoflist
        self.updateRho()



    def updateTransitionTriggerInput(self, successorBias):
        """
        add a vector that biases the shc to start transitions towards that state
        use this to e.g. catch up with states or trigger/suppress transition to a specific state
        """
        _np.copyto(self.transitionTriggerInput, successorBias)
        
    def updatePhasesInput(self, phases):
        """
        add a phase to enslave the system to 
        
        Use this to sync the system with a phase from perception
        """
        _np.copyto(self.phasesInput, phases)

    
    def updateVelocityEnslavementGain(self, gains):    
        """
        Set the gain values to use for each phase transition.
        """
        _np.copyto(self.velocityAdjustmentGain, gains)
        

    def updateTransitionPhaseVelocityExponentInput(self, limits):
        """
        Update the matrix that specifies how fast the given phases should progress
        
        Each element effectively is an exponent with base 2 for adjusting each phase velocity individually
        
        limits[j,i]: exponent for the transition from i to j
        limits[i,i]: 0 (enforced implicitly)
        
        
        While phase velocity can also be controlled by the self.alpha vector directly, 
        large variations to individual states' alpha parameter can alter the 
        convergence behavior and we may lose the stable heteroclinic channel properties
        
        This method here effectly "scales" the timeline during transitions
        """
        _np.copyto(self.phaseVelocityExponentInput, limits)
        _np.fill_diagonal(self.phaseVelocityExponentInput , 0)
    
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
        """
        update the rho matrix, and also compute the full state transition 
        """
        rho = _np.zeros((self.numStates, self.numStates))
        #first, fill the standard inhibitory values:
        for state in range(self.numStates):
            for predecessor in range(self.numStates):
                if state == predecessor:
                    rho[state,predecessor] = self.alpha[state] * self.betaInv[state] #override the special case i==j
                else:
                    rho[state,predecessor] = (self.alpha[state] + self.alpha[predecessor]) * self.betaInv[predecessor]
        #overwrite for the predecessor states:
        self.stateConnectivityMap = _np.zeros((self.numStates, self.numStates))
        for state, predecessorsPerState in enumerate(self.predecessors):
            #precedecessorcount = len(predecessorsPerState)
            for predecessor in predecessorsPerState:
                if state == predecessor: raise ValueError("Cannot set a state ({0}) as predecessor of itself!".format(state))
                rho[state, predecessor] = (self.alpha[state] - (self.alpha[predecessor]/self.nu[predecessor])) * self.betaInv[predecessor]
                self.stateConnectivityMap[state, predecessor] = 1 
        self.rho = rho
        self._staticWeighingMatrix = self.stateConnectivityMap  + _np.identity(self.numStates)  * (0.25**0.5)  #static matrix to properly normalize phase progress values        

        
     
    def _recordState(self):
            self.historyIndex = self.historyIndex + 1
            if self.statehistory.shape[0] < self.historyIndex:
                print("(doubling history buffer)")
                self.statehistory.append(_np.empty(self.statehistory.shape)) #double array size
                self.phasesActivationHistory.append(_np.empty(self.phasesActivationHistory.shape))
                self.phasesProgressHistory.append(_np.empty(self.phasesProgressHistory.shape))                
            self.statehistory[self.historyIndex, 0] = self.t
            self.statehistory[self.historyIndex, 1:self.numStates+1] = self.statevector
            self.phasesActivationHistory[self.historyIndex, :,:] = self.phasesActivation
            self.phasesProgressHistory[self.historyIndex, :,:] = self.phasesProgress

    def getHistory(self):
             return  self.statehistory[:self.historyIndex,:], self.phasesActivationHistory[:self.historyIndex,:,:], self.phasesProgressHistory[:self.historyIndex,:,:]
                  
    
    def getCurrentPhaseProgress(self, ofState=None):
         if ofState is None:
             return _np.sum(self.phasesActivation * self.phasesProgress)
         else:
             return None #not implemented yet
        

        
    def step(self):
            """
            Euler-Maruyama integration scheme
            
            from SHCtoolkit:     Ai = Ai+(Ai.* (alpha-Ai*rho) +mui)*dt+epsiloni.*dWi;
            """
            #advance time
            self.t = self.t + self.dt
                        
            noise_velocity = _np.random.normal(scale = self.epsilonPerDt, size=self.numStates) #discretized wiener process noise

            phaseVelocityAdjustment = 2**_np.sum(self.phasesActivation * self.phaseVelocityExponentInput) #compute adjustment to the instantaneously effective growth factor
            
            #copmute the values for phase enslavement:
            phaseerrors = self.phasesActivation * (self.phasesInput-self.phasesProgress)
            correctiveAction = phaseerrors * self.velocityAdjustmentGain
            statedelta = _np.sum(correctiveAction , axis=1) - _np.sum(correctiveAction, axis=0)
            self.errorHistory[self.historyIndex] = _np.sum(correctiveAction)
            self.statevector = self.statevector #- 0.2 * statedelta
            mu = statedelta
            
            #This is the computation of the PhaSta core:
            excitation = self.alpha - _np.dot(self.rho, self.statevector)             
            velocity = (self.statevector * excitation * phaseVelocityAdjustment  + mu)  #estimate velocity            
            self.dotstatevector = velocity + noise_velocity + self.transitionTriggerInput
            self.statevector = _np.maximum(self.statevector + self.dotstatevector*self.dt , 0) #set the new state and also ensure nonegativity

            #prepare a normalized state vector for the subsequent operations:
            statevector_normalized = self.statevector*self.betaInv
            s = statevector_normalized.reshape((-1,1))  #proper row vector  
            
            #compute the phase activation matrix (Lambda)
            #phasesActivation = 2 * expit(self.gamma * ((s @ s.T) * self.stateConnectivityMap - 0.0)) - 1.
            phasesActivation = betainc(2,5, _np.clip(4 *  (s @ s.T) * self.stateConnectivityMap-0.05 ,0.0,1))

            #compute the state activation and put it into the diagonal:
            self.phasesActivation = phasesActivation + _np.diag(statevector_normalized  * (1.0-_np.sum(phasesActivation)))
            
            #compute the phase progress matrix (Psi)
            s_square = s.repeat(len(statevector_normalized), axis=1)
            self.phasesProgress = betainc(3,3, _np.clip(0.5 + 0.5 * ( s_square - s_square.T), 0.0, 1.0))

            self._recordState()
            return self.statevector       
