# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains the main code for the phase-state machine

"""
import numpy as _np
import pandas as _pd
from scipy.special import expit, betainc
import itertools


class Kernel():
    """
    This class provides a dynamical system that can behave like a state machine.
    The transitions are smooth though, which enables interestingbehaviors like online-synchronisation and negotiation of branch alternatives
    
    
    
    The most important parameters are:
        
        numStates: the number of quasi-discrete states the system should have
        predecessors: a list of lists which defines the preceeding states of each state
            Note: Don't set up mutual predecessors (i.e. a loop with two states). This does not work. You need at least 3 states for a loop.
        alpha: determines the speed at which a state becomes dominant. Effectively speeds up or slows down the machine
        epsilon: "noise" added to the states, which has the effect of reducing the average dwell time for the preceeding states
        
    Less important paramters:     
        beta: scaling factor for the state variable (usually 1.0)
        nu: determines how easy it is to push away from a state (usually 1.5). 
        dt: time step at which the system is simulated (default: 1e-2)

        
    Inputs:
        Observed phase Psi_d: A matrix analogous to the phase matrix, containing phase estimates conditional to the transition or phase being activated
        phase control gain K_p: A matrix analogous to the activation matrix, which indicates how confident the state observation is
        inputbias: vector that signals which state should currently be the next (e.g. from perception)
        
        
    Output:
        stateVector: The actual, evolving state of the dynamical system.
        phase matrix Psi: A (numStates x numStates) matrix aggregating all phase variables for each possible transition, plus the state vector on the diagonal
        activation matrix Lambda: A matrix which contains the corresponding transition activation values. state
        activations correspond to the 1-sum(transition activations), so that sum(matrix) = 1 (i.e.e can be used as a
        weighing matrix)
        
        
        
    """
    
    def __init__(self, **kwargs):
         self.numStates = 0
         self.t = 0.0
         self.statehistorylen = 0
         self.historyIndex = 0
         self.setParameters(**kwargs)
         

    def setParameters(self, numStates=3, predecessors=None, successors=[[1],[2],[0]], alpha=40.0, epsilon=1e-9, nu=1.5,  beta=1.0, dt=1e-2, reset=False, recordSteps=-1):
        """
        Method to set or reconfigure the phase-state-machine
        
        numStates:  The number of states the system should have
        predecessors: A list of lists which contain the state indices of the respective predecessors
        successors: A list of lists which contain the state indices of the respective successors
            Note: use of predecessors and successors parameter is mutually exclusive!
                
        For the meaning of the other parameters, please consult the paper or the code
        """
        oldcount = self.numStates
        #parameters:
        self.dt=dt
        self.dtInv=1./dt
        self.numStates = numStates
        self.alpha = self._sanitizeParam(alpha)
        self.beta = self._sanitizeParam(beta)
        self.betaInv = 1.0/self.beta             #this is used often, so precompute once
        self.nu = self._sanitizeParam(nu)
        self.epsilon = self._sanitizeParam(epsilon) * self.beta #wiener process noise
        self.epsilonPerDt = self.epsilon *_np.sqrt(self.dt)/dt #factor accounts for the accumulation during a time step

        if predecessors is not None:  #convert list of predecessors into list of successors
            self.successors = self._predecessorListToSuccessorList(predecessors)
        else:
            self.successors = successors
            
        self.nonlinearityParamsLambda = (2,5)    #parameters of the beta distribution nonlinearity for computing the Lambda matrix values
        self.nonlinearityParamsPsi    = (3,3)    #parameters of the beta distribution nonlinearity that linearizes phase variables
        self.activationThreshold = 0.05          #clip very small activations below this value to avoid barely activated states

        #inputs:
        self.BiasMatrix = _np.zeros((self.numStates,self.numStates)) #determines transition preferences and state timeout duration
        
        self.phasesInput = _np.zeros((self.numStates,self.numStates)) #input to synchronize state transitions (slower/faster)
        self.velocityAdjustmentGain = _np.zeros((self.numStates,self.numStates))  #gain of the control enslaving the given state transition
        self.phaseVelocityExponentInput = _np.zeros((self.numStates,self.numStates))  #contains values that limit transition velocity
        
        self._updateRho()
        #internal data structures
        if self.numStates != oldcount or reset: #force a reset if number of states change
            self.statevector = _np.zeros((numStates))
            self.dotstatevector = _np.zeros((numStates))
            self.statevector[0] = self.beta[0] #start at state 0
            self.phasesActivation = _np.zeros((self.numStates,self.numStates))
            self.phasesActivationBeta = _np.zeros((self.numStates,self.numStates))
            self.phasesProgress = _np.zeros((self.numStates,self.numStates))
            self.phasesProgressVelocities = _np.zeros((self.numStates,self.numStates))
            self.biases = _np.zeros((self.numStates, self.numStates))
            self._biasMask = (1-_np.eye((self.numStates)))
            
            #these data structures are used to save the history of the system:
            if recordSteps< 0:
                pass
            elif recordSteps == 0:
                self.statehistorylen = 0
                self.historyIndex = 0
            else:
                self.statehistorylen = recordSteps
                self.statehistory = _np.empty((self.statehistorylen, self.numStates+1))
                self.statehistory.fill(_np.nan)
                self.phasesActivationHistory= _np.zeros((self.statehistorylen, self.numStates,self.numStates))
                self.phasesProgressHistory = _np.zeros((self.statehistorylen, self.numStates,self.numStates))
                self.errorHistory = _np.zeros((self.statehistorylen))
                self.historyIndex = 0



    def _updateRho(self):
        """
        internal method to compute the P matrix from preset parameters
        
        also computes the state connectivity matrix
        
        reimplements the computation by the SHCtoolbox code  
        """
        rho = _np.zeros((self.numStates, self.numStates))
        stateConnectivity = _np.zeros((self.numStates, self.numStates))
        #first, fill in the standard inhibitory values:
        for state in range(self.numStates):
            for successor in range(self.numStates):
                if state == successor:
                    rho[successor,state] = self.alpha[successor] * self.betaInv[successor] #override the special case i==j
                else:
                    rho[successor,state] = (self.alpha[successor] + self.alpha[state]) * self.betaInv[state]
        #overwrite for the predecessor states:
        for state, successorsPerState in enumerate(self.successors):
            #precedecessorcount = len(predecessorsPerState)
            for successor in successorsPerState:
                if state == successor: raise ValueError("Cannot set a state ({0}) as successor of itself!".format(state))
                rho[successor,state] = (self.alpha[successor] - (self.alpha[state]/self.nu[state])) * self.betaInv[state]
                stateConnectivity[successor,state] = 1 
        self.rho = rho #save the final result
        self.stateConnectivity = stateConnectivity








    def step(self, period=None, until=None):
            """
            Main algorithm, implementing the integration step, state space decomposition, phase control and velocity adjustment.
            
            """
            #if a period is given, iterate until we finished that period:            
            if period is not None:
                endtime = self.t + period - 0.5*self.dt
                while self.t < endtime:
                    self.step(period=None, until=until)
            if until is not None: 
                while self.t < until:
                    self.step()

            kd = 2**_np.sum(self.phasesActivation * self.phaseVelocityExponentInput) #compute adjustment to the instantaneously effective growth factor
            
            #compute mu for phase control:
            phaseerrors = self.phasesActivation * (self.phasesInput-self.phasesProgress)
            correctiveAction = phaseerrors * self.velocityAdjustmentGain
            statedelta = _np.sum(correctiveAction , axis=1) - _np.sum(correctiveAction, axis=0)
            if self.historyIndex < self.statehistorylen:
                self.errorHistory[self.historyIndex] = _np.sum(correctiveAction)
            self.statevector = self.statevector #- 0.2 * statedelta
            mu = statedelta

            noise_velocity = _np.random.normal(scale = self.epsilonPerDt, size=self.numStates) #discretized wiener process noise

            #compute which transition biases should be applied right now:
            self.biases = _np.dot(self.BiasMatrix, self.statevector)

            
            #This is the core computation and time integration of the dynamical system:
            growth = self.alpha - _np.dot(self.rho, self.statevector)
            velocity = self.statevector * growth * kd  + mu  #estimate velocity
            self.dotstatevector = velocity + noise_velocity + self.biases
            self.statevector = _np.maximum(self.statevector + self.dotstatevector*self.dt , 0) #set the new state and also ensure nonegativity
            
            self.t = self.t + self.dt #advance time

            #prepare a normalized state vector for the subsequent operations:
            statevector_normalized = self.statevector*self.betaInv

            #compute the transition/state activation matrix (Lambda)
            s = statevector_normalized.reshape((-1,1))  #creates a proper row vector
            phasesActivation = 4 * _np.dot(s, s.T) * self.stateConnectivity 
            phasesActivation = betainc(self.nonlinearityParamsLambda[0],self.nonlinearityParamsLambda[1], _np.clip(phasesActivation,0,1))
            phasesActivation = _np.clip( (phasesActivation - self.activationThreshold) / (1.0 - 2*self.activationThreshold) , 0, 1)  #makes sure that we numerically saturate and avoid very small, residual activations
            
            #compute the state activation and put it into the diagonal of Lambda:
            self.phasesActivation = phasesActivation  + _np.clip( _np.diag(statevector_normalized**2) * (1.0-_np.sum(phasesActivation)), 0,1)
            
            #compute the phase progress matrix (Psi)
            s_square = s.repeat(len(statevector_normalized), axis=1)
            newphases = betainc(self.nonlinearityParamsPsi[0],self.nonlinearityParamsPsi[1], _np.clip(0.5 + 0.5 * ( s_square - s_square.T), 0.0, 1.0))
            self.phasesProgressVelocities = (newphases - self.phasesProgress) * self.dtInv
            self.phasesProgress = newphases

            #note the currently most active state/transition (for informative purposes)
            i = _np.argmax(self.phasesActivation)
            self.currentPredecessor = i % self.numStates
            self.currentSuccessor = i // self.numStates


            self._recordState()
            return self.statevector
            



    def get1DState(self):
        """
        return value of a one-dimensional signal that indicates which state we are in, or in which transition
        """
        value = self.currentPredecessor + (self.currentSuccessor - self.currentPredecessor) * self.phasesProgress[self.currentSuccessor,self.currentPredecessor]
        return value
    

    def sayState(self):
        """
        returns a string describing the current state
        """
        if self.currentPredecessor == self.currentSuccessor:
            return "{0}".format(self.currentPredecessor )
        else:
            return "{0}->{1}".format(self.currentPredecessor , self.currentSuccessor)




    def updateSuccessors(self, listoflist):
        """
        recompute the system according to the given list of predecessors
        """
        self.successors=listoflist
        self._updateRho()


    def _predecessorListToSuccessorList(self, predecessors):
        """ helper to convert lists of predecessor states into lists of successor states"""
        successors = [ [] for i in range(self.numStates) ] #create a list of lists
        for i, predecessorsPerState in enumerate(predecessors):
            for pre in predecessorsPerState: 
                successors[pre].append(i)
        return successors
    

    def updatePredecessors(self, listoflist):
        """
        recompute the system according to the given list of predecessors
        """
        self.successors = self._predecessorListToSuccessorList(predecessors)
        self._updateRho()
        
    def getPredecessors(self):
        """
        return the predecessors
        """
        successors = [ [] for i in range(self.numStates) ] #create a list of lists
        for i, predecessorsPerState in enumerate(predecessors):
            for pre in predecessorsPerState: 
                successors[pre].append(i)
        return successors


    def updateB(self, successorBias):
        """
        changes the "bias" input array
        
        Small values bias the system to hasten transitions towards that state
        
        Large, short spikes can be used to override any state and force the system into any state, 
        regardless of state connectivity
        
        successorBias: matrix of biases for each (successor state, current state) 
       
        Note: states cannot be their own successors, so these values ignored!
        """
        bias = _np.asarray(successorBias)
        self.BiasMatrix = self._biasMask * bias 
        
        
    def updateTransitionTriggerInput(self, successorBias):
        """
        changes the "bias" input array (or vector)
        
        Small values bias the system to hasten transitions towards that state
        
        Large, short spikes can be used to override any state and force the system into any state, 
        regardless of state connectivity
        
        successorBias: 
            if scalar:  set all successor biases to the same value
            if vector:  set successor biases to the given vector for every state
            if matrix:  set each (successor state, current state) pair individually
       
        Note: states cannot be their own successors, so these values ignored!                
        """
        bias = _np.asarray(successorBias)
        if bias.ndim == 0:
            self.updateB(bias * self._biasMask)
        if bias.ndim == 1:
            self.updateB(bias[:, _np.newaxis] * self._biasMask)
        elif bias.ndim == 2:
            self.updateB(bias)
        
    def updatePhasesInput(self, phases):
        """
        changes the Psi_d matrix
        
        Use this as phase reference to sync the system with a phase from perception
        """
        _np.copyto(self.phasesInput, phases)
    
    def updateVelocityEnslavementGain(self, gains):    
        """
        changes the K_p matrix
        
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

    def getHistory(self):
        """
        return the historic values for plotting
        """
        if self.statehistorylen == 0:
            raise RuntimeError("no history is being recorded")
        return  (self.statehistory[:self.historyIndex,:],
                  self.phasesActivationHistory[:self.historyIndex,:,:],
                  self.phasesProgressHistory[:self.historyIndex,:,:]
        )

    def _sanitizeParam(self, p):
        """
        internal helper to provide robust handling of lists and numpy array input data
        """
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
     
    def _recordState(self):
        """
        internal helper to save the current state for later plotting
        """
        if self.historyIndex < self.statehistorylen:
            self.statehistory[self.historyIndex, 0] = self.t
            self.statehistory[self.historyIndex, 1:self.numStates+1] = self.statevector
            self.phasesActivationHistory[self.historyIndex, :,:] = self.phasesActivation
            self.phasesProgressHistory[self.historyIndex, :,:] = self.phasesProgress
        if self.historyIndex < self.statehistorylen:
            self.historyIndex = self.historyIndex + 1


