# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains the main code for the phase-state machine

"""
import numpy as _np
import pandas as _pd
import itertools
from numba import jit

import warnings as _warnings


@jit(nopython=True, cache=True)
def _limit(a):
    """ 
    faster version of numpy clip, also modifies array in place
    """
    #numba doesn't support indexing by boolean
    #a[a<lower]=lower
    #a[a>upper]=upper
    shape = a.shape
    for j in range(shape[1]):
        for i in range(shape[0]):
            if a[i,j] < 0.0:
                a[i,j] = 0.0
            if  a[i,j] > 1.0:
                a[i,j] = 1.0


@jit(nopython=True, cache=True)
def _signfunc(x):
     return 1.0-2*(x<0)

@jit(nopython=True, cache=True)
def ReLU(x):
     return 0.5*(_np.abs(x)+x)

# Alternative, differentiable "sign" function
# Also improves stability of the state's sign
#@jit(nopython=True)
#def _signfunc(x, epsilon=1e-3):
#     return _np.tanh(x/epsilon)

#_np.set_printoptions(precision=3, suppress=True)

@jit(nopython=True, cache=True)
def _step(statevector,                      #main state vector.  Input and output, modified in-place
          #outputs, modified in place:
          dotstatevector,                   #velocity of main state vector
          activationMatrix,                 #Activation for each potential state and transition
          phasesMatrix,                     #Phases for each transition
          phaseVelocitiesMatrix,            #Derivative of phases for each transition 
          #inputs:
          phaseVelocityExponentInput,       #input to modify velocity of each transition individually (exponential scale, basis 2)
          BiasMatrix,                       #input to depart / avert departure from states
          stateConnectivityGreedinessAdjustment,  #input to modify how strong a successor state pulls the system towards itself, relative to the predecessor state  
          stateConnectivityCompetingGreedinessAdjustment, #input to adjust greediness in between compeeting successor states
          phasesInput,                      # phase target in case a transition is enslaved to an external phase 
          velocityAdjustmentGain,           # gain related to enslaving phase
          noise_velocity,                   # vector that gets added to state velocity (usually in order to inject some base noise) 
          #parameters:
          numStates,                        #number of states / dimensions
          betaInv,                          #precomputed from beta parameter (state locations / scale)
          stateConnectivityAbs,             #precomputed from state graph
          stateConnectivitySignMap,         #precomputed from state graph
          stateConnectivityIsBidirectional, #precomputed from state graph
          stateConnectivityNrEdges,         #precomputed from state graph 
          rhoZero,                          #coupling values for creating discrete states
          rhoDelta,                         #coupling values for creating stable heteroclinic channels
          alpha,                            #growth rate of states, determines speed of transitioning
          dt,                               # time step duration in seconds
          dtInv,                            #precomputed from dt
          nonlinearityParamsLambda,         #Kumaraswamy distribution parameters to modify gradualness of activation            
          nonlinearityParamsPsi,            #Kumaraswamy distribution parameters to modify gradualness of phase progress
          stateVectorExponent,              #modifies the bending of heteroclinic channels
          speedLimit,                       #safety limit to state velocity
          epsilonLambda,                    #determines the region of zero activation around the coordinates axes 
          #for comparative study:
          emulateHybridAutomaton,           #set this to true to hack phasta into acting like a discrete state graph / hybrid automaton
          triggervalue_successors,          #for HA emulation mode, modified in-place
          ):
        """
        Core phase-state machine computation.

        Written as a function in order to be able to optimize it with numba
        
        Note: The function modifies several arguments (numpy arrays) in place.
        """
        #compute adjustment to the instantaneously effective growth factor
        scaledactivation = activationMatrix * (1.0 / max(1.0, _np.sum(activationMatrix)))
        kd = 2** _np.sum( scaledactivation * phaseVelocityExponentInput)
        
        #compute mu for phase control:
        phaseerrors = activationMatrix * (phasesInput-phasesMatrix)
        correctiveAction = phaseerrors * velocityAdjustmentGain
        correctiveActionPredecessor = _np.zeros((numStates))
        for i in range(numStates):
            correctiveActionPredecessor += correctiveAction[:,i]
        correctiveActionSuccessor = _np.zeros((numStates))
        for i in range(numStates):
            correctiveActionSuccessor += correctiveAction[i,:]
        mu = correctiveActionPredecessor - correctiveActionSuccessor

        statevector_abs = _np.abs(statevector)
        #adjust signs of the bias values depending on the transition direction:
        biases = _np.dot(BiasMatrix * stateConnectivitySignMap * _np.outer(1-statevector_abs,statevector_abs), statevector)
        noise_statevector = noise_velocity * dt
        
        #If requested, decide whether to start a transition using a threshold, and stick to that decision no matter what until the transition finishes
        if emulateHybridAutomaton:
            predecessors = 1.0*(_np.abs(statevector)*betaInv > 0.99)
            successors =  (_np.dot(stateConnectivityAbs,  predecessors) > 0.5 )
            notsuccessors =  (_np.dot(stateConnectivityAbs,  predecessors) < 0.5 )
            triggervalue_successors[notsuccessors] = 0.0
            noise_statevector = _np.zeros((numStates))
            threshold = 0.1
            if _np.any(triggervalue_successors >= threshold ):
                chosensuccessor = _np.argmax(triggervalue_successors)
                value_chosen = triggervalue_successors[chosensuccessor]
                notchosensuccessors = successors.copy()
                notchosensuccessors[chosensuccessor] = 0

                triggervalue_successors[:] = 0.0
                triggervalue_successors[chosensuccessor] = value_chosen
                
                if triggervalue_successors[chosensuccessor] < 1e5:
                    triggervalue_successors[ chosensuccessor ] = 1e6
                    #print(chosensuccessor)
                    noise_statevector[chosensuccessor] = 1.0
            else:
                 triggervalue_successors[:] += biases * dt + noise_velocity
        
        statevector[:] = statevector #for numba

        statesigns = _signfunc(statevector)
        statesignsOuterProduct = _np.outer(statesigns,statesigns) #precompute this, as we need it several times


        #stateVectorExponent=1  #straight channels: |x|  (original SHC by Horchler/Rabinovich)
        #stateVectorExponent=2  #spherical channels: |x|**2 (default for phasta)
        x_gamma = (statevector*statesigns)**stateVectorExponent
                        
        #Compute a mask that ensures the attractor works with negative state values too, that the transition's "sign" is observed, and that unidirectional edges do not accidentally change between positive and negative state values
        #the computation is formulated such that only algebraic and continuous functions (e.g. ReLu) are used
        M_T = ReLU(statesignsOuterProduct*stateConnectivitySignMap) 
        #Appropriate signs for transition-related greediness adjustment, depending on whether a graph edge is bidirectional or not:
        TransitionGreedinessAdjustmentSign = (stateConnectivityNrEdges * ReLU(statesignsOuterProduct) - stateConnectivityIsBidirectional) * stateConnectivitySignMap 
        #sum everything into a transition/greedinesses matrix (T+G):
        T_G = M_T*stateConnectivityAbs + TransitionGreedinessAdjustmentSign*stateConnectivityGreedinessAdjustment + stateConnectivityCompetingGreedinessAdjustment

        #This is the core computation and time integration of the dynamical system:
        growth = alpha + _np.dot(rhoZero, x_gamma) + _np.dot(rhoDelta * T_G, x_gamma)
        dotstatevector[:] = statevector * growth * kd + mu + biases  #estimate velocity. do not add noise to velocity, promp mixer doesnt like jumps

        dotstatevector_L2 = _np.sqrt(_np.sum(dotstatevector**2))
        velocity_limitfactor = _np.minimum(1.0, speedLimit/(1e-8 + dotstatevector_L2))  #limit speed of the motion in state space to avoid extreme phase velocities that a robot cannot         
        statevector[:] = (statevector + dotstatevector*dt*velocity_limitfactor + noise_statevector)   #set the new state 
        
        #prepare a normalized state vector for the subsequent operations:
        statevector_abs = _np.abs(statevector)
        S = statevector_abs.reshape((numStates,1))
        S2 = S*S
        S_plus_P = S + S.T
        statevectorL1 = _np.sum(S)
        statevectorL2 = _np.sum(S2)
        #compute the transition/state activation matrix (Lambda)
        activations = stateConnectivitySignMap * _np.outer(statevector, statevector) * 16 * (statevectorL2) / (S_plus_P**4+statevectorL1**4)
        activationMatrix[:,:] =  activations * stateConnectivityAbs #function shown in visualization_of_activationfunction.py
        _limit(activationMatrix)
        #apply nonlinearity:
        if (nonlinearityParamsLambda[0] != 1.0 or nonlinearityParamsLambda[1] != 1.0 ):
            activationMatrix[:,:] = 1.0-(1.0-activationMatrix**nonlinearityParamsLambda[0])**nonlinearityParamsLambda[1] #Kumaraswamy CDF
        
        #compute the state activation and put it into the diagonal of Lambda:
        residual = max(0.0, 1.0 - _np.sum(activationMatrix))
        stateactivation_normalized = S2/ _np.sum(S2) 
        for i in range(numStates):
            activationMatrix[i,i] =  stateactivation_normalized[i,0] * residual
                
        #compute the phase progress matrix (Psi)
        epsilonPsi = 0.0001
        newphases = (S+epsilonPsi) / (S_plus_P+2*epsilonPsi)
        _limit(newphases)
        #apply nonlinearity:
        if (nonlinearityParamsPsi[0] != 1.0 or nonlinearityParamsPsi[1] != 1.0 ):
            newphases = 1.0-(1.0-newphases**nonlinearityParamsPsi[0])**nonlinearityParamsPsi[1] #Kumaraswamy CDF
        
        phaseVelocitiesMatrix[:,:] = (newphases - phasesMatrix) * dtInv
        phasesMatrix[:,:] = newphases

        return
        

_KumaraswamyCDFParameters = {
    'kumaraswamy1,1': (1.,1.),
    'kumaraswamy2,1': (2.,1.),
    'kumaraswamy1,2': (1.,2.),
    #values for the Kumaraswamy CDF that approximate the given incomplete beta function:
    'beta2,2': (1.913227338072261,2.2301669931409323),
    'beta3,3': (2.561444544688591,3.680069490606511),
    'beta2,5': (1.6666251656562021,5.9340642444701555),
}



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
         

    def setParameters(self, 
            numStates=3, 
            predecessors=None, 
            successors=[[1],[2],[0]], 
            alphaTime=None, 
            alpha=40.0, 
            epsilon=1e-9,
            nu=1.0,  
            beta=1.0, 
            dt=1e-2, 
            stateVectorExponent=2.0,
            speedLimit = _np.inf,
            initialState=0,
            nonlinearityLambda='kumaraswamy1,1',
            nonlinearityPsi='kumaraswamy1,1',
            inputFilterTimeConstant = 0.1,
            reuseNoiseSampleTimes = 10,
            reset=False, 
            recordSteps=-1,
            emulateHybridAutomaton=False):
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
        self.numStates = numStates
        if alphaTime is None:  #backwards compatibility: if no alphatime is provided, use dt-dependent alpha value
            self.alphaTime = self._sanitizeParam(alpha)/dt
        else:
            self.alphaTime = self._sanitizeParam(alphaTime)
            
        self.beta = self._sanitizeParam(beta)
        self.betaInv = 1.0/self.beta             #this is used often, so precompute once
        self.nu = self._sanitizeParam(nu)
        self.nu_term = self.nu/(1 + self.nu)  #equations usually use this term - precompute it
        self.epsilon = self._sanitizeParam(epsilon) * self.beta #Wiener process noise
        self.epsilonLambda=0.01 #regularization parameter of activation function
        self.maxGreediness=10.0  #maximum factor to allow for increasing decisiveness (mainly to guard against input errors)
        self.reuseNoiseSampleTimes = reuseNoiseSampleTimes
        self.stateVectorExponent =stateVectorExponent
        self.speedLimit = speedLimit
        if initialState >= self.numStates:
            raise ValueError()
        self.initialState = initialState
        if predecessors is not None:  #convert list of predecessors into list of successors
            self.successors = self._predecessorListToSuccessorList(predecessors)
        else:
            self.successors = successors
        
        self.updateDt(dt) #also calls self._updateRho
        
        self.nonlinearityParamsLambda = _KumaraswamyCDFParameters[nonlinearityLambda]   #nonlinearity for sparsifying activation values
        self.nonlinearityParamsPsi  = _KumaraswamyCDFParameters[nonlinearityPsi]     #nonlinearity that linearizes phase variables 


        #inputs:
        self.BiasMatrix = _np.zeros((self.numStates,self.numStates)) #determines transition preferences and state timeout duration
        self.BiasMatrixDesired = _np.zeros((self.numStates,self.numStates)) #determines transition preferences and state timeout duration

        self.emulateHybridAutomaton = emulateHybridAutomaton #set this to true to emulate discrete switching behavior on bias input
        self.triggervalue_successors = _np.zeros((self.numStates))
        
        self.phasesInput = _np.zeros((self.numStates,self.numStates)) #input to synchronize state transitions (slower/faster)
        self.velocityAdjustmentGain = _np.zeros((self.numStates,self.numStates))  #gain of the control enslaving the given state transition
        self.phaseVelocityExponentInput = _np.zeros((self.numStates,self.numStates))  #contains values that limit transition velocity
        self.stateConnectivityGreedinessAdjustment = _np.zeros((self.numStates,self.numStates)) #contains values that adjust transition greediness
        self.stateConnectivityCompetingGreedinessAdjustment = _np.zeros((self.numStates,self.numStates)) #contains values that adjust competing transition greediness
        self.stateConnectivityGreedinessTransitions = _np.zeros((self.numStates,self.numStates))
        self.stateConnectivityGreedinessCompetingSuccessors = _np.zeros((self.numStates,self.numStates))

        self.inputfilterK = dt / max(dt , inputFilterTimeConstant)  #how much inputs should be low-passed (to avoid sudden changes in phasta state)
        
        #internal data structures
        if self.numStates != oldcount or reset: #force a reset if number of states change
            self.statevector = _np.zeros((numStates))
            self.dotstatevector = _np.zeros((numStates))
            self.statevector[self.initialState] = self.beta[self.initialState] #start at a state 
            self.phasesActivation = _np.zeros((self.numStates,self.numStates))
            self.phasesProgress = _np.zeros((self.numStates,self.numStates))
            self.phasesProgressVelocities = _np.zeros((self.numStates,self.numStates))
            self.biases = _np.zeros((self.numStates, self.numStates))
            self.noise_velocity = 0.0
            self.noiseValidCounter = 0
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
                self.historyIndex = 0



    def _updateRho(self):
        """
        internal method to compute the P matrix from preset parameters
        
        also computes the state connectivity matrix
        
        reimplements the computation by the SHCtoolbox code  
        """
        stateConnectivityAbs = _np.zeros((self.numStates, self.numStates))
        stateConnectivitySignMap =_np.tri(self.numStates, self.numStates, k=0) - _np.tri(self.numStates, self.numStates, k=-1).T
        for state, successorsPerState in enumerate(self.successors):
            #precedecessorcount = len(predecessorsPerState)
            for successor in successorsPerState:
                if state == successor: raise ValueError("Cannot set a state ({0}) as successor of itself!".format(state))
                stateConnectivityAbs[successor,state] = 1 
                stateConnectivitySignMap[successor,state] = 1
                stateConnectivitySignMap[state, successor] = -1
        self.stateConnectivityAbs = stateConnectivityAbs
        self.stateConnectivitySignMap = stateConnectivitySignMap
        #precompute some things:
        self.stateConnectivityIsBidirectional =  _np.sqrt(self.stateConnectivityAbs * self.stateConnectivityAbs.T) 
        self.stateConnectivityNrEdges = stateConnectivityAbs + stateConnectivityAbs.T

        self.stateConnectivity = self.stateConnectivityAbs
        
        #compute a matrix that has ones for states that have a common predecessor, i.e. pairs of states which compete (except for self-competition)
        self.connectivitySigned = self.stateConnectivitySignMap*self.stateConnectivityAbs
        self.competingStates = _np.dot(self.stateConnectivityAbs, self.stateConnectivityAbs.T) * (1-_np.eye(self.numStates))
        
        #first, fill in the standard values in rhoZero
        # rhoZero = beta^-1 x alpha * (1 - I + alpha^-1 x alpha)
        alphaInv = 1/self.alpha
        s = _np.dot(self.alpha[:,_np.newaxis],self.betaInv[_np.newaxis,:])
        rhoZero = s * (_np.eye(self.numStates) - 1 - _np.dot(self.alpha[:,_np.newaxis],alphaInv[_np.newaxis,:]))
        
        #then fill the rhoDelta:
        rhoDelta = (self.alpha[:,_np.newaxis]*self.betaInv[_np.newaxis,:] / self.nu_term[:,_np.newaxis])
        
        self.rhoZero = rhoZero
        self.rhoDelta = rhoDelta
        successorCountInv = 1.0/_np.maximum(_np.sum(self.stateConnectivityAbs, axis=0)[_np.newaxis,:],1.0)
        self.BiasMeanBalancingWeights = self.stateConnectivityAbs * successorCountInv


    def step(self, until=None, period=None, nr_steps=1):
            """
            Main algorithm, implementing the integration step, state space decomposition, phase control and velocity adjustment.
            
            period: give a period to simulate 
            
            until: give a time until to simulate 
                        
            nr_steps: give the number of steps to simulate at self.dt
            
            If more than one argument is given, then precedence is: until > period > nr_steps
            
            """
            if until is not None: 
                period = until - self.t
                if period < 0.0:
                    raise RuntimeError("argument until is in the past")
            #if a period is given, iterate until we finished that period:            
            if period is not None:
                nr_steps = int(period // self.dt)
            
            for i in range(nr_steps):
                #execute a single step:
                self.t = self.t + self.dt #advance time
                self.noiseValidCounter = self.noiseValidCounter - 1
                if self.noiseValidCounter <= 0: #do not sample every timestep as the dynamical system cannot react that fast anyway. Effectively low-pass-filters the noise.
                    self.noise_velocity = _np.random.normal(scale = self.epsilonPerSample, size=self.numStates) #sample a discretized wiener process noise
                    self.noiseValidCounter = self.reuseNoiseSampleTimes
                #low-pass filter input to avoid sudden jumps in velocity
                self.BiasMatrix += self.inputfilterK * (self.BiasMatrixDesired-self.BiasMatrix) 
                self.stateConnectivityGreedinessAdjustment += self.inputfilterK * (self.stateConnectivityGreedinessTransitions - self.stateConnectivityGreedinessAdjustment)
                self.stateConnectivityCompetingGreedinessAdjustment += self.inputfilterK * (self.stateConnectivityGreedinessCompetingSuccessors -self.stateConnectivityCompetingGreedinessAdjustment)
                
                _step(  #arrays modified in-place:
                        self.statevector, 
                        self.dotstatevector,
                        self.phasesActivation, 
                        self.phasesProgress, 
                        self.phasesProgressVelocities, 
                        #inputs
                        self.phaseVelocityExponentInput, 
                        self.BiasMatrix,
                        self.stateConnectivityGreedinessAdjustment,
                        self.stateConnectivityCompetingGreedinessAdjustment,
                        self.phasesInput, 
                        self.velocityAdjustmentGain, 
                        self.noise_velocity,
                        #parameters
                        self.numStates, 
                        self.betaInv , 
                        self.stateConnectivityAbs,
                        self.stateConnectivitySignMap,
                        self.stateConnectivityIsBidirectional,
                        self.stateConnectivityNrEdges,
                        self.rhoZero, 
                        self.rhoDelta, 
                        self.alpha, 
                        self.dt,
                        self.dtInv, 
                        self.nonlinearityParamsLambda,
                        self.nonlinearityParamsPsi,
                        self.stateVectorExponent,
                        self.speedLimit,
                        self.epsilonLambda,
                        self.emulateHybridAutomaton,
                        self.triggervalue_successors
                )

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


    def updateDt(self, dt):
        """
        upadate the time step used to integrate the dynamical system:
        """
        self.dt  = dt
        self.dtInv = 1.0 / dt
        self.epsilonPerSample = self.epsilon *_np.sqrt(self.dt*self.reuseNoiseSampleTimes)/dt  #factor accounts for the accumulation during a time step (assuming a Wiener process)
        self.alpha = self.alphaTime * self.dt 
        self._updateRho()

    def updateEpsilon(self, epsilon):
        """
        Update the noise vector
        """
        self.epsilon = epsilon
        self.updateDt(self.dt) #need to recompute self.epsilonPerSample
        
 
    def updateSuccessors(self, listoflist):
        """
        recompute the system according to the given list of predecessors
        """
        self.successors=listoflist
        self._updateRho()

    def updateGreediness(self, greedinesses):
        """
        update the greediness for competing transitions / successor states
        
        Low values make the system maintain co-activated transitions for a long time, high values make transitions very competitive.
            0.0: complete indecisiveness (transitions do not compete at all and may not converge towards an exclusive successor state)
            1.0: behavior of the original SHC network by [1]
            20.0: extremely greedy transitions, behaves much like a discrete state machine
            negative values: abort transition and return to the predecessor state
            
        Absolute values less than 1.0 also reduce speed of transitions, 0.0 stops transitions completely.
        
        This value is considered during a transition away from the predecessor state,
        i.e. it influences the transition dynamics while honoring the basic state connectivity
        
        greediness: vector of size self.numStates or matrix of size (numStates,numStates)
            scalar: set a common greediness value for all competing transitions
            vector: greediness values for all competing transitions leading to the related successor state
            matrix: set greediness value for each competing transition individually
        
        """
        greedinesses = _np.asarray(greedinesses)
        if greedinesses.ndim == 1:
            greedinesses = greedinesses[_np.newaxis,:]
        elif greedinesses.ndim == 0:
            greedinesses = _np.full((1, self.numStates),greedinesses)
        
        #adjust the strength / reverse direction of the outgoing shc's according to greedinesses:
        greediness_successorstates = _np.clip((0.5*greedinesses-0.5), -1.0, 0.0) # _np.clip(g, -self.nu_term, 0.0)
        strength = self.stateConnectivityAbs * greediness_successorstates.T #works for (1,-1) transition pairs too
        self.stateConnectivityGreedinessTransitions = strength + strength.T

        #Adjust competition between nodes according to their greediness:
        kappa=0.
#        self.stateConnectivityGreedinessCompetingSuccessors = self.competingStates * 0.5*(1-(1.+kappa)*greedinesses+kappa*greedinesses.T)
        self.stateConnectivityGreedinessCompetingSuccessors = self.competingStates * 0.5*(1-greedinesses)
        


    def updateCompetingTransitionGreediness(self,greedinesses):
        _warnings.warn("Please replace updateCompetingTransitionGreediness with updateGreediness asap!", DeprecationWarning, stacklevel=2)
        self.updateGreediness(greedinesses)
        

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


    def updateBiases(self, successorBias):
        """
        changes the "bias" input array
        
        Small values bias the system to hasten transitions towards that state
        
        Large, short spikes can be used to override any state and force the system into any state, 
        regardless of state connectivity
        
        successorBias: numpy array of biases for each (successor state biased towards, current state) pair

            if scalar:  set all successor biases to the same value
            if vector:  set successor biases to the given vector for every state
            if matrix:  set each (successor state, current state) pair individually
       
        """
        bias = _np.asarray(successorBias)        
        if bias.ndim == 1:
            self.BiasMatrixDesired[:,:] = (self.stateConnectivity) * bias[:,_np.newaxis]
        else:
            self.BiasMatrixDesired[:,:] = bias

    def updateB(self, successorBias):
        _warnings.warn("Please replace updateB() with updateBiases() asap!",stacklevel=2)
        self.updateBiases(successorBias)

    def updateTransitionTriggerInput(self, successorBias):
        _warnings.warn("Please replace updateTransitionTriggerInput() with updateBiases() asap!",stacklevel=2)
        self.updateBiases(successorBias)
        
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
        
        if limits is a vector: treat it as common exponent for transitions of the same predecessor state
        if limits is a scalar: set as common exponent for all transitions
        
        While phase velocity can also be controlled by the self.alpha vector directly, 
        large variations to individual states' alpha parameter can alter the 
        convergence behavior and we may lose the stable heteroclinic channel properties
        
        This method here effectly "scales" the timeline during transitions
        """
        limits = _np.asarray(limits)
        if limits.ndim == 1:
            limits = limits[_np.newaxis,:]
        elif limits.ndim == 0:
            limits = limits[_np.newaxis,_np.newaxis]
        self.phaseVelocityExponentInput[:,:] = limits
        #_np.fill_diagonal(self.phaseVelocityExponentInput , 0.0)

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
        if  _np.isscalar(p):
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


