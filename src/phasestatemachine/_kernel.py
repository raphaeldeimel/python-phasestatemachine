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


@jit(nopython=True)
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


@jit(nopython=True)
def _signfunc(x):
     return 1.0-2*(x<0)

# Alternative, differentiable "sign" function
# Also improves stability of the state's sign
#@jit(nopython=True)
#def _signfunc(x, epsilon=1e-3):
#     return _np.tanh(x/epsilon)


@jit(nopython=True)
def _step(statevector,  #modified in-place
          dotstatevector, #modified in-place 
          activationMatrix, #modified in-place 
          phasesMatrix,  #modified in-place
          phaseVelocitiesMatrix, #modified in-place 
          #inputs:
          phaseVelocityExponentInput, 
          BiasMatrix, 
          phasesInput, 
          velocityAdjustmentGain, 
          noise_velocity,
          #parameters:
          numStates, 
          betaInv, 
          stateConnectivity, 
          rhoZero, 
          rhoDelta,
          alpha, 
          dt, 
          dtInv, 
          nonlinearityParamsLambda,
          nonlinearityParamsPsi,
          stateVectorExponent,
          epsilonLambda,
          emulateHybridAutomaton,  #for HA emulation mode
          triggervalue_successors, #for HA emulation mode, modified in-place
          ):
        """
        Core phase-state machine computation.

        Written as a function in order to be able to optimize it with numba
        
        the function modifies several arguments (numpy arrays) in place.
        """
        #compute adjustment to the instantaneously effective growth factor
        kd = 2**_np.sum(activationMatrix * phaseVelocityExponentInput) 
        
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


        biases = _np.dot(BiasMatrix, statevector)
        velocity_offset = biases * dt 
        noise_statevector = noise_velocity * dt
        
        #compute which transition biases should be applied right now:
        if emulateHybridAutomaton:
            predecessors = 1.0*(_np.abs(statevector)*betaInv > 0.99)
            successors =  (_np.dot(stateConnectivity,  predecessors) > 0.5 )
            notsuccessors =  (_np.dot(stateConnectivity,  predecessors) < 0.5 )
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
                    print( chosensuccessor)
                    noise_statevector[chosensuccessor] = 1.0
            else:
                 triggervalue_successors[:] += biases * dt + noise_velocity


        statesigns = _signfunc(statevector)
        statesignsOuterProduct = _np.outer(statesigns,statesigns)

        #stateVectorExponent=1  #straight channels: |x|  (original SHC)
        #stateVectorExponent=2  #spherical channels: |x|**2
        x_abs = (statevector*statesigns)**stateVectorExponent
        
        #compute the growth rate adjustment depending on the signs of the state and rho:
        #original SHC behavior: alpha_delta=_np.dot(rhoDelta, statesigns*x)
        rhodelta_mask = 1.0 * ( stateConnectivity * statesignsOuterProduct > -0.5) #set rhodelta to zero if state sign flips without us wanting it to
        alpha_delta = _np.dot(rhoDelta*rhodelta_mask, x_abs)

        #This is the core computation and time integration of the dynamical system:
        growth = alpha - _np.dot(rhoZero, x_abs) + alpha_delta
        velocity =  statevector * growth * kd + mu  #estimate velocity  #missing:
        dotstatevector[:] = velocity + velocity_offset #do not add noise to velocity, promp mixer doesnt like it
        statevector[:] = (statevector + dotstatevector*dt + noise_statevector)   #set the new state 
        
        #prepare a normalized state vector for the subsequent operations:
        statevector_scaled = statevector*betaInv
        SP = _np.outer(statevector, statevector)
        P = statevector.reshape((1,numStates))
        P2 = P*P
        S = P.T
        S2 = P2.T
        #compute the transition/state activation matrix (Lambda)
        activations = SP * 8 * (P2 + S2) / ((P + S)**4  + epsilonLambda) 
        activationMatrix[:,:] =  activations * stateConnectivity #function shown in visualization_of_activationfunction.py
        _limit(activationMatrix)
        #apply nonlinearity:
        if (nonlinearityParamsLambda[0] != 1.0 or nonlinearityParamsLambda[1] != 1.0 ):
            activationMatrix[:,:] = 1.0-(1.0-activationMatrix**nonlinearityParamsLambda[0])**nonlinearityParamsLambda[1] #Kumaraswamy CDF
        
        #compute the state activation and put it into the diagonal of Lambda:
        residual = _np.prod(1-activationMatrix) 
        stateactivation_normalized = S2/ _np.sum(S2) 
        for i in range(numStates):
            activationMatrix[i,i] =  stateactivation_normalized[i,0] * residual
                
        #compute the phase progress matrix (Psi)
        epsilonPsi = 0.0001
        newphases = (S+epsilonPsi) / (S+P+2*epsilonPsi)
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
            nu=1.5,  
            beta=1.0, 
            dt=1e-2, 
            stateVectorExponent=2.0,
            nonlinearityLambda='kumaraswamy1,1',
            nonlinearityPsi='kumaraswamy1,1',
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
        self.epsilon = self._sanitizeParam(epsilon) * self.beta #Wiener process noise
        self.epsilonLambda=0.01 #regularization parameter of activation function
        self.reuseNoiseSampleTimes = reuseNoiseSampleTimes
        self.stateVectorExponent =stateVectorExponent
        if predecessors is not None:  #convert list of predecessors into list of successors
            self.successors = self._predecessorListToSuccessorList(predecessors)
        else:
            self.successors = successors
        
        self.updateDt(dt) #also calls self._updateRho
        
        self.nonlinearityParamsLambda = _KumaraswamyCDFParameters[nonlinearityLambda]   #nonlinearity for sparsifying activation values
        self.nonlinearityParamsPsi  = _KumaraswamyCDFParameters[nonlinearityPsi]     #nonlinearity that linearizes phase variables 

        #inputs:
        self.BiasMatrix = _np.zeros((self.numStates,self.numStates)) #determines transition preferences and state timeout duration

        self.emulateHybridAutomaton = emulateHybridAutomaton #set this to true to emulate discrete switching behavior on bias input
        self.triggervalue_successors = _np.zeros((self.numStates))
        
        self.phasesInput = _np.zeros((self.numStates,self.numStates)) #input to synchronize state transitions (slower/faster)
        self.velocityAdjustmentGain = _np.zeros((self.numStates,self.numStates))  #gain of the control enslaving the given state transition
        self.phaseVelocityExponentInput = _np.zeros((self.numStates,self.numStates))  #contains values that limit transition velocity
        
        #internal data structures
        if self.numStates != oldcount or reset: #force a reset if number of states change
            self.statevector = _np.zeros((numStates))
            self.dotstatevector = _np.zeros((numStates))
            self.statevector[0] = self.beta[0] #start at state 0
            self.phasesActivation = _np.zeros((self.numStates,self.numStates))
            self.phasesProgress = _np.zeros((self.numStates,self.numStates))
            self.phasesProgressVelocities = _np.zeros((self.numStates,self.numStates))
            self.biases = _np.zeros((self.numStates, self.numStates))
            self._biasMask = (1-_np.eye((self.numStates)))
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
        stateConnectivity = _np.zeros((self.numStates, self.numStates))
        for state, successorsPerState in enumerate(self.successors):
            #precedecessorcount = len(predecessorsPerState)
            for successor in successorsPerState:
                if state == successor: raise ValueError("Cannot set a state ({0}) as successor of itself!".format(state))
                stateConnectivity[successor,state] = 1 
        self.stateConnectivity = stateConnectivity
        
        #first, fill in the standard values in rhoZero
        # rhoZero = beta^-1 x alpha * (1 - I + alpha^-1 x alpha)
        alphaInv = 1/self.alpha
        s = _np.dot(self.alpha[:,_np.newaxis],self.betaInv[_np.newaxis,:])
        rhoZero = s * (1-_np.eye(self.numStates)+_np.dot(self.alpha[:,_np.newaxis],alphaInv[_np.newaxis,:]))
        
        #then fill the rhoDelta that depends on the state connectivity:
        rhoDelta = self.stateConnectivity * (self.alpha*self.betaInv*(1+1/self.nu))[:,_np.newaxis]
        
        self.rho =  rhoZero - rhoDelta #save the final result
        self.rhoZero = rhoZero
        self.rhoDelta = rhoDelta
        successorCountInv = 1.0/_np.maximum(_np.sum(stateConnectivity, axis=0)[_np.newaxis,:],1.0)
        self.BiasMeanBalancingWeights = stateConnectivity * successorCountInv




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
                _step(  #arrays modified in-place:
                        self.statevector, 
                        self.dotstatevector,
                        self.phasesActivation, 
                        self.phasesProgress, 
                        self.phasesProgressVelocities, 
                        #inputs
                        self.phaseVelocityExponentInput, 
                        self.BiasMatrix, 
                        self.phasesInput, 
                        self.velocityAdjustmentGain, 
                        self.noise_velocity,
                        #parameters
                        self.numStates, 
                        self.betaInv , 
                        self.stateConnectivity, 
                        self.rhoZero, 
                        self.rhoDelta, 
                        self.alpha, 
                        self.dt,
                        self.dtInv, 
                        self.nonlinearityParamsLambda,
                        self.nonlinearityParamsPsi,
                        self.stateVectorExponent,
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
        
        Note 2: predecessor bias is set from the mean bias across succesor states
        """
        bias = _np.asarray(successorBias)
        #balance the means between successor and predecessors:
#        offsets = self.BiasMeanBalancingWeights  * _np.sum( (self.stateConnectivity+_np.eye(self.numStates)) * bias, axis=0)
        offsets = _np.sum( self.BiasMeanBalancingWeights * bias, axis=0)
        self.BiasMatrix = self._biasMask * bias #+ _np.diag(offsets)
        
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
            self.updateB(bias * self.stateConnectivity)
        if bias.ndim == 1:
            self.updateB(bias[:, _np.newaxis] * self.stateConnectivity)
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


