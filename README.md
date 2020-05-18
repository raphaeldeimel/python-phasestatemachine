# Phase-State Machine Reference Implementation


## What is it?

The so called  "Phase-State machine" is a dynamical system designed to govern the decision making process and temporal progress of robot behaviors, and doing so in a continuous, highly reactive and gradual fashion. At the same time it provides an always consistent, high-level (few states) state graph abstraction of the ongoing behavior. Its main application is in human-robot interaction where quick correction of mistakes, fast feedback to the human signals and gradual adaptability of behavior in both time and space are paramount to support diverse interaction idioms, cultural and individual peculiarities, as well as to tolerate technical or situational limitations in perception and actuation.  


The Phase-State machine (nicknamed "Phasta") represents (and emulates) state graphs using high-dimensional dynamical systems. As with discrete automata, states possess the Markov property (independent of prior and future state), but the phase-state machine has time-extended, reversible and smooth transitions instead of instantaneous and irreversible ones. Also, each state and transition has  a smooth, differentiable "activation" signal which allows for gradual blending of behavior to and from states when starting and stopping a transition. Further, each transition has a smooth and differentiable phase signal which provides a notion of progress along each transition. This allows us to associate movement primitives and trajectories with transitions, which both require a notion of time/progress.


If you use this work in scientific publications, please consider citing:

[1] Reactive Interaction Through Body Motion and the Phase-State-Machine, Deimel, Raphael, In Proc. of The International Conference on Intelligent Robots and Systems (IROS), 2019.
[2] A Dynamical System for Governing Continuous, Sequential and Reactive Behaviors, Deimel, Raphael, In Austrian Robotics Workshop, 2019.

The papers also contain an in-depth description and use cases.


This source contains a reference implementation of PhaSta, and the examples used to generate the figures in the paper. Computation is reasonably optimized using Numba, so it should easily scale to dozens of states.


## For the Impatient
 * Go to paper_examples/
 * run `python3 staying_and_leaving_state.py`
 * Have a look at phasestatemachine/_kernel.py


## Folder organization

### phasestatemachine/

Python module containing the phase-state machine, the main algorithm is in _kernel.py

### paper_examples/

Contains minimal examples / tests that use the phase-state machine to illustrate a certain capability

To generate all figures, run `./make_plots.sh`

### paper_examples/figures/

Folder where example scripts will place their output plots


## Notes

The code was tested with python3 on Ubuntu Linux 20.04., using the standard packages.

Tested with:
    Python: 3.5.3 and 2.7.13
    SciPy: 0.18.1 
    Numpy: 1.12.1





## Copyright

All files are distributed under the 2-clause FreeBSD licence. 
