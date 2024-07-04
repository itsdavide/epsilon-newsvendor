# epsilon-newsvendor
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, G. Stabile (2024). 
Newsvendor problem with discrete demand and constrained first moment under ambiguity

# Requirements
The code has been tested on Python 3.10 with the following libraries:
* **matplotlib** 3.7.1
* **numpy** 1.26.4

# File inventory
**epsilon_newsvendor.py**: Computation of the functions:
* _lower_pi(q)_: which is the lower expected profit as a function of _q >= 0_.
    This function should be maximized in the maximin problem.
* _upper_lambda(q)_: which is the upper expected loss as a function of _q >= 0_.
    This function should be minimized in the minimax problem.

**IMPORTANT: If the optimizer is not unique we select the minimum optimizer by convention**

**maxmin.py**: Plots the function _lower_pi(q)_ and the optimizer q*
## Input
* _X_: Range of the discrete random demand in decreasing order
* _P0_: Probability distribution of the random demand referred to the decreasing order of the range
* _r_: unit sales revenue with _r > c > 0_
* _c_: unit purchase cost with _c > 0_
