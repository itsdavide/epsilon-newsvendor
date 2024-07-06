#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, G. Stabile (2024). 
Newsvendor problem with discrete demand and constrained first moment under ambiguity.

INPUT:
   * X: Range of the discrete random demand in decreasing order
   * P0: Probability distribution of the random demand referred to the decreasing order of the range
   * a: unit understocking cost with a > 0
   * b: unit overstocking cost with b > 0

MINIMAX PROBLEM: Minimization of the funtion:
    * upper_lambda(q): which is the upper expected loss as a function of q >= 0
    
IMPORTANT: If the optimizer is not unique we select the minimum optimizer by convention

USAGE: The code below plots the function upper_lambda(q) and the optimizer q*
"""

import numpy as np
import matplotlib.pyplot as plt
import epsilon_newsvendor as env


###############################################################################
################################ PARAMETERS ###################################
###############################################################################

# Example 2
a = 4
b = 2
X = np.array([2500, 2000, 1500, 1000, 500, 0])
P0 = np.array([8, 5, 1, 2, 2, 2])
P0 = P0 / P0.sum()

# Example 3
#a = 2
#b = 2
#X = np.array([1500, 1000, 500, 0])
#P0 = np.array([1/4, 1/4, 1/4, 1/4])

# Sensitivity analysis
#a = 5
#b = 3
#X = np.array([2500, 2000, 1500, 1000, 500, 0])
#P0 = np.array([1, 1, 1, 1, 1, 1])
#P0 = P0 / P0.sum()
#
#a = 3
#b = 5
#X = np.array([2500, 2000, 1500, 1000, 500, 0])
#P0 = np.array([1, 1, 1, 1, 1, 1])
#P0 = P0 / P0.sum()

###############################################################################
###############################################################################
###############################################################################

# Extract the size of the random demand
n = len(X)

# Build the Mobius inverse of nu**
m_ss, alpha, beta = env.mobius_nu_ss(X, P0)

print('*** MINIMAX PROBLEM ***\n')
print('Mobius inverse of nu**:')
print('alpha:', alpha)
print('beta:', beta)
# Remove comment to print the Mobius inverse of nu**
#print('m_nu**:\n', m_ss, '\n')

print('First moment constraint:')
print('E_P0[X] = mu =', env.E_P0(X, P0))
print('C_nu**[X] = mu =', beta * X[n-1] + (1 - beta) * X[0], '\n')


# Build the decompostion of [0, +infty)
decomp = env.decomposition(X, a, b)

step = 0.1
epsilons = np.arange(0, 1.2, 0.2)
colors = ['red', 'green', 'blue', 'orange', 'magenta', 'skyblue']
i_color = 0

plt.figure(figsize=(6.5, 4))
plt.xlabel('$q$')
plt.ylabel(r'${\bb C}_{\overline{\nu}^{**}_\epsilon}[\Lambda(q)]$')

optimizers = []

# Draw the function lower_lambda(q)
for epsilon in epsilons:
    for (q_l, q_u, i_s, j_s) in decomp:
        qs = np.arange(q_l + step, q_u + step, step)
        upper_lambdas = []
        for q in qs:
            upper_lambdas.append(env.upper_lambda(q, epsilon, i_s, j_s, X, P0, m_ss, a, b))
        if q_l == X[n - 1] and q_u != X[n - 1]:
            plt.plot(qs, upper_lambdas, color=colors[i_color], label='$\epsilon=$' + str(round(epsilon,1)))
        else:   
            plt.plot(qs, upper_lambdas, color=colors[i_color])
    (q_min, min_Choq) = env.find_min(decomp, epsilon, X, P0, m_ss, a, b)
    optimizers.append((q_min, min_Choq))
    i_color += 1


# Draw the optimizers
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_range = x_max - x_min
y_range = y_max - y_min
print()
print('Optimizers:')

i_color = 0
for (q_min, min_Choq) in optimizers:
    plt.plot([q_min], [min_Choq], marker="o", markersize=3, markeredgecolor=colors[i_color], markerfacecolor=colors[i_color])
    plt.text(q_min + 0.01*x_range, min_Choq + 0.01*y_range, "min", fontsize=8)
    print('epsilon:', np.round(epsilons[i_color], 2) , 'q_min:', q_min)
    i_color += 1
    
plt.legend()
plt.savefig('MINMAX.png', dpi=300)
