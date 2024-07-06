#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, G. Stabile (2024). 
Newsvendor problem with discrete demand and constrained first moment under ambiguity.

INPUT:
   * X: Range of the discrete random demand in decreasing order
   * P0: Probability distribution of the random demand referred to the decreasing order of the range
   * r: unit sales revenue with r > c > 0
   * c: unit purchase cost with c > 0

MAXIMIN PROBLEM: Maximization of the funtion:
    * lower_pi(q): which is the lower expected profit as a function of q >= 0

IMPORTANT: If the optimizer is not unique we select the minimum optimizer by convention

USAGE: The code below plots the function lower_pi(q) and the optimizer q*
"""


import numpy as np
import matplotlib.pyplot as plt
import epsilon_newsvendor as env


###############################################################################
################################ PARAMETERS ###################################
###############################################################################

# Example 2
r = 6
c = 2
X = np.array([2500, 2000, 1500, 1000, 500, 0])
P0 = np.array([8, 5, 1, 2, 2, 2])
P0 = P0 / P0.sum()

# Example 3
#r = 4
#c = 2
#X = np.array([1500, 1000, 500, 0])
#P0 = np.array([1/4, 1/4, 1/4, 1/4])

# Sensitivity analysis
#r = 8
#c = 3
#X = np.array([2500, 2000, 1500, 1000, 500, 0])
#P0 = np.array([1, 1, 1, 1, 1, 1])
#P0 = P0 / P0.sum()
#
#r = 8
#c = 5
#X = np.array([2500, 2000, 1500, 1000, 500, 0])
#P0 = np.array([1, 1, 1, 1, 1, 1])
#P0 = P0 / P0.sum()

###############################################################################
###############################################################################
###############################################################################

# Extract the size of the random demand
n = len(X)

# Build the Mobius inverse of nu**
m_s, alpha, beta = env.mobius_nu_s(X, P0)

print('*** MAXIMIN PROBLEM ***\n')
print('Mobius inverse of nu*:')
print('alpha:', alpha)
print('beta:', beta)
# Remove comment to print the Mobius inverse of nu*
# print('m_nu*:\n', m_s, '\n')

print('First moment constraint:')
print('E_P0[X] = mu =', env.E_P0(X, P0))
print("C_nu*[X] = mu =", (1 - alpha) * X[n-1] + alpha * X[0], "\n")

epsilons = np.arange(0, 1.2, 0.2)
colors = ['red', 'green', 'blue', 'orange', 'magenta', 'skyblue']
i_color = 0


plt.figure(figsize=(6.5, 4))
plt.xlabel('$q$')
plt.ylabel(r"${\bb C}_{\nu^*_\epsilon}[\Pi(q)]$")

optimizers = []

for epsilon in epsilons:
    i_max = -np.infty
    qs = np.append(np.append([X[0] + 100], X), [0])
    
    lower_pi = []
    for q in qs:
        lower_pi.append(env.lower_pi(q, epsilon, X, P0, alpha, r, c))
    plt.plot(qs, lower_pi, color=colors[i_color], label="$\epsilon=$" + str(round(epsilon,4)))
    
    lower_pi_max = max(lower_pi)
    
    # Selects the minimum of optimizers (which has maximum index since values of X are decreasing)
    i_max = np.max(np.argwhere(np.abs(lower_pi - lower_pi_max) <= 0.000001))
    
    optimizers.append((qs[i_max],lower_pi_max))
    
    i_color +=1
    
    
# Draw the optimizers
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_range = x_max - x_min
y_range = y_max - y_min
print()
print('Optimizers:')

i_color = 0
for (q_max, max_Choq) in optimizers:
    plt.plot([q_max], [max_Choq], marker="o", markersize=3, markeredgecolor=colors[i_color], markerfacecolor=colors[i_color])
    plt.text(q_max + 0.01*x_range, max_Choq + 0.01*y_range, "max", fontsize=8)
    print('epsilon:', np.round(epsilons[i_color], 2) , 'q_max:', q_max)
    i_color += 1

plt.legend()
plt.savefig('MAXMIN.png', dpi=300)
