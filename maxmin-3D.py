#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, G. Stabile (2024). 
Newsvendor problem with discrete demand and constrained first moment under ambiguity.

INPUT:
   * X: Range of the discrete random demand in decreasing order
   * P0: Probability distribution P0 of X referred to the decreasing order of the range

MAXIMIN PROBLEM: Maximization of the funtion:
    * lower_pi(q): which is the lower expected profit as a function of q >= 0

IMPORTANT: If the optimizer is not unique we select the minimum optimizer by convention

USAGE: The code below plots the 3D surface and the countour lines of the optimizer q*
of lower_pi(q) as a function of:
   * r: unit sales revenue with r > c > 0
   * c: unit purchase cost with c > 0
"""


import numpy as np
import matplotlib.pyplot as plt
import epsilon_newsvendor as env
from matplotlib import cm


###############################################################################
################################ PARAMETERS ###################################
###############################################################################

# Uniform distribution over {100, ..., 0}
X = np.arange(100,-1,-1)
P0 = np.ones(len(X)) / len(X)

###############################################################################
###############################################################################
###############################################################################


# Extract the size of the random demand
n = len(X)

# Build the Mobius inverse of nu**
m_s, alpha, beta = env.mobius_nu_s(X, P0)
print('Mobius inverse of nu*:')
print('alpha:', alpha)
print('beta:', beta)
# Remove comment to print the Mobius inverse of nu*
#print('m_nu*:\n', m_s, '\n')

print('First moment constraint:')
print('E_P0[X] = mu =', env.E_P0(X, P0))
print("C_nu*[X] = mu =", (1 - alpha) * X[n-1] + alpha * X[0], "\n")


# Build the 3D surface of lower_pi(q)
step = 0.1 # Set the step to 0.01 to have more precise contour lines
Cs = np.arange(0.1, 5.1 + step, step)
Rs = np.arange(0.1, 5.1 + step, step)

epsilon = 0.2


x = []
y = []
z = []

# Plot the optimal q as a function of r and c
for c in Cs:
    for r in Rs:
        if r > c:
            lower_pi=[]
            for q in X:
                lower_pi.append(env.lower_pi(q, epsilon, X, P0, alpha, r, c))
            lower_pi_max = max(lower_pi)
            
            # Select the minimum of optimizers (which has maximum index since values of X are decreasing)
            i_max = np.max(np.argwhere(np.abs(lower_pi - lower_pi_max) <= 0.000001))
            x.append(r)
            y.append(c)
            z.append(X[i_max])


Xs = np.array(x)
Ys = np.array(y)
Zs = np.array(z)

# 3D plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=190)

surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)

ax.set_xlabel('$r$')
ax.set_ylabel('$c$')

ax.set_title(r'Optimal $q^*$ as a function of $r$ and $c$ ($\epsilon= $' + str(epsilon) + ')')

fig.tight_layout()

plt.show()
fig.savefig('3D_q_star_MAXMIN_surface_epsilon_' + str(epsilon) + '.png', dpi=300)


# Contour plot
plt.clf()
levels = np.arange(0, 100, 10)
fig = plt.figure(figsize=(5,5))
plt.title(r'Contour lines of optimal $q^*$ ($\epsilon= $' + str(epsilon) + ')')
plt.xlabel('$r$')
plt.ylabel('$c$')
plt.tricontour(Xs, Ys, Zs, cmap=cm.jet, levels=levels)
plt.savefig('s_' + str(step) + '_MAXMIN_q_star_CL_epsilon_' + str(epsilon) + '.png', dpi=300)