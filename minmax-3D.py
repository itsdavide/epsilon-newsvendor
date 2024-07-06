#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, G. Stabile (2024). 
Newsvendor problem with discrete demand and constrained first moment under ambiguity.

INPUT:
   * X: Range of the discrete random demand in decreasing order
   * P0: Probability distribution P0 of X referred to the decreasing order of the range

MINIMAX PROBLEM: Minimization of the funtion:
    * upper_lambda(q): which is the upper expected loss as a function of q >= 0
    
IMPORTANT: If the optimizer is not unique we select the minimum optimizer by convention

USAGE: The code below plots the 3D surface and the countour lines of the optimizer q*
of upper_lambda(q) as a function of:
   * a: unit understocking cost with a > 0
   * b: unit overstocking cost with b > 0
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

# Plot the optimal q as a function of a and b
step = 0.1 # Set the step to 0.01 to have more precise contour lines
As = np.arange(0.1, 5.1 + step, step)
Bs = np.arange(0.1, 5.1 + step, step)

epsilon = 0.2

x = []
y = []
z = []

for a in As:
    for b in Bs:
        print('Checking a = ', a, ', b =', b)
        decomp = env.decomposition(X, a, b)
        (q_min, min_Choq) = env.find_min(decomp, epsilon, X, P0, m_ss, a, b)
        print('q_min = ', q_min, '\n')
        x.append(a)
        y.append(b)
        z.append(q_min)


Xs = np.array(x)
Ys = np.array(y)
Zs = np.array(z)

# 3D plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=190)

surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)

ax.set_xlabel('$a$')
ax.set_ylabel('$b$')

ax.set_title(r'Optimal $q^*$ as a function of $a$ and $b$ ($\epsilon= $' + str(epsilon) + ')')

fig.tight_layout()

plt.show()
fig.savefig('3D_q_star_MINMAX_surface_epsilon_' + str(epsilon) + '.png', dpi=300)


# Contour plot
plt.clf()
levels = np.arange(0, 100, 10)
fig = plt.figure(figsize=(5,5))
plt.title(r'Contour lines of optimal $q^*$ ($\epsilon= $' + str(epsilon) + ')')
plt.xlabel('$a$')
plt.ylabel('$b$')
plt.tricontour(Xs, Ys, Zs, cmap=cm.jet, levels=levels)
plt.savefig('s_' + str(step) + '_MINMAX_q_star_CL_epsilon_' + str(epsilon) + '.png', dpi=300)