#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti, G. Stabile (2024). 
Newsvendor problem with discrete demand and constrained first moment under ambiguity.

USAGE: Computation of the functions:
    
    * lower_pi(q): which is the lower expected profit as a function of q >= 0
    This function should be maximized in the maximin problem
    
    * upper_lambda(q): which is the upper expected loss as a function of q >= 0
    This function should be minimized in the minimax problem
"""


import numpy as np


# Compute the expected demand
def E_P0(X, P0):
    return X.dot(P0)

###############################################################################
# MAXIMIN PROBLEM
###############################################################################

# Build the Mobius inverse of nu*
def mobius_nu_s(X, P0):
    n = len(X)
    mu = E_P0(X, P0)
    s = 0
    for k in range(n - 1, -1, -1):
        if X[k] <= mu:
            s = k
        else:
            break
   
    tot = 0
    C2 = [list(range(k,n)) for k in range(1,s+1,1)]
    m_s =[]
    for A in C2:
        if min(A) == s:
            m = (X[s-1]-mu) / (X[s-1]-X[n-1])
            tot += m
            m_s.append((set(A),m))
        else:
            k = min(A)
            m = (X[k-1]-mu) / (X[k-1]-X[n-1])- (X[k]-mu) / (X[k]-X[n-1])
            tot += m
            m_s.append((set(A),m)) 
    beta = tot
    alpha = 1 - beta
    m_s.append(({0}, 1 - beta))

    m_s = np.array(m_s)
    
    return (m_s, alpha, beta)

# Compute the lower_pi(q) functin
def lower_pi(q, epsilon, X, P0, alpha, r, c):
    n = len(X)
    return r * ((1 - epsilon) * np.minimum(X, q).dot(P0) + epsilon * ((1 - alpha) * np.minimum(X[n-1], q) + alpha * np.minimum(X[0], q))) - c * q


###############################################################################
# MINIMAX PROBLEM
###############################################################################

# Function for the decomposition in the minimax problem
def f(x, a, b, x0):
    return (b / (a + b)) * (x + (a / b) * x0)

# Build the Mobius inverse of nu**
def mobius_nu_ss(X, P0):
    n = len(X)
    mu = E_P0(X, P0)
    s = 0
    for k in range(n - 1, -1, -1):
        if X[k] <= mu:
            s = k
        else:
            break
    
    tot = 0
    C1 = [list(range(k+1)) for k in range(s-1, n-1, 1)]
    m_ss = []
    for A in C1:
        if max(A) == s-1:
            if mu != X[s]:
                m = (mu - X[s]) / (X[0] - X[s])
                tot += m
                m_ss.append((set(A), m))
        else:
            k = max(A)
            m = (mu - X[k + 1]) / (X[0] - X[k + 1]) - (mu - X[k]) / (X[0] - X[k])
            tot += m
            m_ss.append((set(A), m))
            
    alpha = tot
    beta = 1 - alpha
    
    m_ss.append(({n-1}, beta))
    
    m_ss = np.array(m_ss)
    return (m_ss, alpha, beta)


# Build the decomposition of [0, +infinity)
def decomposition(X, a, b):
    n = len(X)
    f_X = f(X, a, b, X[0]) 

    nodes = list(set(list(X) + list(f_X)))
    nodes.sort(reverse=True)
    nodes = np.array(nodes)
    
    i_s = 0
    j_s = 0
    
    decomp = []
    
    for k in range(len(nodes)):
        if k == 0:
            decomp.append((nodes[k], nodes[k] + 100, -1, -1))
        else:
            i, = np.where(X == nodes[k - 1])
            j, = np.where(f_X == nodes[k - 1])
            if len(i) != 0:
                i_s = i[0]
            if len(j) != 0:
                j_s = j[0]
            decomp.append((nodes[k], nodes[k - 1], i_s, j_s))
            if k == len(nodes) - 1:
                decomp.append((0, nodes[k], n - 1, n - 1))
                
    return decomp

# Compute the expectation of Lambda_q with respect to P0 on an interval of the decomposition Z
def E_Lambda(q, i_s, j_s, X, P0, a, b):
    n = len(X)
    tot = 0
    if i_s == n - 1:
        for k in range(len(X)):
            tot += a * (X[k] - q) * P0[k]
    else:
        for k in range(len(X)):
            if k <= i_s:
                tot += a * (X[k] - q) * P0[k]
            else:
                tot += b * (q - X[k]) * P0[k]
    return tot

# Compute the Choquet expectation of Lambda_q with respect to nu** on an interval of the decomposition Z
def C_Lambda(q, i_s, j_s, X, m_ss, a, b):
    n = len(X)
    tot = 0
    if i_s == n - 1:
        m_tot_0 = 0
        for (s, m) in m_ss:
            if 0 in s:
                m_tot_0 += m
        m_tot_n_1 = 0
        for (s, m) in m_ss:
            if n - 1 in s:
                m_tot_n_1 += m
        tot += a * (X[0] - q) * m_tot_0 + a * (X[n - 1] - q) * m_tot_n_1
    else:
        for (s, m) in m_ss:
            if 0 in s and max(s) <= j_s:
                tot += a * (X[0] - q) * m
            elif 0 in s and max(s) > j_s:
                tot += b * (q - X[max(s)]) * m
            elif max(s) == n - 1 and min(s) == n - 1:
                tot += b * (q - X[n - 1]) * m
    return tot

# Compute the function upper_lambda(q) on an interval of the decomposition Z
def upper_lambda(q, epsilon, i_s, j_s, X, P0, m_ss, a, b):
    return (1 - epsilon) * E_Lambda(q, i_s, j_s, X, P0, a, b) + epsilon * C_Lambda(q, i_s, j_s, X, m_ss, a, b)

# Compute the optimizer (by convention, in case of non-uniqueness select the minimum of optimizers)
def find_min(decomp, epsilon, X, P0, m_ss, a, b):
    q_min = np.Inf
    min_Choq = np.Inf
    for (q_l, q_u, i_s, j_s) in decomp:
        Choq_u = upper_lambda(q_u, epsilon, i_s, j_s, X, P0, m_ss, a, b)
        if Choq_u <= min_Choq:
            q_min = q_u
            min_Choq = Choq_u
        Choq_l = upper_lambda(q_l, epsilon, i_s, j_s, X, P0, m_ss, a, b)
        if Choq_l <= min_Choq:
            q_min = q_l
            min_Choq = Choq_l
    return (q_min, min_Choq)      




        