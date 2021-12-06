# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 01:00:15 2019

@author: lukas
"""

import numpy as np

n = 11
p0 = 0.0
p = np.full(n, p0)
dp = np.full(n, 10.0)
dp[0] = 0.0

m0 = 0.0
m = np.full(n, m0)
dm = np.full(n, 10.0)
#dm[0] = 0.0

m1 = np.full(n, m0)
dm1 = np.full(n-1, 10.0)

m2 = np.full(n, m0)
dm2 = np.full(n-1, 10.0)
 

def add_source(var, source, dir=1):
    if dir==1:
        diag_matrix = np.tril(np.full((len(var), len(var)), 1.0), k=-1)
    elif dir==-1:
        diag_matrix = np.triu(np.full((len(var), len(var)), 1.0), k=+1)
    var += np.matmul(diag_matrix, source)
    return var

def add_source_2(var, source, dir=1):
    n = len(var)-1
    if len(source) != n:
        raise ValueError('source variable must be of length (var-1)')
    if dir==1:
        diag_matrix = np.tril(np.full((n, n), 1.0))
        var[1:] += np.matmul(diag_matrix, source)
    elif dir==-1:
        diag_matrix = np.triu(np.full((n, n), 1.0))
        var[:-1] += np.matmul(diag_matrix, source)
    return var

p = add_source(p, dp, -1)
m = add_source(m, dm)
print(p)
print(m)

print(add_source_2(m1, dm1))
print(add_source_2(m2, dm2, -1))

