# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, njit, prange
import timeit
import scipy as sp
from scipy.sparse import linalg as sp_la
import dask.array as da


#@jit(nopython=True)
def solve1(a, b):
    return np.linalg.solve(a, b)

def solve2(a, b):
    return np.linalg.tensorsolve(a, b)

def solve3(a, b):
    return sp.linalg.solve(a, b)

def solve4(a, b):
    a_sparse = sp.sparse.csc_matrix(a)
    return sp_la.spsolve(a_sparse, b)

n_cells = 1
n_ele = 50
n_layer = 5
m = n_cells * n_ele * n_layer

n_diags = 8
diags = []
offsets = []
diags.append(np.random.rand(m))
offsets.append(0)
for i in range(1, n_diags):
    diags.append(np.random.rand(m - i))
    diags.append(np.random.rand(m - i))
    offsets.append(i)
    offsets.append(-i)
a_tridiag_sparse = sp.sparse.diags(diags, offsets, format='csr')
a_tridiag = a_tridiag_sparse.toarray()
da_tridiag = da.from_array(a_tridiag, chunks=10000)
# print(a_tridiag)

a_zeros = np.zeros((m, m))
a = np.random.rand(*a_zeros.shape)
b = np.random.rand(m)
da_b = da.from_array(b)

n_iter = 100

#start_time = timeit.default_timer()
#for i in range(n_iter):
#    x5= da.linalg.solve(da_tridiag, b)
#    x5.compute()
#print(timeit.default_timer() - start_time)
#
#start_time = timeit.default_timer()
#for i in range(n_iter):
#    x1 = solve1(a_tridiag, b)
#print(timeit.default_timer() - start_time)
#
#start_time = timeit.default_timer()
#for i in range(n_iter):
#    x2 = solve2(a_tridiag, b)
#print(timeit.default_timer() - start_time)
#
#start_time = timeit.default_timer()
#for i in range(n_iter):
#    x3 = solve3(a_tridiag, b)
#print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    x4= solve4(a_tridiag, b)
print(timeit.default_timer() - start_time)

print(np.sum(x1))
print(np.sum(x2))
print(np.sum(x3))
print(np.sum(x4))
print(np.sum(x5).compute())


