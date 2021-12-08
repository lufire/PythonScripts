# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:46:32 2021

@author: lukas
"""

from numba import jit, njit, prange
import numpy as np
import time

tstart = time.time()

tstop = time.time()
dt_init = tstop - tstart

print('ray initialization [s]: {:.3f}'.format(dt_init))


n = 240
m = 5000


def f():
    a = np.random.rand(m, m)
    b = np.random.rand(m, m)
    c = np.matmul(a, b)
    # return np.sum(c)


@njit(parallel=True)
def f_par(x):
    for i in prange(x):
        a = np.random.rand(m, m)
        b = np.random.rand(m, m)
        # c = np.matmul(a, b)


def wait():
    time.sleep(0.001)
    return None


tstart = time.time()

results = f_par(n)
# print(results) # [0, 1, 4, 9]

tstop = time.time()
dt_par = tstop - tstart
print('numba parallelization [s]: {:.3f}'.format(dt_par))

tstart = time.time()

results = [f() for i in range(n)]
# print(results) # [0, 1, 4, 9]

tstop = time.time()
dt = tstop - tstart
dt_ser = tstop - tstart
print('serial [s]: {:.3f}'.format(dt_ser))
print('numba speedup [-]: {:.3f}'.format(dt_ser/dt_par))

# ray.shutdown()
