# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:46:32 2021

@author: lukas
"""

import ray
import numpy as np
import time

tstart = time.time()

ray.shutdown()
if not ray.is_initialized():
    ray.init()

tstop = time.time()
dt_init = tstop - tstart

print('ray initialization [s]: {:.3f}'.format(dt_init))


n = 200
m = 1000

@ray.remote
def f_par(x):
    a = np.random.rand(m, m)
    b = np.random.rand(m, m)
    c = np.matmul(a, b)
    return np.sum(c)

def f_ser(x):
    a = np.random.rand(m, m)
    b = np.random.rand(m, m)
    c = np.matmul(a, b)
    return np.sum(c)

def wait():
    time.sleep(0.001)
    return None

@ray.remote
def wait_ray():
    time.sleep(0.001)
    return None

tstart = time.time()

results = ray.get([f_par.remote(i) for i in range(n)])
# print(results) # [0, 1, 4, 9]

tstop = time.time()
dt_par = tstop - tstart
print('ray parallelization [s]: {:.3f}'.format(dt_par))

tstart = time.time()

results = [f_ser(i) for i in range(n)]
# print(results) # [0, 1, 4, 9]

tstop = time.time()
dt = tstop - tstart
dt_ser = tstop - tstart
print('serial [s]: {:.3f}'.format(dt_ser))
print('speedup [-]: {:.3f}'.format(dt_ser/dt_par))

#ray.shutdown()
