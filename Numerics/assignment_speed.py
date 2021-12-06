# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 01:34:33 2019

@author: lukas
"""
import numpy as np
import timeit

m = 100
n = 10
n_iter = 100000
array = np.full((m,n), 0.0)
array_rand = np.random.rand(m, n)

start_time = timeit.default_timer()
for i in range(n_iter):
    array = array_rand
print(timeit.default_timer() - start_time)
#print(array[:5])


start_time = timeit.default_timer()
for i in range(n_iter):
    array[:] = array_rand
print(timeit.default_timer() - start_time)
#print(array[:5])


start_time = timeit.default_timer()
for i in range(n_iter):
    array[:] = array_rand[:]
print(timeit.default_timer() - start_time)
#print(array[:5])
