# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:50:06 2019

@author: lukas
"""
import timeit

import numpy as np

n_rows = 1000
n_cols = 300
n_iter = 1000

a = np.random.rand(n_cols)
for i in range(n_rows-1):
    a = np.vstack((a, np.random.rand(n_cols)))
a_t = a.transpose()
b = np.random.rand((n_cols))

print(a)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = a*b
print('Direct multiplication :', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = b*a
print('Reverse direct multiplication :', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = a_t.transpose()*b
print('Direct multiplication + transpose:', timeit.default_timer() - start_time)

b_tiled = np.tile(b, n_rows)
start_time = timeit.default_timer()
for i in range(n_iter):
    c = (a.ravel()*b_tiled).reshape((n_rows, n_cols))
print('Direct multiplication :', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = a_t.transpose()
print('transpose:', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = 0
print('loop:', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = a[:, 0]
print('indexing 1:', timeit.default_timer() - start_time)
#print(c)

start_time = timeit.default_timer()
for i in range(n_iter):
    c = a[0]
print('indexing 1:', timeit.default_timer() - start_time)
#print(c)