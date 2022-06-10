# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
import time


mf = np.asarray([0.21, 0.79])
n = len(mf)
shape = (10, 5)
# scale = np.linspace(0.5, 1.0, shape)
if isinstance(shape, int):
    shape = (shape,)
n_iter = 100000

b = np.multiply.outer(np.ones((n, *shape)), mf)
# b = (np.ones((n, *shape)).transpose() *  mf).transpose()
#b_scale = b * scale

#print(b[:, :, 0])
#print(b.transpose()[0, :, :])
#print(np.moveaxis(b, -1, 0)[0, :, :])

print(b)

# def move_axis(array):
#     if array.ndim > 2:
#         return np.moveaxis(array, -1, 0)    
#     else:
#         return array.transpose()

# start_time = time.time()
# for i in range(n_iter):
#     b = np.moveaxis(b, -1, 0)
# end_time = time.time()

# print('pure moveaxis timing: ', end_time - start_time)

# start_time = time.time()
# for i in range(n_iter):
#     b = b.transpose()
# end_time = time.time()

# print('pure transpose timing: ', end_time - start_time)

# start_time = time.time()
# for i in range(n_iter):
#     if len(shape) > 1:
#         b = np.moveaxis(b, -1, 0)    
#     else:
#         b = b.transpose()
# end_time = time.time()

# print('conditional timing: ', end_time - start_time)

# start_time = time.time()
# for i in range(n_iter):
#     b = move_axis(b)
# end_time = time.time()

# print('conditional function timing: ', end_time - start_time)
