# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import time

   

def add_diag_loop(matrix, n_diag):
    for i in range(n_diag):
        matrix[i,i] = 1.0
    return matrix
    #a = a + 3.0

def add_diag_matrix(matrix, diag):
    b = np.zeros_like(matrix)
    np.fill_diagonal(b, diag)
    matrix = matrix + b
    return matrix
    
n = 1000
n_diag = 100

a = np.zeros((n,n))
a1 = np.copy(a)
b = np.zeros_like(a)

b_diag = np.diagonal(b)
b_diag.setflags(write=True)
for i in range(n_diag):
    b_diag[i] = 1.0
    
start = time.time()
a = add_diag_loop(a, n_diag)
end = time.time()
print(a)
print('loop time: ', end - start)

start = time.time()
a1 = add_diag_matrix(a1, b_diag)
print(a1)
end = time.time()
print('matrix time: ', end - start)

