# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:48:44 2019

@author: lukas
"""

import numpy as np
from scipy import linalg as sp_la
import timeit


def overlapping_vector(vector, reps, overlap_size):
    n_vector = len(vector)
    n_result = reps*n_vector - (reps-1)*overlap_size
    # n_result = reps * (n_vector - overlap_size) + overlap_size
    result = np.zeros(n_result)
    non_overlap_size = int(n_vector-overlap_size)
    for i in range(reps):
        start_id = i * non_overlap_size
        end_id = start_id + n_vector
        result[start_id:end_id] += vector
    return result


def build_1d_conductance_matrix(cond_vector, offset=1):
    n_layer = len(cond_vector) + 1
    cond_matrix = np.full((n_layer, n_layer), 0.)
    center_diag = overlapping_vector(cond_vector, 2, n_layer-2)
    center_diag *= -1.0
    off_diag = np.copy(cond_vector)
    cond_matrix = np.diag(center_diag, k=0) \
        + np.diag(off_diag, k=-offset) \
        + np.diag(off_diag, k=offset)
    return cond_matrix


def build_z_cell_conductance_matrix(cond_array, n_ele):
    mat_layer = build_1d_conductance_matrix(cond_array)
    list_mat = [mat_layer for i in range(n_ele)]
    return sp_la.block_diag(*list_mat)


def build_x_cell_conductance_matrix(cond_vector, n_ele):
    n_layer = len(cond_vector)
    center_diag = np.concatenate([cond_vector for i in range(n_ele)])
    center_diag[n_layer:-n_layer] *= 2.0
    center_diag *= -1.0
    off_diag = np.concatenate([cond_vector for i in range(n_ele-1)])
    return np.diag(center_diag, k=0) \
        + np.diag(off_diag, k=-n_layer) \
        + np.diag(off_diag, k=n_layer)


kx = [1.0, 2.0, 3.0]
kz = [4.0, 5.0]
n_ele = 3
n_layer = 3
cond_vec_x = np.asarray(kx)
cond_vec_z = np.asarray(kz)

z_cond_matrix = build_z_cell_conductance_matrix(cond_vec_z, n_ele)
print(z_cond_matrix)
x_cond_matrix = build_x_cell_conductance_matrix(cond_vec_x, n_ele)
print(x_cond_matrix)
