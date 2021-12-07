# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:24:52 2020

@author: lukas
"""

import numpy as np
from numba import njit, prange
import timeit


@njit
def add_source(var, source, direction=1, tri_mtx=None):
    """
    Add discrete 1d source of length n-1 to var of length n
    :param var: 1d array of quantity variable
    :param source: 1d array of source to add to var
    :param direction: flow direction (1: along array counter, -1: opposite to
    array counter)
    :param tri_mtx: if triangle matrix (2D array, nxn) is not provided,
    it will be created temporarily
    :return:
    """
    n = len(var) - 1
    if len(source) != n:
        raise ValueError('parameter source must be of length (var-1)')
    if direction == 1:
        if tri_mtx is None:
            ones = np.zeros((n, n))
            ones.fill(1.0)
            fwd_mat = np.tril(ones)
        else:
            fwd_mat = tri_mtx
        var[1:] += np.matmul(fwd_mat, source)
    elif direction == -1:
        if tri_mtx is None:
            ones = np.zeros((n, n))
            ones.fill(1.0)
            bwd_mat = np.triu(ones)
        else:
            bwd_mat = tri_mtx
        var[:-1] += np.matmul(bwd_mat, source)
    else:
        raise ValueError('parameter direction must be either 1 or -1')
    return var


@njit
def calc_temp_heat_transfer(wall_temp, fluid_temp, capacity_rate, heat_coeff,
                            flow_direction=1):
    wall_temp = np.asarray(wall_temp)
    fluid_temp = np.asarray(fluid_temp)
    capacity_rate = np.asarray(capacity_rate)
    heat_coeff = np.asarray(heat_coeff)
    assert capacity_rate.shape == wall_temp.shape
    assert heat_coeff.shape == wall_temp.shape
    fluid_temp_avg = np.asarray(fluid_temp[:-1] + fluid_temp[1:]) * .5
    for i in range(len(wall_temp)):
        fluid_avg = fluid_temp_avg[i]
        fluid_out_old = 5e5
        error = 1e3
        iter = 0
        itermax = 10
        while error > 1e-4 and iter <= itermax:
            fluid_in = fluid_temp[i]
            delta_temp = wall_temp[i] - fluid_avg
            q = heat_coeff[i] * delta_temp
            fluid_out = fluid_in + q / capacity_rate[i]
            if fluid_in < wall_temp[i]:
                fluid_out = np.minimum(wall_temp[i] - 1e-3, fluid_out)
            else:
                fluid_out = np.maximum(wall_temp[i] + 1e-3, fluid_out)
            fluid_avg = (fluid_in + fluid_out) * 0.5
            error = np.abs(fluid_out_old - fluid_out) / fluid_out
            fluid_out_old = np.copy(fluid_out)
            iter += 1

        fluid_temp[i + 1] = fluid_out
    fluid_temp_avg = np.asarray(fluid_temp[:-1] + fluid_temp[1:]) * .5
    heat = heat_coeff * (wall_temp - fluid_temp_avg)
    return fluid_temp, heat

@njit(parallel=True)
def parallel_function(wall_temp, fluid_temp, capacity_rate, heat_coeff, n):
    result = []
    for i in prange(n):
        result.append(calc_temp_heat_transfer(wall_temp, fluid_temp,
                                              capacity_rate, heat_coeff))
    return np.asarray(result)


def sequential_function(wall_temp, fluid_temp, capacity_rate, heat_coeff, n):
    result = []
    for i in range(n):
        result.append(calc_temp_heat_transfer(wall_temp, fluid_temp,
                                              capacity_rate, heat_coeff))
    return np.asarray(result)

n = 10
n_loops = 10

fluid_temp = np.full(n, 340.0)
wall_temp = np.full(n-1, 370.0)

g = 0.001 * 4000
k_coeff = 1000.0 * 5e-3 * np.pi * 10e-3

start_time = timeit.default_timer()
result = parallel_function(wall_temp, fluid_temp, g, k_coeff, n_loops)
print('Parallel time: ', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
result = sequential_function(wall_temp, fluid_temp, g, k_coeff, n_loops)
print('Sequential time: ', timeit.default_timer() - start_time)










