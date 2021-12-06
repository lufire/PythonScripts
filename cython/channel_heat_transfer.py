# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:36:22 2020

@author: feierabend
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:35:32 2020

@author: feierabend
"""

import numpy as np

DTYPE = np.float64

def calc_heat_transfer(double[:] wall_temp, double[:] fluid_temp, 
                       double[:] capacity_rate, double[:] heat_coeff,
                       int flow_direction):

    assert capacity_rate.shape == wall_temp.shape
    assert heat_coeff.shape == wall_temp.shape
    assert wall_temp.dtype == DTYPE
    assert fluid_temp.dtype == DTYPE
    assert capacity_rate.dtype == DTYPE
    assert heat_coeff.dtype == DTYPE    
    
    cdef Py_ssize_t elements = wall_temp.shape[0]
    heat = np.zeros(wall_temp.shape, dtype=DTYPE)
    cdef double fluid_avg = 0.0
    cdef double fluid_out_old = 5e5
    cdef double fluid_in = 0.0
    cdef double fluid_out = 0.0
    cdef double error = 1e3
    cdef int iter = 0
    cdef int itermax = 10
    cdef fluid_temp_avg = 0.0
    cdef int i
    if flow_direction == -1:
        for i in reversed(range(elements)):
            fluid_avg = (fluid_temp[i + 1] + fluid_temp[i]) * 0.5
            fluid_out_old = 5e5
            error = 1e3
            iter = 0
            itermax = 10
            while error > 1e-5 and iter <= itermax:
                fluid_in = fluid_temp[i + 1]
                delta_temp = wall_temp[i] - fluid_avg
                q = heat_coeff[i] * delta_temp        
                fluid_out = fluid_in + q/capacity_rate[i]
                if fluid_in < wall_temp[i]:
                    fluid_out = np.minimum(wall_temp[i] - 1e-4, fluid_out)
                else:
                    fluid_out = np.maximum(wall_temp[i] + 1e-4, fluid_out)
                fluid_avg = (fluid_in + fluid_out) * 0.5
                error = abs(fluid_out_old - fluid_out)/fluid_out
                fluid_out_old = fluid_out
                iter += 1             
            fluid_temp[i] = fluid_out
    else:                
        for i in range(elements):
            fluid_avg = (fluid_temp[i + 1] + fluid_temp[i]) * 0.5
            fluid_out_old = 5e5
            error = 1e3
            iter = 0
            itermax = 10
            while error > 1e-4 and iter <= itermax:

                fluid_in = fluid_temp[i]            
                delta_temp = wall_temp[i] - fluid_avg
                q = heat_coeff[i] * delta_temp        
                fluid_out = fluid_in + q/capacity_rate[i]
                if fluid_in < wall_temp[i]:
                    fluid_out = np.minimum(wall_temp[i] - 1e-4, fluid_out)
                else:
                    fluid_out = np.maximum(wall_temp[i] + 1e-4, fluid_out)
                fluid_avg = (fluid_in + fluid_out) * 0.5
                error = abs(fluid_out_old - fluid_out)/fluid_out
                fluid_out_old = fluid_out
                iter += 1
                fluid_temp[i + 1] = fluid_out
            heat[i] = heat_coeff[i] * (wall_temp[i] - fluid_avg)
    return fluid_temp, heat