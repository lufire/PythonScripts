# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:42:07 2021

@author: lukas
"""

# test cython channel heat transfer routine
import numpy as np
import time
import sys
sys.path.append(r"D:\ZBT\Python\cython\compiled")

import channel_heat_transfer as cht
import channel_heat_transfer


def calc_temp_heat_transfer(wall_temp, fluid_temp, capacity_rate, heat_coeff,
                            flow_direction):
    wall_temp = np.asarray(wall_temp)
    fluid_temp = np.asarray(fluid_temp)
    capacity_rate = np.asarray(capacity_rate)
    heat_coeff = np.asarray(heat_coeff)
    assert capacity_rate.shape == wall_temp.shape
    assert heat_coeff.shape == wall_temp.shape
    fluid_temp_avg = np.asarray(fluid_temp[:-1] + fluid_temp[1:]) * .5
    fluid_temp_avg = np.zeros(wall_temp.shape)
    if flow_direction == 1:
        fluid_temp_avg[:] = fluid_temp[0]
    else:
        fluid_temp_avg[:] = fluid_temp[-1]
    id_range = range(len(wall_temp))
    if flow_direction == -1:
        id_range = reversed(id_range)
    for i in id_range:
        fluid_avg = fluid_temp_avg[i]
        fluid_out_old = 5e5
        error = 1e3
        iter = 0
        itermax = 10
        while error > 1e-4 and iter <= itermax:
            if flow_direction == -1:
                fluid_in = fluid_temp[i + 1]
            else:
                fluid_in = fluid_temp[i]
            if (wall_temp[i] - fluid_avg)/(wall_temp[i] - fluid_in) > 0.0:
                delta_temp = wall_temp[i] - fluid_avg
            else:
                delta_temp = wall_temp[i] - fluid_in

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
        if flow_direction == -1:
            fluid_temp[i] = fluid_out
        else:
            fluid_temp[i + 1] = fluid_out
    fluid_temp_avg = np.asarray(fluid_temp[:-1] + fluid_temp[1:]) * .5
    heat = heat_coeff * (wall_temp - fluid_temp_avg)
    return fluid_temp, heat


def calc_heat_transfer(wall_temp, fluid_temp, 
                       capacity_rate, heat_coeff, flow_direction):

    assert tuple(capacity_rate.shape) == tuple(wall_temp.shape)
    assert tuple(heat_coeff.shape) == tuple(wall_temp.shape)  
    
    elements = fluid_temp.shape[0]
    nodes = wall_temp.shape[0]
    heat = np.zeros(elements)
    temp_result = np.zeros(nodes)
    temp_result_view = np.copy(temp_result)
    heat_view = np.copy(heat)
    fluid_avg = 0.0
    fluid_out_old = 5e5
    fluid_in = 0.0
    fluid_out = 0.0
    q = 0.0
    error = 1e3
    delta_temp = 0.0
    iter = 0
    itermax = 10
    fluid_temp_avg = 0.0
    temp_result_view[:] = fluid_temp[:-1]
    if flow_direction == -1:
        for i in reversed(range(elements)):
            print(i)
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
                    fluid_out = min(wall_temp[i] - 1e-4, fluid_out)
                else:
                    fluid_out = max(wall_temp[i] + 1e-4, fluid_out)
                fluid_avg = (fluid_in + fluid_out) * 0.5
                error = abs(fluid_out_old - fluid_out)/fluid_out
                fluid_out_old = fluid_out
                iter += 1             
            temp_result_view[i] = fluid_out
            heat_view[i] = heat_coeff[i] * (wall_temp[i] - fluid_avg)
    else:         
        for i in range(elements):
            fluid_avg = (fluid_temp[i + 1] + fluid_temp[i]) * 0.5
            fluid_out_old = 5e5
            error = 1e3
            iter = 0
            itermax = 10
            while error > 1e-5 and iter <= itermax:

                fluid_in = fluid_temp[i]            
                delta_temp = wall_temp[i] - fluid_avg
                q = heat_coeff[i] * delta_temp        
                fluid_out = fluid_in + q/capacity_rate[i]
                if fluid_in < wall_temp[i]:
                    fluid_out = min(wall_temp[i] - 1e-4, fluid_out)
                else:
                    fluid_out = max(wall_temp[i] + 1e-4, fluid_out)
                fluid_avg = (fluid_in + fluid_out) * 0.5
                error = abs(fluid_out_old - fluid_out)/fluid_out
                fluid_out_old = fluid_out
                iter += 1
            temp_result_view[i + 1] = fluid_out
            heat_view[i] = heat_coeff[i] * (wall_temp[i] - fluid_avg)
    return temp_result, heat

n = 5
m = 1
temp_in = 433.15

wall_temp = np.array([347.412411659, 352.743877026, 353.108570097, 
                      352.910361401, 349.598155234])

temp = np.array([353.149, 353.149, 353.149, 353.149, 353.149, 333.15])
g = np.array([0.000008908, 0.00000958, 0.000010252, 0.000010924, 0.000011581])

k = np.array([0.078058072, 0.080816256, 0.083346148, 0.085674459, 0.085551908])


flow_direction = -1

tstart = time.time()
for i in range(m):
    fluid_temp, heat = \
        calc_temp_heat_transfer(wall_temp, temp, g, k, flow_direction)

tstop = time.time()
dt = tstop - tstart
print('native python time [s]: ', time.time()-tstart)

tstart = time.time()

for i in range(m):
    fluid_temp_cython, heat_cython = \
        cht.calc_heat_transfer(wall_temp, temp, g, k, flow_direction)
tstop = time.time()
dt_cython = tstop - tstart
print('cython time [s]', time.time()-tstart)

rel_dev = np.abs(fluid_temp - fluid_temp_cython) \
    / (fluid_temp + fluid_temp_cython) * 2.0

error = np.sqrt(np.sum(np.square(rel_dev)))
print('fluid temp: ', fluid_temp)
print('cython fluid temp: ', fluid_temp_cython)

#print('speed-up: ', dt/dt_cython)
print('TSRE: ', error)

