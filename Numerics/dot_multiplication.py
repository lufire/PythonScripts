# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:01:05 2019

@author: lukas
"""

import numpy as np

def create_coords(length, direction_vector, start_vector, nx):
    length_vector = length * direction_vector
    end_vector = start_vector + length_vector
    coords = \
        np.asarray([np.linspace(start_vector[i], end_vector[i],
                                nx)
                    for i in range(len(end_vector))])
    return coords

x = np.array([0.0, 1.0, 0.0])
x_norm = x / np.linalg.norm(x)
length = 5.0
start = np.array([1.0, 3.0, 0.0])
nx = 5

coords = create_coords(length, x_norm, start, nx)
cord_length = np.zeros(coords.shape)
cord_length[:, 1:] = np.diff(coords, axis=-1)
print(coords)
print(np.dot(coords.transpose(), x_norm))
print(np.linalg.norm(coords, axis=0))
print(np.linalg.norm(cord_length, axis=0))