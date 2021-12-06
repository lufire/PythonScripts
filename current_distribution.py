# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
import matplotlib.pyplot as plt

def interpolate_1d(array, add_edge_points=False):
    """
    Linear interpolation in between the given array data. If
    add_edge_points is True, the neighbouring value from the settings array is
    used at the edges and the returned array will larger than the settings array.
    """

    interpolated = np.asarray(array[:-1] + array[1:]) * .5
    if add_edge_points:
        first = np.asarray([array[0]])
        last = np.asarray([array[-1]])
        return np.concatenate((first, interpolated, last), axis=0)
    else:
        return interpolated

n_ele = 10
n_nodes = n_ele + 1
x = np.linspace(0, 1, n_nodes)
x_ele = interpolate_1d(x)
dx = np.diff(x)

i_avg = 1.0
a = -0.50
b = 0.0
m = a * (i_avg - b) / (1.0 - np.exp(-a))
i_avg = -m/a*np.exp(-a) + b + m/a
i = m*np.exp(-a*x) + b
print(i_avg)
i_ele = m / (a * dx) * (np.exp(-a * x[:-1]) - np.exp(-a * x[1:])) + b
print(np.average(i_ele, weights=dx))
plt.plot(x, i)
plt.plot(x_ele, i_ele)
plt.show()