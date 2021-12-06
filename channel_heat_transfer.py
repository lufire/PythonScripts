import numpy as np
import scipy as sp
from scipy.interpolate import griddata, interp1d
import timeit
from numba import jit



n = 100
temp_in = 298.15
temp1 = np.full(n, temp_in)
temp2 = np.full(n, temp_in)
temp_layer = np.full(n-1, temp_in + 20.0)
g_cool = 3000.00
k_cool = 20.0
temp2_ele = np.full(n-1, temp_in)

fwd_mat = np.tril(np.full((n-1, n-1), 1.))
bwd_mat = np.triu(np.full((n-1, n-1), 1.))

@jit(nopython=True)
def add_source(var, source, direction=1):
    n = len(var) - 1
    if len(source) != n:
        raise ValueError('source variable must be of length (var-1)')
    if direction == 1:
        var[1:] += np.matmul(fwd_mat, source)
    elif direction == -1:
        var[:-1] += np.matmul(bwd_mat, source)
    return var

def test1():
    def calc_fluid_temp(temp_in, temp_wall, g, k):
        """
        Calculates the linearised fluid outlet temperature
        of an element by the wall temperature and the fluid inlet temperature.
        The function is limited to cases with g > 0.5 * k.
        """
        return (temp_in * (g - .5 * k) + temp_wall * k) / (g + k * .5)
    
    for w in range(1, n):
        temp1[w] =\
            calc_fluid_temp(temp1[w - 1], temp_layer[w - 1], g_cool, k_cool)

def test2():
    dtemp = k_cool / g_cool * (temp_layer - temp2_ele)
    add_source(temp2, dtemp, 1)
    
n_iter = 1000
start_time = timeit.default_timer()
for i in range(n_iter):
    test1()
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    test2()
print(timeit.default_timer() - start_time)