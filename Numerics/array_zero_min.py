# -*- coding: utf-8 -*-
"""
Created on Sun May  2 03:35:40 2021

@author: lukas
"""

import time
import numpy as np

m = 1000
array = np.random.rand(3, 10000)
array -= 0.5


tstart = time.time()
for i in range(m):
    array[array < 0.0] = 0.0

tstop = time.time()
dt = tstop - tstart
print('1', dt)


tstart = time.time()
for i in range(m):
    array.clip(min=0.0, out=array)


tstop = time.time()
dt = tstop - tstart
print('2', dt)