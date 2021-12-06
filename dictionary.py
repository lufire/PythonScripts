# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 01:34:33 2019

@author: lukas
"""
import numpy as np

i = np.full(5, 0.)

print_data = {'Current Density': {'value': i, 'units': '$A/'}}

i[3] = 5.0

print(print_data['Current Density']['value'])
print(print_data)