# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:01:05 2019

@author: lukas
"""

import numpy as np

a = np.array([4.0, 2.0, 3.0, 4.0])
b = np.array([0.5, 0.5, 0.5])
c = np.outer(a, b)
print(c)