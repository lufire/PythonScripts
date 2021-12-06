# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:26:36 2020

@author: lukas
"""
import numpy as np


a = np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
b = np.array((0.0, 1.0, 0.0))
a = np.array(1.0)
print(a[a != 0])
print(b[np.nonzero(b)])