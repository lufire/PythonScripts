# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend
"""

import math
import numpy as np
from scipy import special
from scipy import optimize
from matplotlib import pyplot as plt


def leverett_hi(s):
    return 1.417 * (1.0 - s) - 2.120 * (1.0 - s) ** 2 \
        + 1.263 * (1.0 - s) ** 3
    
def leverett_ho(s):
    return 1.417 * s - 2.120 * s ** 2 + 1.263 * s ** 3

def leverett_j(s, theta):
    if theta < 90.0:
        return leverett_hi(s)
    else:
        return leverett_ho(s)
    
def leverett_p_s(saturation, surface_tension, contact_angle, 
                 porosity, permeability):
    factor = - surface_tension * np.cos(contact_angle * np.pi / 180.0) \
        * np.sqrt(porosity / permeability)
    return factor * leverett_j(saturation, contact_angle)

def leverett_s_p(capillary_pressure, surface_tension, contact_angle, 
                 porosity, permeability):
    factor = surface_tension * np.cos(contact_angle * np.pi / 180.0) \
        * np.sqrt(porosity / permeability)
    def root_leverett_p_s(saturation):
        return factor * leverett_j(saturation, contact_angle) - capillary_pressure
    s_in = np.zeros(np.asarray(capillary_pressure).shape) + 0.0
    saturation = optimize.root(root_leverett_p_s, s_in).x
    return saturation

# parameters
temp = 343.15
surface_tension = 0.07275 * (1.0 - 0.002 * (temp - 291.0))
porosity = 0.5
permeability = 6.2e-12
contact_angle = 80.0

capillary_pressure = np.linspace(-10000.0, 10000.0, 100)
saturation = leverett_s_p(capillary_pressure, surface_tension, contact_angle, 
                 porosity, permeability)
fig, ax = plt.subplots(dpi=150)

ax.plot(capillary_pressure, saturation)

# s = np.linspace(0.0, 1.0, 100)
# j = leverett_j(s, contact_angle)

# fig, ax = plt.subplots(dpi=150)
# ax.plot(s, j)

plt.show()