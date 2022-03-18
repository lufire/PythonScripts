# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend

according to:
Pasaogullari, Ugur, and C. Y. Wang. “Liquid Water Transport in Gas Diffusion
Layer of Polymer Electrolyte Fuel Cells.” Journal of The Electrochemical
Society 151, no. 3 (2004): A399. https://doi.org/10.1149/1.1646148.
"""

import math
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve
# import sympy as sy
from matplotlib import pyplot as plt
import saturation as sat

# s = sy.symbols('s')

# boundary conditions
current_density = np.linspace(100.0, 30000.0, 100)
current_density = [10000.0]
temp = 343.15

# parameters
faraday = 96485.3329
rho_water = 977.8
mu_water = 0.4035e-3
mm_water = 0.018
sigma_water = 0.07275 * (1.0 - 0.002 * (temp - 291.0))

# comparison SGG
thickness = 200e-6
porosity = 0.5
permeability_abs = 6.2e-12
contact_angles = np.asarray([80.0, 100.0])

# numerical discretization
nz = 100
z = np.linspace(0, thickness, nz)
dz = thickness / nz

# saturation bc
s_chl = 0.000

# initial saturation
s_0 = np.ones(z.shape) * s_chl

# channel pressure
p_chl = 101325.0
mu_water = rho_water / mu_water
water_flux = current_density[0] / (2.0 * faraday) * mm_water


def saturation_func(s, theta):
    if theta < 90.0:
        return s ** 4.0 * (-0.2415 + 0.66765 * s - 0.6135 * s ** 2.0)
    else:
        return s ** 4.0 * (0.35425 - 0.8480 * s + 0.6135 * s ** 2.0)


# get constants
constants = []
for contact_angle in contact_angles:
    theta = contact_angle * math.pi / 180.0
    c = saturation_func(s_chl, contact_angle) \
        - water_flux * mu_water * z[-1] / \
        (sigma_water * math.cos(theta) * math.sqrt(porosity * permeability_abs))
    constants.append(c)
saturation_avg = []
saturations = []
for i in range(len(contact_angles)):
    theta = contact_angles[i] * math.pi / 180.0

    def root_saturation_hi(s):
        return s ** 4.0 * (-0.2415 + 0.66765 * s - 0.6135 * s ** 2.0) \
            - water_flux * mu_water * z[1] / \
            (sigma_water * math.cos(theta)
             * math.sqrt(porosity * permeability_abs)) \
            - constants[i]

    def root_saturation_ho(s):
        return s ** 4.0 * (0.35425 - 0.8480 * s + 0.6135 * s ** 2.0) \
            - water_flux * mu_water * z[1] / \
            (sigma_water * math.cos(theta)
             * math.sqrt(porosity * permeability_abs)) \
            - constants[i]

    # s_0 = np.ones(nz) * 0.00
    s_0 = 0.00

    if contact_angles[i] < 90.0:
        saturation = optimize.root(root_saturation_hi, s_0).x
    else:
        saturation = optimize.root(root_saturation_ho, s_0).x

    saturations.append(saturation)
    print('Current density (A/m²): ', current_density[0])

    print('Average saturation (-): ', np.average(saturation))
    print('GDL-channel interface saturation (-): ', saturation[-1])

    saturation_avg.append(np.average(saturation))

saturation_avg = np.asarray(saturation_avg)

# create plots
fig, ax = plt.subplots(dpi=150)

linestyles = ['solid', 'solid', 'solid']
markers = ['.', '.', '.']
colors = ['k', 'r', 'b']
labels = ['Leverett-J {}°'.format(str(int(item))) for item in contact_angles]
labels.append('PSD')
for i in range(len(saturations)):
    ax.plot(z[0]*1e6, saturations[i], linestyle=linestyles[i], marker=markers[i],
            color=colors[i], label=labels[i])
ax.legend()

# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([-2000, 2000.0])
# s = np.linspace(0.0, 1.0, 100)
# j = leverett_j(s, contact_angle)

# fig, ax = plt.subplots(dpi=150)
# ax.plot(s, j)
plt.tight_layout()
plt.show()






