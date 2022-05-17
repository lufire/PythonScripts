# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend
"""

import math
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import interpolate
# import sympy as sy
from matplotlib import pyplot as plt
import saturation as sat
# import analytic_leverett as al

# s = sy.symbols('s')
# SMALL = 1e-8

# boundary conditions
# current_density = np.linspace(100.0, 30000.0, 100)
current_density = 20000.0
temp = 343.15
# saturation bc
s_chl = 1e-3
# channel pressure
p_chl = 101325.0


# parameters
faraday = 96485.3329
rho_water = 977.8
mu_water = 0.4035e-3
mm_water = 0.018
sigma_water = 0.07275 * (1.0 - 0.002 * (temp - 291.0))

# parameters for SGL 34BA (5% PTFE)
thickness = 260e-6
porosity = 0.74
permeability_abs = 1.88e-11

# remaining domain settings
width = 2e-3

# psd specific parameters
r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
F_HI = 0.08
f_k = np.asarray([[0.28, 0.72], [0.28, 0.72]])
s_k = np.asarray([[0.35, 1.0], [0.35, 1.0]])


contact_angles = np.asarray([70.0, 130.0])
contact_angle = contact_angles[1]
# capillary pressure - saturation correlation model ('leverett', 'psd')
saturation_model = 'leverett'
# saturation_models = ['psd']

# numerical discretization
nz = 20
ny = 200
z = np.linspace(0, thickness, nz, endpoint=True)
y = np.linspace(0, width, ny, endpoint=True)
dz = thickness / nz
dy = width / ny

urf = 0.3


# relative permeability
def k_s(saturation):
    saturation = np.copy(saturation)
    saturation[saturation == 0.0] = 1e-6
    return saturation ** 3.0


source = np.zeros((nz-1, ny-1))

k_const = rho_water / mu_water * permeability_abs

capillary_pressure_avg = []
saturation_avg = []
saturations = []
capillary_pressures = []
shape = (nz, ny)
p_c = np.ones(shape)
p_liquid = np.zeros(shape)
p_gas = np.zeros(shape)

s_0 = np.ones(shape) * s_chl
s = np.copy(s_0)

F = np.asarray([F_HI, 1.0 - F_HI])

params_psd = [sigma_water, contact_angles, F, f_k, r_k, s_k]
params_leverett = \
    [sigma_water, contact_angle, porosity, permeability_abs, s]

water_flux = current_density / (2.0 * faraday) * mm_water
# s = np.copy(s_0)
iter_max = 1000
iter_min = 10
eps = np.inf
error_tol = 1e-7
i = 0
s_chl = s[-1]
while i < iter_min or (i < iter_max and eps > error_tol):

    p_gas[:] = p_chl

    if saturation_model == 'leverett':
        p_liquid_chl = \
            sat.leverett_p_s(s_chl, sigma_water, contact_angle,
                             porosity, permeability_abs) + p_chl
    elif saturation_model == 'psd':
        p_liquid_chl = \
            sat.get_capillary_pressure_psd(
                s_chl, params_psd) + p_chl

                # capillary_pressure_prev=p_c[-1]) \
    else:
        raise NotImplementedError
    p_liquid[-1] = p_liquid_chl

    # k_bc = 1e-6
    # p_0 = 1.0
    # p_liquid[:] = p_chl + water_flux * p_0 / k_bc

    k = np.zeros(s.shape)
    k[:] = k_const * k_s(s)
    # k_f = (k[:-1] + k[1:]) * 0.5
    z_f = (z[:-1] + z[1:]) * 0.5
    k_f = interpolate.interp1d(z, k, kind='linear')(z_f)
    # k_chl = k[-1]

    k_0 = k[0]
    # setup main diagonal
    center_diag = np.zeros(ny * (nz - 1))
    center_diag[1:] += k_f[:-1]
    center_diag[:] += k_f
    # boundary conditions (0: Neumann, nz-1: Dirichlet)
    center_diag[0] += k[0]
    # center_diag[-1] = 1.0
    center_diag *= -1

    # setup offset diagonals
    lower_diag = np.zeros(ny * (nz - 2))
    lower_diag[:] = k_f[:-1]

    upper_diag = np.zeros(ny * (nz - 2))
    upper_diag[:] = k_f[:-1]
    upper_diag[0] += k[0]

    # construct tridiagonal matrix
    A_matrix = (np.diag(center_diag, k=0) + np.diag(lower_diag, k=-1)
                + np.diag(upper_diag, k=1)) / dz ** 2.0

    if nz > 200:
        A_matrix = sparse.csr_matrix(A_matrix)

    # setup right hand side
    rhs = np.zeros(ny * (nz-1))
    rhs[:] = - source
    rhs[0] += - water_flux * 2.0 / dz
    rhs[-1] += - k_f[-1] * p_liquid[-1] / dz ** 2.0

    if nz > 200:
        p_liquid[:-1] = spsolve(A_matrix, rhs)
    else:
        p_liquid[:-1] = np.linalg.tensorsolve(A_matrix, rhs)

    p_c_old = np.copy(p_c)

    p_c_new = p_liquid - p_gas

    p_c = p_c_new  # urf * p_c_old + (1.0 - urf) * p_c_new
    s_old = np.copy(s)

    s_new = \
        sat.get_saturation(p_c, params_psd, params_leverett,
                           saturation_model)
    s = urf * s_new + (1.0 - urf) * s_old
    s_diff = s - s_old
    p_diff = p_c - p_c_old
    eps_s = np.dot(s_diff.transpose(), s_diff) / (2.0 * len(s_diff))
    eps_p = np.dot(p_diff.transpose(), p_diff) / (2.0 * len(p_diff))

    eps = eps_s + eps_p
    # p_c_2 = sat.leverett_p_s(s, sigma_water, contact_angle[1],
    #                         porosity, permeability_abs)
    # print(i)
    # print(p_c)
    # print(s)
    i += 1
    # print(eps)
# if i >= iter_max:
#     s *= 0.0
print('Current density (A/m²): ', current_density)
print('Number of iterations: ', i)
print('Error: ', eps)
print('Average saturation (-): ', np.average(s))
print('Average capillary pressure (Pa): ', np.average(p_c))
print('GDL-channel interface saturation (-): ', s[-1])

capillary_pressure_avg.append(np.average(p_c))
saturation_avg.append(np.average(s))
saturations.append(s)
capillary_pressures.append(p_c)

        
capillary_pressure_avg = np.asarray(capillary_pressure_avg)
saturation_avg = np.asarray(saturation_avg)


fig, ax = plt.subplots(dpi=100)
colors = ['k', 'r', 'k', 'r']
linestyles = ['solid', 'solid', 'dashed', 'dashed']
labels = ['F_HI: {}'.format(F_HI)]
# ax.plot(current_density, saturation_avg)
for i in range(len(saturations)):
    ax.plot(z * 1e6, saturations[i]) #, color=colors[i], linestyle=linestyles[i])

# ax.plot(z * 1e6, saturations[0], color=colors[0], linestyle=linestyles[0])
# ax.plot(z * 1e6, saturations[2], color=colors[2], linestyle=linestyles[2])
# ax.plot(z * 1e6, saturations[1], color=colors[1], linestyle=linestyles[1])

# for i in range(len(al.saturations)):
#     ax.plot(z * 1e6, al.saturations[i], color=colors[i], linestyle='dashed')
ax.set_xlabel('GDL Location / µm')
ax.set_ylabel('Saturation / -')
# ax.set_yscale('log')

#ax2.legend(['Pressure'], loc='lower left')

ax.legend(labels, loc='lower left')

# ax.set_ylim(0.0, 1.1)

# plt.plot(z, s_1)
plt.tight_layout()
plt.show()

    
    

    

