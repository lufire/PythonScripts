# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend
"""

import math
import numpy as np
from scipy import optimize
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

# parameters for SGL 34BA (5% PTFE)
thickness = 260e-6
porosity = 0.74
permeability_abs = 1.88e-11

# comparison SGG
thickness = 200e-6
porosity = 0.5
permeability_abs = 6.2e-12

# mixed wettability model parameters
r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
F_HI = 0.0
F = np.asarray([F_HI, 1.0 - F_HI])
f_k = np.asarray([[0.28, 0.72], [1.0, 0.0]])
s_k = np.asarray([[1.0, 0.35], [1.0, 0.35]])
contact_angle = np.asarray([80.0, 95.0])
contact_angle = np.asarray([80.0, 120.0])

# parameters SGG comparison
thickness = 200e-6
porosity = 0.5
permeability_abs = 6.2e-12
contact_angle = np.asarray([80.0, 80.0])

# capillary pressure - saturation correlation model
saturation_model = 'leverett'
# saturation_model = 'psd'

# numerical discretization
nz = 200
z = np.linspace(0, thickness, nz)
dz = thickness / nz

# saturation bc
s_chl = 0.05

# initial saturation
s_0 = np.ones(z.shape) * s_chl

# channel pressure
p_chl = 101325.0

urf = 0.5

# relative permeability
def k_s(saturation):
    return saturation ** 3.0


source = np.zeros(z.shape)

k_const = rho_water / mu_water * permeability_abs

capillary_pressure_avg = []
saturation_avg = []
p_c = np.ones(z.shape) * 1e8
for j in range(len(current_density)):
    
    water_flux = current_density[j] / (2.0 * faraday) * mm_water
    s = np.copy(s_0)
    iter_max = 100
    iter_min = 3
    eps = np.inf
    error_tol = 1e-5
    i = 0
    while i < iter_min or (i < iter_max and eps > error_tol):
        k = np.zeros(s.shape)
        k[:] = k_const * k_s(s)
        k_f = (k[:-1] + k[1:]) * 0.5
        
        k_chl = k_const * k_s(s_chl)
        
        # setup main diagonal    
        center_diag = np.zeros(nz)
        center_diag[:-1] = k_f
        center_diag[1:] += k_f
        center_diag[-1] += 2 * k_chl
        center_diag *= -1
        
        # setup offset diagonals
        off_diag = k_f
           
        # construct tridiagonal matrix        
        A_matrix = (np.diag(center_diag, k=0) + np.diag(off_diag, k=-1)
                    + np.diag(off_diag, k=1)) / dz ** 2.0
            
        # setup right hand side
        rhs = np.zeros(nz)
        rhs[:] = - source
        rhs[0] += - water_flux / dz
        rhs[-1] += - 2 * k_chl / dz ** 2.0 * p_chl
        
        p_liquid = np.linalg.tensorsolve(A_matrix, rhs)
        
        p_gas = p_chl
        p_c_old = np.copy(p_c)
        
        p_c_new = p_liquid - p_gas

        p_c = urf * p_c_old + (1.0 - urf) * p_c_new
        s_old = np.copy(s)
        # s_new = \
        #     sat.get_saturation_psd(p_c, sigma_water, contact_angle,
        #                            F, f_k, r_k, s_k)
        s_new = \
            sat.get_saturation_leverett(p_c, sigma_water, contact_angle[1],
                                        porosity, permeability_abs)
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
    print('Current density (A/m²): ', current_density[j])
    print('Number of iterations: ', i)
    print('Error: ', eps)
    print('Average saturation (-): ', np.average(s))
    print('Average capillary pressure (Pa): ', np.average(p_c))
    print('GDL-channel interface Saturation (-): ', s[-1])

    capillary_pressure_avg.append(np.average(p_c))
    saturation_avg.append(np.average(s))

        
capillary_pressure_avg = np.asarray(capillary_pressure_avg)
saturation_avg = np.asarray(saturation_avg)


fig, ax = plt.subplots(dpi=100)

# ax.plot(current_density, saturation_avg)
ax.plot(z*1e6, s)
# ax.set_ylim(0.0, 1.1)

# plt.plot(z, s_1)
plt.show()

    
    

    

