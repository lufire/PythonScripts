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

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


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
    factor = - surface_tension * np.cos(contact_angle * np.pi / 180.0) \
        * np.sqrt(porosity / permeability)

    def root_leverett_p_s(saturation):
        return factor * leverett_j(saturation, contact_angle) \
               - capillary_pressure
    s_in = np.zeros(np.asarray(capillary_pressure).shape) + 1e-3
    saturation = optimize.root(root_leverett_p_s, s_in).x
    return saturation


def young_laplace(capillary_pressure, sigma, contact_angle):
    return - 1.0 * 2.0 * sigma * np.cos(contact_angle * np.pi / 180.0) \
           / capillary_pressure


def get_critical_radius(capillary_pressure, sigma, contact_angle):
    critical_radius_hydrophilic = \
        np.where(capillary_pressure < 0.0,
                 young_laplace(capillary_pressure, sigma, contact_angle[0]),
                 np.inf)
    critical_radius_hydrophobic = \
        np.where(capillary_pressure < 0.0, np.inf,
                 young_laplace(capillary_pressure, sigma, contact_angle[1]))
    return np.asarray([critical_radius_hydrophilic,
                       critical_radius_hydrophobic])


def get_saturation_leverett(capillary_pressure, params):
    try:
        surface_tension = params[0]
        contact_angle = params[1]
        porosity = params[2]
        permeability = params[3]
    except IndexError:
        raise IndexError('params input list not complete, must include: '
                         'surface_tension:float, contact_angle:float, '
                         'porosity:float, permeability:float')
    saturation = \
        leverett_s_p(capillary_pressure, surface_tension, contact_angle, 
                     porosity, permeability)
    # return saturation
    return np.where(saturation < 0.0, 0.0,
                    np.where(saturation > 1.0, 1.0, saturation))


def get_saturation_psd(capillary_pressure, params):
    try:
        surface_tension = params[0]
        contact_angles = params[1]
        F = params[2]
        f = params[3]
        r = params[4]
        s = params[5]
    except IndexError:
        raise IndexError('params input list not complete, must include: '
                         'surface_tension:float, contact_angle:list, F:list,'
                         'f:list, r:list, s:list')

    critical_radius = get_critical_radius(capillary_pressure, surface_tension,
                                          contact_angles)
    saturation = np.zeros(critical_radius.shape[-1])
    phi = [1, -1]
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            saturation += F[i] * f[i, j] * 0.5 \
                * (1.0 + phi[i] * special.erf((np.log(critical_radius[i])
                                               - np.log(r[i, j]))
                   / (s[i, j] * SQRT_2)))
    return saturation


def get_saturation(capillary_pressure, params_psd, params_leverett, model_type):
    if model_type == 'psd':
        return get_saturation_psd(capillary_pressure, params_psd)
    elif model_type == 'leverett':
        return get_saturation_leverett(capillary_pressure, params_leverett)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    # parameters
    r = [0.1]
    # boundary conditions
    current_density = 100.0
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
    
    # # mixed wettability model parameters
    # r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
    # F_HI = 0.0
    # F = np.asarray([F_HI, 1.0 - F_HI])
    # # f_k = np.asarray([[0.28, 0.72], [0.28, 0.72]])
    # # f_k = np.asarray([[0.0, 1.0], [0.28, 0.72]])
    # f_k = np.asarray([[0.28, 0.72], [0.0, 1.0]])
    # s_k = np.asarray([[1.0, 0.35], [1.0, 0.35]])
    # contact_angle = np.asarray([70.0, 122.0])
    
    # mixed wettability model parameters
    r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
    F_HI = 0.1
    F = np.asarray([F_HI, 1.0 - F_HI])
    f_k = np.asarray([[0.28, 0.72], [0.28, 0.72]])
    s_k = np.asarray([[1.0, 0.35], [1.0, 0.35]])
    thetas = np.asarray([70.0, 122.0])

    p_c = np.linspace(-1000, 1000, 100)
    r_c = get_critical_radius(p_c, sigma_water, thetas)
    # s = get_saturation(r_c, F, f_k, r_k, s_k)
    
    theta = thetas[1]
    
    s_2 = leverett_s_p(p_c, sigma_water, theta, 
                       porosity, permeability_abs)
    s_3 = np.linspace(0.0, 1.0, 100)
    p_c_3 = leverett_p_s(s_3, sigma_water, theta, 
                         porosity, permeability_abs)

    #  plt.plot(p_c, s)
    plt.plot(p_c, s_2)
    plt.plot(p_c_3, s_3)

    plt.show()
