# -*- coding: utf-8 -*-
"""
created on Fri Apr 17 14:40:15 2020

@author: lukas
"""

import numpy as np
from scipy import optimize
import timeit


def overpotential_kulikovsky(j_0, eta_0, j_sigma, j_star, c_star, j_c,
                             b, K, omega):
    j_hat = j_0 / j_star
    a = np.sqrt(2. * j_hat)
    beta = a / (1. + np.sqrt(1.12 * j_hat) * np.exp(a)) \
        + np.pi * j_hat / (2. + j_hat)
    j_beta = (j_star * beta) ** 2.0
    eta_act = b * np.arcsinh((j_0 / j_sigma) ** 2.0
                             / (2.0 * c_star
                                * (1.0 - np.exp(- j_0 / (2.0 * j_star)))))
    eta_diff_cl = K * (j_0 / j_star - np.log(1.0 + j_0 ** 2.0 / j_beta)) \
        / (1.0 - j_0 / j_c)
    eta_diff_gdl = - b * np.log(1.0 - j_0 / j_c)
    eta_mem = omega * j_0
    return eta_act + eta_diff_cl + eta_diff_gdl + eta_mem - eta_0


def der_overpotential_kulikovsky(j_0, eta_0, j_sigma, j_star, c_star, j_c,
                                 b, K, omega):
    der_eta_act = \
        b * (-0.25*(j_0/j_sigma)**2.0*np.exp(-0.5*j_0/j_star)
             / (c_star * j_star * (1.0 - np.exp(-0.5 * j_0 / j_star)) ** 2.0)
             + 1.0 * (j_0 / j_sigma) ** 2.0
             / (c_star * j_0 * (1.0 - np.exp(-0.5*j_0/j_star)))) \
        / np.sqrt(1 + 0.25 * (j_0 / j_sigma) ** 4.0
                  / (c_star ** 2.0
                     * (1.0 - np.exp(-0.5 * j_0 / j_star)) ** 2.0))
    c1 = 0.707106781186547
    c2 = 1.05830052442584
    c3 = 1.4142135623731
    c4 = 0.353553390593274
    c5 = 1.78571428571429
    c6 = -0.529150262212918
    c7 = 0.748331477354788
    c8 = 0.944911182523068
    j_hat = j_0 / j_star
    sqrt_j = np.sqrt(j_hat)
    der_eta_diff_ccl = \
        K * (-(j_0 * (c1 * np.pi * j_0 / (j_star*(j_0 / j_star + 2.0))
                      + j_star * sqrt_j / (c2 * sqrt_j * np.exp(c3 * sqrt_j)
                                           + 1.0)) ** (-2.0)
               + 0.5 * j_0 ** 2.0
               * (c1 * np.pi * j_0 / (j_star*(j_hat + 2.0)) + j_star * sqrt_j
                  / (c2 * sqrt_j * np.exp(c3 * sqrt_j) + 1.0)) ** (-3.0)
               * (c4 * np.pi * j_0/(j_star ** 2.0 * (0.5 * j_hat + 1.0) ** 2.0)
                  - c5 * j_star * sqrt_j * (c6 * sqrt_j * np.exp(c3 * sqrt_j)
                                            / j_0 - c7 * j_hat
                                            * np.exp(c3 * j_hat) / j_0)
                  / (sqrt_j * np.exp(c3 * sqrt_j) + c8) ** 2.0
                  - c3 * np.pi / (j_star * (j_hat + 2.0))
                  - 1.0 * j_star * sqrt_j
                  / (j_0 * (c2 * sqrt_j * np.exp(c3 * sqrt_j) + 1.0))))
             / (0.5 * j_0 ** 2.0
                * (c1 * np.pi * j_0 / (j_star * (j_hat + 2.0)) + j_star*sqrt_j
                   / (c2 * sqrt_j * np.exp(c3 * sqrt_j) + 1.0))
                ** (-2.0) + 1.0)
             + 1.0 / j_star) / (-j_0 / j_c + 1.0) \
        + K * (j_hat - np.log(0.5 * j_0 ** 2.0
                              * (c1 * np.pi * j_0 / (j_star * (j_hat + 2.0))
                                 + j_star * sqrt_j
                                 / (c2 * sqrt_j * np.exp(c3 * sqrt_j) + 1.0))
                              ** (-2.0) + 1.0))/(j_c*(-j_0/j_c + 1.0)**2)
    der_eta_diff_gdl = b / (j_c * (-j_0 / j_c + 1.0))
    return der_eta_act + der_eta_diff_ccl + der_eta_diff_gdl + omega


# j_0 = 10000.0
n_ele = 5
cd_0 = np.full(n_ele, 10000.0)
eta_0 = np.linspace(0.2, 0.2, n_ele)

b = 0.03
i_star = 0.75e6
sigma_t = 3.0
D_ccl = 2.0e-8
D_gdl = 6.75e-6
l_ccl = 10e-6
l_gdl = 250e-6
c_ref = 7.527
c_in = c_ref
c_h = 4.93
omega = 1e-5
F = 96485.3328959

c_star = c_h / c_ref
K = sigma_t * b ** 2.0 / (4.0 * F * D_ccl * c_h)
j_sigma = np.sqrt(2.0 * i_star * sigma_t * b)
j_star = sigma_t * b / l_ccl
j_lim = 4.0 * F * D_gdl * c_in / l_gdl
j_c = j_lim * c_star


# print(overpotential_kulikovsky(1e-6, eta_0[0], j_sigma, j_star, c_star, j_c,
#                                b, K, omega))
# print(overpotential_kulikovsky(1e4, eta_0[0], j_sigma, j_star, c_star, j_c,
#                                b, K, omega))

n_iter = 100
start_time = timeit.default_timer()
for i in range(n_iter):
    res_newton = optimize.newton(overpotential_kulikovsky, cd_0,
                                 args=(eta_0, j_sigma, j_star, c_star, j_c,
                                       b, K, omega))
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
for i in range(n_iter):
    res_newton_f = optimize.newton(overpotential_kulikovsky, cd_0,
                                   fprime=der_overpotential_kulikovsky,
                                   args=(eta_0, j_sigma, j_star, c_star, j_c,
                                         b, K, omega))
print(timeit.default_timer() - start_time)

# start_time = timeit.default_timer()
# for i in range(n_iter):
#     res_fsolve = optimize.fsolve(overpotential_kulikovsky, cd_0,
#                                  args=(eta_0, j_sigma, j_star, c_star, j_c,
#                                        b, K, omega))
# print(timeit.default_timer() - start_time)

# start_time = timeit.default_timer()
# for i in range(n_iter):
#     res_brentq = np.zeros(cd_0.shape)
#     for i in np.ndindex(cd_0.shape):
#         res_brentq[i] = optimize.brentq(overpotential_kulikovsky, 1e-6, 1e4,
#                                         args=(eta_0[i], j_sigma, j_star, c_star,
#                                               j_c, b, K, omega))
# print(timeit.default_timer() - start_time)

print(res_newton)
print(res_newton_f)