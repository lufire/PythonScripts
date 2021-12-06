# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:29:37 2020

@author: lukas
"""
import sympy as sy


def overpotential_kulikovsky(j_0, eta_0, j_sigma, j_star, c_star, j_c,
                             b, K, omega):
    j_hat = j_0 / j_star
    a = sy.sqrt(2.0 * j_0 / j_star)
    beta = sy.sqrt(2.0 * j_0 / j_star) / (1.0 + sy.sqrt(1.12 * j_0 / j_star) * sy.exp(sy.sqrt(2.0 * j_0 / j_star))) \
        + sy.pi * j_0 / j_star / (2.0 + j_0 / j_star)
    j_beta = (j_star * sy.sqrt(2.0 * j_0 / j_star) / (1.0 + sy.sqrt(1.12 * j_0 / j_star) * sy.exp(sy.sqrt(2.0 * j_0 / j_star)))
              + sy.pi * j_0 / j_star / (2.0 + j_0 / j_star)) ** 2.0
    eta_act = b * sy.arcsinh((j_0 / j_sigma) ** 2.0 / (2.0 * c_star * (1.0 - sy.exp(- j_0 / (2.0 * j_star)))))
    eta_diff_cl = K * (j_0 / j_star - sy.log(1.0 + j_0 ** 2.0 / ((j_star * sy.sqrt(2.0 * j_0 / j_star) / (1.0 + sy.sqrt(1.12 * j_0 / j_star) * sy.exp(sy.sqrt(2.0 * j_0 / j_star)))
                                                                  + sy.pi * j_0 / j_star / (2.0 + j_0 / j_star)) ** 2.0))) / (1.0 - j_0 / j_c)
    eta_diff_gdl = - b * sy.log(1.0 - j_0 / j_c)
    eta_mem = omega * j_0
    return eta_act + eta_diff_cl + eta_diff_gdl + eta_mem - eta_0


j_0, eta_0, j_sigma, j_star, c_star, j_c, b, K, omega = sy.symbols('j_0 eta_0 j_sigma j_star c_star j_c b K omega')

der_eta_diff_ccl = sy.diff(K * (j_0 / j_star - sy.log(1.0 + j_0 ** 2.0 / ((j_star * sy.sqrt(2.0 * j_0 / j_star) / (1.0 + sy.sqrt(1.12 * j_0 / j_star) * sy.exp(sy.sqrt(2.0 * j_0 / j_star)))
                                                               + sy.pi * j_0 / j_star / (2.0 + j_0 / j_star)) ** 2.0))) / (1.0 - j_0 / j_c), j_0)

der_eta_act = sy.diff(b * sy.asinh((j_0 / j_sigma) ** 2.0 / (2.0 * c_star * (1.0 - sy.exp(- j_0 / (2.0 * j_star))))), j_0)

der_eta_diff_gdl = sy.diff(- b * sy.log(1.0 - j_0 / j_c), j_0)
print(der_eta_diff_gdl)