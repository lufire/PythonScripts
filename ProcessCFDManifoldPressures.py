# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:50:17 2020

@author: lukas
"""

import os
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt

base_path = r'D:\ZBT\Projekte\Manifold_Modell\Opiat_DummyStack'

p_manifold_file = \
    os.path.join(base_path,
                 'Opiat_Dummy_Inlet_Manifold_PressureDistribution_utf8.txt')
p_channel_file = \
    os.path.join(base_path,
                 'Opiat_Dummy_Channel_PressureDistribution_utf8.txt')
mass_flow_file_path = \
    os.path.join(base_path, 'Opiat_Dummy_MassFlowDistribution_utf8.txt')

flow_dir = 1

manifold_data = np.loadtxt(p_manifold_file).transpose()[:, 1:]
x_manifold = manifold_data[0]
p_manifold = manifold_data[1]

n_junctions = 40
junction_width = 0.004
junction_distance = 0.006825
position_1 = 0.00349946
x_junction = np.asarray([position_1 + junction_distance * i
                         for i in range(n_junctions)])
x_junction_in = x_junction - junction_width * 0.5
x_junction_out = x_junction + junction_width * 0.5
t = 2.0 * np.pi * 1.0 / junction_distance * (x_manifold - x_junction_in[0])
junction_square = 0.5 * signal.square(t) + 0.5

p_manifold_function = \
    interpolate.interp1d(x_manifold, p_manifold, kind='cubic')
p_junction_manifold = p_manifold_function(x_junction)

p_junction_in = p_manifold_function(x_junction_in)
p_junction_out = p_manifold_function(x_junction_out)
dp_junction = p_junction_out - p_junction_in
print(np.sum(dp_junction))

mass_flow_data = np.loadtxt(mass_flow_file_path).transpose()
density = 1.2044
manifold_width = 0.0125
manifold_height = 0.0075
manifold_area = manifold_width * manifold_height
manifold_mass_flow = mass_flow_data[2]
manifold_volume_flow = manifold_mass_flow / density
manifold_velocity = manifold_volume_flow / manifold_area
manifold_velocity = np.append(manifold_velocity, 0.0)
v1 = manifold_velocity[:-1]
v2 = manifold_velocity[1:]
zeta_junction = \
    (2.0 * dp_junction / density - (v2 ** 2.0 - v1 ** 2.0) * flow_dir) \
    / (v1 ** 2.0)

#
x_manifold_res = np.linspace(x_manifold[0], x_manifold[-1], 10000)
p_manifold_res = p_manifold_function(x_manifold_res)
id_manifold_min = signal.argrelmin(p_manifold_res, order=1)[0][:-1]
id_manifold_max = signal.argrelmax(p_manifold_res, order=1)[0]
print(len(id_manifold_min))
print(len(id_manifold_max))
p_manifold_min = p_manifold_res[id_manifold_min]
p_manifold_max = p_manifold_res[id_manifold_max]
x_manifold_min = x_manifold_res[id_manifold_min]
x_manifold_max = x_manifold_res[id_manifold_max]

dp_junction_2 = p_manifold_max - p_manifold_min
zeta_junction_2 = (2.0 * dp_junction_2 / density
                   - (v2[:-1] ** 2.0 - v1[:-1] ** 2.0) * flow_dir) \
    / (v1[:-1] ** 2.0)
# zeta_junction_2 = 2.0 * dp_junction_2 / density \
#     / (v1[:-1] ** 2.0)

if flow_dir == 1:
    zeta_junction_idelchik = 0.4 * (1.0 - v2 / v1) ** 2.0
else:
    zeta_junction_idelchik = np.zeros(zeta_junction.shape)

zeta_junction_fit = 0.4 * (1.0 - v2 / v1) ** 2.0
velocity_ratio = v2/v1

dpi = 200
figsize = (6.4 * 2.0, 4.8 * 2.0)
fig = plt.figure(dpi=dpi, figsize=figsize)
plt.plot(x_junction[:-1], zeta_junction_2, 'k.')
plt.plot(x_junction, zeta_junction_idelchik, 'b.')
# plt.show()
plt.savefig(os.path.join(base_path, 'inlet_x_zeta_junction_manifold.png'))

fig = plt.figure(dpi=dpi, figsize=figsize)
plt.plot(velocity_ratio[:-1], zeta_junction_2, 'k.')
plt.plot(velocity_ratio, zeta_junction_idelchik, 'b.')
# plt.show()
plt.savefig(os.path.join(base_path, 'inlet_zeta_junction_manifold.png'))

fig, ax1 = plt.subplots(dpi=dpi, figsize=figsize)
ax1.plot(x_manifold[-50:], p_manifold[-50:])
ax1.plot(x_manifold[-50:], p_manifold_function(x_manifold[-50:]))
ax2 = ax1.twinx()
ax2.plot(x_manifold[-50:], junction_square[-50:])
# plt.show()
plt.savefig(os.path.join(base_path, 'inlet_manifold_pressure.png'))

channel_data = np.loadtxt(p_channel_file).transpose()
n_channel_data = 5
x_channel = \
    np.asarray([channel_data[3 * i] for i in range(n_channel_data)])
y_channel = \
    np.asarray([channel_data[3 * i + 1] for i in range(n_channel_data)])
p_channel = \
    np.asarray([channel_data[3 * i + 2] for i in range(n_channel_data)])
lin_coeffs = \
    [np.polynomial.polynomial.polyfit(y_channel[i][200:-200],
                                      p_channel[i][200:-200], 1)
     for i in range(n_channel_data)]


def poly(x, coeffs):
    return np.polynomial.polynomial.polyval(x, coeffs)

chl_id = [0, 10, 20, 30, 39]
y_channel_in = --manifold_height * 0.5
p_channel_linear_in = \
    [poly(y_channel_in, lin_coeffs[i]) for i in range(n_channel_data)]
y_channel_out = y_channel[-1] + manifold_height * 0.5
p_channel_linear_out = \
    [poly(y_channel_out, lin_coeffs[i]) for i in range(n_channel_data)]
dp_channel_in = [p_channel_linear_in[i] - p_junction_in[chl_id[i]]
                 for i in range(n_channel_data)]

# zeta_channel_in
fig = plt.figure(dpi=dpi, figsize=figsize)
colors = ['k', 'b', 'r', 'g', 'm']
y_channel_range = y_channel[:, :100]
p_channel_range = p_channel[:, :100]
for i in range(n_channel_data):
    plt.plot(y_channel_range[i], p_channel_range[i], 
             color=colors[i], linestyle='-')
    plt.plot(y_channel_range[i], poly(y_channel_range[i], lin_coeffs[i]),
             color=colors[i], linestyle=':')
plt.show()

fig = plt.figure(dpi=dpi, figsize=figsize)
colors = ['k', 'b', 'r', 'g', 'm']
y_channel_range = y_channel[:, -100:]
p_channel_range = p_channel[:, -100:]
for i in range(n_channel_data):
    plt.plot(y_channel_range[i], p_channel_range[i], 
             color=colors[i], linestyle='-')
    plt.plot(y_channel_range[i], poly(y_channel_range[i], lin_coeffs[i]),
             color=colors[i], linestyle=':')
plt.show()

print('Channel 40 x-position:')
print(x_channel[-1, 0])
print('Channel 40 inlet pressure:')
print(p_channel[-1, 0])
print('Channel 40 manifold pressure:')
print(p_manifold_function(x_channel[-1, 0]))
