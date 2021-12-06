import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import shutil
from PAUDV.publish.plot import save_plot, plot_image_from_grid
from matplotlib import animation
from PAUDV.data_proc import ResultCache
from PAUDV.data_proc.signal_tools import gaussian
from scipy.optimize import curve_fit
import scipy.signal as s
from scipy.stats import norm

def plot_image_from_grid(ax, x, y, val, vmin=3DNone, vmax=3DNone, interpolation=3D'none', **kwargs):
    return ax.imshow(np.swapaxes(val, 0, 1), extent=3D[x.min(), x.max(), y.min(), y.max()], \
		interpolation=3Dinterpolation, origin=3D'lower', vmin=3Dvmin, vmax=3Dvmax, **kwargs)


# get the path to the result cache
path = r"Y:\running_projects\AiF - Zn-Slurry\ZN\Reference_cell_III_xx08_zp001_meas2\results\xx8_PIV_dia_0.0014"
res_cache_list = "2017-06-13_13-40-21_xx8_zp001_y_L10_plane_wave_PIV_2fxx8_PIV_dia_0.0014.res"

# define path for result, transfer folder to ZBT
result_folder_path = r"U:\owncloud_TUD\AIF ZN-Slurry\Messungen\1"

measured_flow_rate = 5.39841397017e-06

# Load resultchache and look at the profiles and STD
res_cache_path = os.path.join(path, res_cache_list)
res_cache = ResultCache.load(res_cache_path)

x_axis = [0, 15e-3]
y_axis = [0, 40e-3]
z_axis = [0, 47e-3]

x = res_cache.position[..., 0].flatten()
y = res_cache.position[..., 1].flatten()

n_points_x = 256
n_points_y = 256

xgrid, ygrid = np.meshgrid(np.linspace(x_axis[0], x_axis[1], n_points_x),
                           np.linspace(y_axis[0], y_axis[1], n_points_y))

v_x_original = res_cache.velocity[:,:,0]
v_y_original = res_cache.velocity[:,:,1]

repetitions = res_cache.shape[0]

v_x_data = []
v_y_data = []
for rep in range(repetitions):
    v_x_data.append(scipy.interpolate.griddata((x, y), v_x_original[rep, : ], (xgrid, ygrid)))
    v_y_data.append(scipy.interpolate.griddata((x, y), v_y_original[rep, : ], (xgrid, ygrid)))

v_x = np.array(v_x_data)
v_y = np.array(v_y_data)

# calculate median flow field exp
v_y_median = np.nanmedian(np.swapaxes(v_y, axis1=1, axis2=2), axis=0)
v_x_median = np.nanmedian(np.swapaxes(v_x, axis1=1, axis2=2), axis=0)

# plot median flow field
fig = plt.figure(dpi=100)
ax = fig.add_subplot(1, 1, 1)
img = plot_image_from_grid(ax, xgrid*1000, ygrid*1000, 1000 * v_y_median, vmin=-2, vmax=20, interpolation='none')
cbar = fig.colorbar(img)
plt.xlim((0,15))
plt.ylim((0,30))
ax.set_xlabel("$x$ / mm")
ax.set_ylabel("$y$ / mm")
cbar.ax.set_ylabel('$v_x$ / mm/s')
save_plot(fig, os.path.join(result_folder_path, res_cache_list + "_median_flow_field"), close=True)

# calculate flow rate
flow_rate_vec = np.nansum(np.nanmedian(v_y[:, :, 0:], axis=0), axis=-1) * x_axis[1] / n_points_x * z_axis[1]
fig = plt.figure(dpi=100)
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.linspace(0, y_axis[1], n_points_y), flow_rate_vec/1.3,
             label = "volume flow, calculated from the flow profile at the specified depth")
plt.axhline(y= measured_flow_rate, xmin=0, xmax=1000e-3, linewidth=2, color = 'k')
plt.legend()
ax.set_xlabel("$y$ / m")
ax.set_ylabel("calculated volume flow / m^3/s")
save_plot(fig, os.path.join(result_folder_path,res_cache_list +  "_flow_rate"), close=True)


# save x,y,v_x,v_y as csv file
np.savetxt(os.path.join(result_folder_path, "x.csv"), x, delimiter=",", fmt = '%1.8f')
np.savetxt(os.path.join(result_folder_path, "y.csv"), y, delimiter=",", fmt = '%1.8f')
np.savetxt(os.path.join(result_folder_path, "v_x.csv"), np.median(v_x_original, axis=0), delimiter=",", fmt = '%1.8f')
np.savetxt(os.path.join(result_folder_path, "v_y.csv"), np.median(v_y_original, axis=0), delimiter=",", fmt = '%1.8f')
np.savetxt(os.path.join(result_folder_path, "v_x_std.csv"), np.std(v_x_original, axis=0, ddof = 1), delimiter=",", fmt = '%1.8f')
np.savetxt(os.path.join(result_folder_path, "v_y_std.csv"), np.std(v_y_original, axis=0, ddof = 1), delimiter=",", fmt = '%1.8f')

path_to_py_file = __file__

shutil.copy(path_to_py_file, result_folder_path)