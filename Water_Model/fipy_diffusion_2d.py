# Solve a two-dimensional diffusion problem in a square domain.

# This example solves a diffusion problem and demonstrates the use of
# applying boundary condition patches.

# .. index::
#    single: Grid2D
import numpy as np
from matplotlib import pyplot as plt
from fipy import CellVariable, FaceVariable, Grid2D, Viewer, TransientTerm, \
    DiffusionTerm
from fipy.tools import numerix
from fipy import input
import saturation as sat
import matplotlib
matplotlib.use('TkAgg')

# physical boundary conditions and parameters
# operating conditions
current_density = 20000.0
temp = 343.15

# saturation at channel gdl interace
s_chl = 0.2

# gas pressure
p_gas = 101325.0

# parameters
faraday = 96485.3329
rho_water = 977.8
mu_water = 0.4035e-3
mm_water = 0.018
sigma_water = 0.07275 * (1.0 - 0.002 * (temp - 291.0))

# water flux due to current density
water_flux = current_density / (2.0 * faraday) * mm_water

# parameters for SGL 34BA (5% PTFE)
thickness = 260e-6
width = 2e-3
porosity = 0.74
permeability_abs = 1.88e-11

contact_angles = np.asarray([70.0, 130.0])
contact_angle = contact_angles[1]
saturation_model = 'leverett'

# collect parameters in lists for each model
params_leverett = \
    [sigma_water, contact_angle, porosity, permeability_abs]
params_psd = None

# numerical discretization
nx = 200
ny = 20

dx = width / nx
dy = thickness / ny

L = dx * nx
W = dy * ny
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# select parameter set according to saturation model
if saturation_model == 'leverett':
    params = params_leverett
elif saturation_model == 'psd':
    params = params_psd
else:
    raise NotImplementedError

# constant factor for saturation "diffusion" coefficient
D_const = rho_water / mu_water * permeability_abs

# We create a :class:`~fipy.variables.cellVariable.CellVariable` and
# initialize it to zero:
p_liq = CellVariable(name="Liquid pressure",
                     mesh=mesh,
                     value=p_gas,
                     hasOld=True)


# and then create a diffusion equation.  This is solved by default with an
# iterative conjugate gradient solver.

# initialize mesh variables
D = CellVariable(mesh=mesh, value=0.0)
D_f = FaceVariable(mesh=mesh, value=D.arithmeticFaceValue())
S = CellVariable(mesh=mesh, value=0.0, hasOld=True)
# set boundary conditions
# top: fixed Dirichlet condition (fixed liquid pressure according to saturation
# boundary condition)
# bottom: Neumann flux condition (according to reaction water flux)
X, Y = mesh.faceCenters
# facesTopLeft = ((mesh.facesLeft & (Y > L / 2))
#                 | (mesh.facesTop & (X < L / 2)))
# facesBottomRight = ((mesh.facesRight & (Y < L / 2))
#                     | (mesh.facesBottom & (X > L / 2)))

# facesTopLeft = (mesh.facesTop & (X < L / 2.0))
facesTopRight = (mesh.facesTop & (X >= L / 2.0))
facesTop = mesh.facesTop
facesBottom = mesh.facesBottom
p_capillary_top = sat.get_capillary_pressure(s_chl, params, saturation_model)
p_liquid_top = p_capillary_top + p_gas
p_liq.setValue(p_liquid_top)
# p_liq.constrain(p_liquid_top, facesTop)
p_liq.constrain(p_liquid_top, facesTopRight)
# p_liq_bot = p_liquid_top + 200.0
# p_liq.constrain(p_liq_bot, facesBottom)
# p_liq.faceGrad.constrain(water_flux, facesBottom)
D.constrain(0.0, facesBottom)

# setup diffusion equation
eq = DiffusionTerm(coeff=D_f) - (facesBottom * water_flux).divergence

# We can solve the steady-state problem
iter_max = 1000
iter_min = 10
error_tol = 1e-7
urf = 0.5
urfs = [0.5]
saturation = np.ones(nx * ny) * s_chl
saturation_old = np.copy(saturation)
residual = np.inf
iter = 0
while iter < iter_min or (iter < iter_max and residual > error_tol):
    # D = D0 * (1-phi)
    # D = CellVariable(name='saturation',
    #                  mesh=mesh,
    #                  value=sat.get_saturation(phi, params_psd, params_leverett,
    #                                           saturation_model))
    D.setValue(D_const * sat.k_s(S))
    D_f.setValue(D.arithmeticFaceValue())
    # p_liq.faceGrad.constrain(water_flux, facesBottom)
    residual = eq.sweep(var=p_liq) #, underRelaxation=urfs[i])
    p_cap = p_liq - p_gas
    saturation_old = np.copy(saturation)
    saturation_new = sat.get_saturation(p_cap, params, saturation_model)
    saturation = urf * saturation_new + (1.0 - urf) * saturation_old
    S.setValue(saturation)

    iter += 1

if __name__ == '__main__':
    viewer = Viewer(vars=S) #, datamin=0., datamax=1.)
    viewer.plot()
    input("Implicit steady-state diffusion. Press <return> to proceed...")
    fig, ax = plt.subplots()
    # for i in range(len(urfs)):
    #     ax.plot(list(range(len(residuals[i]))), residuals[i],
    #             label='urf = ' + str(urfs[i]))
    # ax.set_yscale('log')
    # plt.legend()
    # plt.show()
# .. image:: mesh20x20steadyState.*
#    :width: 90%
#    :align: center
#    :alt: stead-state solution to diffusion problem on a 2D domain with some
#    Dirichlet boundaries

# and test the value of the bottom-right corner cell.
#

# print(numerix.allclose(phi(((L,), (0,))), valueBottom, atol=1e-2))
