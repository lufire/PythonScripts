# Solve a two-dimensional diffusion problem in a square domain.

# This example solves a diffusion problem and demonstrates the use of
# applying boundary condition patches.

# .. index::
#    single: Grid2D
import numpy as np
from fipy import CellVariable, FaceVariable, Grid2D, Viewer, TransientTerm, \
    DiffusionTerm
from fipy.tools import numerix
from fipy import input
import saturation as sat

# physical boundary conditions and parameters
# operating conditions
current_density = 20000.0
temp = 343.15

# saturation at channel gdl interace
s_chl = 1e-3

# parameters
faraday = 96485.3329
rho_water = 977.8
mu_water = 0.4035e-3
mm_water = 0.018
sigma_water = 0.07275 * (1.0 - 0.002 * (temp - 291.0))

# parameters for SGL 34BA (5% PTFE)
thickness = 2e-3 # 260e-6
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
nx = 20
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
phi = CellVariable(name="Liquid pressure",
                   mesh=mesh,
                   value=1.,
                   hasOld=True)


# and then create a diffusion equation.  This is solved by default with an
# iterative conjugate gradient solver.

# initialize diffusion equation
D = CellVariable(mesh=mesh, value=0.0)
D_f = FaceVariable(mesh=mesh, value=0.0)
eq = DiffusionTerm(coeff=D_f)

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
# facesTopRight = (mesh.facesTop & (X >= L / 2.0))
facesTop = mesh.facesTop
facesBottom = mesh.facesBottom
phi_top = sat.get_capillary_pressure(s_chl, params, saturation_model)
phi.constrain(phi_top, facesTop)
# phi.constrain(valueTopRight, facesTopRight)
phi.constrain(0.15, facesBottom)

# We can solve the steady-state problem
iter_max = 1000
iter_min = 10
residual = np.inf
error_tol = 1e-7
i = 0
while i < iter_min or (i < iter_max and residual > error_tol):
    # D = D0 * (1-phi)
    # D = CellVariable(name='saturation',
    #                  mesh=mesh,
    #                  value=sat.get_saturation(phi, params_psd, params_leverett,
    #                                           saturation_model))
    saturation = sat.get_saturation(phi, params, saturation_model)

    D.setValue(sat.get_saturation(phi, params, saturation_model))
    D_f.setValue(D.harmonicFaceValue())
    # D.setValue(sat.get_saturation(phi, params_leverett, saturation_model))
    residual = eq.sweep(var=phi)
    i += 1

if __name__ == '__main__':
    viewer = Viewer(vars=phi, datamin=0., datamax=1.)
    viewer.plot()
    input("Implicit steady-state diffusion. Press <return> to proceed...")

# .. image:: mesh20x20steadyState.*
#    :width: 90%
#    :align: center
#    :alt: stead-state solution to diffusion problem on a 2D domain with some
#    Dirichlet boundaries

# and test the value of the bottom-right corner cell.
#

# print(numerix.allclose(phi(((L,), (0,))), valueBottom, atol=1e-2))
