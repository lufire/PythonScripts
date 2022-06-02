import cantera as ct
import CoolProp.CoolProp as CP
import time
import numpy as np

nx = 100

temperature_init = 293.15
pressure = 101325
x_H2O = 0.2
x_O2 = 0.21 * (1.0 - x_H2O)
x_N2 = 0.79 * (1.0 - x_H2O)
composition = {'O2': x_O2, 'N2': x_N2, 'H2O': x_H2O}


# Initialize
solution = ct.Solution('gri30.yaml')
solution.X = composition
solution.TP = temperature_init, pressure
solution_array = ct.SolutionArray(solution, nx)
temperature = np.linspace(temperature_init, temperature_init + 10.0, nx)

# Loop
n_iter = 10

start_time_ct = time.time()
for i in range(n_iter):
    temperature += 1.0
    solution_array.TP = temperature, pressure
    density = solution_array.density
    viscosity = solution_array.viscosity
    thermal_conductivity = solution_array.thermal_conductivity
    specific_heat = solution_array.cp
end_time_ct = time.time()
print('SolutionArray class timing: ', end_time_ct - start_time_ct)

start_time_ct = time.time()
for i in range(n_iter):
    temperature += 1.0
    density = np.zeros(nx)
    viscosity = np.zeros(nx)
    thermal_conductivity = np.zeros(nx)
    specific_heat = np.zeros(nx)
    for j in range(nx):
        solution.TP = temperature[j], pressure
        density[j] = solution.density
        viscosity[j] = solution.viscosity
        thermal_conductivity[j] = solution.thermal_conductivity
        specific_heat[j] = solution.cp
end_time_ct = time.time()
print('Solution class timing: ', end_time_ct - start_time_ct)

start_time_ct = time.time()
gas_mixture = 'Oxygen&Nitrogen'
HEOS = CP.AbstractState("HEOS", gas_mixture)
HEOS.set_mole_fractions([0.21, 0.79])
for i in range(n_iter):
    temperature += 1.0
    density = np.zeros(nx)
    viscosity = np.zeros(nx)
    thermal_conductivity = np.zeros(nx)
    specific_heat = np.zeros(nx)
    for j in range(nx):
        HEOS.update(CP.PT_INPUTS, pressure, temperature[j])
        density = HEOS.rhomass()
    # viscosity = CP.PropsSI('V', 'T', temperature, 'P', pressure, gas_mixture)
    # thermal_conductivity = \
    #     CP.PropsSI('L', 'T', temperature, 'P', pressure, gas_mixture)
    # specific_heat = CP.PropsSI('C', 'T', temperature, 'P', pressure, gas_mixture)

end_time_ct = time.time()
print('CoolProp timing: ', end_time_ct - start_time_ct)
