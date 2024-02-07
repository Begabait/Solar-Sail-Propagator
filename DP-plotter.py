from RK_Driver import DP45_Integrator, parse_results_doublep
import numpy as np
import matplotlib.pyplot as plt


# M = 5.972e24 -> Earth's mass
M = 1.989e30
G = 6.6743e-11
AU = 1.496e11
base_v = (M * G / AU) ** 0.5
orbital_period = AU / base_v

err_tol = 1e-15 # Error Tolerance for the integrator
range_int = [0.0, 1.11556e8 / orbital_period] #1.261e8 seconds is 4 years
state_init = [1 , 0, 0, 0, 3.0e4 / base_v, 0]

out_file = b"Test_Results.csv" # Filename for the output - must be binary
header = b"T [time], x_less [m], y_less [m], z_less [m], vx_less [m/s], vy_less [m/s], vz_less [m/s]"

DP45_Integrator(err_tol, state_init, range_int, out_file, header)

time, x, y, z, vx, vy, vz = parse_results_doublep(out_file)

time = [orbital_period * i for i in time]
x = [AU * i for i in x]
y = [AU * i for i in y]
z = [AU * i for i in z]
vx = [base_v * i for i in vx]
vy = [base_v * i for i in vy]
vz = [base_v * i for i in vz]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, color='red', label=r'Platform 1')
ax.plot3D([0,0,0], [0,0,0], [-1,0,1], color='blue')

ax.set_xlabel(r'X')
ax.set_ylabel(r'Y')
ax.set_zlabel(r'Z')
ax.legend(loc="upper right")
ax.set_aspect('equalxy')
# plt.xticks(np.arange(2e11, -1e12, 5))
# plt.yticks(np.arange(4e11, -4e11, 5))
plt.show()
