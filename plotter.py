import matplotlib.pyplot as plt
import numpy as np
from RK4 import runge_kutta4_ballistic, runge_kutta4_2d_orbital, runge_kutta4_3d_orbital
from RK4_dimensionless import runge_kutta4_3d_orbital_dimensionless


# def plot_trajectories():
#     time_params = [0.0, 5.0, 1000]
#     vel_i = 30.0
#     theta = 45.0
#
#     time, pos, vel = runge_kutta4_ballistic(time_params, vel_i, theta)
#     time1, pos1, vel1 = runge_kutta4_ballistic(time_params, 20, 75)
#     time2, pos2, vel2 = runge_kutta4_ballistic(time_params, 50, 15)
#
#     x_plot = np.transpose(pos)[0]
#     y_plot = np.transpose(pos)[1]
#
#     x_plot1 = np.transpose(pos1)[0]
#     y_plot1 = np.transpose(pos1)[1]
#
#     x_plot2 = np.transpose(pos2)[0]
#     y_plot2 = np.transpose(pos2)[1]
#
#     fig, ax = plt.subplots()
#     ax.plot(x_plot, y_plot, color='green', label=r'RK4')
#     ax.plot(x_plot1, y_plot1, color='red', label=r'RK4 v2')
#     ax.plot(x_plot2, y_plot2, color='yellow', label=r'RK4 v3')
#
#     end_of_plot(ax, True)


# def plot_earth_orbit_2d():
#     vel_i = [0, 7.778e3]
#
#     time, pos, vel = runge_kutta4_2d_orbital(vel_i)
#
#     x_plot = np.transpose(pos)[0]
#     y_plot = np.transpose(pos)[1]
#
#     fig, ax = plt.subplots()
#     ax.plot(x_plot, y_plot, color='red', label=r'Platform 1')
#
#     circle = matplotlib.patches.Circle((0, 0), 6371000, color='blue')
#     ax.add_artist(circle)
#
#     end_of_plot(ax, False)


def plot_earth_orbit_3d():
    pos_i = [1.496e11, 0, 0]
    vel_i = [0, 2.98e4, 0]
    time_params = (0.0, 3.1e7, 10000)

    time, pos, vel = runge_kutta4_3d_orbital(vel_i=vel_i, pos_i=pos_i, time_params=time_params, body_mass=1.989e30)

    x_plot = np.transpose(pos)[0]
    y_plot = np.transpose(pos)[1]
    z_plot = np.transpose(pos)[2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_plot, y_plot, z_plot, color='red', label=r'Platform 1')

    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Z')
    ax.legend(loc="upper right")
    plt.show()

    with open('rk4_data_with_dimensions.txt', 'w') as file:
        vx = np.transpose(vel)[0] / 1000
        vy = np.transpose(vel)[1] / 1000
        vz = np.transpose(vel)[2] / 1000
        file.write(f'time\t\tx_position\t\ty_position\t\tz_position\t\tx_velocity\t\ty_velocity\t\tz_velocity\n')

        for i in range(110):
            file.write(f'{time[i]}\t{x_plot[i] / 1000}\t{y_plot[i] / 1000}\t{z_plot[i] / 1000}\t{vx[i]}\t{vy[i]}\t{vz[i]}\n')


STEPS = 10000
def plot_earth_orbit_3d_dimensionless():

    pos_i = [1.496e11, 0, 0]
    vel_i = [0, 2.98e4, 0]
    time_params = (0.0, 1.261e8, STEPS)

    time, pos, vel = runge_kutta4_3d_orbital_dimensionless(vel_i, pos_i=pos_i, time_params=time_params)

    x_plot = np.transpose(pos)[0]
    y_plot = np.transpose(pos)[1]
    z_plot = np.transpose(pos)[2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_plot, y_plot, z_plot, color='red', label=r'Platform 1')

    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Z')
    ax.legend(loc="upper right")
    plt.show()

    with open('rk4_data.txt', 'w') as file:
        vx = np.transpose(vel)[0] / 1000
        vy = np.transpose(vel)[1] / 1000
        vz = np.transpose(vel)[2] / 1000

        file.write(f'time\t\tx_position\t\ty_position\t\tz_position\t\tx_velocity\t\ty_velocity\t\tz_velocity\n')
        for i in range(STEPS):
            file.write(f'{time[i]} {x_plot[i] / 1000} {y_plot[i] / 1000} {z_plot[i] / 1000} {vx[i]} {vy[i]} {vz[i]}')
            if i < STEPS - 1:
                file.write('\n')


def end_of_plot(ax, grid):
    ax.axis('equal')
    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.legend(loc="upper right")
    plt.grid(grid)
    plt.show()


# plot_earth_orbit_3d()
plot_earth_orbit_3d_dimensionless()
