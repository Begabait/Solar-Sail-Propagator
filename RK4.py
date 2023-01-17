import numpy as np


g = 9.81
G = 6.6743e-11


def rhs_ballistic_projectile(vel):
    rhs_pos = [vel[0], vel[1]]
    rhs_vel = [0.0, - g]

    return rhs_pos, rhs_vel


def runge_kutta4_ballistic(time_params, vel_i, theta):
    t_i = time_params[0]
    t_f = time_params[1]
    npoints = time_params[2]

    dt = (t_f - t_i)/(npoints - 1)
    time_points = np.linspace(t_i, t_f, npoints)

    vel = np.zeros((npoints, 2))
    pos = np.zeros((npoints, 2))

    pos[0] = [0.0, 0.0]
    vel[0] = [vel_i * np.cos(theta * np.pi / 180), vel_i * np.sin(theta * np.pi / 180)]

    for i in range(npoints - 1):

        k1v = np.zeros(2)
        k2v = np.zeros(2)
        k3v = np.zeros(2)
        k4v = np.zeros(2)

        k1p = np.zeros(2)
        k2p = np.zeros(2)
        k3p = np.zeros(2)
        k4p = np.zeros(2)

        rhs_pos, rhs_vel = rhs_ballistic_projectile(vel[i])
        for j in range(2):
            k1v[j] = rhs_vel[j]
            k1p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_ballistic_projectile([vel[i][0] + dt * k1v[0] / 2, vel[i][1] + dt * k1v[1] / 2])
        for j in range(2):
            k2v[j] = rhs_vel[j]
            k2p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_ballistic_projectile([vel[i][0] + dt * k2v[0] / 2, vel[i][1] + dt * k2v[1] / 2])
        for j in range(2):
            k3v[j] = rhs_vel[j]
            k3p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_ballistic_projectile([vel[i][0] + dt * k3v[0], vel[i][1] + dt * k3v[1]])
        for j in range(2):
            k4v[j] = rhs_vel[j]
            k4p[j] = rhs_pos[j]

        for j in range(2):
            vel[i + 1][j] = vel[i][j] + (k1v[j] + 2 * k2v[j] + 2 * k3v[j] + k4v[j]) * dt / 6
            pos[i + 1][j] = pos[i][j] + (k1p[j] + 2 * k2p[j] + 2 * k3p[j] + k4p[j]) * dt / 6

        if pos[i + 1][1] < 0.0:
            pos[i + 1][1] = 0.0

    return time_points, pos, vel


def rhs_2d_orbital(pos, vel, body_mass=5.972e+24):
    rhs_pos = [vel[0], vel[1]]
    rhs_vel = - G * body_mass * np.array(pos) / (np.linalg.norm(pos) ** 3)

    return rhs_pos, rhs_vel


def runge_kutta4_2d_orbital(vel_i, pos_i=(6771000, 0), time_params=(0.0, 86400.0, 10000), body_mass=5.972e24):
    t_i = time_params[0]
    t_f = time_params[1]
    npoints = time_params[2]

    dt = (t_f - t_i)/(npoints - 1)
    time_points = np.linspace(t_i, t_f, npoints)

    vel = np.zeros((npoints, 2))
    pos = np.zeros((npoints, 2))

    pos[0] = list(pos_i)
    vel[0] = vel_i

    for i in range(npoints - 1):

        k1v = np.zeros(2)
        k2v = np.zeros(2)
        k3v = np.zeros(2)
        k4v = np.zeros(2)

        k1p = np.zeros(2)
        k2p = np.zeros(2)
        k3p = np.zeros(2)
        k4p = np.zeros(2)

        rhs_pos, rhs_vel = rhs_2d_orbital(pos[i], vel[i], body_mass)
        for j in range(2):
            k1v[j] = rhs_vel[j]
            k1p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_2d_orbital([pos[i][0] + dt * k1p[0] / 2, pos[i][1] + dt * k1p[1] / 2], [vel[i][0] + dt * k1v[0] / 2, vel[i][1] + dt * k1v[1] / 2], body_mass)
        for j in range(2):
            k2v[j] = rhs_vel[j]
            k2p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_2d_orbital([pos[i][0] + dt * k2p[0] / 2, pos[i][1] + dt * k2p[1] / 2], [vel[i][0] + dt * k2v[0] / 2, vel[i][1] + dt * k2v[1] / 2], body_mass)
        for j in range(2):
            k3v[j] = rhs_vel[j]
            k3p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_2d_orbital([pos[i][0] + dt * k3p[0], pos[i][1] + dt * k3p[1]], [vel[i][0] + dt * k3v[0], vel[i][1] + dt * k3v[1]], body_mass)
        for j in range(2):
            k4v[j] = rhs_vel[j]
            k4p[j] = rhs_pos[j]

        for j in range(2):
            vel[i + 1][j] = vel[i][j] + (k1v[j] + 2 * k2v[j] + 2 * k3v[j] + k4v[j]) * dt / 6
            pos[i + 1][j] = pos[i][j] + (k1p[j] + 2 * k2p[j] + 2 * k3p[j] + k4p[j]) * dt / 6

    return time_points, pos, vel


def rhs_3d_orbital(pos, vel, body_mass=5.972e+24):
    rhs_pos = [vel[0], vel[1], vel[2]]
    rhs_vel = - G * body_mass * np.array(pos) / (np.linalg.norm(pos) ** 3)

    return rhs_pos, rhs_vel


def runge_kutta4_3d_orbital(vel_i, pos_i=(6771000, 0, 0), time_params=(0.0, 86400.0, 10000), body_mass=5.972e24):
    t_i = time_params[0]
    t_f = time_params[1]
    npoints = time_params[2]

    dt = (t_f - t_i)/(npoints - 1)
    time_points = np.linspace(t_i, t_f, npoints)

    vel = np.zeros((npoints, 3))
    pos = np.zeros((npoints, 3))

    pos[0] = list(pos_i)
    vel[0] = vel_i

    for i in range(npoints - 1):

        k1v = np.zeros(3)
        k2v = np.zeros(3)
        k3v = np.zeros(3)
        k4v = np.zeros(3)

        k1p = np.zeros(3)
        k2p = np.zeros(3)
        k3p = np.zeros(3)
        k4p = np.zeros(3)

        rhs_pos, rhs_vel = rhs_3d_orbital(pos[i], vel[i], body_mass)
        for j in range(3):
            k1v[j] = rhs_vel[j]
            k1p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_3d_orbital([pos[i][0] + dt * k1p[0] / 2, pos[i][1] + dt * k1p[1] / 2, pos[i][2] + dt * k1p[2] / 2],
                                          [vel[i][0] + dt * k1v[0] / 2, vel[i][1] + dt * k1v[1] / 2, vel[i][2] + dt * k1v[2] / 2], body_mass)
        for j in range(3):
            k2v[j] = rhs_vel[j]
            k2p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_3d_orbital([pos[i][0] + dt * k2p[0] / 2, pos[i][1] + dt * k2p[1] / 2, pos[i][2] + dt * k2p[2] / 2],
                                          [vel[i][0] + dt * k2v[0] / 2, vel[i][1] + dt * k2v[1] / 2, vel[i][2] + dt * k2v[2] / 2], body_mass)
        for j in range(3):
            k3v[j] = rhs_vel[j]
            k3p[j] = rhs_pos[j]

        rhs_pos, rhs_vel = rhs_3d_orbital([pos[i][0] + dt * k3p[0], pos[i][1] + dt * k3p[1], pos[i][2] + dt * k3p[2]],
                                          [vel[i][0] + dt * k3v[0], vel[i][1] + dt * k3v[1], vel[i][2] + dt * k3v[2]], body_mass)
        for j in range(3):
            k4v[j] = rhs_vel[j]
            k4p[j] = rhs_pos[j]

        for j in range(3):
            vel[i + 1][j] = vel[i][j] + (k1v[j] + 2 * k2v[j] + 2 * k3v[j] + k4v[j]) * dt / 6
            pos[i + 1][j] = pos[i][j] + (k1p[j] + 2 * k2p[j] + 2 * k3p[j] + k4p[j]) * dt / 6

    return time_points, pos, vel
