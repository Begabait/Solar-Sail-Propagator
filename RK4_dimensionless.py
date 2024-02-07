import numpy as np


# M = 5.972e24  -> Earth mass; we use it when the orbit is around the earth
g = 9.81
G = 6.6743e-11
R = 6.96e8
T = 5772
M = 1.989e30
st_bo = 5.67e-8
c = 3e8
S = 2.7e4
m = 100
beta = 0
earth_radius = 6.371e6
sun_radius = 6.963e8
AU = 1.496e11
# S_M_ratio = S / m
base_v = (M * G / AU) ** 0.5
orbital_period = AU / base_v
coef = 1.95 * (st_bo * (R ** 2) * (T ** 4) * S * np.cos(beta)) / (c * m * M * G)



def rhs_3d_orbital(pos, vel):
    rhs_pos = [vel[0], vel[1], vel[2]]
    rhs_vel = (- np.array(pos) * (1 - coef)) / (np.linalg.norm(pos) ** 3)

    return rhs_pos, rhs_vel

def rhs_3d_orbital___(pos, vel):
    rhs_pos = [vel[0], vel[1], vel[2]]
    rhs_vel = - np.array(pos) / (np.linalg.norm(pos) ** 3)
    return rhs_pos, rhs_vel


def runge_kutta4_3d_orbital_dimensionless(vel_i, pos_i=(6771000, 0, 0), time_params=(0.0, 86400, 10000)):
    t_i = time_params[0]
    t_f = time_params[1] / orbital_period
    npoints = time_params[2]

    dt = (t_f - t_i)/(npoints - 1)
    time_points = np.linspace(t_i, t_f, npoints)

    vel = np.zeros((npoints, 3))
    pos = np.zeros((npoints, 3))

    for i in range(3):
        pos[0][i] = pos_i[i] / AU
        vel[0][i] = vel_i[i] / base_v

    for i in range(npoints - 1):

        k1v = np.zeros(3)
        k2v = np.zeros(3)
        k3v = np.zeros(3)
        k4v = np.zeros(3)

        k1p = np.zeros(3)
        k2p = np.zeros(3)
        k3p = np.zeros(3)
        k4p = np.zeros(3)

        k1p, k1v = rhs_3d_orbital(pos[i], vel[i])

        k2p, k2v = rhs_3d_orbital([pos[i][0] + dt * k1p[0] / 2, pos[i][1] + dt * k1p[1] / 2, pos[i][2] + dt * k1p[2] / 2],
                                  [vel[i][0] + dt * k1v[0] / 2, vel[i][1] + dt * k1v[1] / 2, vel[i][2] + dt * k1v[2] / 2])

        k3p, k3v = rhs_3d_orbital([pos[i][0] + dt * k2p[0] / 2, pos[i][1] + dt * k2p[1] / 2, pos[i][2] + dt * k2p[2] / 2],
                                  [vel[i][0] + dt * k2v[0] / 2, vel[i][1] + dt * k2v[1] / 2, vel[i][2] + dt * k2v[2] / 2])

        k4p, k4v = rhs_3d_orbital([pos[i][0] + dt * k3p[0], pos[i][1] + dt * k3p[1], pos[i][2] + dt * k3p[2]],
                                  [vel[i][0] + dt * k3v[0], vel[i][1] + dt * k3v[1], vel[i][2] + dt * k3v[2]])

        for j in range(3):
            vel[i + 1][j] = vel[i][j] + (k1v[j] + 2 * k2v[j] + 2 * k3v[j] + k4v[j]) * dt / 6
            pos[i + 1][j] = pos[i][j] + (k1p[j] + 2 * k2p[j] + 2 * k3p[j] + k4p[j]) * dt / 6

    return time_points * orbital_period, pos * AU, vel * base_v
