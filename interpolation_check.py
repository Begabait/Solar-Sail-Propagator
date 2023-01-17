import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from plotter import STEPS


my_data = {'t':[], 'x':[], 'y':[], 'z':[]}
gmat_rk4 = {'t':[], 'x':[], 'y':[], 'z':[]}
gmat_sog = {'t':[], 'x':[], 'y':[], 'z':[]}

with open('rk4_data.txt', 'r') as file:
    line = file.readline()
    for i in range(STEPS):
        line = file.readline()
        my_data['t'].append(float(line.split()[0]))
        my_data['x'].append(float(line.split()[1]))
        my_data['y'].append(float(line.split()[2]))
        my_data['z'].append(float(line.split()[3]))

with open('jgm3.txt', 'r') as file:
    line = file.readline()
    for i in range(STEPS):
        line = file.readline()
        hours = float(line[12:14]) - 12
        minutes = float(line[15:17])
        time = hours * 3600 + minutes * 60
        gmat_rk4['t'].append(time)
        gmat_rk4['x'].append(float(line.split()[4]))
        gmat_rk4['y'].append(float(line.split()[5]))
        gmat_rk4['z'].append(float(line.split()[6]))

with open('gravity_change_and_friction.txt', 'r') as file:
    line = file.readline()
    for i in range(STEPS):
        line = file.readline()
        hours = float(line[12:14]) - 12
        minutes = float(line[15:17])
        time = hours * 3600 + minutes * 60
        gmat_sog['t'].append(time)
        gmat_sog['x'].append(float(line.split()[4]))
        gmat_sog['y'].append(float(line.split()[5]))
        gmat_sog['z'].append(float(line.split()[6]))

gmat_interpolation = {'x':interp1d(gmat_rk4['t'], gmat_rk4['x'], kind='cubic'), 'y':interp1d(gmat_rk4['t'], gmat_rk4['y'], kind='cubic'), 'z':interp1d(gmat_rk4['t'], gmat_rk4['z'], kind='cubic')}
gmat_sog_interpolation = {'x':interp1d(gmat_sog['t'], gmat_sog['x'], kind='cubic'), 'y':interp1d(gmat_sog['t'], gmat_sog['y'], kind='cubic'), 'z':interp1d(gmat_sog['t'], gmat_sog['z'], kind='cubic')}

inter_data_rk4 = {'x':gmat_interpolation['x'](my_data['t']), 'y':gmat_interpolation['y'](my_data['t']), 'z':gmat_interpolation['z'](my_data['t'])}
inter_data_sog = {'x':gmat_sog_interpolation['x'](my_data['t']), 'y':gmat_sog_interpolation['y'](my_data['t']), 'z':gmat_sog_interpolation['z'](my_data['t'])}

diff_gmat = np.zeros(STEPS)

diff_gmat_sog = np.zeros(STEPS)

for i in range(STEPS):
    diff_gmat[i] = (((gmat_rk4['x'][i] - inter_data_rk4['x'][i]) ** 2) + ((gmat_rk4['y'][i] - inter_data_rk4['y'][i]) ** 2) + ((gmat_rk4['z'][i] - inter_data_rk4['z'][i]) ** 2)) ** 0.5

for i in range(STEPS):
    diff_gmat_sog[i] = (((gmat_sog['x'][i] - inter_data_sog['x'][i]) ** 2) + ((gmat_sog['y'][i] - inter_data_sog['y'][i]) ** 2) + ((gmat_sog['z'][i] - inter_data_sog['z'][i]) ** 2)) ** 0.5

fig, ax = plt.subplots()
ax.plot(gmat_rk4['t'], diff_gmat, color='blue', label=r'Error')
ax.plot(gmat_rk4['t'], diff_gmat_sog, color='red')
ax.set_xlabel(r't [s]')
ax.set_ylabel(r'd [km]')
plt.show()
