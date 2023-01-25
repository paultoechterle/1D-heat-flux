# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:07:26 2022

@author: Paul
"""

import numpy as np
import onediff_v0_1 as od
import matplotlib.pyplot as plt

def analytical_solution(T_surf, z, D, time):
    
    from scipy.special import erf
    T_analytical = T_surf * (1 - erf(z / (2 * np.sqrt(D * time))))
    
    return T_analytical

def analytical_solution_monthly(T_surf, z, D, Nt):
    
    sec_in_year = 365*24*3600
    time = np.linspace(1,sec_in_year*Nt, Nt*12)
    
    T_data = np.zeros((len(z), Nt*12))
    for i, t in enumerate(time):
        T_data[:,i] = analytical_solution(T_surf, z, D, t)
    return T_data

def plot_Tdata(T, vmin=-20, vmax=20, isoline=[0], save=False, title=''):
    
    f, a = plt.subplots()
    
    a.set_title(title)

    im = a.imshow(T, cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax)
    
    a.contour(T, levels=isoline, colors='k')

    a.set_ylabel('Depth interval')
    a.set_xlabel('Runtime [months]')

    cbar = plt.colorbar(im)
    cbar.set_label('Temperature [°C]')
    return None

# initialize layer
Nt = 100               # runtime (in years)
depth_s = 1             # snow layer thickness (m)
depth_t = 250           # total thickness of model domain (atmosphere to ground)
Nzs = 2               # number of vertical steps for snow - no unit
Nzg = 50              # number of vertical steps for ground - no unit
Tins = 0                # surface initial temperature - deg C
T_forcing = -5

snow = np.zeros(Nt*12)
temps = np.full((Nt*12), T_forcing)

L = od.Layer(depth_s, depth_t, Nzs, Nzg, Tins, snow, temps,
             Diff_air=1.8e-6, Diff_ground=1.8e-6, Diff_snow=1.8e-6,
             print_status=True)

# diffusion in snow layer
L.diffusion(Tins)

# sec_in_year = 365*24*3600
# time = Nt * sec_in_year

an = analytical_solution_monthly(T_forcing, L.z, 1.8e-6, Nt)

plot_Tdata(L.T_data, vmin=-5, vmax=5, isoline=[-0.1], title='numerical solution')
plot_Tdata(L.T_data-an, vmin=-1, vmax=1, isoline=[-0.1,0.1], title='num. - anal. solution')

ts = np.linspace(0, Nt*12-1, 4).astype('int')
f, a = plt.subplots(dpi=150)
a.set_title(f'Nzs={Nzs} Nzg={Nzg} depth_s={depth_s} depth_t={depth_t}')
for t in ts:
    diff = an[:,t] - L.T_data[:,t]
    a.plot(diff, -L.z, label=f'difference t={t}')
a.set_ylabel('Depth [m]')
a.set_xlabel('Temperature [°C]')
a.legend()

zs = [0,3,13,23,33,43,53]
f, a = plt.subplots(dpi=150)
a.set_title(f'Nzs={Nzs} Nzg={Nzg} depth_s={depth_s} depth_t={depth_t}')
for z in zs:
    diff = an[z,:] - L.T_data[z,:]
    a.plot(np.linspace(0,Nt,Nt*12), diff, label=f'difference z={z}')
a.set_ylabel('Temperature [°C]')
a.set_xlabel('Runtime [years]')
a.legend()