# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:55:54 2022

@author: Paul Töchterle
"""

import time
import onediff_v0_1 as od
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Set up Model Function
# =============================================================================
temps = np.array([ 3.6,  6. ,  7.2,  8.5, 10.4, 13.4, 16.7, 16. , 13.8,  9.9,  6.1,
         5.7, 5.8,  6.2,  5.9, 10. , 11.8, 14.2, 14.9, 16.6, 13.5, 10. , 8.1,
         5. ])
snow = np.zeros(24)
    
def run_model(Diff_ground, Nts=50):
    # temperature and snow data from HadUK 1km grid 
    # https://data.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.3.0/1km
    temps_ts = np.tile(temps, Nts)
    snow_ts = np.tile(snow, Nts)
    
    # initialize layer
    depth_s = 1            # snow layer thickness (m)
    depth_t = 100          # total thickness of model domain (atmosphere to ground)
    Nzs_i = 2              # number of vertical steps for snow - no unit
    Nzg_i = 20             # number of vertical steps for ground - no unit
    Tins = 9               # surface initial temperature - deg C
    
    L = od.Layer(depth_s, depth_t, Nzs_i, Nzg_i, Tins, snow_ts, temps_ts,
                 Diff_air=18.46e-6, Diff_ground=Diff_ground, Diff_snow=2.92e-7,
                 print_status=False)
    
    # diffusion in snow layer
    L.diffusion(Tins)
    
    return L

# =============================================================================
# Run Model
# 
# Calibration targed is a seasonal amplitude of 0.9 °C, offset by approx. 6 
# months and an annual mean of 10.35 °C at a depth of approx. 15m below surface
# =============================================================================
ttic = time.perf_counter() # timer start

diff = np.linspace(1e-7, 1e-5, 5)

i = 1
results = {}

for d in diff:
    tic = time.perf_counter() # timer start
    
    print(f'running scenario number {i}')
    run = run_model(Diff_ground=d)
    index = f'run_{i}'
    results[index] = run
    
    # run.plot_Tdata(title=f'D = {d.round(7)}', isoline=9, vmin=temps.min(),
    #                vmax=temps.max())
    MAST = run.get_MAST(year=50)
    DZAA = run.get_DZAA(threshold=0.9, year=50)
    MAGT = run.get_mat(DZAA[1], year=50)
    print(f'D = {d.round(7)}')
    print(f'Depth of 0.9°C ampl. = {DZAA[0].round(1)}m')
    print(f'MAST = {MAST.round(2)} °C')
    print(f'MAGT = {MAGT.round(2)} °C @ {DZAA[0].round(1)}m')
    i += 1
    tac = time.perf_counter() # timer end
    print(f'finished scenario number {i} in {tac-tic:0.2f} seconds')
    print('-----------------------------------------------------------') 
    
ttac = time.perf_counter() # timer end
print(f' *** Modelling Completed in {ttac-ttic:0.2f} seconds ***')

# =============================================================================
# Result: Diff_ground ~ 1.8 +- 0.4 e-6
# =============================================================================
# plot modelled depths of 0.9 °C amplitude as function of D_ground

DZs = [results[i].get_DZAA(threshold=0.9)[0] for i in results]
Diffs = [results[i].D[-1,0] for i in results]

f, a = plt.subplots()
a.scatter(Diffs, DZs)
a.set_xlabel('ground diffusivity')
a.set_ylabel('level of 0.9 °C amplitude')