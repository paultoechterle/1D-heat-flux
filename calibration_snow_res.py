# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:51:19 2022

@author: Anna
"""

import time
import onediff_v0_1 as od
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Variables that are bothering me more than they should
# =============================================================================

depth_s = 2    # Depth of the snow layer (m)

# =============================================================================
# Set up Model Function
# =============================================================================

def run_model(snow, temps, title, res_snow):
    
    tic = time.perf_counter() # timer start
    
    # Layer
    depth_t = 100              # Total depth of the layer - atmosphere to ground (m)
    Nzs_i = res_snow           # number of vertical steps for snow - no unit
    Nzg_i = 20                 # number of vertical steps for ground - no unit
    Tins = 9
    
    L = od.Layer(depth_s, depth_t, Nzs_i, Nzg_i, Tins, snow, temps,
                 Diff_air=18.46e-6, Diff_ground=1.0e-6, Diff_snow=2.92e-7,
                 print_status=False)
    
    # diffusion in snow layer
    L.diffusion(Tins)
    
    tac = time.perf_counter() # timer end
    print(f'*** run completed in {tac-tic:0.2f} seconds ***')
    return L

# =============================================================================
# Define Scenarios
# =============================================================================
# load snow/temp time series data from file
ts = pd.read_csv('snow_data.csv')
snow = np.tile(ts.snow, 5)
temps = np.tile(ts.temp, 5)
# Scenarios

offset_snow = [0.1, 0.5, 1, 1.5, 2]
offset_t = [-5, -15, 0, 5, 15]
Scenarios = []

for deltas in offset_snow:
    for deltat in offset_t:
        Scenarios.append(
            {'snow':snow*deltas,
             'temps':temps+deltat,
             'title':f'Scenario delta_snow:{deltas}, delta_t:{deltat}','res_snow':10})

# that means having
# 5 scenarios in a row with the same amount of snow but varying temperature offset
# then 5 with the next snow offset and all the temepratures offsets again, ...        

# =============================================================================
# Run Model
# =============================================================================

Nzs_i = [16]

MAAT = np.zeros((5,5,len(Nzs_i)))
MAST = np.zeros((5,5,len(Nzs_i)))
SSD = np.zeros((5,5,len(Nzs_i)))
slopes = []
intercepts = []

tic = time.perf_counter() # timer start

# k = different resolutions cycles
for k, res_snow in enumerate(Nzs_i):
    
    print(f'running model set {k} of {Nzs_i} at snow resolution of {res_snow} intervals')
    
    plt.figure()
    # i = different snow amount cycles
    for i in np.arange(5):
        
        # j = different temperature offset cycles
        for j in np.arange(5):
            S = Scenarios[5*i+j]
            S['res_snow'] = res_snow
            layer = run_model(**S)        # running model with given scenario
            
            snowpars = layer.get_snow_params()
            SSD[i,j,k] = snowpars[0]                # snow parameters
            MAAT[i,j,k] = layer.get_mat(0, year=5)  # MAAT
            MAST[i,j,k] = layer.get_MAST(year=5)    # MAST
        
        plt.scatter(MAAT[0,:,0], MAST[i,:,k], label = f'{SSD[i,0,k]}')
    
    # Linear regression
    m, q = np.polyfit(MAAT[:,:,k].flatten(), MAST[:,:,k].flatten(), 1)
    y_pred = np.sort(MAAT[0,:,k])*m + q
    slopes.append(m)
    intercepts.append(q)
    
    # Plotting
    plt.plot(np.sort(MAAT[0,:,k]), y_pred, '--', color = 'black')
    plt.xlabel('MAAT [deg C]')
    plt.ylabel('MAST [deg C]')
    plt.title(f'Snow layer depth: {depth_s}m - #sublayers: {res_snow}'
              '\n'
              f'y = {round(m,2)}x+{round(q,2)}')
    plt.legend(title = 'SSD(cm)', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    print('--------------------------------------------------------')
    
tac = time.perf_counter() # timer end
print(f'*** total runtime in {tac-tic:0.2f} seconds ***')

# =============================================================================
# comparison plot
# =============================================================================
# one final plot comparing all the linear regressions for that snow depth

plt.figure()    
for k, res_snow in enumerate(Nzs_i):
    plt.plot(np.sort(MAAT[0,:,k]), (slopes[k]*np.sort(MAAT[0,:,k]) + intercepts[k]),
             label = f'{res_snow} -> {round(slopes[k],3)}')

plt.title(f'Snow layer depth: {depth_s}m')
plt.xlabel('MAAT [deg C]')
plt.ylabel('MAST [deg C]')
plt.plot(np.sort(MAAT[0,:,0]), np.sort(MAAT[0,:,0]), color = 'black', linestyle = 'dotted',
         label = '1:1 line')
plt.legend(title = '# Layers -> Slope', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



