# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:44:02 2022

@author: Anna Baldo and Paul Töchterle
        (with code extracts by Alexander H. Jarosch (research@alexj.at)
        'A 1D implementation of thermal conductivity through soil and air')
"""

# =============================================================================
# Import libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# =============================================================================
# To do list - please add/remove as necessary
# =============================================================================
''' 
-   check if indices and depths align between z array, D array and interface 
    definitions.

-   Figure out how to make diffusion through air go quasi-instantly (or at 
    least fast enough so it won't delay the monthly temperature shifts)
        - adding lost temperature to the input? or similar work-arounds
        
-   Add option for geothermal heat flux (modify boundary condition)
'''

# =============================================================================
# Global Variables
# =============================================================================
# time scale
sec_in_year = 365*24*3600
dt = sec_in_year / 12 # timestep

# =============================================================================
# Classes
# =============================================================================
class Layer:
    
    def __init__(self, depth_s, depth_t, Nzs, Nzg, Tin, snow_ts, temp_ts,
                 Diff_air=18.46e-6, Diff_ground=1.0e-6, Diff_snow=2.92e-7,
                 print_status=True):
        
        # input checks
        if max(snow_ts) > depth_s:
            print('WARNING: max. snow height is larger than the snow layer domain. Make sure max(snow_ts) < depth_s')
        if len(temp_ts) != len(snow_ts):
            print('WARNING: snow_ts and temp_ts must be the same size.')
        if len(temp_ts)%12 != 0:
            print('WARNING: time series inputs (temp_ts/snow_ts contain incomplete years. Length must be a multiple of 12 (months)!')
        
        # set attributes        
        self.depth_s = depth_s      # snow layer depth (m) - not actual snow depth!
        self.depth_t = depth_t      # total depth (m)
        self.Nz_s = Nzs             # number vertical steps snow
        self.Nz_g = Nzg             # number vertical steps ground
        self.Nz_res = Nzs + Nzg     # total number of vertical steps
        self.Tin = Tin              # initial surface temperature - deg C
        self.Nt = np.rint(len(temp_ts)/12).astype('int')   # runtime (Number of years)
        self.snow_ts = snow_ts      # time series of monthly snow depth
        self.temp_ts = temp_ts      # time series of monthly temperature
        self.print_status = print_status # print year xx of xx statement on/off
        
        # set space domain
        self.z, self.dz, self.Nz = self.compute_space_domain()
        
        # set up diffusivity matrix (monthly profiles)
        dimz = self.Nz_res+2        # depth dimension (m)
        dimt = self.snow_ts.size    # time dimension (months) 
        self.D = np.full((dimz, dimt), Diff_ground)  
        
        for i in range(dimt):
            percentage = 1 - self.snow_ts[i]/self.depth_s
            self.interface1 = np.rint(percentage*self.Nz_s).astype('int') # interface air/snow
            self.interface2 = np.rint(self.Nz_s).astype('int') # interface snow/ground
            
            self.D[:self.interface1,[i]] = Diff_air
            self.D[self.interface1:self.interface2,[i]] = Diff_snow
        
        # generate temperature array to be filled later on in the
        # diffusion steps:
        self.T_data = np.zeros((dimz, dimt))
        # set initial T conditions (year 0, month 0, all depth steps)
        self.T_data[:, 0] = self.Tin

        return None
    
    def compute_space_domain(self):
        """
        Creates the space domain of the layer,
        taking into account different resolutions for snow and ground.
        Results are stored in z_layer, dz_layer and Nz
        Returns
        -------
        z_layer : array-like
            Array containing the depth grid points.
        dz_layer: float
            Resolution of the layer (distance between consecutive grid points).
        Nz: float
            Number of elements contained in z_layers,
            that is the physical grid points plus the ones needed for BCs
        """ 
        
        # snow
        [zs, dzs] = np.linspace(0, self.depth_s, self.Nz_s, endpoint = False, retstep=True)
        # ground
        self.depth_g = self.depth_t - self.depth_s
        [zg0, dzg0] = np.linspace(self.depth_s + dzs, self.depth_s + dzs + self.depth_g,
                                  self.Nz_g-1, endpoint = False, retstep=True)
        [zg, dzg] = np.linspace(self.depth_s + dzs, self.depth_s + dzs + self.depth_g+2*dzg0, 
                                self.Nz_g+1, endpoint = False, retstep=True)
        
        zg_upper = np.linspace(self.depth_s, self.depth_s + dzs, 1, endpoint = False)
        zg = np.concatenate((zg_upper, zg), axis = None)
        z_layer = np.concatenate((zs, zg), axis = None)
        dz_layer = np.concatenate((dzs, dzg), axis = None)

        Nz = len(z_layer)
        
        return z_layer, dz_layer, Nz #, dz_total
    
    def diffusion(self, T_top):
        """
        Calculates the heat transfer based on the input parameter specified
        in the Layer object. The results are stored in the T_data matrix.

        Parameters
        ----------
        T_top : float
            Initial temperature of the topmost depth interval.

        Returns
        -------
        T_data : array-like
            Matrix containing temperatures after diffusion. Dimensions are
            T_data[year, month, z], where z is the index of the respective
            depth interval.

        """        
        # copy the first time, first month,  all layers
        T = copy.deepcopy(self.T_data[:,0])
        
        # make i index array
        index_i = np.arange(1, self.Nz-1)
        # make i plus array
        index_i_plus = np.arange(2, self.Nz)
        # make i minus array
        index_i_minus = np.arange(0, self.Nz-2)

        # overall time in seconds
        t = 0
        t_month = sec_in_year/12
        
        for month in range(self.temp_ts.size): # iteration over runtime 
            
            if self.print_status==True:
                if self.temp_ts.size < 120:
                    if month % 12 == 0:
                        print(f'running model year {int(month/12)} of {self.Nt}...')
                elif month % 120 == 0:
                    print(f'running model year {int(month/12)} of {self.Nt}...')
                else:
                    pass
                
            t_stab = 0
            while t_stab < t_month:
                
                To = copy.deepcopy(T)
                
                # Get grid spacing
                h_up = (self.z[index_i_plus] - self.z[index_i]) # m
                h_down = (self.z[index_i] - self.z[index_i_minus]) # m
    
                # update diffusivity array
                D = self.D[:,month] # m²/s
                
                # D_up = (D[index_i]+D[index_i_plus])*0.5 # m²/s
                # D_down = (D[index_i_minus]+D[index_i])*0.5 # m²/s
                D_up = D[1:self.Nz-1]
                D_down = D[:self.Nz-2]
    
                # stability according to CFL
                dt_stab = (0.5 * self.dz[0]**2) / max(max(abs(D_up)), max(abs(D_down))) # s
                # time step to actually use
                dt_use = min(dt_stab, dt - t_stab)
                # counting stability time
                t_stab = t_stab + dt_use
                # update passing of time
                t = t + dt_use
                
                # upper BC
                # (this needs to be the surf. temp of month)
                T[0] = self.temp_ts[month]
                
                # compute D*gradT
                D_div_T = (D_down*((T[index_i_minus]-T[index_i])/h_down) - (D_up*((T[index_i]-T[index_i_plus])/h_up)) ) / (0.5*(h_up+h_down))
                # time step: 
                T[index_i] = To[index_i] + D_div_T*dt_use
                
                # lower BC
                T[self.Nz-1] = T[self.Nz-2]
                    
            # monthly timestep
            self.T_data[:, month] = T
        
        return None
    
    def export_csv(self, fname=''):
        
        seasonality = self.temp_ts.max()-self.temp_ts.min()
        snow = self.snow_ts.max().round(2)
        
        if fname == '':
            fname= f'Nt{self.Nt}_Nzsg{self.Nz_s}_{self.Nz_g}_Tin{self.Tin}_seas{seasonality.round(0)}_snow{snow}'
        filename = 'exports/'+fname+'.csv'
        header = self.input_params()
        np.savetxt(filename, self.T_data.T.round(1), delimiter=',',fmt='%.2e', 
                   header=header)
        return None
        
    def get_snow_params(self):
        """
        Calculate the Sum of Snow Depth (SSD) and Snow Cover Days (SCD)
        parameters according to Chen et al (2021,
        https://doi.org/10.1016/j.gloplacha.2020.103394) based on a given snow 
        cover.

        Returns
        -------
        SSD, float
            Sum of (daily) snow depths in cm.
        SCD, float
            Days with snow cover in a year.

        """
        
        self.SSD = sum(self.snow_ts*100*30)
        self.SCD = sum([30 for i in self.snow_ts[self.snow_ts>0.05]])
        
        return self.SSD, self.SCD
    
    def get_MAST(self, year=1):
        """
        calculate mean annual temperature at surface level.

        Parameters
        ----------
        year : int, optional
            modelled year to calculate. The default is 1.

        Returns
        -------
        MAST, float
            Mean annual surface temperature of the selected year.

        """
        if year > self.Nt:
            print('parsed year outside of modelled range')
            
        bounds = (year*12-12, year*12) #from months to month
        T = self.T_data[:, bounds[0]:bounds[1]]
        
        T_last_snow = T[self.Nz_s-1, :].mean().round(2)
        T_first_ground = T[self.Nz_s, :].mean().round(2)
        T_2interpol = np.concatenate((T_last_snow, T_first_ground), axis = None)
        x_interpol = self.Nz_s * self.dz[0]
        x_original = np.concatenate(((self.Nz_s-1)*self.dz[0]+self.dz[0],
                                     (self.Nz_s+1)*self.dz[0]-self.dz[0]), axis = None)
        self.MAST = np.interp(x_interpol, x_original, T_2interpol)

        return self.MAST
    
    def get_mat(self, level, year=1):
        """
        Calculate the average temperature for a given year at a given 
        depth interval.

        Parameters
        ----------
        level : int
            Depth interval to calculate.
        year : int, optional
            Time slice to calculate. The default is -2 (i.e. the last year of
            a given model run).

        Returns
        -------
        MAT : float
            Mean Annual Temperature.

        """
        if year > self.Nt:
            print('parsed year outside of modelled range')
            
        bounds = (year*12-12, year*12) #from months to month
        self.MAT = self.T_data[level, bounds[0]:bounds[1]].mean().round(2)
        
        return self.MAT
    
    def get_offset(self, year=1):
        """
        Calculate Temperature offset between topmost and lowermost depth
        interval of the Layer object (after diffusion).

        Parameters
        ----------
        year : int, optional
            Time slice to calculate. The default is -2 (i.e. the last year of
            a given model run).

        Returns
        -------
        offset, float
            Temperature offset T_top-T_bottom.

        """
        if year > self.Nt:
            print('parsed year outside of modelled range')
            
        bounds = (year*12-12, year*12) #from months to month
        T = self.T_data[:, bounds[0]:bounds[1]]
        
        T_top = T[:,0].mean() # top layer average
        T_bot = T[:,-1].mean() # bottom layer average
        self.offset = T_top-T_bot
        
        return self.offset
    
    def get_DZAA(self, year=1, threshold=0.1):
        """
        Find the depth interval below which the seasonal temperature amplitude
        does not exceed a certain threshold value.

        Parameters
        ----------
        year : int, optional
            year to be analysed. The default is 1.
        threshold : float, optional
            threshold value for the seasonal amplitude. The default is 0.1.

        Returns
        -------
        z : float
            Depth of temperature threshold.
        i : int
            Index of depth interval.

        """
        if year > self.Nt:
            print('parsed year outside of modelled range')
            
        bounds = (year*12-12, year*12) #from months to month
        T = self.T_data[:, bounds[0]:bounds[1]]
        
        for i in range(self.Nz_res):
            ts = T[i, :]
            amplitude = ts.max() - ts.min()
            if amplitude<threshold:
                # print(f'DZAA = {self.z[i]}')
                break
        return self.z[i], i
            
    def plot_seasonality(self, z=0, year=1, save=False, title=''):  
        """
        Generate a time series (seasonality) plot of the model outputs.

        Parameters
        ----------
        z : int, optional
            Index of depth interval to plot. The default is 0 (i.e. uppermost
            interval).
        year : integer, optional
            Index of modelled year to plot. The default is 1.
        save : boolean, optional
            If True, results are saved to a subfolder /plots in the project 
            root directory. The default is False.
        title : string, optional
            Title text of the plot. The default is ''.

        Returns
        -------
        None.

        """
        bounds = (year*12-12, year*12) #from months to month
        T = self.T_data[:, bounds[0]:bounds[1]]
        
        x = np.linspace(0,11,12)
        y1 = self.snow_ts[-12:]
        y2 = self.temp_ts[-12:]
        y3 = T[z,:]

        f, a1 = plt.subplots(figsize=(10,6))
        a1.set_title(title)
        
        a1.plot(x, y1, label='snow cover', c='k', ls='--')
        
        a2 = plt.twinx(a1)
        
        a2.fill_between(x, [-2 for i in x], [0 for i in x],
                        color='k', alpha=0.2, label='CCC formation')
        
        a2.plot(x, y2, label='atm. temp.', c='k')
        a2.plot(x, y3, label=f'T @ z={self.z[z].round(1)} m', c='k', ls=':')

        ticks = x[0::2]
        a1.set_xticks(ticks)
        a1.set_xticklabels(['JAN', 'MAR', 'MAY', 'JUL', 'SEP', 'NOV'])
        a1.set_ylabel('Snow Depth (m)')
        a2.set_ylabel('Temperature (°C)')
        a2.text(0, 0, f'MAGT = {y3.mean().round(1)} °C @ z={self.z[z].round(1)} m')
        
        
        f.legend()
        plt.tight_layout()
        
        if save == True:
            fname = 'plots/ts_' + title + '.png'
            plt.savefig(fname)
        
        return None
    
    def plot_ts(self, level=0):
        
        x = np.linspace(1, self.Nt, self.temp_ts.size)
        y = self.T_data[level, :]
        
        f, a = plt.subplots()
        a.plot(x, y)
        a.set_ylabel(f'Temp @ {self.z[level]:0.2f} m [°C]')
        a.set_xlabel('Runtime [years]')
        return None
              
    def plot_input(self, title='', save=False):
        """
        Plot the input time series of temperature and snow.

        Parameters
        ----------
        title : str, optional
            Plot title. The default is ''.
        save : boolean, optional
            Set TRUE if you want to save the figure. The default is False.

        Returns
        -------
        None.

        """
        
        f, (a1,a2) = plt.subplots(2, 1, sharex=True)
        a1.set_title(title)
        
        x = np.linspace(1, self.Nt, self.temp_ts.size)
        a1.plot(x, self.temp_ts, lw=1)
        a2.plot(x, self.snow_ts, lw=1)
        
        a1.set_ylabel('Temperature [°C]')
        a2.set_ylabel('Snow height [m]')
        a2.set_xlabel('runtime [years]')
        
        if save == True:
            fname = 'plots/input_' + title + '.png'
            plt.savefig(fname)
            
        return None
    
    def plot_Tdata(self, vmin=-20, vmax=20, isoline=[0], save=False, title=''):
        """
        Generate a contour plot showing the temperature profiles of
        the entire model run against time.

        Parameters
        ----------
        save : boolean, optional
            If True, results are saved to a subfolder /plots in the project 
            root directory. The default is False.
        vmin : float, optional
            Lower limit of the temperature scale. The default is -20.
        vmax : float, optional
            Upper limit of the temperature scale. The default is 20.
        title : str, optional
            Title text of the plot. The default is ''.

        Returns
        -------
        None.

        """
        T = self.T_data
        
        f, a = plt.subplots()
        
        a.set_title(title)
 
        im = a.imshow(T, cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax,
                      extent=[0, self.Nt, T.shape[0], 0])
        
        a.contour(T, levels=isoline, extent=[0, self.Nt, 0, T.shape[0]],
                  colors='k')
        # a.contour(T, levels=[0])
        # a.clabel(CS, CS.levels, inline=True)
        
        a.hlines(self.interface2, 0, self.Nt, ls='--', lw=1, 
                 color='k', label='ground surface')
        a.legend()
    
        a.set_ylabel('Depth interval')
        a.set_xlabel('Runtime [years]')

        cbar = plt.colorbar(im)
        cbar.set_label('Temperature [°C]')
        
        if save == True:
            fname = 'plots/T_data_' + title + '.png'
            plt.savefig(fname)
        
        return None
    
    def plot_profile(self, year=1, title=''):
        
        bounds = (year*12-12, year*12) #from months to month
        T = self.T_data[:, bounds[0]:bounds[1]]
        
        f, a = plt.subplots()
        a.set_title(title)
        a.set_ylabel('Depth [m]')
        a.set_xlabel('Temperature [°C]')
        a.vlines(0, 0, -self.depth_t, ls='--', color='k')
        
        for i in T.T:
            a.plot(i, -self.z, color='0.5')
        
        return None
        
    def input_params(self):        
        ipt = f''' depth_s = {self.depth_s} m \n depth_t = {self.depth_t} m \n Nz_s = {self.Nz_s} steps \n Nz_g = {self.Nz_g} steps \n dz_s, dz_g = {self.dz} m \n Tin = {self.Tin} °C \n Nt = {self.Nt} years'''
        return ipt
    
# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # initialize layer
    Nt = 100                # runtime (in years)
    depth_s = 1             # snow layer thickness (m)
    depth_t = 200           # total thickness of model domain (atmosphere to ground)
    Nzs_i = 3              # number of vertical steps for snow - no unit
    Nzg_i = 50              # number of vertical steps for ground - no unit
    Tins = -5                # surface initial temperature - deg C
    
    # load snow/temp time series data from file
    ts = pd.read_csv('snow_data.csv')
    snow = np.tile(ts.snow, Nt)
    temps = np.tile(ts.temp, Nt)
    
    L = Layer(depth_s, depth_t, Nzs_i, Nzg_i, Tins, snow, temps,
              Diff_air=18.46e-6, Diff_ground=1.8e-6, Diff_snow=2.92e-7,
              print_status=True)
    
    # diffusion in snow layer
    L.diffusion(Tins)
    
    # plot results
    print(L.input_params())
    L.plot_Tdata(title='', vmin=-20, vmax=20, isoline=[-2,0])
    dzaa = L.get_DZAA(year=Nt)
    print(f'DZAA = {dzaa[0].round(0)} m')
    L.plot_ts(level=dzaa[1])
    print(f'MAT @ {L.z[dzaa[1]].round(0)} m = {L.get_mat(dzaa[1], year=Nt).round(2)} °C')
    L.plot_input()
    L.plot_seasonality(z=dzaa[1], year=Nt)
    for i in np.linspace(1,Nt,2).astype('int'):
        L.plot_profile(year=i, title=f'year = {i}')
    L.export_csv()