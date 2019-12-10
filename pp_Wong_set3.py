#General Modules
import numpy as np
from scipy import integrate
import os
import unittest
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import floatpy.derivatives.explicit.first as first_der

import floatpy_helpers as fh

#Local Modules
import floatpy.readers.moving_grid_data_reader as mgdr

# Set up default plotting
params = {'legend.fontsize': 14,
          'legend.handlelength': 2,
          'font.size': 16}
matplotlib.rcParams.update(params)


########################################################################
###################### Functions ##############################
def plot_set1(tsteps, tvec, path):
    reader = mgdr.MovingGridDataReader(path)
    print("Domain Size: {} ".format(reader._domain_size))
    print("Variables available: {}".format(reader._var_names))
    Nx, Ny, Nz = reader._domain_size

    middle_selection = 0.12 #select middle 25% of domain in x
    middle_offset = 0.016
    x1 = int(Nx/2 - middle_selection/2*Nx + middle_offset*Nx)
    x2 = int(Nx/2 + middle_selection/2*Nx + middle_offset*Nx)
    reader.setSubDomain(((x1,0,0),(x2,Ny-1,Nz-1)))

    # Get coordinates based on subdomain
    x,y,z = reader.readCoordinates()
    x_center = np.mean(x[:,0,0])
    nx, ny, nz = reader._subdomain_hi-reader._subdomain_lo+1

    W        = np.zeros(len(tsteps))
    t_legend = []
    
    for tind, step in enumerate(tsteps):

        #Reading in
        reader.setStep(step)
        print("reading in Mass Fraction at step {}.".format(step))
        YHeavy       = np.squeeze(np.array(reader.readData('mass fraction 0')))
        print("reading in density at step {}.".format(step))
        density    = np.squeeze(np.array(reader.readData('density')))
        print("reading in pressure at step {}.".format(step))
        pressure   = np.squeeze(np.array(reader.readData('pressure')))
        
        #Calculating Stats
	print("Computing Diffusion Properties")
        rho_bar = np.mean(np.mean(density,axis=2),axis=1)

        Cp = YHeavy*Cp_Heavy + (1-YHeavy)*Cp_air
        Cv = YHeavy*Cv_Heavy + (1-YHeavy)*Cv_air
        gamma = Cp/Cv
	del Cp, Cv
 
        c = (gamma*pressure/density)**0.5

        del gamma

        M = (YHeavy/M_Heavy+(1-YHeavy)/M_air)**(-1)
        T = M*pressure/(density*Ru)

        #Find inner mixing zone (IMZ)
        IMZ_thresh = 0.9
        XHeavy = M/M_Heavy*YHeavy
        XHeavy_bar = np.mean(np.mean(XHeavy,axis=2),axis=1)
        IMZ_crit = 4*XHeavy_bar*(1-XHeavy_bar) - IMZ_thresh

        for ind in range(len(IMZ_crit)):
            if IMZ_crit[ind] >= 0:
                IMZ_lo = ind
                break

        for ind in range(len(IMZ_crit)):
            ind2 = nx-ind-1
            if IMZ_crit[ind2] >= 0:
                IMZ_hi = ind2
                break

        IMZ_mid = np.argmax(IMZ_crit)
        
        del XHeavy

        #Density Spectra
        t_legend.append("t = {} ms".format(step*1e-2))

        k_rad, rho_spec_rad = fh.radial_spectra(x[:,0,0],y[0,:,0],z[0,0,:],density[IMZ_lo:IMZ_hi+1,:,:])

        plt.figure(2)
        plt.loglog(k_rad, rho_spec_rad)
        plt.xlabel('Radial Wavenumber [m-1]')
        plt.ylabel("Density spectra")


        #Energy Spectra
        velocity   = np.squeeze(np.array(reader.readData('velocity')))
        u_tilde = np.mean(np.mean(velocity[0,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        u_doubleprime = velocity[0,:,:,:]-u_tilde.reshape((nx,1,1))
        energy_for_spectra = density*u_doubleprime/(np.reshape(rho_bar,(nx,1,1)))**0.5
        k_rad, rhoU_spec_rad = fh.radial_spectra(x[:,0,0],y[0,:,0],z[0,0,:],energy_for_spectra[IMZ_lo:IMZ_hi+1,:,:])

        plt.figure(3)
        plt.loglog(k_rad, rhoU_spec_rad)
        plt.xlabel('Radial Wavenumber [m-1]')
        plt.ylabel("Energy Spectra") 


    #Format plots
    print("Formatting Plots")
    t_legend.append("-3/2 slope")
    for ind in plt.get_fignums():
        plt.figure(ind)
        plt.loglog(k_rad, 1e2*k_rad**(-1.5),'k--')
        plt.legend(t_legend)
        plt.tight_layout()


    return None

########################################################################
########################################################################
########################################################################

if __name__ == '__main__':

    ########################################################################
    ########################################################################
    #### PROBLEM SPECIFIC STUFF ####
    from Wong_setup import *
    
    #Choose the time steps to use
    tsteps = (125,135,155,180) #all steps
    
    #path = "/work/05428/tg847275/Wong/uniform_data/3D_Poggi_RMI_RD/case_1_1/grid_B"
    path = "/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B"

    tvec = 1e-5*np.array(tsteps)

    plot_set1(tsteps,tvec,path)

    #### END PROBLEM SPECIFIC STUFF ####
    ########################################################################
    ########################################################################

    plt.show()
