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
import floatpy.readers.miranda_reader as mir

# Set up default plotting
params = {'legend.fontsize': 14,
          'legend.handlelength': 2,
          'font.size': 16}
matplotlib.rcParams.update(params)


########################################################################
###################### Functions ##############################
def plot_set2(tsteps, tvec, path):
    reader = mir.MirandaReader(path, periodic_dimensions=(False,True,True),verbose=True)

    print("Domain Size: {} ".format(reader._domain_size))
    print("Variables available: {}".format(reader.varNames))
    Nx, Ny, Nz = reader._domain_size

    W        = np.zeros(len(tsteps))
    
    t_legend = []

    for tind, step in enumerate(tsteps):

        ##################################
        #Reading in full fields to figure out where subdomain and interface are
        reader.setStep(step)
        reader.setSubDomain(((0,0,0),(Nx-1,Ny-1,Nz-1)))
        print("Figuring out subdomain at step {}".format(step))
        YN2  = np.squeeze(np.array(reader.readData('Ideal1_01')))
        YO2  = np.squeeze(np.array(reader.readData('Ideal1_02')))
        YSF6 = np.squeeze(np.array(reader.readData('Ideal1_03')))
        YAc  = np.squeeze(np.array(reader.readData('Ideal1_04')))
        YHeavy = YSF6+YAc

        M = (YN2/M_N2+YO2/M_O2+YSF6/M_SF6+YAc/M_Ac)**(-1)

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
            ind2 = Nx-ind-1
            if IMZ_crit[ind2] >= 0:
                IMZ_hi = ind2
                break

        IMZ_mid = np.argmax(IMZ_crit)

        middle_selection = 0.40 #select middle 20% of domain in x
        x1 = IMZ_mid-int(middle_selection/2*Nx)
        x2 = min(IMZ_mid+int(middle_selection/2*Nx),Nx-1)
        reader.setSubDomain(((x1,0,0),(x2,Ny-1,Nz-1)))

        # Get coordinates based on subdomain
        x,y,z = reader.readCoordinates()
        x = x/100.0 #convert from cm-> m
        y = y/100.0 
        z = z/100.0
        nx = reader.chunk[0][1] - reader.chunk[0][0]
        ny = reader.chunk[1][1] - reader.chunk[1][0]
        nz = reader.chunk[2][1] - reader.chunk[2][0]

        ##################################

        # Reading in on subdomain  
        reader.setStep(step)
        print("reading in Mass Fraction at step {}.".format(step))
        YN2  = np.squeeze(np.array(reader.readData('Ideal1_01')))
        YO2  = np.squeeze(np.array(reader.readData('Ideal1_02')))
        YSF6 = np.squeeze(np.array(reader.readData('Ideal1_03')))
        YAc  = np.squeeze(np.array(reader.readData('Ideal1_04')))
        YHeavy = YSF6+YAc
        
        print("reading in density at step {}.".format(step))
        density    = 1e3*np.squeeze(np.array(reader.readData('density')))
        print("reading in pressure at step {}.".format(step))
        pressure   = 1e-1*np.squeeze(np.array(reader.readData('pressure')))
        
        #Calculating Stats
	print("Computing Diffusion Properties")
        rho_bar = np.mean(np.mean(density,axis=2),axis=1)

        Cp = YN2*Cp_N2 +YO2*Cp_O2 +YSF6*Cp_SF6 +YAc*Cp_Ac
        Cv = YN2*Cv_N2 +YO2*Cv_O2 +YSF6*Cv_SF6 +YAc*Cv_Ac

        gamma = Cp/Cv
        M = (YN2/M_N2+YO2/M_O2+YSF6/M_SF6+YAc/M_Ac)**(-1)
        T = M*pressure/(density*Ru)
        c = (gamma*pressure/density)**0.5
       
        del YAc, gamma

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
        
        #Mole Fraction Variance
        XHeavy_prime = XHeavy-XHeavy_bar.reshape((nx,1,1))
        XHeavy_variance = np.mean(np.mean(XHeavy_prime**2,axis=2),axis=1)

        #Compute Turbulent Mach Number
        print("reading in velocity at step {}.".format(step))
        velocity = np.zeros((3,nx,ny,nz))
        velocity[0,:,:,:]   = 1e-2*np.squeeze(np.array(reader.readData('velocity-0')))
        velocity[1,:,:,:]   = 1e-2*np.squeeze(np.array(reader.readData('velocity-1')))
        velocity[2,:,:,:]   = 1e-2*np.squeeze(np.array(reader.readData('velocity-2')))
        u_tilde = np.mean(np.mean(velocity[0,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)

        u_doubleprime = velocity[0,:,:,:]-u_tilde.reshape((nx,1,1))

        #Density Spectra
        t_legend.append("t = {} ms".format(step*1e-2))

        k_rad, rho_spec_rad = fh.radial_spectra(x[:,0,0],y[0,:,0],z[0,0,:],density[IMZ_lo:IMZ_hi+1,:,:])

        plt.figure(2)
        plt.loglog(k_rad, rho_spec_rad)
        plt.xlabel('Radial Wavenumber [m-1]')
        plt.ylabel("Density spectra")


        #Energy Spectra
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
    from Tritschler_setup import *
    
    #Choose the time steps to use
    tsteps = (22,27,50)
    tvec   = 1e-4*np.array(tsteps)
    path = "/scratch/05428/tg847275/Tritschler/RM_CTR_3D_256/plot.mir"
    path = "/home/jrwest/Research/FloATPy_moving_grid/data/Tritschler/RM_CTR_3D_64/plot.mir"

    plot_set2(tsteps,tvec,path)


    #### END PROBLEM SPECIFIC STUFF ####
    ########################################################################
    ########################################################################

    plt.show()
