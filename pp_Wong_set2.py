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
    t_legend1 = []
    t_legend2 = []
    
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
        
        # Compute Mixing Width 
	print("Computing Mixing Width and Mixedness")
        integrand = 4*XHeavy_bar*(1-XHeavy_bar)
        W[tind] = integrate.simps(integrand,x[:,0,0])
        xstar= (x[:,0,0]-x[IMZ_mid,0,0])/W[tind]

        #Mole Fraction Variance
        XHeavy_prime = XHeavy-XHeavy_bar.reshape((nx,1,1))
        XHeavy_variance = np.mean(np.mean(XHeavy_prime**2,axis=2),axis=1)

        if step < 125:
            #before reshock
            plt.figure(1)
            t_legend1.append("t = {} ms".format(step*1e-2))
        else:
            #after reshock
            plt.figure(2)
            t_legend2.append("t = {} ms".format(step*1e-2))

        plt.plot(xstar,XHeavy_variance**0.5)
        plt.xlim([-2,2])
        plt.xlabel('x*')
        plt.ylabel("X'rms (SF6)")
        plt.ylim([0,0.3])

        del XHeavy

        #Compute Turbulent Mach Number
        print("reading in velocity at step {}.".format(step))
	print("Computing Turbulent Mach Number")
        velocity   = np.squeeze(np.array(reader.readData('velocity')))
        u_tilde = np.mean(np.mean(velocity[0,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        v_tilde = np.mean(np.mean(velocity[1,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        w_tilde = np.mean(np.mean(velocity[2,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)

        u_doubleprime = velocity[0,:,:,:]-u_tilde.reshape((nx,1,1))
        v_doubleprime = velocity[1,:,:,:]-v_tilde.reshape((nx,1,1))
        w_doubleprime = velocity[2,:,:,:]-w_tilde.reshape((nx,1,1))

        del u_tilde, v_tilde, w_tilde

        TKE = 0.5*density*(u_doubleprime**2+v_doubleprime**2+w_doubleprime**2)

        #Anisotropy (23 and 24)
        R11 = np.mean(np.mean(density*u_doubleprime**2,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        Rkk = np.mean(np.mean(2*TKE,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        b11 = R11/Rkk - 1.0/3.0

        Ma_t = (((np.mean(np.mean(2*TKE/density,axis=2),axis=1))**0.5)/
             np.mean(np.mean(c,axis=2),axis=1))

        del u_doubleprime,v_doubleprime,w_doubleprime, TKE

        if step < 125:
            #before reshock
            plt.figure(3)
        else:
            #after reshock
            plt.figure(4)

        plt.plot(xstar,b11)
        plt.xlabel('x*')
        plt.ylabel('b11')
        plt.xlim([-1.5,1.5])
        plt.ylim([-2.0/3.0,2.0/3.0])
    
        #Compute Effective Atwood Number
        rho_prime = density-rho_bar.reshape((nx,1,1))

        At_e = ((np.mean(np.mean(rho_prime**2,axis=2),axis=1))**0.5)/rho_bar

        # Compute Covariances of Density
        M_bar = np.mean(np.mean(M,axis=2),axis=1)
        M_prime = M-M_bar.reshape((nx,1,1))
        cov_rho_M = np.mean(np.mean(rho_prime*M_prime,axis=2),axis=1)/(rho_bar*M_bar)

        p_bar = np.mean(np.mean(pressure,axis=2),axis=1)
        p_prime = pressure-p_bar.reshape((nx,1,1))
        cov_rho_p = np.mean(np.mean(rho_prime*p_prime,axis=2),axis=1)/(rho_bar*p_bar)

        Tinv_bar = np.mean(np.mean(1.0/T,axis=2),axis=1)
        Tinv_prime = 1.0/T-Tinv_bar.reshape((nx,1,1))
        cov_rho_Tinv = np.mean(np.mean(rho_prime*Tinv_prime,axis=2),axis=1)/(rho_bar*Tinv_bar)

        del rho_prime, M, M_prime, T, Tinv_prime, p_prime

        if (step == 80) or (step == 135):
            plt.figure(step)
            plt.plot(xstar,cov_rho_p)
            plt.plot(xstar,cov_rho_M)
            plt.plot(xstar,cov_rho_Tinv)
            plt.xlabel('x*')
            plt.xlim([-2.0,2.0])
            plt.ylim([0,0.25])
            plt.legend(['rho-p','rho-M','rho-1/T'])
            plt.ylabel('Density Covariance')
            plt.title("t = {} ms".format(step*1e-2))

        #Density Reconstruction
        rho_Heavy = np.mean(density[0,:,:])
        rho_air = np.mean(density[-1,:,:])
        rho_recon_bar = (rho_Heavy-rho_air)*XHeavy_bar+rho_air
        
        if step < 125:
            #before reshock
            plt.figure(5)
        else:
            #after reshock
            plt.figure(6)

        plt.plot(xstar,rho_recon_bar/rho_bar)
        plt.xlabel('x*')
        plt.ylabel('rho / incompressible rho')
        
        del density, YHeavy

    #Format plots
    print("Formatting Plots")
    for ind in range(1,7):
        plt.figure(ind)
        if ind%2 == 1:
            plt.legend(t_legend1)
        else:
            plt.legend(t_legend2)

    for ind in plt.get_fignums():
        plt.figure(ind)
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
    tsteps = (10,45,80,115,125,135,155,180) #all steps
    
    #path = "/work/05428/tg847275/Wong/uniform_data/3D_Poggi_RMI_RD/case_1_1/grid_B"
    path = "/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B"

    tvec = 1e-5*np.array(tsteps)

    plot_set1(tsteps,tvec,path)

    #### END PROBLEM SPECIFIC STUFF ####
    ########################################################################
    ########################################################################

    plt.show()
