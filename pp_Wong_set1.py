#General Modules
import numpy as np
from scipy import integrate
import os
import unittest
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
matplotlib.rcParams['text.usetex'] = True

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

    Ma_t_IMZ = np.zeros(len(tsteps))
    At_e_IMZ = np.zeros(len(tsteps))
    At_e_centerplane = np.zeros(len(tsteps))
    W        = np.zeros(len(tsteps))
    Theta    = np.zeros(len(tsteps))
    TKE_int  = np.zeros(len(tsteps))
    Diss_int = np.zeros(len(tsteps))
    Enstrophy_int = np.zeros(len(tsteps))
    b11_IMZ  = np.zeros(len(tsteps))

    for tind, step in enumerate(tsteps):

        #Reading in
        reader.setStep(step)
        print("reading in fields at step {}.".format(step))
        YHeavy       = np.squeeze(np.array(reader.readData('mass fraction 0')))
        density    = np.squeeze(np.array(reader.readData('density')))
        pressure   = np.squeeze(np.array(reader.readData('pressure')))
        
        #Calculating Stats
        rho_bar = np.mean(np.mean(density,axis=2),axis=1)

        Cp = YHeavy*Cp_Heavy + (1-YHeavy)*Cp_air
        Cv = YHeavy*Cv_Heavy + (1-YHeavy)*Cv_air
        gamma = Cp/Cv 
        c = (gamma*pressure/density)**0.5

        del Cp, Cv, gamma

        M = (YHeavy/M_Heavy+(1-YHeavy)/M_air)**(-1)
        T = M*pressure/(density*Ru)
        Tij = T/eps_ij
        Omega_D = A*Tij**B + C*np.exp(D*Tij) + E*np.exp(F*Tij) + G*np.exp(H*Tij)
        D_binary = (0.0266/Omega_D)*(T**1.5)/(sigma_ij**2*pressure*M_ij**0.5)
        
        del Tij, Omega_D

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
        integrand = 4*XHeavy_bar*(1-XHeavy_bar)
        W[tind] = integrate.simps(integrand,x[:,0,0])
        
        # Compute Mixedness
        integrand = XHeavy*(1-XHeavy)
        integrand = np.mean(np.mean(integrand,axis=2),axis=1)
        numer = integrate.simps(integrand,x[:,0,0])

        denom = W[tind]/4.0
        Theta[tind] = numer/denom
        
        xstar= (x[:,0,0]-x[IMZ_mid,0,0])/W[tind]

        del XHeavy

        #Compute Turbulent Mach Number
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
        b11_IMZ[tind] = np.mean(b11[IMZ_lo:IMZ_hi+1])

        del u_doubleprime,v_doubleprime,w_doubleprime
    
        Ma_t = (((np.mean(np.mean(2*TKE/density,axis=2),axis=1))**0.5)/
             np.mean(np.mean(c,axis=2),axis=1))

        Ma_t_IMZ[tind] = np.mean(Ma_t[IMZ_lo:IMZ_hi+1])

        #Compute Effective Atwood Number
        rho_prime = density-rho_bar.reshape((nx,1,1))

        At_e = ((np.mean(np.mean(rho_prime**2,axis=2),axis=1))**0.5)/rho_bar
        At_e_IMZ[tind] = np.mean(At_e[IMZ_lo:IMZ_hi+1])
        At_e_centerplane[tind] = At_e[IMZ_mid]
        
        del rho_prime, M, T, pressure

        # Compute Integrated TKE
        TKE_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            TKE,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        del TKE
        
        # Compute Integrated Scalar Dissipation Rate
        dx = x[1,0,0]-x[0,0,0]
        dy = y[0,1,0]-y[0,0,0]
        dz = z[0,0,1]-z[0,0,0]
        dYdx = first_der.differentiateSixthOrderFiniteDifference(YHeavy, dx, 0, None, True, 3, 'C')
        dYdy = first_der.differentiateSixthOrderFiniteDifference(YHeavy, dy, 1, None, True, 3, 'C')
        dYdz = first_der.differentiateSixthOrderFiniteDifference(YHeavy, dz, 2, None, True, 3, 'C')

        Diss = D_binary*(dYdx**2+dYdy**2+dYdz**2)
        Diss_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            Diss,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        del dYdx, dYdy, dYdz, Diss

        # Compute Integrated Enstrophy
        dwdy = first_der.differentiateSixthOrderFiniteDifference(velocity[2,:,:,:], dy, 1, None, True, 3, 'C')
        dvdx = first_der.differentiateSixthOrderFiniteDifference(velocity[1,:,:,:], dx, 0, None, True, 3, 'C')
        omega1 = dwdy-dvdx
        del dwdy, dvdx

        dudz = first_der.differentiateSixthOrderFiniteDifference(velocity[0,:,:,:], dz, 2, None, True, 3, 'C')
        dwdx = first_der.differentiateSixthOrderFiniteDifference(velocity[2,:,:,:], dx, 0, None, True, 3, 'C')
        omega2 = dudz-dwdx
        del dudz, dwdx

        dvdx = first_der.differentiateSixthOrderFiniteDifference(velocity[1,:,:,:], dx, 0, None, True, 3, 'C')
        dudy = first_der.differentiateSixthOrderFiniteDifference(velocity[0,:,:,:], dy, 1, None, True, 3, 'C')
        omega3 = dvdx-dudy
        del dvdx, dudy, velocity

        Enstrophy = density*(omega1**2+omega2**2+omega3**2)
        Enstrophy_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            Enstrophy,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        del omega1, omega2, omega3, Enstrophy
        del density
        del YHeavy

    #Plots
    plt.figure(1)
    plt.plot(tvec*1000,W*1000,'-o')
    plt.ylabel('W [mm]')
    plt.xlabel('time [ms]')

    plt.figure(2)
    plt.plot(tvec*1000,Theta,'-o')
    plt.ylabel(r'\Theta')
    plt.xlabel('time [ms]')

    plt.figure(3)
    plt.semilogy(tvec*1000,TKE_int,'-o')
    plt.ylabel(r'\int TKE' + '~dydz~[kg~m~s^{-2}]')
    plt.xlabel('time [ms]')

    plt.figure(4)
    plt.semilogy(tvec*1000,Diss_int,'-o')
    plt.ylabel(r'\int \chi' + '~dydz~[m^2~s^{-1}]')
    plt.xlabel('time [ms]')

    plt.figure(5)
    plt.semilogy(tvec*1000,Enstrophy_int,'-o')
    plt.ylabel(r'\int \Omega' +'~dydz~[kg ~m^{-1} ~s^{-2}]')
    plt.xlabel('time [ms]')

    plt.figure(6)
    plt.plot(tvec*1000,Ma_t_IMZ,'-o')
    plt.ylabel(r'\langle Ma_t \rangle')
    plt.xlabel('time [ms]')

    plt.figure(7)
    plt.plot(tvec*1000,At_e_IMZ,'-o')
    plt.ylabel(r'\langle At_e \rangle')
    plt.xlabel('time [ms]')

    plt.figure(8)
    plt.plot(tvec*1000,b11_IMZ,'-o')
    plt.ylabel(r'\langle b11 \rangle')
    plt.xlabel('time [ms]')

    #Format plots
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
    tsteps_B = (0,10,45,80,115,125,135,145,155,165,180) #all steps
    tsteps_C = (0,10,45,80,115,125)
    tsteps_D = (0,10,45,80)
    
    path_to_B = "/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B"
    path_to_C = "/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B"
    path_to_D = "/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B"

    paths_vec = (path_to_B,path_to_C,path_to_D)
    tsteps_vec = (tsteps_B,tsteps_C,tsteps_D)
    tvec_vec   = (1e-5*np.array(tsteps_vec[0]),1e-5*np.array(tsteps_vec[1]),0.5e-5*np.array(tsteps_vec[2]))

    for i,path in enumerate(paths_vec):
        plot_set1(tsteps_vec[i],tvec_vec[i],path)

    for ind in plt.get_fignums():
        plt.figure(ind)
        plt.legend(['gridB','grid C','grid D'])

    #### END PROBLEM SPECIFIC STUFF ####
    ########################################################################
    ########################################################################

    plt.show()
