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
def plot_set1(tsteps, tvec, path):
    reader = mir.MirandaReader(path, periodic_dimensions=(False,True,True),verbose=True)

    print("Domain Size: {} ".format(reader._domain_size))
    print("Variables available: {}".format(reader.varNames))
    Nx, Ny, Nz = reader._domain_size

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

	del M

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

        middle_selection = 0.35 #select middle 20% of domain in x
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
        
        D_SF6 = np.squeeze(np.array(reader.readData('D_Ideal1_03')))

        print("reading in density at step {}.".format(step))
        density    = np.squeeze(np.array(reader.readData('density')))
        print("reading in pressure at step {}.".format(step))
        pressure   = np.squeeze(np.array(reader.readData('pressure')))
        
        #Calculating Stats
	print("Computing Diffusion Properties")
        rho_bar = np.mean(np.mean(density,axis=2),axis=1)

        Cp = YN2*Cp_N2 +YO2*Cp_O2 +YSF6*Cp_SF6 +YAc*Cp_Ac
        Cv = YN2*Cv_N2 +YO2*Cv_O2 +YSF6*Cv_SF6 +YAc*Cv_Ac

        gamma = Cp/Cv
        M = (YN2/M_N2+YO2/M_O2+YSF6/M_SF6+YAc/M_Ac)**(-1)
        T = M*pressure/(density*Ru)
        c = (gamma*pressure/density)**0.5
       
        del T, YAc, gamma

        # Compute Integrated Scalar Dissipation Rate
	print("Computing Dissipation")
        dx = x[1,0,0]-x[0,0,0]
        dy = y[0,1,0]-y[0,0,0]
        dz = z[0,0,1]-z[0,0,0]
        dYdx = first_der.differentiateSixthOrderFiniteDifference(YSF6, dx, 0, None, True, 3, 'C')
        dYdy = first_der.differentiateSixthOrderFiniteDifference(YSF6, dy, 1, None, True, 3, 'C')
        dYdz = first_der.differentiateSixthOrderFiniteDifference(YSF6, dz, 2, None, True, 3, 'C')

        Diss = D_SF6*(dYdx**2+dYdy**2+dYdz**2)
        Diss_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            Diss,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        del dYdx, dYdy, dYdz, Diss, D_SF6

        #Find inner mixing zone (IMZ)
        IMZ_thresh = 0.9
        XHeavy = M/M_Heavy*YHeavy
        XHeavy_bar = np.mean(np.mean(XHeavy,axis=2),axis=1)
        IMZ_crit = 4*XHeavy_bar*(1-XHeavy_bar) - IMZ_thresh

	del M

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
        
        # Compute Mixedness
        integrand = XHeavy*(1-XHeavy)
        integrand = np.mean(np.mean(integrand,axis=2),axis=1)
        numer = integrate.simps(integrand,x[:,0,0])

        denom = W[tind]/4.0
        Theta[tind] = numer/denom
        
        xstar= (x[:,0,0]-x[IMZ_mid,0,0])/W[tind]

        del XHeavy

        #Compute Turbulent Mach Number
        print("reading in velocity at step {}.".format(step))
	print("Computing Turbulent Mach Number")
        velocity = np.zeros((3,nx,ny,nz))
        velocity[0,:,:,:]   = np.squeeze(np.array(reader.readData('velocity-0')))
        velocity[1,:,:,:]   = np.squeeze(np.array(reader.readData('velocity-1')))
        velocity[2,:,:,:]   = np.squeeze(np.array(reader.readData('velocity-2')))
        u_tilde = np.mean(np.mean(velocity[0,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        v_tilde = np.mean(np.mean(velocity[1,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        w_tilde = np.mean(np.mean(velocity[2,:,:,:]*density,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)

        u_doubleprime = velocity[0,:,:,:]-u_tilde.reshape((nx,1,1))
        v_doubleprime = velocity[1,:,:,:]-v_tilde.reshape((nx,1,1))
        w_doubleprime = velocity[2,:,:,:]-w_tilde.reshape((nx,1,1))

        del u_tilde, v_tilde, w_tilde

        TKE = 0.5*density*(u_doubleprime**2+v_doubleprime**2+w_doubleprime**2)

        # Compute Integrated TKE
        TKE_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            TKE,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        #Anisotropy (23 and 24)
        R11 = np.mean(np.mean(density*u_doubleprime**2,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        Rkk = np.mean(np.mean(2*TKE,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        b11 = R11/Rkk - 1.0/3.0
        b11_IMZ[tind] = np.mean(b11[IMZ_lo:IMZ_hi+1])

        Ma_t = (((np.mean(np.mean(2*TKE/density,axis=2),axis=1))**0.5)/
             np.mean(np.mean(c,axis=2),axis=1))

        Ma_t_IMZ[tind] = np.mean(Ma_t[IMZ_lo:IMZ_hi+1])

        del u_doubleprime,v_doubleprime,w_doubleprime, TKE
    
        #Compute Effective Atwood Number
        rho_prime = density-rho_bar.reshape((nx,1,1))

        At_e = ((np.mean(np.mean(rho_prime**2,axis=2),axis=1))**0.5)/rho_bar
        At_e_IMZ[tind] = np.mean(At_e[IMZ_lo:IMZ_hi+1])
        At_e_centerplane[tind] = At_e[IMZ_mid]
        
        del rho_prime

        # Compute Integrated Enstrophy
	print("Computing Enstrophy")
	Enstrophy = 0
        dwdy = first_der.differentiateSixthOrderFiniteDifference(velocity[2,:,:,:], dy, 1, None, True, 3, 'C')
        dvdx = first_der.differentiateSixthOrderFiniteDifference(velocity[1,:,:,:], dx, 0, None, True, 3, 'C')
        Enstrophy += density*(dwdy-dvdx)**2
        del dwdy, dvdx

        dudz = first_der.differentiateSixthOrderFiniteDifference(velocity[0,:,:,:], dz, 2, None, True, 3, 'C')
        dwdx = first_der.differentiateSixthOrderFiniteDifference(velocity[2,:,:,:], dx, 0, None, True, 3, 'C')
        Enstrophy += density*(dudz-dwdx)**2
        del dudz, dwdx

        dvdx = first_der.differentiateSixthOrderFiniteDifference(velocity[1,:,:,:], dx, 0, None, True, 3, 'C')
        dudy = first_der.differentiateSixthOrderFiniteDifference(velocity[0,:,:,:], dy, 1, None, True, 3, 'C')
        Enstrophy += density*(dvdx-dudy)**2
        
	del dvdx, dudy, velocity

        Enstrophy_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            Enstrophy,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        del Enstrophy, density, YHeavy

    print("Plotting")
    #Plots
    plt.figure(1)
    plt.plot(tvec*1000,W*1000,'-o')
    plt.ylabel('W [mm]')
    plt.xlabel('time [ms]')

    plt.figure(2)
    plt.plot(tvec*1000,Theta,'-o')
    plt.ylabel('Mixedness [-]')
    plt.xlabel('time [ms]')

    plt.figure(3)
    plt.semilogy(tvec*1000,TKE_int,'-o')
    plt.ylabel('Int. TKE [kg m s-2]')
    plt.xlabel('time [ms]')

    plt.figure(4)
    plt.semilogy(tvec*1000,Diss_int,'-o')
    plt.ylabel('Int. Dissipation Rate [m2 s-1]')
    plt.xlabel('time [ms]')

    plt.figure(5)
    plt.semilogy(tvec*1000,Enstrophy_int,'-o')
    plt.ylabel(r'Int. Enstrophy [kg m-1 s-2]')
    plt.xlabel('time [ms]')

    plt.figure(6)
    plt.plot(tvec*1000,Ma_t_IMZ,'-o')
    plt.ylabel('Turbulent Ma')
    plt.xlabel('time [ms]')

    plt.figure(7)
    plt.plot(tvec*1000,At_e_IMZ,'-o')
    plt.ylabel('Effective At')
    plt.xlabel('time [ms]')

    plt.figure(8)
    plt.plot(tvec*1000,b11_IMZ,'-o')
    plt.ylabel('b11')
    plt.xlabel('time [ms]')

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
    tsteps = (15,20,22,27,50)
    path_to_64  = "/home/jrwest/Research/FloATPy_moving_grid/data/Tritschler/RM_CTR_3D_64/plot.mir"
    path_to_128 = "/home/jrwest/Research/FloATPy_moving_grid/data/Tritschler/RM_CTR_3D_64/plot.mir"
    path_to_256 = "/home/jrwest/Research/FloATPy_moving_grid/data/Tritschler/RM_CTR_3D_64/plot.mir"

    paths_vec = (path_to_64,path_to_128,path_to_256)
    tvec   = 1e-4*np.array(tsteps)

    for i,path in enumerate(paths_vec):
        plot_set1(tsteps,tvec,path)

    #Format plots
    print("Formatting Plots")
    for ind in plt.get_fignums():
        plt.figure(ind)
        plt.legend(['gridB','grid C','grid D'])
        plt.tight_layout()


    #### END PROBLEM SPECIFIC STUFF ####
    ########################################################################
    ########################################################################

    plt.show()
