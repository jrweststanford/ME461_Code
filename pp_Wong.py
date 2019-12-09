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

if __name__ == '__main__':

    #problem params (see table V in Wong and Lele 2019)
    Cp_air       = 1040.50 #[J kg-1 K-1] 
    Cv_air       = 743.697 #[J kg-1 K-1]
    M_air        = 28.0135 #[kg kmol-1]
    eps_on_k_air = 78.6 #[K]
    sigma_air    = 3.711

    Cp_SF6       = 668.286 #[J kg-1 K-1]
    Cv_SF6       = 611.359 #[J kg-1 K-1]
    M_SF6        = 146.055 #[kg kmol-1]
    eps_on_k_SF6 = 221 #[K]
    sigma_SF6    = 5.128

    Ly = 0.025 #[m]
    Lz = 0.025 #[m]

    Ru = 8314 #[J kmol-1 K-1]

    #calculate some constants
    eps_ij = (eps_on_k_air*eps_on_k_SF6)**0.5
    sigma_ij = 0.5*(sigma_air+sigma_SF6)
    M_ij = 2.0/(1/M_air+1/M_SF6)
    A = 1.06036 
    B = -0.1561
    C = 0.19300
    D = -0.47635
    E = 1.03587
    F = -1.52996
    G = 1.76474
    H = -3.89411

    #Read in a file and plot something
    reader = mgdr.MovingGridDataReader("/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B")

    print("Domain Size: {} ".format(reader._domain_size))
    Nx, Ny, Nz = reader._domain_size

    print("Variables available: {}".format(reader._var_names))

    middle_selection = 0.12 #select middle 25% of domain in x
    middle_offset = 0.016
    x1 = int(Nx/2 - middle_selection/2*Nx + middle_offset*Nx)
    x2 = int(Nx/2 + middle_selection/2*Nx + middle_offset*Nx)
    reader.setSubDomain(((x1,0,0),(x2,Ny-1,Nz-1)))

    # Get coordinates based on subdomain
    x,y,z = reader.readCoordinates()
    x_center = np.mean(x[:,0,0])
    nx, ny, nz = reader._subdomain_hi-reader._subdomain_lo+1

    #tsteps = (0,10,45,80,115,125,135,145,155,165,180) #all steps
    #tsteps = (10,125,180)
    #tsteps = (125,145,165,180) #for spectra
    tsteps = (10,45,80,115,125,145,165,180) #use for density reconstruction and anisotropy
    #tsteps = (10,125) #for covariances

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
        reader.setStep(step)
        print("reading in fields at step {}.".format(step))
        YSF6       = np.squeeze(np.array(reader.readData('mass fraction 0')))
        density    = np.squeeze(np.array(reader.readData('density')))
        pressure   = np.squeeze(np.array(reader.readData('pressure')))
        
        rho_bar = np.mean(np.mean(density,axis=2),axis=1)

        Cp = YSF6*Cp_SF6 + (1-YSF6)*Cp_air
        Cv = YSF6*Cv_SF6 + (1-YSF6)*Cv_air
        gamma = Cp/Cv 
        c = (gamma*pressure/density)**0.5

        del Cp, Cv, gamma

        M = (YSF6/M_SF6+(1-YSF6)/M_air)**(-1)
        T = M*pressure/(density*Ru)
        Tij = T/eps_ij
        Omega_D = A*Tij**B + C*np.exp(D*Tij) + E*np.exp(F*Tij) + G*np.exp(H*Tij)
        D_binary = (0.0266/Omega_D)*(T**1.5)/(sigma_ij**2*pressure*M_ij**0.5)
        
        del Tij, Omega_D

        #Find inner mixing zone (IMZ)
        IMZ_thresh = 0.9
        XSF6 = M/M_SF6*YSF6
        XSF6_bar = np.mean(np.mean(XSF6,axis=2),axis=1)
        IMZ_crit = 4*XSF6_bar*(1-XSF6_bar) - IMZ_thresh

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
        print(IMZ_lo,IMZ_mid,IMZ_hi)
        
        # Compute Mixing Width 
        integrand = 4*XSF6_bar*(1-XSF6_bar)
        W[tind] = integrate.simps(integrand,x[:,0,0])
        
        # Compute Mixedness
        integrand = XSF6*(1-XSF6)
        integrand = np.mean(np.mean(integrand,axis=2),axis=1)
        numer = integrate.simps(integrand,x[:,0,0])

        denom = W[tind]/4.0
        Theta[tind] = numer/denom
        
        xstar= (x[:,0,0]-x[IMZ_mid,0,0])/W[tind]

        ##Mole Fraction Variance
        #XSF6_prime = XSF6-XSF6_bar.reshape((nx,1,1))
        #XSF6_variance = np.mean(np.mean(XSF6_prime**2,axis=2),axis=1)

        #if step < 125:
        #    #before reshock
        #    plt.figure(1)
        #else:
        #    #after reshock
        #    plt.figure(2)

        #plt.plot(x[:,0,0],XSF6_variance)

        #TODO: Prediction of mole fraction variance with density

        ##XSF6 Spectra
        #k_rad, XSF6_spec_rad = fh.radial_spectra(x[:,0,0],y[0,:,0],z[0,0,:],XSF6[IMZ_lo:IMZ_hi+1,:,:])

        #plt.figure(1)
        #plt.title("XSF6 spectra".format(step)) 
        #plt.loglog(k_rad, XSF6_spec_rad)
        #if step==125:
        #    plt.loglog(k_rad, 1e2*k_rad**(-1.5),'k--')

        del XSF6

        ##Density Spectra
        #k_rad, rho_spec_rad = fh.radial_spectra(x[:,0,0],y[0,:,0],z[0,0,:],density[IMZ_lo:IMZ_hi+1,:,:])

        #plt.figure(2)
        #plt.title("Density spectra")
        #plt.loglog(k_rad, rho_spec_rad)

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

        #TODO: Anisotropy (23 and 24)
        R11 = np.mean(np.mean(density*u_doubleprime**2,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        Rkk = np.mean(np.mean(2*TKE,axis=2),axis=1)/np.mean(np.mean(density,axis=2),axis=1)
        b11 = R11/Rkk - 1.0/3.0
        b11_IMZ[tind] = np.mean(b11[IMZ_lo:IMZ_hi+1])

        if step < 125:
            #before reshock
            plt.figure(1)
        else:
            #after reshock
            plt.figure(2)

        plt.plot(xstar,b11)
        plt.xlabel('x*')
        plt.xlim([-1.5,2.0])
        plt.ylim([-2.0/3.0,2.0/3.0])

        ##Energy Spectra
        #energy_for_spectra = density*u_doubleprime/(np.reshape(rho_bar,(nx,1,1)))**0.5
        #k_rad, rhoU_spec_rad = fh.radial_spectra(x[:,0,0],y[0,:,0],z[0,0,:],energy_for_spectra[IMZ_lo:IMZ_hi+1,:,:])

        #plt.figure(3)
        #plt.title("KE Spectra") 
        #plt.loglog(k_rad, rhoU_spec_rad)
        #if step==125:
        #    plt.loglog(k_rad, 1e2*k_rad**(-1.5),'k--')

        del u_doubleprime,v_doubleprime,w_doubleprime
    
        Ma_t = (((np.mean(np.mean(2*TKE/density,axis=2),axis=1))**0.5)/
             np.mean(np.mean(c,axis=2),axis=1))

        Ma_t_IMZ[tind] = np.mean(Ma_t[IMZ_lo:IMZ_hi+1])

        #Compute Effective Atwood Number
        rho_prime = density-rho_bar.reshape((nx,1,1))

        At_e = ((np.mean(np.mean(rho_prime**2,axis=2),axis=1))**0.5)/rho_bar
        At_e_IMZ[tind] = np.mean(At_e[IMZ_lo:IMZ_hi+1])
        At_e_centerplane[tind] = At_e[IMZ_mid]

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

        del rho_prime, M, M_prime, T, pressure, Tinv_prime, p_prime

        #plt.figure()
        #plt.plot(xstar,cov_rho_p)
        #plt.plot(xstar,cov_rho_M)
        #plt.plot(xstar,cov_rho_Tinv)
        #plt.xlabel('x*')
        #plt.xlim([-1.5,2.0])

        # Compute Integrated TKE
        TKE_int[tind] = integrate.simps(integrate.simps(integrate.simps(
            TKE,z[0,0,:],axis=2),y[0,:,0],axis=1),x[:,0,0],axis=0)

        del TKE
        
        # Compute Integrated Scalar Dissipation Rate
        dx = x[1,0,0]-x[0,0,0]
        dy = y[0,1,0]-y[0,0,0]
        dz = z[0,0,1]-z[0,0,0]
        dYdx = first_der.differentiateSixthOrderFiniteDifference(YSF6, dx, 0, None, True, 3, 'C')
        dYdy = first_der.differentiateSixthOrderFiniteDifference(YSF6, dy, 1, None, True, 3, 'C')
        dYdz = first_der.differentiateSixthOrderFiniteDifference(YSF6, dz, 2, None, True, 3, 'C')

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

        ##Density Reconstruction
        #rho_SF6 = np.mean(density[0,:,:])
        #rho_air = np.mean(density[-1,:,:])

        #if step < 125:
        #    #before reshock
        #    plt.figure(1)
        #else:
        #    #after reshock
        #    plt.figure(2)

        #rho_recon_bar = (rho_SF6-rho_air)*XSF6_bar+rho_air
        #plt.plot(x[:,0,0],rho_recon_bar/rho_bar)

        del density

        ##Plots
        #plt.figure()
        #plt.title("Mass Fraction: t = {}E-5 sec".format(step)) 
        #plt.pcolor(x[:,:,128],y[:,:,128],YSF6[:,:,128])

        #Clean up variables
        del YSF6

    #for ind in plt.get_fignums():
    #    plt.figure(ind)
    #    plt.tight_layout()
    #    plt.xlabel('x [m]')
    #    plt.ylabel('y [m]')
    #    ax = plt.gca()
    #    ax.set_aspect('equal')
    #    plt.colorbar()

    #Non-dimensionalization (see Wong and Lele, 2019)
    tvec = (1e-5)*np.array(tsteps)
    lambda_3D = 1.02e-3 #[m]
    eta_dot_3D_imp = 18.4 #[m s-1]
    tau_3D = lambda_3D/eta_dot_3D_imp
    tau_c = tau_3D
    tstar = tvec/tau_c

    ##Plots
    #plt.figure()
    #plt.plot(tvec,W)
    #plt.title('Mixing Width')
    #plt.xlabel('time [s]')

    #plt.figure()
    #plt.plot(tvec,Theta)
    #plt.title('Mixedness')
    #plt.xlabel('time [s]')

    #plt.figure()
    #plt.semilogy(tvec,TKE_int)
    #plt.title('Integrated TKE')
    #plt.xlabel('time [s]')

    #plt.figure()
    #plt.semilogy(tvec,Diss_int)
    #plt.title('Integrated Dissipation Rate')
    #plt.xlabel('time [s]')

    #plt.figure()
    #plt.semilogy(tvec,Enstrophy_int)
    #plt.title('Integrated Enstrophy')
    #plt.xlabel('time [s]')

    #plt.figure()
    #plt.plot(tstar,Ma_t_IMZ)
    #plt.title('Turbulent Mach #')
    #plt.xlabel('t* [-]')

    #plt.figure()
    #plt.plot(tstar,At_e_IMZ)
    #plt.title('Effective Atwood #')
    #plt.xlabel('t* [-]')

    plt.figure()
    plt.plot(tstar,b11_IMZ)
    plt.title('b11')
    plt.xlabel('t* [-]')

    plt.show()



