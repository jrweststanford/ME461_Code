import numpy as np

def radial_spectra(x,y,z,f):
    #assumes periodic in y and z, then averages over x
    #best designed for the IMZ in the RMI problem
    #note: x/y/z are 1D
    fhat = np.fft.fft(f,axis=2) 
    fhat = np.fft.fft(fhat,axis=1)
    f_spec = np.real(fhat*np.conj(fhat))
    fspec_yz = np.mean(f_spec,axis=0)

    #Loop thru radial bins and lump
    Lz = z[-1]-z[0]
    Nz = len(z)
    Ly = y[-1]-y[0]
    Ny = len(y)
    kz = 2*np.pi/(Lz)*np.arange(0,Nz/2+1)
    ky = 2*np.pi/(Ly)*np.arange(0,Ny/2+1)

    kmin = (ky[1]**2+kz[1]**2)**0.5 #intentionally discard 0 wavenumber
    kmax = (ky[-1]**2+kz[-1]**2)**0.5

    Nbins = Ny/3
    kbin_edges = np.linspace(kmin,kmax,Nbins+1)
    k_bins = (kbin_edges[0:-1]*kbin_edges[1:])**0.5 #geometric mean
    fspec_rad = np.zeros(Nbins)

    for indy in range(Ny/2):
        for indz in range(Nz/2):
            kmag = (ky[indy]**2+kz[indz]**2)**0.5

            #assign to bin
            for indbin in range(Nbins):
                if (kmag<kbin_edges[indbin+1]) and (kmag>kbin_edges[indbin]):
                    fspec_rad[indbin] += fspec_yz[indy,indz]
                    break

    fspec_rad = fspec_rad/(np.trapz(fspec_rad,k_bins))

    return k_bins, fspec_rad
