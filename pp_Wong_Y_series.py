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

    middle_selection = 0.20 #select middle 25% of domain in x
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

        plt.figure()
        plt.title("t = {} ms".format(step*1e-2)) 
        plt.pcolor(1000*x[:,:,int(Nz/2)],1000*y[:,:,int(Nz/2)],YHeavy[:,:,int(Nz/2)])
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.colorbar()
        plt.clim([0,1])

    #Format plots
    print("Formatting Plots")
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
    tsteps = (80,115,125,155,180)
    
    path = "/work/05428/tg847275/Wong/uniform_data/3D_Poggi_RMI_RD/case_1_1/grid_C"
    #path = "/home/jrwest/Research/FloATPy_moving_grid/data/Wong/grid_B"

    tvec = 1e-5*np.array(tsteps)

    plot_set1(tsteps,tvec,path)

    #### END PROBLEM SPECIFIC STUFF ####
    ########################################################################
    ########################################################################

    plt.show()
