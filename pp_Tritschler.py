#General Modules
import numpy as np
import os
import unittest
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

#Local Modules
import floatpy.readers.miranda_reader as mir

# Set up default plotting
params = {'legend.fontsize': 14,
          'legend.handlelength': 2,
          'font.size': 16}
matplotlib.rcParams.update(params)

if __name__ == '__main__':

    #Read in a file and plot something
    reader = mir.MirandaReader("/home/jrwest/Research/FloATPy_moving_grid/data/Tritschler/RM_CTR_3D_64/plot.mir", periodic_dimensions=(False,True,True),verbose=True)

    print("Domain Size: {} ".format(reader._domain_size))

    x,y,z = reader.readCoordinates()

    print("Variables available: {}".format(reader.varNames))

    tsteps = (15,20,22,27,50)

    for step in tsteps:
        reader.setStep(step)
        YN2  = np.array(reader.readData('Ideal1_01'))
        YO2  = np.array(reader.readData('Ideal1_02'))
        YSF6 = np.array(reader.readData('Ideal1_03'))
        YAc  = np.array(reader.readData('Ideal1_04'))
        YHeavy = YSF6+YAc
        YLight = YN2+YO2

        plt.figure()
        plt.title("t = {}E-4 sec".format(step)) 
        plt.pcolor(x[:,:,32],y[:,:,32],YLight[0,:,:,32])
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.colorbar()

    plt.show()



