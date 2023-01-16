
# Imports
import sys
import numpy as np
sys.path.insert(0, '/home/carterrhea/Documents/LUCI/')  # Location of Luci
from LuciBase import Luci

#Set Parameters
# Using Machine Learning Algorithm for Initial Guess
Luci_path = '/home/carterrhea/Documents/LUCI/'
cube_dir = '/mnt/carterrhea/carterrhea/NGC4449'  # Path to data cube
cube_name = 'NGC4449_SN3'  # don't add .hdf5 extension
object_name = 'NGC4449'
redshift = 0.00068  # Redshift
resolution = 5000

# Create Luci object
cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, mdn=True)

cube.visualize()
