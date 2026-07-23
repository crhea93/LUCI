
# Imports
import sys
import numpy as np
sys.path.insert(0, '/home/crhea/Projects/LUCI/')  # Location of Luci
from LuciBase import Luci

#Set Parameters
# Using Machine Learning Algorithm for Initial Guess
Luci_path = '/home/crhea/Projects/LUCI/'
# Initialize paths and set parameters
cube_dir = '/home/crhea/Documents/LUCI_DATA'  # Path to data cube
#cube_dir = '/mnt/carterrhea/carterrhea/NGC628'  # Full path to data cube (example 2)
cube_name = 'M86_SN4'  # don't add .hdf5 extension
object_name = 'M86'
filter_name = 'SN4'
redshift = -0.0010201857421032745  # Redshift of object
resolution = 4800 # The actual resolution is 400, but we don't have ML algorithms for that resolution, so use 1000

# Create Luci object
cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, mdn=True)

cube.visualize()
