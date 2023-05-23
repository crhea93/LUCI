#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import sys
sys.path.insert(0, '/home/crhea/home/LUCI/')  # Location of Luci
from LuciBase import Luci
import LUCI.LuciPlotting as lplt
import matplotlib.pyplot as plt


# In[2]:


#Set Parameters
# Using Machine Learning Algorithm for Initial Guess
Luci_path = '/home/crhea/home/LUCI/'
cube_dir = '/home/crhea/home/SITELLE/NGC4449'  # Path to data cube
cube_name = '2475286z'  # don't add .hdf5 extension
object_name = 'NGC4449'
redshift = 0.00068  # Redshift
resolution = 5000


# In[3]:


# Create Luci object
cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, mdn=False)


# In[ ]:


# Create Deep Image
cube.create_deep_image()


# In[ ]:


bkg_axis, bkg_sky = cube.extract_spectrum_region(cube_dir+'/bkg.reg', mean=True)  # We use mean=True to take the mean of the emission in the region instead of the sum




vel_map, broad_map, flux_map, chi2_fits = cube.fit_cube(['Halpha', 'NII6583', 'NII6548', 'SII6716', 'SII6731'],
                                                        'sincgauss',
                                                        [1,1,1,2,2], [1,1,1,2,2],
                                                        450, 1650 ,450, 1650,
                                                        #900,1000,900,1000,
                                                        binning=1,
                                                        bkg=bkg_sky,
                                                        n_threads=100,
                                                        n_stoch = 1,
                                                        uncertainty_bool = True

                                    )


# In[ ]:


lplt.plot_map(flux_map[:,:,0], 'flux', cube_dir, cube.header, clims=[-17, -13])


# In[ ]:


lplt.plot_map(vel_map[:,:,0], 'velocity', cube_dir, cube.header, clims=[-10,90])


# In[ ]:


lplt.plot_map(broad_map[:,:,0], 'broadening', cube_dir, cube.header, clims=[0,80])


# ###

# In[ ]:
