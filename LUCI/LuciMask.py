"""
Code for masking out the background pixels in LUCI
"""
import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits 
import numpy.ma as ma
from LuciPlotting import plot_map_no_coords
# Imports
import os
import sys
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt

# Get location of LUCI 
path = os.path.abspath('/home/carterrhea/Documents/LUCI')  
sys.path.insert(0, path)  # add LUCI to the available paths

from LuciBase import Luci
import LUCI.LuciPlotting as lplt

def find_background_pixels(deep_image, sigma_threshold=0.25, plot_mask=True):
    """
    This algorithm uses a sigma thresholding algorithm on the deep image to determine which pixels belong to the background.
    """

    zscore = (deep_image - deep_image.mean())/deep_image.std()  # Calculate z-score for each pixel
    idx = np.argwhere(np.abs(zscore) > sigma_threshold)  # Get indices in array where z-score is above sigma_threshold
    if plot_mask == True:
        zscore_masked = np.where(np.abs(zscore) < sigma_threshold, zscore, 0)  
        plot_map_no_coords(zscore_masked, 'zscore')
        plt.show()
    return idx

deep = fits.open('/mnt/carterrhea/carterrhea/NGC4449/Luci_outputs/NGC4449_deep.fits')[0].data
idx = find_background_pixels(deep)

#Set Parameters
# Using Machine Learning Algorithm for Initial Guess
Luci_path = '/home/carterrhea/Documents/LUCI/'
cube_dir = '/mnt/carterrhea/carterrhea/NGC4449'  # Path to data cube
cube_name = 'NGC4449_SN3'  # don't add .hdf5 extension
object_name = 'NGC4449'
redshift = 0.00068  # Redshift
resolution = 5000

cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, mdn=False)

print(len(idx))
if len(idx) > 500000:
    exit
bkg_spectra = [cube.cube_final[index[0], index[1]] for index in idx if index[0]<2048 and index[1]<2064]
print(len(bkg_spectra))
n_components = 5
pca = decomposition.PCA(n_components=n_components)
pca.fit(bkg_spectra)
X = pca.transform(bkg_spectra)

# Plot the primary components
plt.figure(figsize=(18, 16))
# plot the mean first
l = plt.plot(1e7/cube.spectrum_axis, pca.mean_-1,linewidth=3)
c = l[0].get_color()
plt.text(6, -0.9, 'mean emission', color=c,fontsize='xx-large')
shift = 0.6
for i in range(n_components):
    l = plt.plot(1e7/cube.spectrum_axis, pca.components_[i] + (i*shift), linewidth=3)
    c = l[0].get_color()
    plt.text(6, i*shift+0.1, "component %i" % (i + 1), color=c, fontsize='xx-large')
plt.xlabel('nm', fontsize=24)
plt.ylabel('Normalized Emission + offset',  fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('./PCA.png')
plt.clf()