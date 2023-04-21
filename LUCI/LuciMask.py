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
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources
# Get location of LUCI 
path = os.path.abspath('/home/carterrhea/Documents/LUCI')  
sys.path.insert(0, path)  # add LUCI to the available paths

from LuciBase import Luci
import LUCI.LuciPlotting as lplt

def find_background_pixels(deep_image, sigma_threshold=1, plot_mask=True):
    """
    This algorithm uses a sigma thresholding algorithm on the deep image to determine which pixels belong to the background.
    """
    bkg_estimator = MedianBackground()
    bkg = Background2D(deep_image, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    deep_image -= bkg.background  # subtract the background
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    convolved_data = convolve(deep_image, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    #zscore = (deep_image - deep_image.mean())/deep_image.std()  # Calculate z-score for each pixel
    #idx = np.argwhere(np.abs(zscore) > sigma_threshold)  # Get indices in array where z-score is above sigma_threshold
    if plot_mask == True:
        #zscore_masked = np.where(np.abs(zscore) < sigma_threshold, zscore, np.nan)
        #zscore_masked = np.where(np.abs(zscore_masked) > sigma_threshold, zscore_masked, 1)
        #plt.imshow(zscore_masked)
        plt.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
        plt.ylim(0, 2048)
        plt.xlim(0, 2048)
        #plot_map_no_coords(zscore_masked, '/home/carterrhea/Downloads/zscore')
        plt.savefig('Background.png')
    return idx
deep = fits.open('/mnt/carterrhea/carterrhea/Perseus/SITELLE/Luci_outputs/NGC1275_lowres_deep.fits')[0].data
#fits.open('/mnt/carterrhea/carterrhea/NGC4449/Luci_outputs/NGC4449_deep.fits')[0].data
deep = deep[200:2000, 200:2000]
idx = find_background_pixels(deep)

#Set Parameters
# Using Machine Learning Algorithm for Initial Guess
Luci_path = '/home/carterrhea/Documents/LUCI/'
cube_dir = '/mnt/carterrhea/carterrhea/Perseus/SITELLE'  # Path to data cube
cube_name = 'A0426_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
object_name = 'NGC1275'
redshift = 0.0179  # Redshift
resolution = 1800

cube = Luci(Luci_path, cube_dir+'/'+cube_name, cube_dir, object_name, redshift, resolution, mdn=False)

print(len(idx))
if len(idx) > 500000:
    print('exiting')
    exit
bkg_spectra = [cube.cube_final[index[0], index[1]] for index in idx if index[0]<2048 and index[1]<2064]
print(len(bkg_spectra))
n_components = 50
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
#plt.savefig('./PCA.png')
#plt.clf()
plt.show()
mu = np.mean(bkg_spectra, axis=0)

pca = decomposition.PCA()
pca.fit(bkg_spectra)

nComp = 5
projectedPCAValues = pca.transform(bkg_spectra)[:,1:nComp]  # PCA Coefficient values in PCA space
Xhat = np.dot(projectedPCAValues, pca.components_[1:nComp,:])
Xhat += mu

lplt.plot_spectrum(1e7/cube.spectrum_axis, np.median(Xhat, axis=0))
plt.show()

plt.plot(projectedPCAValues[:,1], projectedPCAValues[:,2])
plt.show()