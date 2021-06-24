from orcs.process import SpectralCube
import orb
import numpy as np
from astropy.io import fits
from LuciFit import Fit
from numba import njit, prange, jit
import progressbar
from joblib import Parallel, delayed
import time

cube_dir = '/media/carterrhea/carterrhea/M33'  # Path to data cube
cube_name = 'M33_Field7_SN3.merged.cm1.1.0'  # don't add .hdf5 extension

cube = SpectralCube(cube_dir+'/'+cube_name+'.hdf5')
deep_file = '/media/carterrhea/carterrhea/M33/M33_deep'  # Path to deep image fits file: required for header


spectrum_axis, sky_spectrum = cube.extract_spectrum(1100, 1100, 3)

# Machine Learning Reference Spectrum
ref_spec = fits.open('Reference-Spectrum-R5000.fits')[1].data
channel = []
counts = []
for chan in ref_spec:  # Only want SN3 region
    channel.append(chan[0])
    counts.append(np.real(chan[1]))
min_ = np.argmin(np.abs(np.array(channel)-14700))
max_ = np.argmin(np.abs(np.array(channel)-15600))
wavenumbers_syn = channel[min_:max_]

# Define spatial regions of interest
x_min = 1250
x_max = 1350#cube.shape[0]
y_min = 1750
y_max = 1850#cube.shape[1]
# Read in data
dat = cube.get_data(x_min,x_max,y_min,y_max,0,cube.shape[2])
dat = dat.astype('float32')
# Step through spectra
VEL = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32)
start = time.time()
def SNR_calc(SNR, i):
    vel_local = []
    for j in range(y_max-y_min):
        # Calculate velocity
        fit = Fit(dat[i,j], spectrum_axis, wavenumbers_syn, 'gaussian', ['Halpha'],
                'R5000-PREDICTOR-I', Plot_bool = False)
        fit_dict = fit.fit()
        vel_local.append(fit_dict['velocity'])
    VEL[i] = vel_local
n_threads = 8
Parallel(n_jobs=n_threads, backend="threading", batch_size=int((x_max-x_min)/n_threads))(delayed(SNR_calc)(VEL,i) for i in range(x_max-x_min));
end = time.time()
print(end-start)
# Save
reg_fits = fits.open(deep_file+'.fits')
header = reg_fits[0].header
fits.writeto('VEL.fits', VEL.T, header, overwrite=True)