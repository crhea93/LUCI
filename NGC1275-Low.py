import numpy as np
from astropy.io import fits
from LuciFit import Fit
from joblib import Parallel, delayed
import time
from tqdm import tqdm
import keras
import warnings
import regions
from astropy.io import fits
import h5py
from astropy.wcs import WCS
warnings.filterwarnings("ignore")

cube_dir = '/media/carterrhea/carterrhea/Benjamin'  # Path to data cube
cube_name = 'A0426_SN3.merged.cm1.1.0'  # don't add .hdf5 extension
redshift = 0.017284
#cube = SpectralCube(cube_dir+'/'+cube_name+'.hdf5')
#deep_file = '/media/carterrhea/carterrhea/M33/M33_deep'  # Path to deep image fits file: required for header

file =  h5py.File(cube_dir+'/'+cube_name+'.hdf5', 'r')
div_nb = 3
quad_number = 1

quad_nb = file.attrs['quad_nb']#div_nb**2
dimx = file.attrs['dimx']
dimy = file.attrs['dimy']
dimz = file.attrs['dimz']
def get_quadrant_dims(quad_number):
    if (quad_number < 0) or (quad_number > quad_nb - 1):
        raise StandardError("quad_number out of bounds [0," + str(quad_nb- 1) + "]")
        return "SOMETHING FAILED"

    index_x = quad_number % div_nb
    index_y = (quad_number - index_x) / div_nb

    x_min = index_x * np.floor(dimx / div_nb)
    if (index_x != div_nb - 1):
        x_max = (index_x  + 1) * np.floor(dimx / div_nb)
    else:
        x_max = dimx

    y_min = index_y * np.floor(dimy / div_nb)
    if (index_y != div_nb - 1):
        y_max = (index_y  + 1) * np.floor(dimy / div_nb)
    else:
        y_max = dimy
    return int(x_min), int(x_max), int(y_min), int(y_max)

cube_final = np.zeros((dimx, dimy, dimz))
for iquad in range(quad_nb):
    #print(iquad)
    xmin,xmax,ymin,ymax=get_quadrant_dims(iquad)
    #print(xmin, xmax, ymin, ymax)
    iquad_data = file['quad00%i'%iquad]['data'][:]
    #print(iquad_data.shape)
    iquad_data[(np.isfinite(iquad_data) == False)]= 1e-22 # Modifs
    iquad_data[(iquad_data < -1e-16)]= -1e-22 # Modifs
    iquad_data[(iquad_data > 1e-9)]= 1e-22 # Modifs
    cube_final[xmin:xmax, ymin:ymax, :] = iquad_data


# HEADER STUFF
hdr_dict = {}
header_cols = [str(val[0]).replace("'b",'').replace("'", "").replace("b",'') for val in list(file['header'][()])]
header_vals = [str(val[1]).replace("'b",'').replace("'", "").replace("b",'') for val in list(file['header'][()])]
header_types = [val[3] for val in list(file['header'][()])]
for header_col, header_val, header_type in zip(header_cols,header_vals, header_types):
    if 'bool' in str(header_type):
        hdr_dict[header_col] = bool(header_val)
    if 'float' in str(header_type):
        hdr_dict[header_col] = float(header_val)
    if 'int' in str(header_type):
        hdr_dict[header_col] = int(header_val)
    #print(header_type)
    else:
        try:
            hdr_dict[header_col] = float(header_val)
        except:
            hdr_dict[header_col] = str(header_val)
hdr_dict['CTYPE3'] = 'WAVE-SIP'
hdr_dict['CUNIT3'] = 'm'

# Make WCS
wcs_data = WCS(hdr_dict, naxis=2)
header = wcs_data.to_header()

# Create spectrum axis
len_wl = cube_final.shape[2]
print(cube_final.shape)
start = hdr_dict['CRVAL3']
end = start + len_wl*hdr_dict['CDELT3']
step = hdr_dict['CDELT3']
spectrum_axis = np.linspace(start, end, len_wl)*(redshift+1)
#print(spectrum_axis)
#spectrum_axis, sky_spectrum = cube.extract_spectrum(1100, 1100, 3)

# Machine Learning Reference Spectrum
ref_spec = fits.open('Reference-Spectrum-R1800.fits')[1].data
channel = []
counts = []
for chan in ref_spec:  # Only want SN3 region
    channel.append(chan[0])
    counts.append(np.real(chan[1]))
min_ = np.argmin(np.abs(np.array(channel)-14700))
max_ = np.argmin(np.abs(np.array(channel)-15600))
wavenumbers_syn = channel[min_:max_]

# Define spatial regions of interest
x_min = 1000
x_max = 1500#cube.shape[0]
y_min = 250
y_max = 750#cube.shape[1]
# Read in data
#dat = cube.get_data(x_min,x_max,y_min,y_max,0,cube.shape[2])
#dat = dat.astype('float32')
# Step through spectra
VEL = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32)
BROAD = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32)
#print(cube_final[1350, 600,:])
model_ML = keras.models.load_model('R1800-PREDICTOR-I')  # Read in
start = time.time()
def SNR_calc(VEL, BROAD, i):
    x_pix = x_min+i
    vel_local = []
    broad_local = []
    #if i%10 == 0:
        #print('Step %i'%i)
    for j in range(y_max-y_min):
        y_pix = y_min+j
        #print(i,j)
        #exit()
        # Calculate velocity
        sky =cube_final[x_pix,y_pix,:]# np.real(cube_final[i,j,:])
        #print(sky)
        good_sky_inds = [~np.isnan(sky)]
        sky = sky[good_sky_inds]
        axis = spectrum_axis[good_sky_inds]
        #print(wavenumbers_syn)
        #print(axis)
        #print(sky)
        #print(sky)
        fit = Fit(sky, axis, wavenumbers_syn, 'gaussian', ['Halpha', 'NII6583', 'NII6548', 'SII6716', 'SII6731'],
                model_ML, Plot_bool = False)
        fit_dict = fit.fit()
        vel_local.append(fit_dict['velocity'])
        broad_local.append(fit_dict['broadening'])
    VEL[i] = vel_local
    BROAD[i] = broad_local

n_threads = 1
#for i in range(x_max-x_min):
    #SNR_calc(VEL, BROAD, i)
#Parallel(n_jobs=n_threads, backend="threading", batch_size=int((x_max-x_min)/n_threads))(delayed(SNR_calc)(VEL, BROAD, i) for i in range(x_max-x_min));
Parallel(n_jobs=n_threads, backend="threading")(delayed(SNR_calc)(VEL, BROAD, i) for i in tqdm(range(x_max-x_min)));
end = time.time()
print(end-start)
# Save
fits.writeto('VEL.fits', VEL.T, header, overwrite=True)
fits.writeto('BROAD.fits', BROAD.T, header, overwrite=True)
