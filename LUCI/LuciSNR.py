import astropy.stats as astrostats
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import numpy.ma as ma
from astropy.io import fits

def create_snr_map_function(cube_final, spectrum_axis, hdr_dict, output_dir, object_name, header, x_min=0, x_max=2048, y_min=0, y_max=2064, method=1, n_threads=2):
    """
    Create signal-to-noise ratio (SNR) map of a given region. If no bounds are given,
    a map of the entire cube is calculated.

    Args:
        x_min: Minimal X value (default 0)
        x_max: Maximal X value (default 2048)
        y_min: Minimal Y value (default 0)
        y_max: Maximal Y value (default 2064)
        method: Method used to calculate SNR (default 1; options 1 or 2)
        n_threads: Number of threads to use
    Return:
        snr_map: Signal-to-Noise ratio map

    """
    # Calculate bounds for SNR calculation
    # Step through spectra
    # SNR = np.zeros((x_max-x_min, y_max-y_min), dtype=np.float32).T
    SNR = np.zeros((2048, 2064), dtype=np.float32).T
    # start = time.time()
    # def SNR_calc(SNR, i):
    flux_min = 0;
    flux_max = 0;
    noise_min = 0;
    noise_max = 0  # Initializing bounds for flux and noise calculation regions
    if hdr_dict['FILTER'] == 'SN3':
        flux_min = 15150;
        flux_max = 15300;
        noise_min = 14250;
        noise_max = 14400
    elif hdr_dict['FILTER'] == 'SN2':
        flux_min = 19500;
        flux_max = 20750;
        noise_min = 18600;
        noise_max = 19000
    elif hdr_dict['FILTER'] == 'SN1':
        flux_min = 26550;
        flux_max = 27550;
        noise_min = 25300;
        noise_max = 25700
    else:
        print('SNR Calculation for this filter has not been implemented')

    def SNR_calc(i):
        y_pix = y_min + i
        snr_local = np.zeros(2048)
        for j in range(x_max - x_min):
            x_pix = x_min + j
            # Calculate SNR
            # Select spectral region around Halpha and NII complex
            min_ = np.argmin(np.abs(np.array(spectrum_axis) - flux_min))
            max_ = np.argmin(np.abs(np.array(spectrum_axis) - flux_max))
            in_region = cube_final[x_pix, y_pix, min_:max_]
            flux_in_region = np.nansum(cube_final[x_pix, y_pix, min_:max_])
            # Subtract off continuum estimate
            clipped_spec = astrostats.sigma_clip(cube_final[x_pix, y_pix, min_:max_], sigma=2, masked=False,
                                                 copy=False, maxiters=3)
            # Now take the mean value to serve as the continuum value
            cont_val = np.median(clipped_spec)
            flux_in_region -= cont_val * (max_ - min_)  # Need to scale by the number of steps along wavelength axis
            # Select distance region
            min_ = np.argmin(np.abs(np.array(spectrum_axis) - noise_min))
            max_ = np.argmin(np.abs(np.array(spectrum_axis) - noise_max))
            out_region = cube_final[x_pix, y_pix, min_:max_]
            std_out_region = np.nanstd(cube_final[x_pix, y_pix, min_:max_])
            if method == 1:
                signal = np.nanmax(in_region) - np.nanmedian(in_region)
                noise = np.abs(np.nanstd(out_region))
                snr = float(signal / np.sqrt(noise))
                if snr < 0:
                    snr = 0
                else:
                    snr = snr / (np.sqrt(np.nanmean(np.abs(in_region))))
            else:
                snr = float(flux_in_region / std_out_region)
                if snr < 0:
                    snr = 0
                else:
                    pass
            snr_local[x_pix] = snr
        return snr_local, i

    res = Parallel(n_jobs=n_threads, backend="threading")(delayed(SNR_calc)(i) for i in tqdm(range(y_max - y_min)));
    # Save
    for snr_ind in res:
        snr_vals, step_i = snr_ind
        SNR[y_min + step_i] = snr_vals
    fits.writeto(output_dir + '/' + object_name + '_SNR.fits', SNR, header, overwrite=True)

    # Save masks for SNr 3, 5, and 10
    for snr_val in [3, 5, 10]:
        mask = ma.masked_where(SNR >= snr_val, SNR)
        np.save("%s/SNR_%i_mask.npy" % (output_dir, snr_val), mask.mask)

    return None