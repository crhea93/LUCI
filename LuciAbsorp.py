import scipy as sp
import numpy as np

def extract_spectrum_deredshift(cube, x_min, x_max, y_min, y_max, vel_map, region = None, bkg=None, binning=None, mean=False):
        """
        Extract spectrum in region, including deredshifting velocity correction. This is primarily used to extract background regions.
        The spectra in the region are summed and then averaged (if mean is selected).
        Using the 'mean' argument, we can either calculate the total summed spectrum (False)
        or the averaged spectrum for background spectra (True).

        Args:
            x_min: Lower bound in x
            x_max: Upper bound in x
            y_min: Lower bound in y
            y_max: Upper bound in y
            vel_map: Velocity map (must be same size as given bounds)
            region: Name of ds9 region file (e.x. 'region.reg'). You can also pass a boolean mask array.
            bkg: Background Spectrum (1D numpy array; default None)
            binning:  Value by which to bin (default None)
            region: 
            mean: Boolean to determine whether or not the mean spectrum is taken. This is used for calculating background spectra.
        Return:
            X-axis (redshifted) and spectral axis of region.

        """
        axis_all = []
        spectrum_all = []
        avg_vel = np.nanmean(vel_map)
        
        mask = None
        
        if region is not None:
            if isinstance(region,str) and '.reg' in region:  # If passed a .reg file
                if binning != None and binning > 1:
                    header = self.header_binned
                else:
                    header = self.header
                header.set('NAXIS1', 2064)  # Need this for astropy
                header.set('NAXIS2', 2048)
                mask = reg_to_mask(region, header)
            elif isinstance(region,str) and '.npy' in region:  # If passed numpy file
                mask = np.load(region).T
            else:  # If passed numpy array
                mask = region.T
                print(mask)
        
        bool_corr = True
        
        spec_ct = 0
        axis = np.array(cube.spectrum_axis[:cube.cube_final.shape[2]])  # Initialize

        # Initialize fit solution arrays
        # if binning != None and binning != 1:
        if binning != None:
            cube.bin_cube(cube.cube_final, cube.header, binning, x_min, x_max, y_min,
                          y_max)
            x_max = int((x_max - x_min) / binning)
            y_max = int((y_max - y_min) / binning)
            x_min = 0
            y_min = 0
        for i in range(y_max - y_min):
            y_pix = y_min + i
            for j in range(x_max - x_min):
                x_pix = x_min + j
                
                if mask is not None:  # Check if there is a mask
                    # print(mask[x_pix, y_pix])
                    if mask[x_pix, y_pix]:  # Check that the mask is true
                        bool_corr = True
                    else:
                        bool_corr = False
                    
                # if binning is not None and binning != 1:
                if binning is not None:
                    sky = cube.cube_binned[x_pix, y_pix, :]
                else:
                    sky = cube.cube_final[x_pix, y_pix, :]
                if bkg is not None:
                    if binning:
                        sky -= bkg * binning ** 2  # Subtract background spectrum
                    else:
                        sky -= bkg  # Subtract background spectrum
                
                if bool_corr and (np.abs(vel_map[j,i]) <= 500): #Check that we should include this pixel                        
                    vel = vel_map[j,i] 
                    beta = vel/299800
                    axis_corr = axis*np.sqrt((1+beta)/(1-beta)) #velocity correction

                    axis_all.append(axis_corr)
                    spectrum_all.append(sky)
                    spec_ct += 1
        
        axis_all = np.concatenate(axis_all)
        spectrum_all = np.concatenate(spectrum_all)
        
        axis_lam = np.flip(10**8/axis_all)
        spectrum_flip = np.flip(spectrum_all)
        
        beta_avg = -avg_vel/299800

        bins = np.linspace((10**8/axis[-1])*np.sqrt((1+beta_avg)/(1-beta_avg)), (10**8/axis[0])*np.sqrt((1+beta_avg)/(1-beta_avg)), len(axis))
        
        mean_axis = np.zeros(len(axis))
        mean_spec = np.zeros(len(axis))
        inds = np.digitize(axis_lam, np.float64(bins))
        for i in range(1,len(bins)+1):
            cond = np.where(inds == i)
            mean_axis[i-1] = np.nanmean(axis_lam[cond])
            mean_spec[i-1] = np.nanmean(spectrum_flip[cond])
        
        print(spec_ct)
        if not mean:
            mean_spec *= spec_ct
        return mean_axis, mean_spec


def interpolate(datax, datay, new_step):
    #Calculate the length required so that the new spectrum has a number of steps given by 'resolution'.
    new_length = int(round(((datax[-1]-datax[0]))/new_step))
    #Create the interpolated arrays
    newx = np.linspace(np.array(datax).min(), np.array(datax).max(), new_length)
    newy = sp.interpolate.interp1d(datax, datay, kind='linear')(newx)
    return newx, newy


def assemble_spectrum(sn2axis, sn2spec, sn3axis, sn3spec, new_step):
    #Because spectra all have different steps between their x-axis value, we must create a new spectrum containing many interpolated points between each previous data point,
    #so that the new step is the same for all the cubes.
    
    interx_sn3, intery_sn3 = interpolate(sn3axis, sn3spec, new_step)
    interx_sn2, intery_sn2 = interpolate(sn2axis, sn2spec, new_step)
    len_empty = int((interx_sn3[0]-interx_sn2[-1])/new_step)
    spectrum_total = np.append(intery_sn2,np.zeros(len_empty))
    spectrum_total = np.append(spectrum_total,intery_sn3)
    axis_total=np.linspace(interx_sn2[0],interx_sn3[-1],len(spectrum_total))
    
    return axis_total, spectrum_total

