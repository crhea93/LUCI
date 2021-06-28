

class Gaussian:
    def __init__(self, channel, params):
        A = params[0]; x = params[1]; sigma = params[2]
        self.func = A*np.exp((-(channel-x)**2)/(2*sigma**2))

# Update the model
def gaussian_model(channel, theta, models):
    """
    """
    f1 = 0.0
    for model_num in range(len(models)):
        params = theta[model_num*3:(model_num+1)*3]
        f1 += Gaussian(channel, params).func
    return f1

def log_likelihood(theta, x, y, yerr, model):
    """
    theta - list of parameters for gaussian fit
    """
    #if model == 1:
    #    A_,B_,x_,sigma_ = theta
    #    model = gaussian_model(x, A_, B_, x_, sigma_)
    #elif model == 2:
    #    A_,B_,x_,sigma_, A2_, x2_, sigma2_ = theta
    #    model = gaussian_model2(x, A_, B_, x_, sigma_, A2_, x2_, sigma2_)
    model = gaussian_model(x, theta, model)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def line_vals_estimate(spec, line_name):
    line_theo = line_dict[line_name]
    line_pos_est = 1e7/((vel_ml/3e5)*line_theo + line_theo)  # Estimate of position of line in cm-1
    line_ind = np.argmin(np.abs(np.array(axis)-line_pos_est))
    line_amp_est = spec[line_ind]#np.mean([sky[line_ind-1], sky[line_ind], sky[line_ind+1]])
    return line_amp_est, line_pos_est


def basic_fit(axis, sky, wavenumber_syn, model):
    """
    Apply basic fit. Must supply the uninterpolated spectrum
    """
    # Clean up spectrum
    good_sky_inds = [~np.isnan(sky)]
    sky = sky[good_sky_inds]
    axis = axis[good_sky_inds]

    # Interpolate
    f = interpolate.interp1d(axis, sky, kind='slinear')
    sky_corr = (f(wavenumbers_syn))
    sky_corr_scale = np.max(sky_corr)
    sky_corr = sky_corr/sky_corr_scale

    # Calculate estimates
    Spectrum = sky_corr.reshape(1, sky_corr.shape[0], 1)
    predictions = model(Spectrum, training=False)
    vel_ml = float(predictions[0][0])
    broad_ml = float(predictions[0][1])
    print(vel_ml, broad_ml)
    line_halpha = 656.28
    line_ml = 1e7/((vel_ml/3e5)*line_halpha + line_halpha)
    print(line_ml)
    # Calculate flux estimate
    line_ind = np.argmin(np.abs(np.array(axis)-line_ml))
    flux_est = np.max([sky[line_ind-1], sky[line_ind], sky[line_ind+1]])

    # Set bounds
    A_min = 0#1e-19
    A_max = 1.#1e-15
    x_min = 14700
    x_max = 15400
    sigma_min = 0
    sigma_max = 100

    # Create line dictionary
    line_dict = {'Halpha': 656.280, 'NII6583': 658.341, 'NII6548': 654.803, 'SII6716': 671.647, 'SII6731': 673.085}

    # Define models
    model = ['Halpha', 'NII6583', 'NII6548', 'SII6716', 'SII6731']
    nll = lambda *args: -log_likelihood(*args)
    initial = np.ones((3*len(model)))
    # Normalize Spectrum
    sky_scale = np.max(sky)
    sky_norm = sky/sky_scale
    bounds_ = []#np.zeros((3*len(model), 2))
    #Set bounds
    for mod in range(len(model)):
        val = 3*mod + 1
        amp_est, vel_est = line_vals_estimate(sky_norm, model[mod])
        initial[3*mod] = amp_est
        initial[3*mod + 1] = vel_est
        initial[3*mod + 2] = 2
        bounds_.append((A_min, A_max))
        bounds_.append((x_min, x_max))
        bounds_.append((sigma_min, sigma_max))
    bounds_l = [val[0] for val in bounds_]
    bounds_u = [val[1] for val in bounds_]
    bounds = Bounds(bounds_l, bounds_u)
    # Set constraints
    cons = (#{'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[1]-line_dict['Halpha'])/(1e7/x[1]))},
            {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[11]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[14]},
            {'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[7]-line_dict['NII6548'])/(1e7/x[7]))},
            {'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[10]-line_dict['SII6716'])/(1e7/x[10]))},
            {'type': 'eq', 'fun': lambda x: 3e5*((1e7/x[4]-line_dict['NII6583'])/(1e7/x[4])) - 3e5*((1e7/x[13]-line_dict['SII6731'])/(1e7/x[13]))})
    # Solve
    soln = minimize(nll, initial, method='SLSQP',
                    options={'disp': True}, bounds=bounds, tol=1e-16,
                    args=(axis, sky_norm, 1e-2, model))#, constraints=cons)
    # Update solution
    parameters = soln.x
    for i in range(len(model)):
        parameters[i*3] *= sky_corr_scale
    # Make fit vector
    final_model = gaussian_model(axis, parameters, model)
    
