"""
Set of functions required for Bayesian analysis using LUCI
"""
from LuciFit import Fit


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
    model = Fit.gaussian_model(x, theta, model)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(theta, model):
    A_min = 0#1e-19
    A_max = 1.#1e-15
    x_min = 14700
    x_max = 15400
    sigma_min = 0
    sigma_max = 10
    for model_num in range(len(model)):
        params = theta[model_num*3:(model_num+1)*3]
    within_bounds = True  # Boolean to determine if parameters are within bounds
    for ct, param in enumerate(params):
        if ct%3 == 0:  # Amplitude parameter
            if param > A_min and param < A_max:
                pass
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct%3 == 1:  # velocity parameter
            if param > x_min and param < x_max:
                pass
            else:
                within_bounds = False  # Value not in bounds
                break
        if ct%3 == 2:  # sigma parameter
            if param > sigma_min and param < sigma_max:
                pass
            else:
                within_bounds = False  # Value not in bounds
                break
    if within_bounds:
        return 0.0
    else:
        return -np.inf
    #A_,x_,sigma_ = theta
    #if A_min < A_ < A_max and x_min < x_ < x_max and sigma_min < sigma_ < sigma_max:
    #    return 0.0#np.log(1/((t_max-t_min)*(rp_max-rp_min)*(b_max-b_min)))
    #return -np.inf



def log_probability(theta, x, y, yerr, model):
    lp = log_prior(theta, model)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, model)
