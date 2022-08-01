
def fit_absorption(self):
  # Define log likelihood function
  def log_likelihood(theta):
    """
    Calculate log likelihood function given a set of parameters theta.
    Theta = [amplitude, position, sigma, continuum]
    """
    # Define model function
    model = Gaussian(self.freeze).evaluate(self.axis_restricted, theta[0:3], 'Halpha')
    sigma2 = self.noise ** 2
    return -0.5 * np.sum((self.spectrum_restricted - model) ** 2 / sigma2) + np.log(2 * np.pi * sigma2)



  # Define negative log likelihood function
  nll = lambda *args: -self.log_likelihood(*args)  # Negative Log Likelihood function

  # Define decent initial guess
  ampl_init = -0.2  # Say 20% is absorbed
  pos_init = 15350  # Position of non-shifted Halpha in cm^-1
  pos_sigma = 1  # For a decently wide absorption line
  cont_init = 1  # Assuming the continuum is the largest feature in the normalized spectrum.

  # Call minimization code
  soln = minimize(nll, initial,
                        method='SLSQP',
                        options={'disp': False, 'maxiter': 30},
                        tol=1e-2,
                        args=()
                        )
  parameters = soln.x  # This is the list of parameters you get out [ampl, pos, sigma, cont]
