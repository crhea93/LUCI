import numpy as np
from scipy import special as sps
 

class Gaussian:
    def __init__(self, channel, params):
        A = params[0];
        x = params[1];
        sigma = params[2]
        self.func = A * np.exp((-(channel - x) ** 2) / (2 * sigma ** 2))


class Sinc:
    def __init__(self, channel, params):
        p0 = params[0];
        p1 = params[1];
        p2 = params[2]
        u = (channel - p1) / p2
        self.func = p0 * (np.sin(u) / u)


class SincGauss:
    def __init__(self, channel, params, sinc_width):
        p0 = params[0]
        p1 = params[1]
        p2 = sinc_width
        p3 = params[2]
        a = p3/(np.sqrt(2)*p2)
        b = (channel-p1)/(np.sqrt(2)*p3)
        self.func = p0*np.exp(-b**2)*((sps.erf(a-1j*b)+sps.erf(a+1j*b))/(2*sps.erf(a)))
