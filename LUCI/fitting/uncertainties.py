"""Numerical Hessian used to derive parameter uncertainties."""

import numpy as np


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def hessianComp(func, initial, delta=1e-1):
    """
    Calculate the hessian using finite differences. The function was taken from https://rh8liuqy.github.io/Finite_Difference.html.

    Choice of delta does affect the result if you pick delta too small or too large.
    See https://math.stackexchange.com/questions/1039428/finite-difference-method
    """
    f = func
    initial = np.array(initial, dtype=float)
    n = len(initial)
    output = np.matrix(np.zeros(n * n))
    output = output.reshape(n, n)
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            ej = np.zeros(n)
            ej[j] = 1
            f1 = f(initial + delta * ei + delta * ej)
            f2 = f(initial + delta * ei - delta * ej)
            f3 = f(initial - delta * ei + delta * ej)
            f4 = f(initial - delta * ei - delta * ej)
            numdiff = (f1 - f2 - f3 + f4) / (4 * delta * delta)
            output[i, j] = numdiff
    return output
