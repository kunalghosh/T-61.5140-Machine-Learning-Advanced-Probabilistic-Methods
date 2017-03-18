# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: antti.kangasraasio@aalto.fi, 2016

from numpy import zeros
from numpy.random import randn


def generate_X(ndata, pdata):
    """
        Return a matrix of normally distributed random values.
    """
    X = randn(ndata, pdata)
    return X


def generate_YZ(X, distribution):
    """
        Draw observations Y and latent variable values Z from a distribution.
    """
    ndata = len(X)
    Y = zeros(ndata)
    Z = zeros(ndata)
    for i in range(ndata):
        Y[i], Z[i] = distribution.draw(X[i])
    return Y, Z


def get_hyperp():
    """
        Return model hyperparameters.
    """
    return {
            "alpha_s20": 5.0,
            "beta_s20" : 1.0,
            "lbd_phi0" : 1.0,
            "mu_phi0"  : 0.0,
            "alpha_w0" : 3.0,
            "beta_w0"  : 3.0,
            }

