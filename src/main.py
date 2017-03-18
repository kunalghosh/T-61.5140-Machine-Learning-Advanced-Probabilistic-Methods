# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: kunal.ghosh@aalto.fi, 2016
# Author: Jussi.k.ojala@aalto.fi, 2016
# Author: antti.kangasraasio@aalto.fi, 2016

from em_algo_mm import EM_algo_MM
from em_algo_lm import EM_algo_LM
from generator import generate_X, generate_YZ, get_hyperp

import matplotlib.pyplot as plt
from numpy import arange, min, max, sqrt, mean, std
from scipy.spatial.distance import cosine
import numpy as np
import sys
import pprint

def mse(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    print("MSE:")
    # pprint.pprint(zip(a,b))
    print("%.3f" % mean((a-b)**2))

def main():
    """
        Executed when program is run.
    """
    ndims = 1
    data_t = 100
    # ndims = int(sys.argv[1])
    # data_t = int(sys.argv[2])
    data_v = 50
    print("Mixture Model")
    print("")
    test_MM_model(ndims=ndims, data_t=data_t, data_v = data_v)
    print("")
    print("Starting program")
    print("")
    test_LM_model(ndims=ndims, data_t=data_t, data_v = data_v)
    print("")


def test_LM_model(ndims, data_t, data_v):
    """
        Example that demonstrates how to call the model.
    """
    # get hyperparameters for model
    hyperp = get_hyperp()
    # generate 50 training data and 20 validation data locations of dim=1
    ndata = data_t  # Training Data
    ndata_v = data_v  # Validation Data
    pdata = ndims  # K
    X = generate_X(ndata, pdata)
    X_v = generate_X(ndata_v, pdata)
    # intialie true model randomly and draw observations from it
    true_model = EM_algo_LM(hyperp, ndata=ndata, pdata=pdata)
    Y, Z = generate_YZ(X, true_model) # TODO : change distribution.draw to draw samples from the mixture model
    Y_v, Z_v = generate_YZ(X_v, true_model)
    print("Generated %d training data and %d validation data from true model:" % \
            (ndata, ndata_v))
    true_model.print_p()
    print("")

    # generate a model for estimating the parameters of the
    # true model based on the observations (X, Y) we just made
    model = EM_algo_LM(hyperp, X, Y)
    i, logl, r = model.EM_fit()
    print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
            (logl, i, r))
    print("")
    print("MAP estimate of true model parameters:")
    model.print_p()
    print("")

    # crossvalidate the estimated model with the validation data
    fit_params = model.get_p()
    model_v = EM_algo_LM(hyperp, X_v, Y_v)
    model_v.set_p(fit_params)
    logl, ll = model_v.logl()
    print("Crossvalidated logl: %.2f" % (logl))

    # if possible, plot samples, true model and estimated model
    if pdata != 1:
        return
    plt.scatter(X, Y, s=20, c='black', label="Training data")
    plt.scatter(X_v, Y_v, s=20, c='orange', label="Validation data")
    x = arange(min(X)-0.1, max(X)+0.1, 0.1)
    print_linear_model(x, true_model.get_p()["phi"], \
            true_model.get_p()["sigma2"], 'red', "True model")
    print_linear_model(x, model.get_p()["phi"], \
            model.get_p()["sigma2"], 'blue', "Predicted model")
    plt.legend(loc=1)
    plt.xlim(min(x), max(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def test_MM_model(ndims, data_t, data_v):
    """
        Example that demonstrates how to call the model.
    """
    # get hyperparameters for model
    hyperp = get_hyperp()
    # generate 50 training data and 20 validation data locations of dim=1
    ndata = data_t # Training Data
    ndata_v = data_v # Validation Data
    pdata = ndims # K
    X = generate_X(ndata, pdata)
    X_v = generate_X(ndata_v, pdata)
    # intialize true model randomly and draw observations from it
    true_model = EM_algo_MM(hyperp, ndata=ndata, pdata=pdata)
    # true_model.print_p()
    Y, Z = generate_YZ(X, true_model) # TODO : change distribution.draw to draw samples from the mixture model
    Y_v, Z_v = generate_YZ(X_v, true_model)
    print("Generated %d training data and %d validation data from true model:" % \
            (ndata, ndata_v))
    true_model.print_p()
    print("")

    # generate a model for estimating the parameters of the
    # true model based on the observations (X, Y) we just made
    model = EM_algo_MM(hyperp, X, Y)
    # model.print_p()
    i, logl, r = model.EM_fit()

    # plt.plot(model.get_loglVals())
    # plt.show()

    # plt.plot(vals)
    # plt.show()
    print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
            (logl, i, r))
    print("")
    print("MAP estimate of true model parameters:")
    model.print_p()
    print("")

    # crossvalidate the estimated model with the validation data
    fit_params = model.get_p()
    model_v = EM_algo_MM(hyperp, X_v, Y_v)
    model_v.set_p(fit_params)
    logl, ll = model_v.logl()
    print("Crossvalidated logl: %.2f" % (logl))
    # print("DEBUG MSE")
    # print zip((Z_v*(X_v.dot(fit_params["phi_1"]))),((1-Z_v)*(X_v.dot(fit_params["phi_2"]))))
    mse((Z_v*X_v.dot(model_v.get_p()["phi_1"]))+((1-Z_v)*X_v.dot(model_v.get_p()["phi_2"])), Y_v)
    # if possible, plot samples, true model and estimated model
    if pdata != 1:
        return
    plt.scatter(X, Y, s=20, c='black', label="Training data")
    # plt.scatter(X_v, Y_v, s=20, c='orange', label="Validation data")
    x = arange(min(X)-0.1, max(X)+0.1, 0.1)
    print_mixture_model(x, true_model.get_p()["phi_1"], true_model.get_p()["phi_2"], \
            true_model.get_p()["sigma2_1"], true_model.get_p()["sigma2_2"], 'red', "True model")
    print_mixture_model(x, model.get_p()["phi_1"], model.get_p()["phi_2"], \
                        model.get_p()["sigma2_1"], model.get_p()["sigma2_2"], 'blue', "Predicted model")
    plt.legend(loc=1)
    plt.xlim(min(x), max(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print "end"

def print_mixture_model(x, phi_1, phi_2 , sigma2_1, sigma2_2, color, label):
    """
        Print mixture model mean and 95% confidence interval.
    """
    y1 = phi_1 * x;
    y2 = phi_2 * x;
    plt.plot(x, y1, color, label=label)
    plt.plot(x, y2, color, label=label)
    plt.fill_between(x, y1 + 1.96 * sqrt(sigma2_1), y1 - 1.96 * sqrt(sigma2_1), \
            alpha=0.25, facecolor=color, interpolate=True)
    plt.fill_between(x, y2 + 1.96 * sqrt(sigma2_2), y2 - 1.96 * sqrt(sigma2_2), \
                     alpha=0.25, facecolor=color, interpolate=True)


def print_linear_model(x, phi, sigma2, color, label):
    """
        Print linear model mean and 95% confidence interval.
    """
    y = phi * x
    plt.plot(x, y, color, label=label)
    plt.fill_between(x, y + 1.96 * sqrt(sigma2), y - 1.96 * sqrt(sigma2), \
            alpha=0.25, facecolor=color, interpolate=True)


if __name__ == "__main__":
    main()

