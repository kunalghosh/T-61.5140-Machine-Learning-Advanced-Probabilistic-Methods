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

def main():
    """
        Executed when program is run.
    """
    ndims = int(sys.argv[1])
    data_t = int(sys.argv[2])
    data_v = 50

    # -------- Generating Data ---------
    # get hyperparameters for model
    hyperp = get_hyperp()
    # generate 50 training data and 20 validation data locations of dim=1
    ndata = data_t  # Training Data
    ndata_v = data_v  # Validation Data
    pdata = ndims  # K
    X = generate_X(ndata, pdata)
    X_v = generate_X(ndata_v, pdata)
    # intialie true model randomly and draw observations from it
    true_model = EM_algo_MM(hyperp, ndata=ndata, pdata=pdata)
    phi1 = true_model.get_p()["phi_1"]
    phi2 = true_model.get_p()["phi_2"]
    print(phi1)
    print(phi2)
    print("\x1b[31;m COSINE SIMILARITY = %.3f \x1b[0m" % (1-cosine(phi1,phi2)))
    Y, Z = generate_YZ(X, true_model) # TODO : change distribution.draw to draw samples from the mixture model
    Y_v, Z_v = generate_YZ(X_v, true_model)
    print("Generated %d training data and %d validation data from true model: %s" % \
            (ndata, ndata_v, true_model.get_model_type()))
    true_model.print_p()
    print("")
    # ----------------------------------
    print("Mixture Model")
    print("")
    test_MM_model(ndims=ndims, data_t=data_t, data_v = data_v, hyperp=hyperp,X=X,Y=Y,Z=Z,X_v=X_v,Y_v=Y_v,Z_v=Z_v,true_model=true_model, iters=10)
    print("")
    print("Starting program")
    print("")
    test_LM_model(ndims=ndims, data_t=data_t, data_v = data_v,hyperp=hyperp,X=X,Y=Y,Z=Z,X_v=X_v,Y_v=Y_v,Z_v=Z_v,true_model=true_model, iters=10)
    print("")


def test_LM_model(ndims, data_t, data_v,hyperp,X,Y,Z,X_v,Y_v,Z_v,true_model,iters):
    """
        Example that demonstrates how to call the model.
    """
    print("RUNNING LINEAR MODEL CODE ...... ")
    # get hyperparameters for model
    hyperp = hyperp
    # generate 50 training data and 20 validation data locations of dim=1
    ndata = data_t  # Training Data
    ndata_v = data_v  # Validation Data
    pdata = ndims  # K
    X = X
    X_v = X_v 
    # intialie true model randomly and draw observations from it
    true_model = true_model
    Y, Z = Y,Z # TODO : change distribution.draw to draw samples from the mixture model
    Y_v, Z_v = Y_v, Z_v 
    
    lowest_mse = 999
    params = None
    for i in range(iters):
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
        error = model.get_mse(X_v, Y_v, Z_v)
        if error < lowest_mse:
            lowest_mse = error
            params = fit_params

    print("\x1b[31;m Linear Model LOWEST MSE = %.3f \x1b[0m" % lowest_mse)
    # print("Linear Model MSE : %.3f" % model.get_mse(X_v,Y_v,Z_v))
    print("Lowest MSE Model Params :")
    testmodel = EM_algo_LM(hyperp, ndata=ndata, pdata=pdata)
    testmodel.set_p(params)
    testmodel.print_p()
    testmodel = None

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

def test_MM_model(ndims, data_t, data_v,hyperp,X,Y,Z,X_v,Y_v,Z_v,true_model,iters):
    """
        Example that demonstrates how to call the model.
    """
    
    print("RUNNING MIXTURE MODEL CODE ...... ")
    # get hyperparameters for model
    hyperp = hyperp
    # generate 50 training data and 20 validation data locations of dim=1
    ndata = data_t # Training Data
    ndata_v = data_v # Validation Data
    pdata = ndims # K
    X = X
    X_v = X_v 
    # intialize true model randomly and draw observations from it
    true_model = true_model 
    # true_model.print_p()
    Y, Z = Y,Z # TODO : change distribution.draw to draw samples from the mixture model
    Y_v, Z_v = Y_v,Z_v

    lowest_mse = 999
    params = None
    for i in xrange(iters):
        # generate a model for estimating the parameters of the
        # true model based on the observations (X, Y) we just made
        model = EM_algo_MM(hyperp, X, Y)
        # model.print_p()
        i, logl, r = model.EM_fit()

        print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
                (logl, i, r))
        print("")
        # print("MAP estimate of true model parameters:")
        # model.print_p()
        print("")

        # crossvalidate the estimated model with the validation data
        fit_params = model.get_p()
        model_v = EM_algo_MM(hyperp, X_v, Y_v)
        model_v.set_p(fit_params)
        logl, ll = model_v.logl()
        print("Crossvalidated logl: %.2f" % (logl))
        # print("DEBUG MSE")
        # print zip((Z_v*(X_v.dot(fit_params["phi_1"]))),((1-Z_v)*(X_v.dot(fit_params["phi_2"]))))
        error = model.get_mse(X_v,Y_v,Z_v)
        if error < lowest_mse:
            lowest_mse = error
            params = fit_params
    # print("\x1b[%sm%s\x1b[0m;31;MIXTURE MODEL LOWEST MSE = %.3f" % lowest_mse)
    print("\x1b[31;m Mixture Model LOWEST MSE = %.3f \x1b[0m" % lowest_mse)
    print("Lowest MSE model Params :")
    
    testmodel = EM_algo_MM(hyperp, ndata=ndata, pdata=pdata)
    testmodel.set_p(params)
    testmodel.print_p()
    testmodel = None

    print("TRUE Model:\nParameters:")
    true_model.print_p()
    print("MSE:")
    print("%.3f" % true_model.get_mse(X_v,Y_v,Z_v))

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

