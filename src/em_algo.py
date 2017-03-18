# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: antti.kangasraasio@aalto.fi, 2016

import copy
import numpy as np

class EM_algo():
    """
        A superclass for different EM-fitted models.
    """

    def __init__(self, hyperparams, X=None, Y=None, ndata=0, pdata=0):
        """
            Initialize model based either on given data (X, Y) or
            on given data dimensionality (ndata, pdata).
        """
        if X != None and Y != None:
            self.X = X
            self.Y = Y
            self.ndata = len(self.X)
            self.pdata = len(self.X[0])
        if ndata and pdata:
            self.X = None
            self.Y = None
            self.ndata = ndata
            self.pdata = pdata
        self.h = hyperparams
        self.p = dict() # model parameters
        self.loglVals = []
        self.reset()
        if X != None and Y != None:
            self.current_logl, self.cll = self.logl()


    def reset(self):
        """
            Reset priors and draw parameter estimates from prior.
        """
        raise NotImplementedError("Subclass implements")


    def draw(self, item):
        """
            Draw a data sample from the current predictive distribution.
            Returns the drawn y and z-values.
        """
        raise NotImplementedError("Subclass implements")


    def logl(self):
        """
            Calculates the full log likelihood for this model.
            Returns the logl (and the values of each term for debugging purposes)
        """
        raise NotImplementedError("Subclass implements")


    def EM_iter(self):
        """
            Executes a single round of EM updates for this model.
        """
        raise NotImplementedError("Subclass implements")


    def EM_fit(self, alim=1e-10, maxit=1e4):
        """
            Calls the EM_iter repeatedly until the log likelihood
            of the model increases less than 'alim' in absolute
            value or after 'maxit' iterations have been done.

            Returns the number of EM-iterations, final log likelihood
            value and a string that explains the end condition.
        """
        self.loglVals = []
        logl, ll = self.logl()
        for i in range(int(maxit)):
            self.EM_iter()
            logl2, ll2 = self.logl()
            self.loglVals.append(logl2)
            adiff = abs(logl2 - logl)
            if adiff < alim:
                return i+1, logl2, "alim"
            logl = logl2

        return maxit, logl2, "maxit"


    def assert_logl_increased(self, event):
        """
            Checks that the log likelihood increased since model
            initialization or the time this function was last called.
        """
        newlogl, ll = self.logl()
        if self.current_logl - newlogl > 1e-3:
            self.debug_logl(self.cll, ll)
            raise ValueError("logl increased after %s" % (event))
        self.current_logl, self.cll = newlogl, ll


    def get_p(self):
        """
            Returns a copy of the model parameters.
        """
        return copy.deepcopy(self.p)

    def get_loglVals(self):
        return self.loglVals


    def set_p(self, p):
        """
            Sets the model parameters.
        """
        self.p = p.copy()


    def print_p(self):
        """
            Prints the model parameters, one at each line.
        """
        for k, v in self.p.items():
            print("%s = %s" % (k, v))


    def pretty_vector(self, x):
        """
            Returns a formatted version of a vector.
        """
        s = ["("]
        s.extend(["%.2f, " % (xi) for xi in x[:-1]])
        s.append("%.2f)" % (x[-1]))
        return "".join(s)


    def debug_logl(self, ll1, ll2):
        """
            Prints an analysis of the per-term change in
            log likelihood from ll1 to ll2.
        """
        print("Logl      before     after")
        for v1, v2, i in zip(ll1, ll2, range(len(ll1))):
            if v1 > v2:
                d = ">"
            elif v2 > v1:
                d = "<"
            else:
                d = "="
            print("Term %02d: %7.3f %s %7.3f" % (i, v1, d, v2))
        print("Total    %7.3f   %7.3f" % (sum(ll1), sum(ll2)))

    def get_model_type(self):
        """
        Returns linear or Mixture Model
        """
        return None

    def mse(self,a,b):
        a = np.asarray(a)
        b = np.asarray(b)
        # print("MSE:")
        # pprint.pprint(zip(a,b))
        mse = np.mean((a-b)**2)
        # print("%.3f" % mse)
        return mse

