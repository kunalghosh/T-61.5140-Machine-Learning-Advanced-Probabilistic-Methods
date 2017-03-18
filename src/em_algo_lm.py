# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: kunal.ghosh@aalto.fi, 2016
# Author: Jussi.k.ojala@aalto.fi, 2016
# Author: antti.kangasraasio@aalto.fi, 2016

from numpy import outer, eye, ones, zeros, log, sqrt, exp, pi
from numpy.linalg import inv, solve
from numpy.random import multivariate_normal as mvnormal, normal, gamma, beta, binomial
from scipy.special import gammaln

from em_algo import EM_algo

class EM_algo_LM(EM_algo):
    """
        A linear gaussian model.
    """

    def reset(self):
        """
            Reset priors and draw parameter estimates from prior.
        """
        # priors
        self.lbd_phi0       = self.h["lbd_phi0"] # lambda
        self.alpha_s20      = self.h["alpha_s20"] # alpha sigma sq
        self.beta_s20       = self.h["beta_s20"] # beta sigma sq
        self.sigma_phi0     = eye(self.pdata) * self.h["lbd_phi0"]
        self.sigma_phi0_inv = eye(self.pdata) / self.h["lbd_phi0"]
        self.mu_phi0        = ones(self.pdata) * self.h["mu_phi0"]

        # initial parameter estimates drawn from prior
        self.p           = dict()
        self.p["sigma2"] = 1.0 / gamma(self.alpha_s20, 1.0 / self.beta_s20) # inverse gamma # possible numerical problems ?
        self.p["phi"]    = mvnormal(self.mu_phi0, self.p["sigma2"] * self.sigma_phi0)


    def draw(self, item):
        """
            Draw a data sample from the current predictive distribution.
            Returns the y-value (and a constant z-value for compatibility)
        """
        mean = float(item.dot(self.p["phi"]))
        std  = sqrt(self.p["sigma2"]) # sigma square
        return normal(mean, std), 1 # means the data comes from the first component (z_t1=1) always


    def logl(self):
        """
            Calculates the full log likelihood for this model.
            Returns the logl (and the values of each term for debugging purposes)
        """
        ll    = zeros(8)
        phie  = self.p["phi"] - self.mu_phi0
        err   = (self.X.dot(self.p["phi"]) - self.Y) ** 2
        # p(y)
        ll[0] = - 0.5 * log(2 * pi * self.p["sigma2"]) * self.ndata
        ll[1] = sum(- 0.5 * err / self.p["sigma2"])
        # p(phi)
        ll[2] = - 0.5 * log(2 * pi * self.lbd_phi0 * self.p["sigma2"]) * self.pdata
        ll[3] = - 0.5 * phie.T.dot(phie) / (self.lbd_phi0 * self.p["sigma2"])
        # p(sigma2)
        ll[4] = self.alpha_s20 * log(self.beta_s20)
        ll[5] = - gammaln(self.alpha_s20)
        ll[6] = - (self.alpha_s20 + 1.0) * log(self.p["sigma2"])
        ll[7] = - self.beta_s20 / self.p["sigma2"]
        return sum(ll), ll


    def EM_iter(self): # TODO : Change this to get updates of all model parameters
        """
            Executes a single round of EM updates for this model.

            Has checks to make sure that updates increase logl and
            that parameter values stay in sensible limits.
        """
        # phi
        sumxx         = self.X.T.dot(self.X)
        sumxy         = self.X.T.dot(self.Y)
        sigma_mu      = self.sigma_phi0_inv.dot(self.mu_phi0)
        sigma_phi_inv = self.sigma_phi0_inv + sumxx
        self.p["phi"] = solve(sigma_phi_inv, sigma_mu + sumxy)
        self.assert_logl_increased("phi update")

        # sigma2
        phie = (self.p["phi"] - self.mu_phi0) ** 2
        err  = (self.X.dot(self.p["phi"]) - self.Y) ** 2
        num  = self.beta_s20 + 0.5 * sum(err) + 0.5 * sum(phie) / self.lbd_phi0
        den  = self.alpha_s20 + 1.0 + 0.5 * (self.ndata + self.pdata)
        self.p["sigma2"] = num / den
        if self.p["sigma2"] < 0.0:
            raise ValueError("sigma2 < 0.0")
        self.assert_logl_increased("sigma2 update")


    def print_p(self): # TODO : Change this to print out all the model parameters.
        """
            Prints the model parameters, one at each line.
        """
        print("phi    : %s" % (self.pretty_vector(self.p["phi"])))
        print("sigma2 : %.3f" % (self.p["sigma2"]))

    def get_model_type(self):
        return "Linear model"

    def get_mse(self,X_v,Y_v,Z_v):
        return self.mse(X_v.dot(self.get_p()["phi"]), Y_v)
