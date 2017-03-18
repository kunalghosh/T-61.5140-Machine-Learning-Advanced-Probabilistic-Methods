# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: kunal.ghosh@aalto.fi, 2016
# Author: Jussi.k.ojala@aalto.fi, 2016
# Author: antti.kangasraasio@aalto.fi, 2016

from numpy import outer, eye, ones, zeros, diag, log, sqrt, exp, pi, size, shape, sum, transpose, logaddexp
from numpy.linalg import inv, solve
from numpy.random import multivariate_normal as mvnormal, normal, gamma, beta, binomial
from scipy.special import gammaln
from scipy.stats import norm

from em_algo import EM_algo

class EM_algo_MM(EM_algo):
    """
        A mixture of two linear models.
    """
    def reset(self):
        """
            Reset priors and draw parameter estimates from prior.
        """
        # priors
        self.lbd_phi0       = self.h["lbd_phi0"]
        self.alpha_s20      = self.h["alpha_s20"]
        self.beta_s20       = self.h["beta_s20"]
        self.alpha_w       = self.h["alpha_w0"]
        self.beta_w        = self.h["beta_w0"]
        self.sigma_phi0     = eye(self.pdata) * self.h["lbd_phi0"]
        self.sigma_phi0_inv = eye(self.pdata) / self.h["lbd_phi0"]
        self.mu_phi0         = ones(self.pdata) * self.h["mu_phi0"]

        # initial parameter estimates drawn from prior
        self.p             = dict()
        self.p["sigma2_1"] = 1.0 / gamma(self.alpha_s20 , (1.0 / self.beta_s20) )  # inverse gamma
        self.p["phi_1"]    = mvnormal(self.mu_phi0 , self.p["sigma2_1"] * self.sigma_phi0)
        self.p["sigma2_2"] = 1.0 / gamma(self.alpha_s20 , (1.0 / self.beta_s20))  # inverse gamma
        self.p["phi_2"]    = mvnormal(self.mu_phi0, self.p["sigma2_2"] * self.sigma_phi0)
        self.p["w"]        = beta(self.alpha_w, self.beta_w)  # beta distribution
        
#     def reset(self):
#         """
#         Reset priors and draw parameter estimates from prior.
#         """
#         # priors
#         self.lbd_phi0 = self.h["lbd_phi0"]  # lambda
#         self.alpha_s20 = self.h["alpha_s20"]  # alpha sigma sq
#         self.beta_s20 = self.h["beta_s20"]  # beta sigma sq
#         self.sigma_phi0 = eye(self.pdata) * self.h["lbd_phi0"]
#         self.sigma_phi0_inv = eye(self.pdata) / self.h["lbd_phi0"]
#         self.mu_phi0 = ones(self.pdata) * self.h["mu_phi0"]
#         self.alpha_w = self.h["alpha_w0"]
#         self.beta_w = self.h["beta_w0"]
# 
#         # initial parameter estimates drawn from prior
#         self.p = dict()
#         self.p["sigma2_1"] = 1.0 / gamma(self.alpha_s20,
#                                          1.0 / self.beta_s20)  # inverse gamma # possible numerical problems ?
#         self.p["sigma2_2"] = 1.0 / gamma(self.alpha_s20,
#                                          1.0 / self.beta_s20)  # inverse gamma # possible numerical problems ?
#         self.p["phi_1"] = mvnormal(self.mu_phi0, self.p["sigma2_1"] * self.sigma_phi0)
#         self.p["phi_2"] = mvnormal(self.mu_phi0, self.p["sigma2_2"] * self.sigma_phi0)
#         self.p["w"]     = beta(self.alpha_w, self.beta_w)

#     def draw(self, item):
#         """
#             Draw a data sample from the current predictive distribution.
#             Returns the y-value (and a constant z-value for compatibility)
#         """
#         # draw a z
#         z = binomial(n=1,p=self.p["w"])
#         #print(z)
#         if(z==0):
#             mean = float(item.dot(self.p["phi_1"]))
#             std = sqrt(self.p["sigma2_1"])  # sigma square
#         else:
#             mean = float(item.dot(self.p["phi_2"]))
#             std = sqrt(self.p["sigma2_2"])  # sigma square
#      
#         return normal(mean, std), z # z is in {0,1}

    def draw(self, item):
        """
            Draw a data sample from the current predictive distribution.
        """
        zt = binomial(1, self.p["w"])
        mean_n1 = float(item.dot(self.p["phi_1"]))
        std_n1 = sqrt(self.p["sigma2_1"])
        mean_n2 = float(item.dot(self.p["phi_2"]))
        std_n2 = sqrt(self.p["sigma2_2"])
        # y = normal(mean_n1, std_n1) ** zt + normal(mean_n2, std_n2) ** (1 - zt)
        y = normal(mean_n1, std_n1) * zt + normal(mean_n2, std_n2) * (1 - zt)
        return y, zt
    
    
    def logl(self):
        """
            Calculates the full log likelihood for this model.
            Returns the logl (and the values of each term for debugging purposes)
    
            We should be able to calculate log-likelihood for any arbitrary X
            so we should not use the Z values, we obtained while generating the
            dataset.
        """
    
        ll = zeros(14)
        phie_1 = self.p["phi_1"] - self.mu_phi0
        phie_2 = self.p["phi_2"] - self.mu_phi0
        w = self.p["w"]
        normpdf = norm.pdf
    
        # p(y)

        # ll[0] = sum([log(w * normpdf(yt, loc=xt_phi1, scale=self.p["sigma2_1"])
        #              + (1-w) * normpdf(yt, loc=xt_phi2, scale=self.p["sigma2_2"])) for (xt_phi1, xt_phi2, yt) in zip(self.X.dot(self.p["phi_1"]),
        #                                                                              self.X.dot(self.p["phi_1"]),
        #                                                                              self.Y)])
        ll[0] = sum([logaddexp(log(w) - 0.5*log(2*pi*self.p["sigma2_1"]) - 0.5 * (1/self.p["sigma2_1"]) * (xt_phi1 - yt)**2
                     ,log(1-w) - 0.5*log(2*pi*self.p["sigma2_2"]) - 0.5 * (1/self.p["sigma2_2"]) * (xt_phi2 - yt)**2) for (xt_phi1, xt_phi2, yt) 
                     in zip(self.X.dot(self.p["phi_1"]),self.X.dot(self.p["phi_1"]),self.Y)])
        # p(phi_1)
        ll[1] = - 0.5 * log(2 * pi * self.lbd_phi0 * self.p["sigma2_1"]) * self.pdata
        ll[2] = - 0.5 * phie_1.T.dot(phie_1) / (self.lbd_phi0 * self.p["sigma2_1"])
    
        # p(phi_2)
        ll[3] = - 0.5 * log(2 * pi * self.lbd_phi0 * self.p["sigma2_2"]) * self.pdata
        ll[4] = - 0.5 * phie_2.T.dot(phie_2) / (self.lbd_phi0 * self.p["sigma2_2"])
    
        # Common code for p(sigma2_1) and p(sigma2_2)
        const1 = self.alpha_s20 * log(self.beta_s20)
        const2 = - gammaln(self.alpha_s20)
    
        # p(sigma2_1)
        ll[5] = const1
        ll[6] = const2
        ll[7] = - (self.alpha_s20 + 1.0) * log(self.p["sigma2_1"])
        ll[8] = - self.beta_s20 / self.p["sigma2_1"]
    
        # p(sigma2_2)
        ll[9] = const1
        ll[10] = const2
        ll[11] = - (self.alpha_s20 + 1.0) * log(self.p["sigma2_2"])
        ll[12] = - self.beta_s20 / self.p["sigma2_2"]
        return sum(ll), ll    
    
    def EM_iter(self):
        """
        Executes a single round of EM updates for this model.

        Has checks to make sure that updates increase logl and
        that parameter values stay in sensible limits.
        """
        #update of gamma i.e expectation of z i.e. expectation of latent variables
        mean_Z1=self.X.dot(self.p["phi_1"])
        mean_Z2=self.X.dot(self.p["phi_2"])
        #pdf_t_z1=norm.pdf(self.Y,mean_Z1,self.p["sigma2_1"]*ones(self.Y.shape))

        # pdf_t_z1=norm.pdf(mean_Z1,self.Y,sqrt(self.p["sigma2_1"]*ones(self.Y.shape)))
        pdf_t_z1 = norm(mean_Z1,sqrt(self.p["sigma2_1"] * ones(self.Y.shape))).pdf(self.Y)

        #pdf_t_z1=norm.pdf(mean_Z1,self.Y,self.p["sigma2_1"]*ones(self.Y.shape))
        #pdf_t_z0=norm.pdf(self.Y,mean_Z2,self.p["sigma2_2"]*ones(self.Y.shape))

        # pdf_t_z0=norm.pdf(mean_Z2,self.Y,sqrt(self.p["sigma2_2"]*ones(self.Y.shape)))
        pdf_t_z0 = norm(mean_Z2,sqrt(self.p["sigma2_2"] * ones(self.Y.shape))).pdf(self.Y)

        #pdf_t_z0=norm.pdf(mean_Z2,self.Y,self.p["sigma2_2"]*ones(self.Y.shape))
        gamma_Z=(self.p["w"]*pdf_t_z1)/(self.p["w"]*pdf_t_z1+(1-self.p["w"])*pdf_t_z0)
        #print(gamma_Z)
        #print(shape(gamma_Z))
  
        # update of phi-1
        myTSigma    = self.mu_phi0.T.dot(self.sigma_phi0_inv)
        SumgammayxT_S = sum(self.X.T*self.Y * gamma_Z ,axis=1)+ myTSigma 
        SumgammaxxT_S     = (gamma_Z*self.X.T).dot(self.X) +self.sigma_phi0_inv
        
        #print( self.p["phi_1"] )
        self.p["phi_1"] = SumgammayxT_S.dot(inv(SumgammaxxT_S ))
        #print( self.p["phi_1"] )
        
        #self.assert_logl_increased("phi_1 update")
        #print("---------")
        SumgammayxT_S2 = sum(self.X.T*self.Y * (1-gamma_Z) ,axis=1)+ myTSigma 
        SumgammaxxT_S2     = ((1-gamma_Z)*self.X.T).dot(self.X) +self.sigma_phi0_inv
        
        #print( self.p["phi_2"] )
        self.p["phi_2"] = SumgammayxT_S2.dot(inv(SumgammaxxT_S2 ))
        #print( self.p["phi_2"] )
        #self.assert_logl_increased("phi_2 update")
        # sigma2
        #print('---A---')
        
        phiSphi = ((self.mu_phi0-self.p["phi_1"]).T.dot(self.sigma_phi0_inv)).dot((self.mu_phi0-self.p["phi_1"])) 
        #print(shape(phiSphi))
        #print(phiSphi)
        err22= (self.X.dot(self.p["phi_1"])-self.Y)**2
        sum_err = sum(err22*gamma_Z)
        num=phiSphi+2*self.beta_s20+sum_err 
        den=self.pdata+2*(self.alpha_s20+1)+sum(gamma_Z)
        #print(num)
        #print(den)
        #print(self.p["sigma2_1"])
        self.p["sigma2_1"]=num/den
        #print(self.p["sigma2_1"])
        if self.p["sigma2_1"] < 0.0 :
            raise ValueError("sigma2_1 < 0.0")
        #self.assert_logl_increased("sigma2_1 update")   
        #print('---B---')
        
        phiSphi2 = ((self.mu_phi0-self.p["phi_2"]).T.dot(self.sigma_phi0_inv)).dot((self.mu_phi0-self.p["phi_2"])) 
        #print(shape(phiSphi))
        #print(phiSphi)
        err222= (self.X.dot(self.p["phi_2"])-self.Y)**2
        sum_err2 = sum(err222*(1-gamma_Z))
        num2=phiSphi2+2*self.beta_s20+sum_err2 
        den2=self.pdata+2*(self.alpha_s20+1)+sum((1-gamma_Z))
        #print(num2)
        #print(den2)
        #print(self.p["sigma2_2"])
        self.p["sigma2_2"]=num2/den2
        #print(self.p["sigma2_2"])

        if self.p["sigma2_2"] < 0.0 :
            raise ValueError("sigma2_2 < 0.0")        
        #self.assert_logl_increased("sigma2_2 update")   
        #print(self.p["w"])
        self.p["w"]=(self.alpha_w-1+sum(gamma_Z))/(self.alpha_w+self.beta_w-2+self.ndata)
        #print(self.p["w"])
        #self.assert_logl_increased("w update") 
 
 
    def print_p(self):  # TODO : Change this to print out all the model parameters.
        """
            Prints the model parameters, one at each line.
        """
        print("w        : %s" % (self.p["w"]))
        print("phi_1    : %s" % (self.pretty_vector(self.p["phi_1"])))
        print("phi_2    : %s" % (self.pretty_vector(self.p["phi_2"])))
        print("sigma2_1 : %.3f" % (self.p["sigma2_1"]))
        print("sigma2_2 : %.3f" % (self.p["sigma2_2"]))    
   
    def get_model_type(self):
        return "Mixture model"

    def get_mse(self,X_v,Y_v,Z_v):
        return self.mse((Z_v*X_v.dot(self.get_p()["phi_1"]))+((1-Z_v)*X_v.dot(self.get_p()["phi_2"])), Y_v)
