import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, digamma, polygamma
from scipy.optimize import newton
import cython
cimport numpy as np

ctypedef np.float64_t FLOAT64

#The model core object contains all the learned parameters that are optimized
#during a fit. The StudentMixture class creates and stores a StudentMixtureModelCore
#object and performs all fthe necessary checks on the validity of the data and
#the fitted model before placing any calls to the model core. When fitting with
#multiple restarts, the StudentMixture class can create more than one model core
#and save the best one.
cdef class StudentMixtureModelCore():
    cdef public np.ndarray mix_weights_
    cdef public np.ndarray loc_
    cdef public np.ndarray scale_
    cdef public np.ndarray scale_cholesky_
    cdef public np.ndarray scale_inv_cholesky_
    cdef public bint converged_
    cdef public int n_iter_
    cdef public np.ndarray df_
    cdef public int random_state
    cdef public bint fixed_df
    cdef public float dim
    cdef public float reg_covar
    cdef public int n_components

    def __init__(self, np.ndarray[FLOAT64, ndim=2] X,
                        float start_df, float tol,
                        int n_components, float reg_covar, int max_iter,
                        bint verbose,
                        int random_state, bint fixed_df = True):
        self.mix_weights_ = np.empty((X.shape[1]))
        self.dim = float(X.shape[1])
        self.loc_ = np.empty((n_components, X.shape[1]))
        self.scale_ = np.empty((X.shape[1], X.shape[1], n_components))
        self.scale_cholesky_ = np.empty((X.shape[1], X.shape[1], n_components))
        self.scale_inv_cholesky_ = np.empty((X.shape[1], X.shape[1], n_components))
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = np.full((n_components), start_df, dtype = np.float64)
        self.reg_covar = reg_covar
        self.fixed_df = fixed_df
        self.random_state = random_state
        self.n_components = n_components
        self.fit(X, tol, max_iter, verbose)


    #Function for fitting the model to user supplied data. The user-specified
    #fitting parameters (df, n_components, max_iter etc) are stored in and 
    #supplied by the mixt_model class and passed to its model core when it does a
    #fit.
    cdef fit(self, np.ndarray[FLOAT64, ndim=2] X, float tol, int max_iter, bint verbose):
        self.initialize_params(X)
        lower_bound = -np.inf
        cdef int i = 0
        cdef np.ndarray[FLOAT64, ndim=2] resp = np.empty((X.shape[0], self.n_components))
        cdef np.ndarray[FLOAT64, ndim=2] maha_dist = np.empty((X.shape[0], self.n_components))
        cdef float current_bound = 0
        cdef float change = 0
        for i in range(max_iter):
            resp, u, current_bound = self.Estep(X, maha_dist)
            self.Mstep(X, resp, u, reg_covar)
            change = current_bound - lower_bound
            if abs(change) < tol:
                self.converged_ = True
                break
            lower_bound = current_bound
            if verbose:
                #print("Current bound: %s"%current_bound)
                print("Change: %s"%change)
        return current_bound


    #The e-step in mixture fitting. Calculates responsibilities for each datapoint
    #and E[u] for the formulation of the t-distribution as a 
    #Gaussian scale mixture. It returns the responsibilities (NxK array),
    #E[u] (NxK array), the mahalanobis distance (NxK array), and the log
    #of the determinant of the scale matrix.
    cdef Estep(self, np.ndarray[FLOAT64, ndim=2] &X,
            np.ndarray[FLOAT64, ndim=2] &maha_dist):
        cdef np.ndarray[FLOAT64, ndim=1] log_prob_norm = np.empty((X.shape[0]))
        cdef np.ndarray[FLOAT64, ndim=2] u = np.empty((X.shape[0], self.n_components))
        cdef np.ndarray[FLOAT64, ndim=2] resp = np.empty((X.shape[0], self.n_components))
        self.maha_distance(X, maha_dist)
        resp = self.get_training_loglik(X, maha_dist)
        resp = resp + np.log(np.clip(self.mix_weights_, a_min=1e-9, a_max=None))[np.newaxis,:]
        log_prob_norm = logsumexp(resp, axis=1)
        with np.errstate(under="ignore"):
            resp = np.exp(resp - log_prob_norm[:, np.newaxis])
        u = (self.df_[np.newaxis,:] + X.shape[1]) / (self.df_[np.newaxis,:] + maha_dist)
        return resp, u, np.mean(log_prob_norm)

    #The M-step in mixture fitting. Calculates the ML value for the scale matrix
    #location and mixture weights (we are using fixed df, otherwise df would also
    #have to be optimized). We calculate self.loc_ -- the mean, resulting array
    #is KxP for K components and P dimensions; self.scale_, array is PxPxK;
    #self.scale_cholesky_, the cholesky decomposition of the scale matrix; and
    #self.scale_inv_cholesky, the cholesky decomposition of the precision
    #matrix (also self.mix_weights_, the component mixture weights).
    cdef Mstep(self, np.ndarray[FLOAT64, ndim=2] &X, np.ndarray[FLOAT64, ndim=2] &resp,
                np.ndarray[FLOAT64, ndim=2] &u, float &reg_covar):
        cdef np.ndarray[FLOAT64, ndim=2] ru = np.empty((resp.shape[0], resp.shape[1]))
        cdef np.ndarray[FLOAT64, ndim=1] resp_sum = np.empty((resp.shape[1]))
        cdef int i = 0
        self.mix_weights_ = np.mean(resp, axis=0)
        ru = resp * u
        self.loc_ = np.dot((ru).T, X)
        resp_sum = np.sum(ru, axis=0) + 10 * np.finfo(resp.dtype).eps
        self.loc_ = self.loc_ / resp_sum[:,np.newaxis]
        for i in range(self.mix_weights_.shape[0]):
            scaled_x = X - self.loc_[i,:][np.newaxis,:]
            self.scale_[:,:,i] = np.dot((ru[:,i:i+1] * scaled_x).T,
                            scaled_x) / resp_sum[i]
            self.scale_[:,:,i].flat[::X.shape[1] + 1] += reg_covar
            self.scale_cholesky_[:,:,i] = np.linalg.cholesky(self.scale_[:,:,i])
        #For mahalanobis distance we really need the cholesky decomposition of
        #the precision matrix (inverse of scale), but do not want to take 
        #the inverse directly to avoid possible numerical stability issues.
        #We get what we want using the cholesky decomposition of the scale matrix
        #from the following function call.
        self.get_scale_inv_cholesky()
        if self.fixed_df == False:
            self.optimize_df(X, resp, u)


    #Optimizes the df parameter using Newton Raphson.
    cdef optimize_df(self, np.ndarray[FLOAT64, ndim=2] X, np.ndarray[FLOAT64, ndim=2] resp,
                np.ndarray[FLOAT64, ndim=2] u):
        cdef int i = 0
        for i in range(self.mix_weights_.shape[0]):
            self.df_[i] = newton(self.dof_first_deriv, x0 = self.df_[i],
                                 fprime = self.dof_second_deriv,
                                 fprime2 = self.dof_third_deriv,
                                 args = (u, resp, X.shape[1], i),
                                 full_output = False, disp=False, tol=1e-3)

    # First derivative of the complete data log likelihood w/r/t df.
    def dof_first_deriv(self, dof, u, resp, dim, i):
        grad = 1.0 - digamma(dof * 0.5) + np.log(0.5 * dof)
        grad += (1 / resp[:,i].sum(axis=0)) * (resp[:,i] * (np.log(u[:,i]) - u[:,i])).sum(axis=0)
        return grad + digamma(0.5 * (self.df_[i] + dim)) - np.log(0.5 * (self.df_[i] + dim))

    #Second derivative of the complete data log likelihood w/r/t df.
    def dof_second_deriv(self, dof, u, resp, dim, i):
        return -0.5 * polygamma(1, 0.5 * dof) + 1 / dof

    #Third derivative of the complete data log likelihood w/r/t df.
    def dof_third_deriv(self, dof, u, resp, dim, i):
        return -0.25 * polygamma(2, 0.5 * dof) - 1 / (dof**2)



    #Calculates the mahalanobis distance for X to all components. Returns an
    #array of dim N x K for N datapoints, K mixture components.
    cdef maha_distance(self, np.ndarray[FLOAT64, ndim=2] &X,
                    np.ndarray[FLOAT64, ndim=2] &maha_dist):
        cdef int i = 0
        cdef np.ndarray[FLOAT64, ndim=2] y = np.empty((X.shape[0], X.shape[1]))
        for i in range(self.mix_weights_.shape[0]):
            y = np.dot(X, self.scale_inv_cholesky_[:,:,i])
            y = y - np.dot(self.loc_[i,:], self.scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            maha_dist[:,i] = np.sum(y**2, axis=1)


    #Gets the inverse of the cholesky decomposition of the scale matrix.
    cdef get_scale_inv_cholesky(self):
        cdef int i = 0
        for i in range(self.mix_weights_.shape[0]):
            self.scale_inv_cholesky_[:,:,i] = solve_triangular(self.scale_cholesky_[:,:,i],
                    np.eye(self.scale_cholesky_.shape[0]), lower=True).T

    #Calculates log p(X | theta) using the mixture components formulated as
    #multivariate t-distributions. This function is the same as get_loglik with a
    #few key differences -- it is cdef and takes arguments by reference, so it is used
    #only during training, when speed is critical, rather than during prediction when
    #it is not.
    cdef get_training_loglik(self, np.ndarray[FLOAT64, ndim=2] &X,
                    np.ndarray[FLOAT64, ndim=2] &maha_dist):
        cdef np.ndarray[FLOAT64, ndim=2] y = np.empty((maha_dist.shape[0], maha_dist.shape[1]))
        cdef np.ndarray[FLOAT64, ndim=1] const_term = np.empty((self.df_.shape[0]))
        cdef int i = 0
        y = 1 + maha_dist / self.df_[np.newaxis,:]
        y = -0.5*(self.df_[np.newaxis,:] + self.dim)*np.log(y)
        const_term = gammaln(0.5*(self.df_ + self.dim)) - gammaln(0.5*self.df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(self.df_) + np.log(np.pi))
        scale_logdet = [np.sum(np.log(np.diag(self.scale_cholesky_[:,:,i])))
                        for i in range(self.mix_weights_.shape[0])]
        scale_logdet = np.asarray(scale_logdet)
        return -scale_logdet[np.newaxis,:] + const_term[np.newaxis,:] + y


    #This function fulfills the same purpose as maha_distance but is used for making
    #predictions (it is not cdef).
    def offline_maha_distance(self, np.ndarray[FLOAT64, ndim=2] X):
        cdef int i = 0
        cdef np.ndarray[FLOAT64, ndim=2] y = np.empty((X.shape[0], X.shape[1]))
        cdef np.ndarray[FLOAT64, ndim=2] maha_dist = np.empty((X.shape[0], self.n_components))
        for i in range(self.n_components):
            y = np.dot(X, self.scale_inv_cholesky_[:,:,i])
            y = y - np.dot(self.loc_[i,:], self.scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            maha_dist[:,i] = np.sum(y**2, axis=1)
        return maha_dist


    #Calculates log p(X | theta) using the mixture components formulated as 
    #multivariate t-distributions. This function is not cdef and so is only used
    #for making predictions (the faster get_training_loglik is used during training).
    def get_loglik(self, X):
        maha_dist = 1 + self.offline_maha_distance(X) / self.df_[np.newaxis,:]
        maha_dist = -0.5*(self.df_[np.newaxis,:] + self.dim)*np.log(maha_dist)
        const_term = gammaln(0.5*(self.df_ + self.dim)) - gammaln(0.5*self.df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(self.df_) + np.log(np.pi))
        scale_logdet = [np.sum(np.log(np.diag(self.scale_cholesky_[:,:,i])))
                        for i in range(self.mix_weights_.shape[0])]
        scale_logdet = np.asarray(scale_logdet)
        loglik = -scale_logdet[np.newaxis,:] + const_term + maha_dist
        return loglik


    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center to get the starting locations.
    cdef initialize_params(self, np.ndarray[FLOAT64, ndim=2] &X):
        cdef float ncomp = float(self.n_components)
        cdef int i = 0
        cdef int next_center_id = 0
        cdef np.ndarray[FLOAT64, ndim=1] dist_arr = np.empty((X.shape[0]))
        cdef np.ndarray[FLOAT64, ndim = 2] distmat = np.empty((X.shape[0], 
                      max([self.n_components - 1, 1]) ))
        cdef np.ndarray[FLOAT64, ndim=1] min_dist = np.empty((X.shape[0]))
        np.random.seed(self.random_state)
        self.loc_[0,:] = X[np.random.randint(0, X.shape[0]-1), :]
        self.mix_weights_.fill(1 / ncomp)
        if self.n_components > 1:
            for i in range(1, self.n_components):
                distmat[:,i-1] = np.sum((X - self.loc_[i-1])**2, axis=1)
                min_dist = np.min(distmat[:,:i], axis=1)
                min_dist = min_dist / np.sum(min_dist)
                next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
                self.loc_[i,:] = X[next_center_id,:]

        #For initialization, set all covariance matrices to I.
        for i in range(self.n_components):
            self.scale_[:,:,i] = np.eye(X.shape[1])
        self.scale_cholesky_ = np.copy(self.scale_)
        self.scale_inv_cholesky_ = np.copy(self.scale_)


    '''The remaining functions are used for prediction and do not need static typing for
    speed optimization'''

    #Returns log p(X | theta) + log mix_weights.
    def get_weighted_loglik(self, X, precalc_dist=None):
        loglik = self.get_loglik(X, precalc_dist)
        return loglik + np.log(self.mix_weights_)[np.newaxis,:]

    #Returns the probability that the input data belongs to each component. Used
    #for making predictions.
    def get_component_probability(self, X):
        weighted_loglik = self.get_weighted_loglik(X)
        with np.errstate(under="ignore"):
            loglik = weighted_loglik - logsumexp(weighted_loglik, axis=1)[:,np.newaxis]
        return np.exp(loglik)

    #Gets an average (over samples) log likelihood (useful for AIC, BIC).
    def score(self, X):
        return np.mean(self.score_samples(X))

    #Gets a per sample net log likelihood.
    def score_samples(self, X):
        return logsumexp(self.get_weighted_loglik(X), axis=1)

       
    #Returns the dimensionality of the training data.
    def get_data_dim(self):
        return self.loc_.shape[1]

    #Checks whether model converged.
    def check_modelcore_convergence(self):
        return self.converged_

    #Returns the model locations.
    def get_location(self):
        return self.loc_

    #Returns the model scale matrices.
    def get_scale(self):
        return self.scale_

    #Returns the mixture weights.
    def get_mix_weights(self):
        return self.mix_weights_

    #Gets the number of parameters (useful for AIC & BIC calculations). Note that df is only
    #treated as a parameter if df is not fixed.
    def get_num_parameters(self):
        num_parameters = self.mix_weights_.shape[0] + self.loc_.shape[0] * self.loc_.shape[1]
        num_parameters += 0.5 * self.scale_.shape[0] * (self.scale_.shape[1] + 1) * self.scale_.shape[2]
        if self.fixed_df:
            return num_parameters
        else:
            return num_parameters + self.df_.shape[0]
