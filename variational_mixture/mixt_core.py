import numpy as np, random
from scipy.linalg import solve_triangular, cholesky
from scipy.special import gammaln, logsumexp
from copy import copy
from scipy.stats import multivariate_t
from scipy.spatial.distance import mahalanobis

#The model core object contains all the learned parameters that are optimized
#during a fit. The StudentMixture class creates and stores a StudentMixtureModelCore
#object and performs all the necessary checks on the validity of the data and
#the fitted model before placing any calls to the model core. When fitting with
#multiple restarts, the StudentMixture class can create more than one model core
#and save the best one.
class StudentMixtureModelCore():

    def __init__(self):
        self.mix_weights_ = None
        self.loc_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None

    #Function for fitting the model to user supplied data. The user-specified
    #fitting parameters (df, n_components, max_iter etc) are stored in and 
    #supplied by the mixt_model class and passed to its model core when it does a
    #fit.
    def fit(self, X, df, tol, n_components, reg_covar, max_iter):
        self.df_ = df
        self.initialize_params(X, n_components)
        lower_bound = -np.inf
        for i in range(max_iter):
            resp, u, maha_dist, current_bound = self.Estep(X)
            current_bound = np.mean(np.log(resp))
                            #self.get_loglik_lower_bound(X, resp, u,
                            #maha_dist, scale_logdet)
            self.Mstep(X, resp, u, reg_covar)
            change = current_bound - lower_bound
            if abs(change) < tol:
                self.converged_ = True
                break
            lower_bound = copy(current_bound)
        s = self.loc_[:,0]
        return current_bound


    #The e-step in mixture fitting. Calculates responsibilities for each datapoint
    #and E[u] for the formulation of the t-distribution as a 
    #Gaussian scale mixture. It returns the responsibilities (NxK array),
    #E[u] (NxK array), the mahalanobis distance (NxK array), and the log
    #of the determinant of the scale matrix.
    def Estep(self, X):
        maha_dist = self.maha_distance(X)
        logprobs, scale_logdet = self.get_logprob_tdist(X, maha_dist)
        current_bound = np.mean(logsumexp(logprobs, axis=1))
        logprobs = logprobs - np.max(logprobs, axis=1)[:,np.newaxis]
        with np.errstate(under="ignore"):
            probs = np.exp(logprobs) / np.sum(np.exp(logprobs), axis=1)[:,np.newaxis]
        u = (self.df_ + X.shape[1]) / (self.df_ + maha_dist)
        return probs, u, maha_dist, current_bound

    #The M-step in mixture fitting. Calculates the ML value for the scale matrix
    #location and mixture weights (we are using fixed df, otherwise df would also
    #have to be optimized). We calculate self.loc_ -- the mean, resulting array
    #is KxP for K components and P dimensions; self.scale_, array is PxPxK;
    #self.scale_cholesky_, the cholesky decomposition of the scale matrix; and
    #self.scale_inv_cholesky, the cholesky decomposition of the precision
    #matrix (also self.mix_weights_, the component mixture weights).
    def Mstep(self, X, resp, u, reg_covar):
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
            self.scale_cholesky_[:,:,i] = cholesky(self.scale_[:,:,i], lower=True)
        #For mahalanobis distance we really need the cholesky decomposition of
        #the precision matrix (inverse of scale), but do not want to take 
        #the inverse directly to avoid possible numerical stability issues.
        #We get what we want using the cholesky decomposition of the scale matrix
        #from the following function call.
        self.get_scale_inv_cholesky()


    #Optimizes the df parameter using Newton Raphson.
    def optimize_df(self, X, resp, u):
        pass


    #Calculates the mahalanobis distance for X to all components. Returns an
    #array of dim N x K for N datapoints, K mixture components.
    def maha_distance(self, X):
        maha_dist = []
        for i in range(self.mix_weights_.shape[0]):
            y = np.dot(X, self.scale_inv_cholesky_[:,:,i])
            y = y - np.dot(self.loc_[i,:], self.scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            maha_dist.append(np.sum(y**2, axis=1))
        return np.stack(maha_dist, axis=-1)


    #Calculates log p(X | theta) using the mixture components formulated as 
    #multivariate t-distributions (for specific steps in the algorithm it 
    #is preferable to formulate each component as a Gaussian scale mixture).
    #The function can take precalculated mahalanobis distance to save time if
    #calculated elsewhere. The function returns an array of dim N x K for
    #N datapoints, K mixture components.
    def get_logprob_tdist(self, X, precalc_dist=None):
        if precalc_dist is None:
            maha_dist = 1 + self.maha_distance(X) / self.df_
        else:
            maha_dist = 1 + precalc_dist / self.df_
        maha_dist = -0.5*(self.df_ + X.shape[1])*np.log(maha_dist)
        const_term = gammaln(0.5*(self.df_ + X.shape[1])) - gammaln(0.5*self.df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(self.df_) + np.log(np.pi))
        scale_logdet = [np.sum(np.log(np.diag(self.scale_cholesky_[:,:,i])))
                        for i in range(self.mix_weights_.shape[0])]
        scale_logdet = np.asarray(scale_logdet)
        output = -scale_logdet[np.newaxis,:] + const_term + maha_dist
        output = output + np.log(np.clip(self.mix_weights_[np.newaxis,:], a_min=1e-9, a_max=None))
        return output, scale_logdet
        

    #Gets the inverse of the cholesky decomposition of the scale matrix,
    #(don't use np.linalg.inv!)
    def get_scale_inv_cholesky(self):
        for i in range(self.mix_weights_.shape[0]):
            self.scale_inv_cholesky_[:,:,i] = solve_triangular(self.scale_cholesky_[:,:,i],
                    np.eye(self.scale_cholesky_.shape[0]), lower=True).T

    
    #Gets the Jensen's inequality lower bound on the log likelihood.
    #Only used during training.
    def get_loglik_lower_bound(self, X, resp, u, maha_dist, scale_logdet):
        lower_bound = np.log(np.clip(self.mix_weights_, a_min=1e-12, a_max=None))
        lower_bound -= 0.5 * X.shape[1] * np.log(2*np.pi)
        lower_bound = lower_bound[np.newaxis,:] - scale_logdet - 0.5 * u * maha_dist
        lower_bound += 0.5 * self.df_ * np.log(self.df_ * 0.5)
        lower_bound += gammaln(self.df_ * 0.5)
        lower_bound += 0.5 * self.df_ * (np.log(u) - u) - np.log(u)
        lower_bound = resp * lower_bound - np.log(np.clip(resp, a_min=1e-12, a_max=None))
        return np.sum(lower_bound)
    

    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center to get the starting locations.
    def initialize_params(self, X, n_components):
        self.loc_ = [X[random.randint(0, X.shape[0]-1), :]]
        self.mix_weights_ = np.empty(n_components)
        self.mix_weights_.fill(1/n_components)
        dist_arr_list = []
        for i in range(1, n_components):
            dist_arr = np.sum((X - self.loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            self.loc_.append(X[next_center_id[0],:])

        self.loc_ = np.stack(self.loc_)
        assignments = np.argmin(distmat, axis=1)
        #For initialization, set all covariance matrices to I.
        self.scale_ = [np.eye(X.shape[1]) for i in range(n_components)]
        self.scale_ = np.stack(self.scale_, axis=-1)
        self.scale_cholesky_ = np.copy(self.scale_)
        self.scale_inv_cholesky_ = np.copy(self.scale_)
       
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

    #Gets the number of parameters (useful for AIC & BIC calculations).
    def get_num_parameters(self):
        num_parameters = self.mix_weights_.shape[0] * self.loc.shape[1]
        num_parameters = num_parameters * (self.loc.shape[1] + 1) / 2
        return num_parameters
