import numpy as np, math
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, digamma, polygamma
from scipy.optimize import newton

#The model core object contains all the learned parameters that are optimized
#during a fit. The StudentMixture class creates and stores a StudentMixtureModelCore
#object and performs all the necessary checks on the validity of the data and
#the fitted model before placing any calls to the model core. When fitting with
#multiple restarts, the StudentMixture class can create more than one model core
#and save the best one.
class FiniteModelCore():

    def __init__(self, random_state, fixed_df):
        self.mix_weights_ = None
        self.loc_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None
        self.fixed_df = fixed_df
        self.random_state = random_state


    #Function for fitting the model to user supplied data. The user-specified
    #fitting parameters (df, n_components, max_iter etc) are stored in and 
    #supplied by the mixt_model class and passed to its model core when it does a
    #fit.
    def fit(self, X, start_df, tol, n_components, reg_covar, max_iter, verbose):
        self.df_ = np.full((n_components), start_df, dtype=np.float64)
        self.initialize_params(X, n_components)
        lower_bound = -np.inf
        for i in range(max_iter):
            resp, u, maha_dist, current_bound = self.Estep(X)
            self.Mstep(X, resp, u, reg_covar)
            change = current_bound - lower_bound
            if abs(change) < tol:
                self.converged_ = True
                break
            lower_bound = current_bound
            if verbose:
                print("Change in lower bound: %s"%change)
                print("Actual lower bound: %s" % current_bound)
        return current_bound


    #The e-step in mixture fitting. Calculates responsibilities for each datapoint
    #and E[u] for the formulation of the t-distribution as a 
    #Gaussian scale mixture. It returns the responsibilities (NxK array),
    #E[u] (NxK array), the mahalanobis distance (NxK array), and the log
    #of the determinant of the scale matrix.
    def Estep(self, X):
        maha_dist = self.maha_distance(X)
        loglik = self.get_loglik(X, maha_dist)
        weighted_log_prob = loglik + np.log(np.clip(self.mix_weights_,
                                        a_min=1e-9, a_max=None))[np.newaxis,:]
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            resp = np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])
        u = (self.df_[np.newaxis,:] + X.shape[1]) / (self.df_[np.newaxis,:] + maha_dist)
        return resp, u, maha_dist, np.mean(log_prob_norm)

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
    def optimize_df(self, X, resp, u):
        for i in range(self.mix_weights_.shape[0]):
            self.df_[i] = newton(self.dof_first_deriv, x0 = self.df_[i],
                                 fprime = self.dof_second_deriv,
                                 fprime2 = self.dof_third_deriv,
                                 args = (u, resp, X.shape[1], i),
                                 full_output = False, disp=False, tol=1e-3)
            #It may occasionally happen that newton does not converge.
            #If so, reset to the default value for this iteration.
            if math.isnan(self.df_[i]):
                self.df_[i] = 4.0
            #DF should never be less than 1 but can go arbitrarily high.
            if self.df_[i] < 1:
                self.df_[i] = 1.0

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
    def maha_distance(self, X):
        maha_dist = np.empty((X.shape[0], self.mix_weights_.shape[0]))
        for i in range(self.mix_weights_.shape[0]):
            y = np.dot(X, self.scale_inv_cholesky_[:,:,i])
            y = y - np.dot(self.loc_[i,:], self.scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            maha_dist[:,i] = np.sum(y**2, axis=1)
        return maha_dist


    #Calculates log p(X | theta) using the mixture components formulated as 
    #multivariate t-distributions (for specific steps in the algorithm it 
    #is preferable to formulate each component as a Gaussian scale mixture).
    #The function can take precalculated mahalanobis distance to save time if
    #calculated elsewhere. The function returns an array of dim N x K for
    #N datapoints, K mixture components.
    def get_loglik(self, X, precalc_dist=None):
        if precalc_dist is None:
            maha_dist = 1 + self.maha_distance(X) / self.df_[np.newaxis,:]
        else:
            maha_dist = 1 + precalc_dist / self.df_[np.newaxis,:]
        maha_dist = -0.5*(self.df_[np.newaxis,:] + X.shape[1])*np.log(maha_dist)
        const_term = gammaln(0.5*(self.df_ + X.shape[1])) - gammaln(0.5*self.df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(self.df_) + np.log(np.pi))
        scale_logdet = [np.sum(np.log(np.diag(self.scale_cholesky_[:,:,i])))
                        for i in range(self.mix_weights_.shape[0])]
        scale_logdet = np.asarray(scale_logdet)
        return -scale_logdet[np.newaxis,:] + const_term[np.newaxis,:] + maha_dist


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

    #Gets the inverse of the cholesky decomposition of the scale matrix.
    def get_scale_inv_cholesky(self):
        for i in range(self.mix_weights_.shape[0]):
            self.scale_inv_cholesky_[:,:,i] = solve_triangular(self.scale_cholesky_[:,:,i],
                    np.eye(self.scale_cholesky_.shape[0]), lower=True).T

    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center to get the starting locations.
    def initialize_params(self, X, n_components):
        np.random.seed(self.random_state)
        self.loc_ = [X[np.random.randint(0, X.shape[0]-1), :]]
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

    #Returns the mixture weights.
    def get_mix_weights(self):
        return self.mix_weights_
    
    #Returns the mixture df.
    def get_df(self):
        return self.df_

    #Gets the number of parameters (useful for AIC & BIC calculations). Note that df is only
    #treated as a parameter if df is not fixed.
    def get_num_parameters(self):
        num_parameters = self.mix_weights_.shape[0] + self.loc_.shape[0] * self.loc_.shape[1]
        num_parameters += 0.5 * self.scale_.shape[0] * (self.scale_.shape[1] + 1) * self.scale_.shape[2]
        if self.fixed_df:
            return num_parameters
        else:
            return num_parameters + self.df_.shape[0]
