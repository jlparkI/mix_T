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
class InfiniteModelCore():

    def __init__(self, random_state, fixed_df):
        self.mix_weights_ = None
        self.loc_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None
        self.random_state = random_state
        self.fixed_df_ = fixed_df

    #Function for fitting the model to user supplied data. The user-specified fitting
    #parameters (df, max_components, max_iter etc) are stored in and supplied by the
    #caller and passed to ModelCore when it does a fit.
    def fit(self, X, start_df, tol, max_components, reg_covar, max_iter, 
            verbose, hyperparams):
        self.df_ = np.full((max_components), start_df, dtype=np.float64)
        self.initialize_params(X, max_components)
        lower_bound = -np.inf
        for i in range(max_iter):
            #We refer to "E-step" and "M-step" here to emphasize the shared similarities
            #with EM. Although this is in fact a mean field variational approximation
            #we find our local maximum through a stepwise approach that exhibits some
            #shared structure.
            resp, maha_dist, Eu = self.Estep(X, hyperparams, resp, eta1, eta2)
            self.Mstep(X, resp, Eu, hyperparams)
            change = current_bound - lower_bound
            if abs(change) < tol:
                self.converged_ = True
                break
            if verbose:
                print("Change in lower bound: %s"%change)
                print("Actual lower bound: %s" % current_bound)
        return current_bound

    #TODO: Clean this up, split into separate functions where appropriate,
    #ensure none of these calculations are redundant with lower bound.
    def Estep(self, X, hyperparams, resp, eta1, eta2):
        beta1, beta2 = self.update_beta_values(resp, eta1, eta2)
        maha_dist = self.update_maha_dist(X)
        resp_sum = hyperparams.dof_prior + np.sum(resp, axis=0)
        expect_maha_dist = X.shape[1] / hyperparams.lambda_prior + 
        vj1 = 0.5 * (resp * X.shape[1] + self.df_[np.newaxis,:])
        vj2 = 0.5 * (resp * expect_maha_dist + self.df_[np.newaxis,:])
        Eu = vj1 / vj2
        Elogu = digamma(vj1) - log(vj2)

        logV, log1minusV = self.mix_weights_expect(beta1, beta2)
        resp = [np.sum(np.log(np.diag(self.scale_inv_cholesky[:,:,i]))) for i in 
            range(self.scale_inv_cholesky.shape[2])]
        resp = np.asarray(resp) - 0.5 * X.shape[1] * np.log(2 * np.pi)
        resp = resp + 0.5 * X.shape[1] * Elogu - 0.5 * logu * expect_maha_dist
        resp_norm = logsumexp(resp, axis=1)
        with np.errstate(under="ignore"):
            resp = np.exp(resp - resp_norm[:,np.newaxis])
        return resp, maha_dist, Eu
        

    def Mstep(self, X, resp, Eu, hyperparams):
        weighted_resp = resp * Eu
        resp_sum = np.sum(weighted_resp, axis=0)
        Xweighted = X[:,np.newaxis,:] * weighted_resp[:,:,np.newaxis]
        Xaverage = np.mean(Xweighted, axis=0)
        self.loc_ = hyperparams.lambda_prior * hyperparams.mean_prior + Xaverage
        self.loc_ /= (resp_sum + hyperparams.lambda_prior)[:,np.newaxis]
        
        Xcentered = Xweighted - Xaverage[np.newaxis,:,:]
        Xweighted = Xweighted - Xaverage[np.newaxis,:]
        Xscatter = (1 / X.shape[0]) * np.matmul(Xweighted.T, Xweighted)

        #Add degrees of freedom optimization here later:
        if self.fixed_df_:
            pass

    def mix_weights_expect(self, beta1, beta2):
        beta_sum = digamma(beta1 + beta2)
        logV = digamma(beta1) - beta_sum
        log1minusV = digamma(beta2) - beta_sum
        return logV, log1minusV

    def update_beta_values(self, resp, eta1, eta2):
        resp_sum = np.sum(resp, axis=0)
        beta_1 = resp_sum + 1
        alpha_expect = eta1 / eta2
        beta_2 = np.zeros((resp_sum.shape[0]))
        beta_2[:-1] = np.cumsum(resp[:-1][::-1])[::-1]
        beta_2 += alpha_expect
        return beta_1, beta_2


    #Some of the calculations here are redundant. TODO: Clean this up.
    def update_lower_bound(self):
        pass

    #Returns the squared mahalanobis distance for all datapoints to all clusters.
    def update_maha_dist(self, X):
        maha_dist = np.empty((X.shape[0], self.mix_weights_.shape[0]))
        for i in range(self.mix_weights_.shape[0]):
            y = np.dot(X, self.scale_inv_cholesky_[:,:,i])
            y = y - np.dot(self.loc_[i,:], self.scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            maha_dist[:,i] = np.sum(y**2, axis=1)
        return maha_dist

    #Gets the inverse of the cholesky decomposition of the scale matrix.
    def get_scale_inv_cholesky(self):
        for i in range(self.mix_weights_.shape[0]):
            self.scale_inv_cholesky_[:,:,i] = solve_triangular(self.scale_cholesky_[:,:,i],
                    np.eye(self.scale_cholesky_.shape[0]), lower=True).T

    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center to get the starting locations.
    def initialize_params(self, X, max_components):
        np.random.seed(self.random_state)
        self.loc_ = [X[np.random.randint(0, X.shape[0]-1), :]]
        self.mix_weights_ = np.empty(max_components)
        self.mix_weights_.fill(1/max_components)
        dist_arr_list = []
        for i in range(1, max_components):
            dist_arr = np.sum((X - self.loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            self.loc_.append(X[next_center_id[0],:])

        self.loc_ = np.stack(self.loc_)
        #For initialization, set all covariance matrices to I.
        self.scale_ = [np.eye(X.shape[1]) for i in range(max_components)]
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

