import numpy as np
from .variational_hyperparams import VariationalMixHyperparams

class VariationalStudentMixture():

    def __init__(self, max_components = 10, tol=1e-3,
            reg_covar=1e-06, max_iter=500, n_init=1,
            df = 4.0, random_state=123, mean_prior = None,
            scale_prior = None, degrees_of_freedom_prior = None, 
            weight_concentration_prior = None, verbose=False):
        self.check_user_params(max_components, tol, reg_covar, max_iter, 
                        n_init, df, random_state)
        #General model parameters specified by user.
        self.start_df_ = float(df)
        self.max_components = max_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        #Hyperparameters. Any that are None will be automatically calculated 
        #when fit is called.
        self.hyperparams = InfiniteMixHyperparams(mean_prior, scale_prior, degrees_of_freedom_prior,
                        weight_concentration_prior)
        #The model fit parameters (if a model has been fitted) are stored in the model
        #core. If we do multiple restarts we can create multiple core objects and store
        #the best one.
        self.model_core = None
        self.verbose = verbose


    #Function to check the user specified model parameters for validity.
    def check_user_params(self, max_components, tol, reg_covar, max_iter, n_init, df, random_state):
        try:
            max_components = int(max_components)
            tol = float(tol)
            n_init = int(n_init)
            random_state = int(random_state)
            max_iter = int(max_iter)
            reg_covar = float(reg_covar)
        except:
            raise ValueError("n_components, tol, max_iter, n_init, reg_covar and random state should be numeric.")
        if df > 1000:
            raise ValueError("Very large values for dof will give results essentially identical to a Gaussian mixture."
                    "DF = 4 is suggested as a good default. If fixed_df is False, the df will be "
                             "optimized.")
        if df < 1:
            raise ValueError("Inappropriate starting value for df!")
        if max_iter < 1:
            raise ValueError("Inappropriate value for the maximum number of iterations! Must be >= 1.")
        if n_init < 1:
            raise ValueError("Inappropriate value for the number of restarts! Must be >= 1.")
        if tol <= 0:
            raise ValueError("Inappropriate value for tol! Must be greater than 0.")
        if max_components < 1:
            raise ValueError("Inappropriate value for max_components! Must be >= 1.")
        if reg_covar < 0:
            raise ValueError("Reg covar must be >= 0.")


    #Function to check whether the input has the correct dimensionality.
    def check_inputs(self, X):
        if isinstance(X, np.ndarray) == False:
            raise ValueError("X must be a numpy array.")
        #Check first whether model has been fitted. If not, model_core will be None.
        self.check_model()
        if X.dtype != "float64":
            raise ValueError("The input array should be of type float64.")
        if len(X.shape) > 2:
            raise ValueError("Only 1d or 2d arrays are accepted as input.")
        x = X
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if x.shape[1] != self.model_core.get_data_dim():
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.model_core.get_data_dim())
        return x


    #Function to check whether the model has been fitted yet.
    def check_model(self):
        if self.model_core is None:
            raise ValueError("The model has not been successfully fitted yet.")


    #Check data supplied for fitting to make sure it meets basic
    #criteria. We require that N > 2*D and N > 3*n_components.
    def check_fitting_data(self, X):
        if isinstance(X, np.ndarray) == False:
            raise ValueError("X must be a numpy array.")
        if X.dtype != "float64":
            raise ValueError("The input array should be of type float64.")
        if len(X.shape) > 2:
            raise ValueError("This class only accepts 1d or 2d arrays as inputs.")
        if len(X.shape) == 1:
            x = X.reshape(-1,1)
        else:
            x = X
        if x.shape[0] <= 2*x.shape[1]:
            raise ValueError("Too few datapoints for dataset "
            "dimensionality. You should have at least 2 datapoints per "
            "dimension (preferably more).")
        if x.shape[0] <= 3*self.n_components:
            raise ValueError("Too few datapoints for number of components "
            "in mixture. You should have at least 3 datapoints per mixture "
            "component (preferably more).")
        return x


    #Function for fitting a model using the parameters the user
    #selected when creating the model object.
    def fit(self, X):
        x = self.check_fitting_data(X)
        self.hyperparams.check_hyperparameters(x)
        best_lower_bound = -np.inf
        self.model_core = None
        model_cores = []
        for i in range(self.n_init):
            #Increment random state so that each random initialization is different from the
            #rest but so that the overall chain is reproducible.
            model_core = InfiniteModelCore(self.random_state + i, self.fixed_df)
            lower_bound = model_core.fit(x, self.start_df_, self.tol,
                    self.max_components, self.reg_covar, self.max_iter, self.verbose)
            model_cores.append(model_core)
            if self.verbose:
                print("Restart %s now complete"%i)
            if model_core.check_modelcore_convergence() == False:
                print("Restart %s did not converge!"%(i+1))
            elif lower_bound > best_lower_bound:
                self.model_core = model_core
                best_lower_bound = lower_bound
        if self.model_core is None:
            print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")


    #Returns a categorical component assignment for each sample in the input.
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    #Returns the probability that each sample belongs to each component.
    def predict_proba(self, X):
        self.check_model()
        x = self.check_inputs(X)
        probs = self.model_core.get_component_probabilities(x)
        return probs


    #Returns the average log likelihood (i.e. averaged over all datapoints).
    def score(self, X, run_model_checks=True):
        if run_model_checks:
            self.check_model()
            X = self.check_inputs(X)
        return self.model_core.score(X)

    #Returns the per sample log likelihood. Useful if fitting a class conditional classifier
    #with a mixture for each class.
    def score_samples(self, X, run_model_checks=True):
        if run_model_checks:
            self.check_model()
            X = self.check_inputs(X)
        return self.model_core.score_samples(X)
        
    #Simultaneously fits and makes predictions for the input dataset.
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    #Gets the locations for the fitted mixture model.
    def get_cluster_centers(self):
        self.check_model()
        return self.model_core.get_location()

    #Gets the scale matrices for the fitted mixture model.
    def get_cluster_scales(self):
        self.check_model()
        return self.model_core.get_scale()

    #Gets the mixture weights for a fitted model.
    def get_weights(self):
        self.check_model()
        return self.model_core.get_mix_weights()

    #Gets the degrees of freedom for the fitted mixture model.
    def get_df(self):
        self.check_model()
        return self.model_core.get_df()


import numpy as np, math
from scipy.linalg import solve_triangular, solve
from scipy.special import gammaln, logsumexp, digamma, polygamma, loggamma
from scipy.optimize import newton

#The model core object contains all the learned parameters that are optimized
#during a fit. The StudentMixture class creates and stores a StudentMixtureModelCore
#object and performs all the necessary checks on the validity of the data and
#the fitted model before placing any calls to the model core. When fitting with
#multiple restarts, the StudentMixture class can create more than one model core
#and save the best one.
class VariationalModelCore():

    def __init__(self, random_state, fixed_df):
        self.mix_weights_ = None
        self.loc_ = None
        self.wishart_scale_ = None
        self.scale_inv_ = None
        self.wishart_scale_cholesky_ = None
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
        self.initialize_params(X, max_components)
        resp, E_log_mixweights, E_logdet_scale, E_log_u, E_u, E_maha_dist, R_m = self.get_starting_expectation_values(X)
        lower_bound = -np.inf
        for i in range(max_iter):
            resp = self.update_resp(resp, E_log_mixweights, E_logdet_scale, E_log_u, E_u, E_sq_maha_dist)
            Nk = np.sum(resp, axis=0)
            E_log_mixweights = self.update_log_mixweights(Nk, hyperparams)
            E_u, E_log_u = self.update_u(resp, E_maha_dist)
            E_logdet_scale_inv, ru = self.update_scale(X, resp, Nk, E_u, R_m, hyperparams)
            E_mean_outer_prod = self.update_loc(X, ru, hyperparams)
            E_sq_maha_dist = self.update_maha_dist(E_mean_outer_prod)

            lower_bound = self.update_lower_bound(resp, maha_dist, Eu, Elogu)
            change = current_bound - lower_bound
            if abs(change) < tol:
                self.converged_ = True
                break
            if verbose:
                print("Change in lower bound: %s"%change)
                print("Actual lower bound: %s" % current_bound)
        return current_bound


    def update_resp(self, E_log_mixweights, E_logdet_scale, E_log_u, E_u, E_maha_dist):
        resp = 0.5 * E_maha_dist * E_u
        resp = resp + E_log_mixweights + 0.5 * E_logdet_scale + 0.5 * E_log_u
        resp = resp - self.loc_.shape[1] * 0.5 * np.log(2 * np.pi)
        log_resp_norm = logsumexp(resp, axis=1)
        with np.errstate(under="ignore"):
            resp = np.exp(resp - log_resp_norm[:,np.newaxis])
        return resp

    def update_log_mixweights(self, Nk, hyperparams):
        alpha_k = Nk + hyperparams.alpha_m
        alpha_0 = Nk + self.mix_weights_.shape[0] * hyperparams.alpha_m
        return digamma(alpha_k) - digamma(alpha_0)

    def update_u(self, resp, E_maha_distance):
        a_nm = 0.5 * (self.df[:,np.newaxis] + resp * resp.shape[0])
        b_nm = 0.5 * (self.df[:,np.newaxis] + resp * E_maha_distance)
        E_u = a_nm / b_nm
        E_log_u = digamma(a_nm) - np.log(b_nm)
        return E_u, E_log_u

    def update_scale(self, X, resp, E_u, Nk, R_m, hyperparams):
        wishart_dof_updated = hyperparams.wishart_v0 + Nk
        ru = resp * E_u
        for i in range(self.loc_.shape[0]):
            x_adj = X - self.loc_[i:i+1,:]
            scatter_mat = np.dot((ru[:,i] * x_adj).T, x_adj)
            self.wishart_scale_[:,:,i] = hyperparams.wishart_scale_inv + scatter_mat
            self.wishart_scale_cholesky_[:,:,i] = np.linalg.cholesky(self.wishart_scale_[:,:,i])
        #The above calculation gives us the updated scale matrix. We actually need the cholesky 
        #decomposition of its inverse.
        self.get_scale_inv_cholesky(Nk + hyperparams.wishart_v0)
        E_logdet_scale_inv = -np.asarray([2 * np.sum(np.log(np.diag(self.wishart_scale_cholesky_[:,:,i]))) 
                                for i in range(self.mix_weights_.shape[0])])
        E_logdet_scale_inv += self.mix_weights_.shape[0] * np.log(2)
        digamma_sum_term = np.tile(np.arange(0, self.mix_weights_.shape[0] - 1)) + hyperparams.wishart_v0
        digamma_sum_term += Nk[:,np.newaxis] + 1
        digamma_sum_term = np.sum(digamma(digamma_sum_term * 0.5), axis=1)
        return E_logdet_scale_inv + digamma_sum_term


    def update_loc(self, X, ru, hyperparams):
        Rm = np.sum(ru, axis=0)
        Rm = np.stack([ru[i] * self.scale_inv_[:,:,i] + hyperparams.mean_cov_prior * 
                        np.eye(self.mix_weights_.shape[0])
                        for i in range(0,self.mix_weights_.shape[0])], axis=2) 
        
        updated_loc = np.sum(ru * X, axis=0)
        updated_loc = np.stack([updated_loc[i] * self.scale_inv_[:,:,i] + 
                            hyperparams.mean_cov_prior * hyperparams.loc for 
                            i in range(0,self.mix_weights_.shape[0]), axis=2])
        
        E_mean_outer_prod = []
        for i in range(self.mix_weights.shape[0]):
            self.loc_[i,:] = solve(Rm[:,:,i], updated_loc[:,:,i])
            E_mean_outer_prod.append(np.outer(self.loc_[:,i], self.loc_[:,i]) +
                            Rm[:,:,i])
        return np.stack(E_mean_outer_prod, axis=2)

    #Updates the expected squared mahalanobis distance term.
    def update_maha_dist(self, E_mean_outer_prod, X):
        E_sq_maha_dist = []
        for i in range(self.mix_weights_.shape[0]):
            dist_array = np.matmul(X, self.scale_inv_[:,:,i])
            dist_array = np.matmul(dist_array, X.T) + 2 * np.matmul(dist_array,
                    self.loc_[i,:])
            dist_array += np.trace(np.matmul(E_mean_outer_prod, self.scale_inv_[:,:,i]))
            E_sq_maha_dist.append(dist_array)
        return E_sq_maha_dist
    
    #Calculates the squared mahalanobis distance for X to all components. Returns an
    #array of dim N x K for N datapoints, K mixture components. We use the cholesky
    #decomposition of the precision matrix to avoid having to invert anything.
    def vectorized_sq_maha_distance(self, X, loc_, scale_inv_cholesky_):
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        y1 = np.matmul(X, np.transpose(scale_inv_cholesky_, (2,0,1)))
        y2 = np.sum(loc_.T[:,np.newaxis,:] * scale_inv_cholesky_, axis=0)
        y = np.transpose(y1, (1,0,2)) - y2.T
        return np.sum(y**2, axis=2)

    
    #Updates the variational lower bound so we can assess convergence. We leave out
    #constant terms for simplicity. We evaluate each term of the overall lower bound
    #formula separately and then sum them to get the lower bound. This is 
    #unavoidably messy -- the variational lower bound has ten contributing terms,
    #and there's only so far that we can simplify it.
    def update_lower_bound(self, X, resp, Nk, E_sq_maha_dist, E_u,
            E_log_u, E_logdet_scale_inv, E_log_mixweights, hyperparams):
        log_px = E_logdet_scale_inv + X.shape[1] * E_log_u
        log_px = 0.5 * np.sum(log_px - E_u * E_sq_maha_dist)

        log_ploc = -hyperparams.mean_cov_prior * 0.5 * _________________________

        log_pscale = np.asarray([-0.5 * np.trace(np.matmul(hyperparams.wishart_scale_inv,
                self.scale_inv_[:,:,i])) for i in range(self.mix_weights_.shape[0])])
        log_pscale += 0.5 * (hyperparams.wishart_v0 - X.shape[1] - 1) * E_logdet_scale_inv
        log_pscale = np.sum(log_pscale)

        log_pu = 0.5 * self.df_ * np.log(0.5 * self.df_) - loggamma(0.5 * self.df_)
        log_pu = np.sum(X.shape[0] * log_pu)
        log_pu += np.sum( (0.5 * self.df_ - 1) * E_log_u - 0.5 * self.df_ * E_u)
        
        log_pmixweights = (hyperparams.alpha_m - 1) * E_log_mixweights - loggamma(hyperparams.alpha_m)
        log_pmixweights = np.sum(log_pmixweights)

        log_ps = np.sum(resp * E_log_mixweights[np.newaxis,:])
        log_qmu = np.sum(0.5 * logdet_Rm)
        eta_m = Nk + wishart_v0
        log_qscale = np.asarray([self.wishart_norm(self.scale_inv_cholesky_[:,:,i] / np.sqrt(eta_m[i]), eta_m[i])
                    for i in range(self.mix_weights_.shape[0])])
        log_qscale += (eta_m - X.shape[1] - 1) * E_logdet_scale_inv * 0.5
        log_qscale = np.sum(log_qscale - eta_m * 0.5 * X.shape[1])

        log_qu = 

        log_qs = np.sum(resp * np.log(resp))


    #Gets the normalization term for the Wishart distribution (only required for 
    #calculating the variational lower bound). The normalization term is calculated
    #for one component only, so caller must loop over components to get the normalization
    #term for all components as required for the variational lower bound.
    def wishart_norm(self, W_cholesky, eta):
        logdet_term = -eta * np.sum(np.log(np.diag(W_cholesky)))
        inverse_term = np.asarray([loggamma( 0.5 * (eta + 1 - i)) for i in range(W_cholesky.shape[0] - 1)])
        inverse_term = np.prod(inverse_term) * (2 ** (0.5 * eta * W_cholesky.shape[0]))
        inverse_term *= np.pi ** (W_cholesky.shape[0] * (W_cholesky.shape[0] - 1) / 4)
        return logdet_term / inverse_term

    #Gets the inverse of the cholesky decomposition of the scale matrix and the inverse scale (aka "precision")
    #matrix.
    def get_scale_inv_cholesky(self, wishart_dof):
        for i in range(self.mix_weights_.shape[0]):
            self.scale_inv_cholesky_[:,:,i] = solve_triangular(self.wishart_scale_cholesky_[:,:,i],
                    np.eye(self.wishart_scale_cholesky_.shape[0]), lower=True).T
            self.scale_inv_cholesky_[:,:,i] *= np.sqrt(wishart_dof[i])
            self.scale_inv_[:,:,i] = np.matmul(self.scale_inv_cholesky_[:,:,i], self.scale_inv_cholesky_[:,:,i].T)

    
    #Calculates log p(X | theta) using the mixture components formulated as 
    #multivariate t-distributions (for specific steps in the algorithm it 
    #is preferable to formulate each component as a Gaussian scale mixture).
    #The function can take precalculated mahalanobis distance to save time if
    #calculated elsewhere. The function returns an array of dim N x K for
    #N datapoints, K mixture components. Primarily used for 
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


