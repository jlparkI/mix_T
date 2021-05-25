import numpy as np, math
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, digamma, polygamma
from scipy.optimize import newton


#This class is used to fit a finite student's t mixture using the EM algorithm (see
#documentation for usage, derivation of update equations). EM is a maximum likelihood
#approach so we get a point estimate and do not require a prior. For a more Bayesian
#approach, use the variational model instead. This class takes as inputs:
#n_components -- the number of components in the mixture
#tol          -- if the change in the lower bound between iterations is less than tol, 
#                this restart has converged
#reg_covar    -- a value added to the diagonal of all scale matrices to provide 
#                regularization and ensure they are positive definite
#max_iter     -- the maximum number of iterations per restart before we just assume
#                this restart simply didn't converge. Set this number low enough
#                that if the model isn't converging we stop instead of trying forever,
#                but high enough to give the model the iterations needed to find a good
#                solution.
#n_init       -- the maximum number of fitting restarts. EM finds a local maximum
#                so more restarts increases our chances of finding an optimal solution
#                but increases computational cost.
#fixed_df     -- a boolean indicating whether df should be optimized or "fixed" to the
#                user-specified value.
#random_state -- Seed to the random number generator to ensure restarts are reproducible.
#verbose      -- Print updates to keep user updated on fitting.

#Parameters that are stored:
#mix_weights  -- The mixture weights for each component of the mixture; sums to one.
#loc_         -- Equivalent of mean for a gaussian; the center of each component's 
#                distribution. For a student's t distribution, this is called the "location"
#                not the mean. Shape is K x D for K components, D dimensions.
#scale_       -- Equivalent of covariance for a gaussian. Shape is D x D x K for D dimensions,
#                K components.
#scale_cholesky_ -- The cholesky decomposition of the scale matrix. Shape is D x D x K.
#scale_inv_cholesky_ -- Equivalent to the cholesky decomposition of the precision matrix (inverse
#                       of the scale matrix). Shape is D x D x K. 
#df_          -- The degrees of freedom parameter of each student's t distribution. 
#                Shape is K for K components. User can either specify a fixed value or
#                allow algorithm to optimize.
#converged_   -- Whether the fit converged.

class FiniteStudentMixture():

    def __init__(self, n_components = 2, tol=1e-3,
            reg_covar=1e-06, max_iter=500, n_init=1,
            df = 4.0, fixed_df = True, random_state=123, verbose=False):
        self.check_user_params(n_components, tol, reg_covar, max_iter, n_init, df, random_state)
        #General model parameters specified by user.
        self.start_df_ = float(df)
        self.fixed_df = fixed_df
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        #the number of restarts -- this is different from max_iter, which is
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        #The model fit parameters are all initialized to None and will be set 
        #if / when a model is fitted.
        self.mix_weights_ = None
        self.loc_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None

    #Function to check the user specified model parameters for validity.
    def check_user_params(self, n_components, tol, reg_covar, max_iter, n_init, df, random_state):
        try:
            n_components = int(n_components)
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
        if n_components < 1:
            raise ValueError("Inappropriate value for n_components! Must be >= 1.")
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
        if x.shape[1] != self.get_data_dim():
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.model_core.get_data_dim())
        return x


    #Function to check whether the model has been fitted yet. The parameters
    #are all set at the same time, so we need merely check any one of them (e.g. df_)
    #to see whether the model has been fitted.
    def check_model(self):
        if self.df_ is None:
            raise ValueError("The model has not been successfully fitted yet.")


    #Check data supplied for fitting to make sure it meets basic
    #criteria. We require that N > 2*D and N > 3*n_components. If the user
    #supplied a 1d input array, we reshape it to 2d -- this enables us to
    #cluster 1d input arrays without forcing the user to reshape them -- a
    #little bit more user-friendly than scikitlearn's classes, which will just raise
    #a value error if you input a 1d array as X.
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
        best_lower_bound = -np.inf
        #We use self.n_init restarts and save the best result. More restarts = better 
        #chance to find the best possible solution, but also higher cost.
        for i in range(self.n_init):
            #Increment random state so that each random initialization is different from the
            #rest but so that the overall chain is reproducible.
            lower_bound, convergence, loc_, scale_, scale_inv_cholesky_, mix_weights_,\
                    df_, scale_cholesky_ = self.fitting_iteration(x, self.random_state + i)
            if self.verbose:
                print("Restart %s now complete"%i)
            if convergence == False:
                print("Restart %s did not converge!"%(i+1))
            #If this is the best lower bound we've seen so far, update our saved
            #parameters.
            elif lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                self.df_, self.loc_, self.scale_ = df_, loc_, scale_
                self.scale_inv_cholesky_ = scale_inv_cholesky_
                self.scale_cholesky_ = scale_cholesky_
                self.mix_weights_ = mix_weights_
                self.converged_ = True
        if self.converged_ == False:
            print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")
    

    #A single fitting restart. Returns the final parameters, its lower bound
    #and its convergence state. If the fit converges on this attempt,
    #and if lower bound is better than any so far achieved, the model object
    #will use these parameters as its best to date.
    def fitting_iteration(self, X, random_state):
        df_ = np.full((self.n_components), self.start_df_, dtype=np.float64)
        loc_, scale_, mix_weights_, scale_cholesky_, scale_inv_cholesky_ = \
                self.initialize_params(X, random_state)
        lower_bound, convergence = -np.inf, False
        #For each iteration, we run the E step calculations then the M step
        #calculations, update the lower bound then check for convergence.
        for i in range(self.max_iter):
            resp, u, current_bound = self.Estep(X, df_, loc_, scale_inv_cholesky_, 
                                scale_cholesky_, mix_weights_)
            
            mix_weights_, loc_, scale_, scale_cholesky_, scale_inv_cholesky_,\
                            df_ = self.Mstep(X, resp, u, scale_, 
                                scale_cholesky_, df_, scale_inv_cholesky_)
            change = current_bound - lower_bound
            #IN GENERAL, for EM, the lower bound will always increase, and this is in
            #fact a useful debugging tool; in testing for this package this was in fact
            #always true. However, in the event that due to floating point
            #error or a situation where Newton-Raphson did not converge for one or more df so
            #not all df were updated, we do not want to generate what might from the user's
            #perspective be a rather mystifying error, so we use abs(change) rather than 
            #change and do not check the sign. scikitlearn's gaussian mixture does the same!
            if abs(change) < self.tol:
                convergence = True
                break
            lower_bound = current_bound
            if self.verbose:
                print("Change in lower bound: %s"%change)
                print("Actual lower bound: %s" % current_bound)
        return current_bound, convergence, loc_, scale_, scale_inv_cholesky_,\
                mix_weights_, df_, scale_cholesky_



    #The e-step in mixture fitting. Calculates responsibilities for each datapoint
    #and E[u] for the formulation of the t-distribution as a 
    #Gaussian scale mixture. It returns the responsibilities (NxK array),
    #E[u] (NxK array), the squared mahalanobis distance (NxK array), and the log
    #of the determinant of the scale matrix.
    def Estep(self, X, df_, loc_, scale_inv_cholesky_, scale_cholesky_, mix_weights_):
        sq_maha_dist = self.sq_maha_distance(X, loc_, scale_inv_cholesky_)
        
        loglik = self.get_loglik(X, sq_maha_dist, df_, 
                scale_cholesky_, mix_weights_)

        weighted_log_prob = loglik + np.log(np.clip(mix_weights_,
                                        a_min=1e-9, a_max=None))[np.newaxis,:]
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            resp = np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])
        u = (df_[np.newaxis,:] + X.shape[1]) / (df_[np.newaxis,:] + sq_maha_dist)
        return resp, u, np.mean(log_prob_norm)



    #The M-step in mixture fitting. Calculates the ML value for the scale matrix
    #location and mixture weights. We calculate loc_ -- resulting array
    #is KxP for K components and P dimensions; scale_, array is PxPxK;
    #scale_cholesky_, the cholesky decomposition of the scale matrix; and
    #scale_inv_cholesky, the cholesky decomposition of the precision
    #matrix (also mix_weights_, the component mixture weights).
    def Mstep(self, X, resp, u, scale_, scale_cholesky_, df_,
                scale_inv_cholesky_):
        mix_weights_ = np.mean(resp, axis=0)
        ru = resp * u
        loc_ = np.dot((ru).T, X)
        resp_sum = np.sum(ru, axis=0) + 10 * np.finfo(resp.dtype).eps
        loc_ = loc_ / resp_sum[:,np.newaxis]
        for i in range(mix_weights_.shape[0]):
            scaled_x = X - loc_[i,:][np.newaxis,:]
            scale_[:,:,i] = np.dot((ru[:,i:i+1] * scaled_x).T,
                            scaled_x) / resp_sum[i]
            scale_[:,:,i].flat[::X.shape[1] + 1] += self.reg_covar
            scale_cholesky_[:,:,i] = np.linalg.cholesky(scale_[:,:,i])
        #We really need the cholesky decomposition of
        #the precision matrix (inverse of scale), but do not want to take 
        #the inverse directly to avoid possible numerical stability issues.
        #We get what we want using the cholesky decomposition of the scale matrix
        #from the following function call.
        scale_inv_cholesky_ = self.get_scale_inv_cholesky(scale_cholesky_,
                            scale_inv_cholesky_)
        if self.fixed_df == False:
            df_ = self.optimize_df(X, resp, u, df_)
        return mix_weights_, loc_, scale_, scale_cholesky_, scale_inv_cholesky_, df_



    #Optimizes the df parameter using Newton Raphson.
    def optimize_df(self, X, resp, u, df_):
        for i in range(self.n_components):
            optimal_df = newton(self.dof_first_deriv, x0 = df_[i],
                                 fprime = self.dof_second_deriv,
                                 fprime2 = self.dof_third_deriv,
                                 args = (u, resp, X.shape[1], i, df_),
                                 full_output = False, disp=False, tol=1e-3)
            #It may occasionally happen that newton does not converge.
            #If so, ignore the result for this iteration (keep the pre-existing
            #df value).
            if math.isnan(df_[i]) == False:
                df_[i] = optimal_df
            #DF should never be less than 1 but can go arbitrarily high.
            if df_[i] < 1:
                df_[i] = 1.0
        return df_


    # First derivative of the complete data log likelihood w/r/t df.
    def dof_first_deriv(self, dof, u, resp, dim, i, df_):
        grad = 1.0 - digamma(dof * 0.5) + np.log(0.5 * dof)
        grad += (1 / resp[:,i].sum(axis=0)) * (resp[:,i] * (np.log(u[:,i]) - u[:,i])).sum(axis=0)
        return grad + digamma(0.5 * (df_[i] + dim)) - np.log(0.5 * (df_[i] + dim))


    #Second derivative of the complete data log likelihood w/r/t df.
    def dof_second_deriv(self, dof, u, resp, dim, i, df_):
        return -0.5 * polygamma(1, 0.5 * dof) + 1 / dof


    #Third derivative of the complete data log likelihood w/r/t df.
    def dof_third_deriv(self, dof, u, resp, dim, i, df_):
        return -0.25 * polygamma(2, 0.5 * dof) - 1 / (dof**2)



    #Calculates the squared mahalanobis distance for X to all components. Returns an
    #array of dim N x K for N datapoints, K mixture components.
    def sq_maha_distance(self, X, loc_, scale_inv_cholesky_):
        sq_maha_dist = np.empty((X.shape[0], scale_inv_cholesky_.shape[2]))
        for i in range(sq_maha_dist.shape[1]):
            y = np.dot(X, scale_inv_cholesky_[:,:,i])
            y = y - np.dot(loc_[i,:], scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            sq_maha_dist[:,i] = np.sum(y**2, axis=1)
        return sq_maha_dist


    #Gets the inverse of the cholesky decomposition of the scale matrix.
    def get_scale_inv_cholesky(self, scale_cholesky_, scale_inv_cholesky_):
        for i in range(scale_cholesky_.shape[2]):
            scale_inv_cholesky_[:,:,i] = solve_triangular(scale_cholesky_[:,:,i],
                    np.eye(scale_cholesky_.shape[0]), lower=True).T
        return scale_inv_cholesky_


    #Calculates log p(X | theta) using the mixture components formulated as 
    #multivariate t-distributions (for specific steps in the algorithm it 
    #is preferable to formulate each component as a Gaussian scale mixture).
    #The function returns an array of dim N x K for N datapoints, K mixture components.
    #It expects to receive the squared mahalanobis distance and the model 
    #parameters (since during model fitting these are still being updated).
    def get_loglik(self, X, sq_maha_dist, df_, scale_cholesky_, mix_weights_):
        sq_maha_dist = 1 + sq_maha_dist / df_[np.newaxis,:]
        
        #THe rest of this is just the calculations for log probability of X for the
        #student's t distributions described by the input parameters broken up
        #into three convenient chunks that we sum on the last line.
        sq_maha_dist = -0.5*(df_[np.newaxis,:] + X.shape[1]) * np.log(sq_maha_dist)
        const_term = gammaln(0.5*(df_ + X.shape[1])) - gammaln(0.5*df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(df_) + np.log(np.pi))
        scale_logdet = [np.sum(np.log(np.diag(scale_cholesky_[:,:,i])))
                        for i in range(mix_weights_.shape[0])]
        scale_logdet = np.asarray(scale_logdet)
        return -scale_logdet[np.newaxis,:] + const_term[np.newaxis,:] + sq_maha_dist
    

    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center to get the starting locations.
    def initialize_params(self, X, random_seed):
        np.random.seed(random_seed)
        loc_ = [X[np.random.randint(0, X.shape[0]-1), :]]
        mix_weights_ = np.empty(self.n_components)
        mix_weights_.fill(1/self.n_components)
        dist_arr_list = []
        for i in range(1, self.n_components):
            dist_arr = np.sum((X - loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            loc_.append(X[next_center_id[0],:])

        loc_ = np.stack(loc_)
        #For initialization, set all covariance matrices to I.
        scale_ = [np.eye(X.shape[1]) for i in range(self.n_components)]
        scale_ = np.stack(scale_, axis=-1)
        scale_cholesky_ = np.copy(scale_)
        scale_inv_cholesky_ = np.copy(scale_)
        return loc_, scale_, mix_weights_, scale_cholesky_, scale_inv_cholesky_




    '''The remaining functions are called for a fitted model. Each of the functions
    that the user is expected to call (i.e. the ones described in the documentation)
    check that the model has been fitted using self.check_model() before performing any
    calculations; if the model has not been fitted they raise a value error.'''


    #Returns a categorical component assignment for each sample in the input. It calls
    #predict_proba which performs a model fit check, then assigns each datapoint to
    #the component with the largest likelihood for that datapoint.
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    #Returns the probability that each sample belongs to each component. Used by
    #self.predict.
    def predict_proba(self, X):
        self.check_model()
        x = self.check_inputs(X)
        probs = self.get_component_probabilities(x)
        return probs


    #Returns the average log likelihood (i.e. averaged over all datapoints). Calls
    #self.score_samples to do the actual calculation. Useful for AIC, BIC. It has
    #the option to not perform model checks since it is called by AIC and BIC which
    #perform model checks before calling score.
    def score(self, X, perform_model_checks = True):
        return np.mean(self.score_samples(X, perform_model_checks))

    #Returns the per sample log likelihood. Useful if fitting a class conditional classifier
    #with a mixture for each class.
    def score_samples(self, X, perform_model_checks = True):
        if perform_model_checks:
            self.check_model()
            X = self.check_inputs(X)
        return logsumexp(self.get_weighted_loglik(X), axis=1)
        
    #Simultaneously fits and makes predictions for the input dataset.
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    #Gets the locations for the fitted mixture model. self.check_model ensures
    #model has been fitted and will raise a value error if it hasn't.
    def get_cluster_centers(self):
        self.check_model()
        return self.loc_

    #Gets the scale matrices for the fitted mixture model.
    def get_cluster_scales(self):
        self.check_model()
        return self.scale_

    #Gets the mixture weights for a fitted model.
    def get_weights(self):
        self.check_model()
        return self.mix_weights_

    #Gets the degrees of freedom for the fitted mixture model.
    def get_df(self):
        self.check_model()
        return self.df_

    #Returns the Akaike information criterion (AIC) for the input dataset.
    #Useful in selecting the number of components.
    def aic(self, X):
        self.check_model()
        x = self.check_inputs(X)
        n_params = self.get_num_parameters()
        score = self.score(x, perform_model_checks = False)
        return 2 * n_params - 2 * score * X.shape[0]

    #Returns the Bayes information criterion (BIC) for the input dataset.
    #Useful in selecting the number of components, more heavily penalizes
    #n_components than AIC.
    def bic(self, X):
        self.check_model()
        x = self.check_inputs(X)
        score = self.score(x, perform_model_checks = False)
        n_params = self.get_num_parameters()
        return 2 * n_params * np.log(x.shape[0]) - 2 * score * x.shape[0]


    #Returns log p(X | theta) + log mix_weights. This is called by other class
    #functions which check before calling it that the model has been fitted.
    def get_weighted_loglik(self, X):
        sq_maha_dist = self.sq_maha_distance(X, self.loc_, self.scale_inv_cholesky_)
        loglik = self.get_loglik(X, sq_maha_dist, self.df_, self.scale_cholesky_,
                        self.mix_weights_)
        return loglik + np.log(self.mix_weights_)[np.newaxis,:]


    #Returns the probability that the input data belongs to each component. Used
    #for making predictions. This is called by other class functions which check before
    #calling it that the model has been fitted.
    def get_component_probabilities(self, X):
        weighted_loglik = self.get_weighted_loglik(X)
        with np.errstate(under="ignore"):
            loglik = weighted_loglik - logsumexp(weighted_loglik, axis=1)[:,np.newaxis]
        return np.exp(loglik)


    #Gets the number of parameters (useful for AIC & BIC calculations). Note that df is only
    #treated as a parameter if df is not fixed. This function is only used by AIC and BIC
    #which check whether the model has been fitted first so no need to check here.
    def get_num_parameters(self):
        num_parameters = self.mix_weights_.shape[0] + self.loc_.shape[0] * self.loc_.shape[1]
        num_parameters += 0.5 * self.scale_.shape[0] * (self.scale_.shape[1] + 1) * self.scale_.shape[2]
        if self.fixed_df:
            return num_parameters
        else:
            return num_parameters + self.df_.shape[0]



    #Returns the dimensionality of the training data.
    def get_data_dim(self):
        if self.loc_ is None:
            raise ValueError("The model has not been fitted successfully yet! no "
                            "parameters have been saved.")
        return self.loc_.shape[1]
