'''Finite mixture of Student's t-distributions fit using variational mean-field.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>
#License: MIT

import numpy as np, math
from scipy.linalg import solve_triangular, solve
from scipy.special import gamma, logsumexp, digamma, polygamma, loggamma
from scipy.optimize import newton
from .variational_hyperparams import VariationalMixHyperparams as Hyperparams
from .parameter_bundle import ParameterBundle
from squaredMahaDistance import squaredMahaDistance




#################################################################################

#This class is used to fit a finite student's t mixture using variational mean-field
#(for details, see the docs). This is a Bayesian approach unlike EM and therefore
#can be used to approximate the posterior (a crude approximation though,
#so in general variational techniques are better for prediction than for
#inference). This shares many elements
#in common with EM, in particular the "stepwise" approach where some parameters
#are held fixed while others are updated, but the update calculations are in some
#respects different, also the lower bound calculations are quite different.
#
#INPUTS:
#n_components   --  the number of components in the mixture. With variational mean-field,
#                   unlike EM, this is an upper bound, since variational mean-field can 
#                   "kill off" unneeded components depending on the hyperparameters selected.
#tol            --  if the change in the lower bound between iterations is less than tol, 
#                   this restart has converged
#max_iter       --  the maximum number of iterations per restart before we just assume
#                   this restart simply didn't converge.
#n_init         --  the maximum number of fitting restarts.
#fixed_df       --  a boolean indicating whether df should be optimized or "fixed" to the
#                   user-specified value.
#random_state   --  Seed to the random number generator to ensure restarts are reproducible.
#verbose        --  Print updates to keep user updated on fitting.
#init_type      --  The type of initialization algorithm -- either "k++" for kmeans++ or 
#                   "kmeans" for kmeans.
#scale_inv_prior    --  The prior for the inverse of the scale matrices. Defaults to None.
#                   If None, this class will use a reasonable data-driven default for the 
#                   scale prior. If it is not None, it must be of shape D x D x K, 
#                   for D dimensions and K components. This component should never be
#                   set to all zeros because it provides some regularization and ensures the
#                   scale matrices are positive definite, similar to reg_covar in EM.
#loc_prior      --  The prior for the location of each component. If None it will
#                   be set to the mean of the data.
#mean_cov_prior --  The diagonal value for the covariance matrix of the prior for the
#                   location of the components of the mixture. This essentially determines
#                   how much weight the model puts on the prior -- a very large value
#                   will force the model to push all components onto the prior mean!
#                   Generally use a small nonzero value.
#weight_conc_prior  --  The weight concentration prior. This is crucial to determining
#                   the behavior of the algorithm and is one of the most important
#                   user-determined values. A high value indicates many clusters are
#                   expected, while a low value indicates only a few are expected.
#                   If this value is low, the algorithm will tend to "kill off"
#                   unneeded components. The user may need to tune this for a specific
#                   problem.
#wishart_dof_prior  --  The dof parameter for the Wishart prior on the scale matrices.
#                   Generally can be safely left as None, in which case it will be set to
#                   the dimensionality of the input.
#max_df         --  The maximum value for the degrees of freedom parameter. If there
#                   is no max, the df can gradually creep higher for distributions that
#                   are approximately normal, leading to very slow convergence since
#                   the lower bound is still changing, but without really improving the
#                   fit, since df values > 100 are approximately equivalent to a Gaussian.
#                   The default is 100; set to np.inf to allow for no maximum df.

#PARAMETERS FROM FIT:
#mix_weights    --  The mixture weights for each component of the mixture; sums to one.
#location_      --  Equivalent of mean for a gaussian; the center of each component's 
#                   distribution. For a student's t distribution, this is called the "location"
#                   not the mean. Shape is K x D for K components, D dimensions.
#scale_         --  Equivalent of covariance for a gaussian. Shape is D x D x K for D dimensions,
#                   K components.
#scale_cholesky_ -- The cholesky decomposition of the scale matrix. Shape is D x D x K.
#scale_inv_cholesky_ -- Equivalent to the cholesky decomposition of the precision matrix (inverse
#                       of the scale matrix). Shape is D x D x K. 
#df_            --  The degrees of freedom parameter of each student's t distribution. 
#                   Shape is K for K components. User can either specify a fixed value or
#                   allow algorithm to optimize.
#converged_     --  Whether the fit converged.

class VariationalStudentMixture():

    def __init__(self, n_components = 2, tol=1e-5, max_iter=2000, n_init=1,
            df = 4.0, fixed_df = True, random_state=123, verbose=False,
            init_type = "kmeans", scale_inv_prior=None, loc_prior=None,
            mean_cov_prior = 1e-2, weight_conc_prior=None, wishart_dof_prior = None,
            max_df = 100):
        self.max_df = max_df
        if self.max_df is None:
            raise ValueError("max_df cannot be None! Set max_df to np.inf if you would like "
                             "to have no maximum for df.")
        if self.max_df < 1:
            raise ValueError("max_df cannot be < 1!")
        
        self.check_user_params(n_components, tol, 1e-3, max_iter, n_init, df, random_state,
                init_type)
        #General model parameters specified by user.
        self.start_df = float(df)
        self.fixed_df = fixed_df
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.init_type = init_type
        #Hyperparameters. These are unique to the variational mixture.
        #Any that are None will be calculated using a call to Hyperparams.initialize()
        #when fit is called.
        self.scale_inv_prior = scale_inv_prior
        self.loc_prior = loc_prior
        self.mean_cov_prior = mean_cov_prior
        self.weight_conc_prior = weight_conc_prior
        self.wishart_dof_prior = wishart_dof_prior
        #the number of restarts -- this is different from max_iter, which is the maximum 
        #number of iterations per restart.
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        #The model fit parameters are all initialized to None and will be set 
        #if / when a model is fitted.
        self.mix_weights_ = None
        self.location_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None
        #We save the variational lower bound in case the user wants to do model
        #comparisons.
        self.final_lower_bound = None
        self.resp = None

    #Function to check the user specified model parameters for validity.
    def check_user_params(self, n_components, tol, reg_covar, max_iter, n_init, df, random_state,
            init_type):
        try:
            n_components = int(n_components)
            tol = float(tol)
            n_init = int(n_init)
            random_state = int(random_state)
            max_iter = int(max_iter)
            reg_covar = float(reg_covar)
            init_type = str(init_type)
        except:
            raise ValueError("n_components, tol, max_iter, n_init, reg_covar and random state should be numeric; "
                    "init_type should be a string.")
        if df > self.max_df:
            raise ValueError("Very large values for dof will give results essentially identical to a Gaussian mixture."
                    "DF = 4 is suggested as a good default. If fixed_df is False, the df will be "
                             "optimized. Currently you have chosen a starting value for df that is "
                             "greater than the max_df value you specified.")
        if df > 1000:
            raise ValueError("Very large values for dof will give results essentially identical to a Gaussian mixture."
                    "DF = 4 is suggested as a good default. If fixed_df is False, the df will be "
                             "optimized. Currently you have chosen a very large value for df > 1000.")
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
        if init_type not in ["k++", "kmeans"]:
            raise ValueError("init_type must be one of either 'k++' or 'kmeans'.")


    #Function to check whether the input has the correct dimensionality. This function is
    #used to check data supplied to self.predict, NOT fitting data, and assumes the model
    #has already been fitted -- it compares the dimensionality of the input to the dimensionality
    #of training data. To check training data for validity, self.check_fitting_data is used instead.
    def check_inputs(self, X):
        if isinstance(X, np.ndarray) == False:
            raise ValueError("X must be a numpy array.")
        #Check first whether model has been fitted.
        self.check_model()
        if X.dtype != "float64":
            raise ValueError("The input array should be of type float64.")
        if len(X.shape) > 2:
            raise ValueError("Only 1d or 2d arrays are accepted as input.")
        x = X
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if x.shape[1] != self.location_.shape[1]:
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.location_.shape[1])
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
    #
    #INPUTS
    #X              --  The raw data for fitting. This must be either a 1d array, in which case
    #                   self.check_fitting_data will reshape it to a 2d 1-column array, or
    #                   a 2d array where each column is a feature and each row a datapoint.
    #use_score      --  A boolean. If True, instead of using the variational lower bound
    #                   to select the best model from n_init, we use the model score.
    #                   If False, we keep the model with the best variational lower bound.
    #                   Using score instead prioritizes models with larger likelihood.
    #                   Generally both approaches should give the same result, but 
    #                   in some situations there may be interesting differences.
    #                   Defaults to True.
    def fit(self, X, use_score = False):
        x = self.check_fitting_data(X)
        best_lower_bound = -np.inf
        #Check the user specified hyperparams and for any that are None (indicating user
        #wanted us to initialize them), initialize them. The Hyperparams object stores
        #them in a convenient bundle that can be passed to all the update functions.
        hyperparams = Hyperparams(x, self.loc_prior, self.scale_inv_prior, self.weight_conc_prior,
                                       wishart_v0 = self.wishart_dof_prior, 
                                       mean_covariance_prior = self.mean_cov_prior,
                                       n_components = self.n_components)
        #We use self.n_init restarts and save the best result. More restarts = better 
        #chance to find the best possible solution, but also higher cost.
        for i in range(self.n_init):
            #Increment random state so that each random initialization is different from the
            #rest but so that the overall chain is reproducible.
            lower_bound, convergence, param_bundle, score = self.fitting_restart(x, 
                                self.random_state + i, hyperparams)
            if self.verbose:
                print("Restart %s now complete"%i)
            if convergence == False:
                print("Restart %s did not converge!"%(i+1))
                continue
            #If this is the best lower bound we've seen so far, update our saved
            #parameters using the parameter bundle and then discard it.
            if lower_bound > best_lower_bound and use_score == False:
                self.transfer_fit_params(X, param_bundle, hyperparams)
                del param_bundle
                self.converged_ = True
                best_lower_bound = lower_bound
                self.final_lower_bound = lower_bound
            #If the user wants to use the score instead, use this to choose the model
            #instead of the lower bound.
            elif use_score == True and score > best_lower_bound:
                self.transfer_fit_params(X, param_bundle, hyperparams)
                del param_bundle
                self.converged_ = True
                best_lower_bound = score
                self.final_lower_bound = lower_bound

        if self.converged_ == False:
            print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")
    

    #A single fitting restart.
    #
    #INPUTS
    #X              --  The raw data. Must be a 2d array where each column is a feature and
    #                   each row is a datapoint. The caller (self.fit) ensures this is true.
    #random_state   --  The seed for the random number generator.
    #hyperparams    --  A copy of the hyperparameters object passed by the class.
    #
    #RETURNED PARAMETERS        
    #current_bound  --  The lower bound for the current fitting iteration. The caller (self.fit)
    #                   keeps the set of parameters that have the best associated lower bound.
    #convergence    --  A boolean indicating convergence or lack thereof.
    #param_bundle   --  Object containing all fit parameters and all values needed to calculate
    #                   the lower bound.
    def fitting_restart(self, X, random_state, hyperparams):
        params = ParameterBundle(X, self.n_components, self.start_df, random_state,
                        self.init_type)
        
        #The param_bundle has several expectations that need to be initialized before
        #the first fitting iteration -- this is done by the following function call.
        params = self.initialize_expectations(X, params, hyperparams)
        old_lower_bound, convergence = -np.inf, False
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        #For each iteration, we run the E step calculations then the M step
        #calculations, update the lower bound then check for convergence.
        for i in range(self.max_iter):
            params, sq_maha_dist, score = self.VariationalEStep(X, params,
                                            sq_maha_dist)
            params = self.VariationalMStep(X, params, hyperparams)
            new_lower_bound = self.update_lower_bound(X, params, hyperparams,
                                                          sq_maha_dist)
            change = new_lower_bound - old_lower_bound
            if abs(change) < self.tol:
                convergence = True
                break
            old_lower_bound = new_lower_bound
            if self.verbose:
                #print(change)
                print("Actual lower bound: %s" % new_lower_bound)
        return new_lower_bound, convergence, params, score



    #This function initializes the expectation values needed before the first fitting
    #iteration using simple maximum likelihood procedures (in the mean-field formulation
    #of this problem, they are all interdependent, so maximum likelihood while holding
    #location_ and scale_inv_ fixed offers a convenient way to get starting estimates).
    #INPUTS:
    #X              --  The raw data. Must be a 2d array where each column is a feature and
    #                   each row is a datapoint.
    #params         --  Object of class ParameterBundle holding slots for all
    #                   of the parameters and expectations that will be needed.
    #
    #OUTPUTS:
    #params         --  The updated ParameterBundle.
    def initialize_expectations(self, X, params, hyperparams):
        params.eta_m = np.full(shape=int(self.n_components), fill_value = hyperparams.eta0,
                               dtype=np.float64)
        
        params.kappa_m = np.full(shape=int(self.n_components), fill_value = hyperparams.kappa0,
                                 dtype=np.float64)
        params.E_log_weights = digamma(params.kappa_m) - digamma(np.sum(params.kappa_m))
        params.wishart_vm = X.shape[0] * np.full(shape=self.n_components,
                    fill_value = 1 / self.n_components) + hyperparams.wishart_v0

        
        params.E_logdet = 2 * np.asarray([np.sum(np.log(np.diag(params.scale_inv_chole_[:,:,i])))
                                                for i in range(self.n_components)])
        params.E_logdet += X.shape[1] * np.log(2)
        params.E_logdet += np.sum([digamma(0.5 * (params.wishart_vm[0] - i)) for i in
                                   range(self.n_components)])

        params.resp = np.zeros((X.shape[0], self.n_components))
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        squaredMahaDistance(X, params.loc_, params.scale_inv_chole_,
                sq_maha_dist)
        #sq_maha_dist = self.sq_maha_distance(X, params.loc_, params.scale_inv_chole_)
        
        assigned_comp = np.argmin(sq_maha_dist, axis=1)
        params.resp[np.arange(X.shape[0]), assigned_comp] = 1.0
        params.a_nm = 0.5 * (params.resp * X.shape[1] + params.df_[np.newaxis,:])
        params.b_nm = 0.5 * (params.resp * sq_maha_dist + params.df_[np.newaxis,:])
        params.E_gamma = params.a_nm / params.b_nm
        params.E_log_gamma = digamma(params.a_nm) - np.log(np.clip(params.b_nm, a_min=1e-12,
                                                                                 a_max=None))
        return params


    #At the end of fitting, the parameters need to be transferred from the ParameterBundle
    #class to the VariationalStudentMixture class so the user can access them easily and
    #so they can be used to make predictions. In addition, certain parameters that are
    #not calculated during fitting (the actual mixture weights for example) must
    #be calculated.
    #INPUTS:
    #params         --  Object of class ParameterBundle holding all of the fit parameters.
    def transfer_fit_params(self, X, params, hyperparams):
        self.scale_, self.scale_inv_cholesky_, self.scale_cholesky_ = np.empty_like(params.scale_),\
                                                np.empty_like(params.scale_), np.empty_like(params.scale_)
        for i in range(self.n_components):
            self.scale_[:,:,i] = params.scale_[:,:,i] / params.wishart_vm[i]
            self.scale_cholesky_[:,:,i] = np.linalg.cholesky(self.scale_[:,:,i])
        self.scale_inv_cholesky_ = self.get_scale_inv_cholesky(self.scale_cholesky_, 
                                        self.scale_inv_cholesky_)
        self.df_ = params.df_
        self.location_ = params.loc_
        updated_alpha = params.kappa_m
        self.mix_weights = params.kappa_m / (self.n_components * hyperparams.kappa0 + 
                X.shape[0])
        self.resp = params.resp



    #The equivalent of the E step in the EM algorithm but for the variational algorithm.
    #Despite some superficial resemblances and shared calculations, this algorithm makes 
    #very different assumptions.
    #
    #Parameters that are updated:
    #sq_maha_dist   --  the squared mahalanobis distance, N x K for K components;
    #params.resp    --  the responsibility of each component for each datapoint, N x K;
    #params.a_nm    --  the first parameter of the gamma distribution that generates
    #                   the "hidden variable" in the Gaussian scale mixture rep of the
    #                   student's t for each datapoint. Shape K for K components.
    #params.b_nm    --  the second parameter of the gamma distribution that etc. N x K
    #                   for N samples, K components.
    #params.E_gamma --  The expectation of the gamma hidden variable. N x K for K
    #                   components. Mean of a gamma distribution is just a / b.
    #params.E_log_gamma --  The expectation of the log of the hidden variable.
    def VariationalEStep(self, X, params, sq_maha_dist):
        squaredMahaDistance(X, params.loc_, params.scale_inv_chole_,
                        sq_maha_dist)
        #sq_maha_dist = self.sq_maha_distance(X, params.loc_, params.scale_inv_chole_)
        
        sq_maha_dist = sq_maha_dist * params.wishart_vm[np.newaxis,:] + \
                       X.shape[1] / params.eta_m[np.newaxis,:]
        weighted_loglik = self.variational_loglik(X, params, sq_maha_dist)
        logprobnorm = logsumexp(weighted_loglik, axis=1)
        with np.errstate(under="ignore"):
            params.resp = weighted_loglik - logprobnorm[:,np.newaxis]
            params.resp = np.exp(params.resp)
        params.a_nm = 0.5 * (params.resp * X.shape[1] + params.df_[np.newaxis,:]) + \
                                      10 * np.finfo(params.resp.dtype).eps
        params.b_nm = 0.5 * params.resp * sq_maha_dist + 0.5 * params.df_[np.newaxis,:] + \
                                      10 * np.finfo(params.resp.dtype).eps
        params.E_gamma = params.a_nm / params.b_nm
        params.E_log_gamma = digamma(params.a_nm) - np.log(params.b_nm)
        #Return the model score (like the variational lower bound, provides another
        #way to compare models)
        score = np.mean(logprobnorm)
        return params, sq_maha_dist, score

    #This function is used by the E-step to calculate responsibilities during training. It is slightly
    #different from the more straightforward likelihood calculation employed for a fitted model
    #(which is identical to EM) -- remember, despite the superficial similarities, the 
    #variational model is derived in a very different way, and has some important differences!
    def variational_loglik(self, X, params, sq_maha_dist):
        loglik = 0.5 * params.E_logdet[np.newaxis,:] + 0.5 * X.shape[1] * \
                 (-np.log(2 * np.pi) + params.E_log_gamma)
        loglik += params.E_log_weights
        loglik -= 0.5 * params.E_gamma * sq_maha_dist
        return loglik
   

    #The equivalent of the M step in the EM algorithm but for the variational algorithm.
    #Despite some superficial resemblances and shared calculations, this algorithm makes 
    #different assumptions.
    #
    #Parameters that are updated:
    #params.kappa_m     --  The updated parameters of the dirichlet distribution. Shape is K
    #                       for K components.
    #params.wishart_vm  --  The updated dof parameters of the Wishart distributions
    #                       that generate the scale matrices for each component. Shape is K.
    #params.eta_m       --  The updated mean covariance prior. SHape is K.
    #params.loc_        --  The location of each component (analogous to mean of a Gaussian). Shape
    #                       is K x D for K components, D dimensions.
    #params.scale_      --  The updated scale matrices (analogous to covariance for a Gaussian).
    #                       Shape is D x D x K for K components, D dimensions.
    #                       Both the cholesky decomposition of the scale matrices and the inverse of this
    #                       cholesky decomposition are stored as well.
    #params.df_         --  Degrees of freedom for each component, shape K. Updated if user
    #                       so specified.
    def VariationalMStep(self, X, params, hyperparams):
        ru = params.E_gamma * params.resp
        ru_sum = np.sum(ru, axis=0) + 10 * np.finfo(params.resp.dtype).eps
        resp_sum = np.sum(params.resp, axis=0) + 10 * np.finfo(params.resp.dtype).eps

        params.kappa_m = resp_sum + hyperparams.kappa0
        params.wishart_vm = resp_sum + hyperparams.wishart_v0
        params.eta_m = ru_sum + hyperparams.eta0
        
        params.E_log_weights = digamma(params.kappa_m) - digamma(np.sum(params.kappa_m))
        
        params.loc_ = np.dot((ru).T, X)
        params.loc_ = params.loc_ / ru_sum[:,np.newaxis]
        
        for i in range(self.n_components):
            scaled_x = X - params.loc_[i,:][np.newaxis,:]
            params.scale_[:,:,i] = np.dot((ru[:,i:i+1] * scaled_x).T, scaled_x)
            loc_vs_prior = params.loc_[i,:] - hyperparams.loc_prior
            params.scale_[:,:,i] += ru_sum[i] * hyperparams.eta0 * np.outer(loc_vs_prior,
                                                loc_vs_prior) / params.eta_m[i]
            params.scale_[:,:,i] += hyperparams.S0
            params.scale_chole_[:,:,i] = np.linalg.cholesky(params.scale_[:,:,i])

            params.loc_[i,:] = params.loc_[i,:] * ru_sum[i] + hyperparams.eta0 * hyperparams.loc_prior
            params.loc_[i,:] = params.loc_[i,:] / params.eta_m[i]

        params.scale_inv_chole_ = self.get_scale_inv_cholesky(params.scale_chole_, 
                                        params.scale_inv_chole_)
        params.E_logdet = 2 * np.asarray([np.sum(np.log(np.diag(params.scale_inv_chole_[:,:,i])))
                                                for i in range(self.n_components)])
        for i in range(self.n_components):
            params.E_logdet[i] += np.sum([digamma(0.5 * (params.wishart_vm[i] - j)) for j in
                                   range(X.shape[1])]) + X.shape[1] * np.log(2)
        if self.fixed_df == False:
            params = self.optimize_df(X, params, ru_sum, resp_sum)
        return params
        

    

    #Updates the variational lower bound so we can assess convergence. We leave out
    #constant terms for simplicity. We evaluate each term of the overall lower bound
    #formula separately and then sum them to get the lower bound. This is 
    #unavoidably messy -- the variational lower bound has eight contributing terms,
    #and there's only so far that we can simplify it.
    #
    #TODO: Currently this is written in an unnecessarily verbose and cumbersome format
    #in order to aid in troubleshooting (it closely parallels the formula for the variational
    #lower bound derived on paper). Later it may be helpful to simplify this as far as that
    #is possible. Constant terms can be removed for starters.
    def update_lower_bound(self, X, params, hyperparams, sq_maha_dist):
        #compute terms involving p(X | params)
        E_log_pX = -X.shape[1] * np.log(2 * np.pi) + X.shape[1] * params.E_log_gamma
        E_log_pX += params.E_logdet[np.newaxis,:] - params.E_gamma * sq_maha_dist
        E_log_pX = 0.5 * np.sum(params.resp * E_log_pX)

        #compute terms involving p(u | other params)
        half_df = params.df_ * 0.5
        E_log_p_u = (half_df[np.newaxis,:] - 1) * params.E_log_gamma
        E_log_p_u -= half_df[np.newaxis,:] * params.E_gamma
        E_log_p_u = np.sum(E_log_p_u, axis=0)
        E_log_p_u += X.shape[0] * (half_df * np.log(half_df) - loggamma(half_df))
        E_log_p_u = np.sum(E_log_p_u)
        
        #Compute terms involving p_mu, p_scale
        E_log_pthetaS = 0.5 * X.shape[1] * np.log(hyperparams.eta0 / (2 * np.pi))
        E_log_pthetaS += ((hyperparams.wishart_v0 - X.shape[1]) * 0.5 * params.E_logdet)
        E_log_pthetaS -= hyperparams.eta0 * X.shape[1] / (2 * params.eta_m)
        E_log_pthetaS = np.sum(E_log_pthetaS)
        for i in range(self.n_components):
            mean_offset = np.dot(params.loc_[i,:], params.scale_inv_chole_[:,:,i])
            mean_offset = mean_offset - np.dot(hyperparams.loc_prior, params.scale_inv_chole_[:,:,i])
            mean_offset = 0.5 * hyperparams.eta0 * params.wishart_vm[i] * np.sum(mean_offset**2)
            trace_term = np.trace(np.matmul(hyperparams.S0, np.matmul(params.scale_inv_chole_[:,:,i],
                                                                      params.scale_inv_chole_[:,:,i].T)))
            E_log_pthetaS -= 0.5 * params.wishart_vm[i] * trace_term
            E_log_pthetaS -= mean_offset
        E_log_pthetaS = 0.5 * E_log_pthetaS


        #Compute terms involving p(z)
        E_log_pz = np.sum(params.resp * params.E_log_weights[np.newaxis,:])

        #Compute terms involving q_z
        E_log_qz = np.sum(params.resp * np.log(np.clip(params.resp, a_min=1e-12, a_max=None)))
        
        #Compute terms involving q(u)
        E_log_qu = (params.a_nm - 1) * digamma(params.a_nm)
        E_log_qu += -loggamma(params.a_nm) - params.a_nm
        E_log_qu = np.sum(E_log_qu + np.log(params.b_nm))

        #Compute terms involving q_thetaS
        E_log_qthetaS = X.shape[1] * 0.5 * np.log(np.clip(params.wishart_vm /
                                                          (2 * np.pi), a_min=1e-12, a_max=None))
        Cnw = [self.wishart_norm(params.scale_inv_chole_[:,:,i], params.wishart_vm[i],
                    return_log = True) for i in range(self.n_components)]
        E_log_qthetaS += 0.5 * (params.wishart_vm - X.shape[1]) * params.E_logdet
        E_log_qthetaS += (params.wishart_vm + 1) * X.shape[1] * 0.5
        E_log_qthetaS = np.sum(E_log_qthetaS)
        E_log_qthetaS += np.sum(Cnw)

        #Compute terms involving p(mix_weights) and q(mix_weights)
        E_log_pweights = loggamma(hyperparams.kappa0 * self.n_components)
        E_log_pweights += np.sum((hyperparams.kappa0 - 1) * params.E_log_weights - \
                                 loggamma(hyperparams.kappa0))
        E_log_qweights = loggamma(np.sum(params.kappa_m))
        E_log_qweights += np.sum((params.kappa_m - 1) * params.E_log_weights - \
                                 loggamma(params.kappa_m))
        E_log_qweights = np.sum(E_log_qweights)

        #Add it all up....                                             
        lower_bound = E_log_pX + E_log_p_u + E_log_pthetaS - E_log_qu - E_log_qthetaS  \
                - E_log_qz + E_log_pz + E_log_pweights - E_log_qweights

        return lower_bound


    #Gets the normalization term for the Wishart distribution (only required for 
    #calculating the variational lower bound). The normalization term is calculated
    #for one component only, so caller must loop over components to get the normalization
    #term for all components as required for the variational lower bound.
    def wishart_norm(self, W_cholesky, eta, return_log = False):
        logdet_term = -eta * np.sum(np.log(np.diag(W_cholesky)))
        inverse_term = np.sum([loggamma( 0.5 * (eta - i)) for i in 
                    range(W_cholesky.shape[0])])
        inverse_term += np.log(2) * (0.5 * eta * W_cholesky.shape[0])
        inverse_term += np.log(np.pi) * (W_cholesky.shape[0] * (W_cholesky.shape[0] - 1) / 4)
        if return_log:
            return logdet_term - inverse_term
        return np.exp(logdet_term - inverse_term)



    #Optimizes the df parameter using Newton Raphson.
    def optimize_df(self, X, params, ru_sum, resp_sum):
        #First calculate the constant term of the degrees of freedom optimization
        #expression so that it does not need to be recalculated on each iteration.
        #Notice some subtle differences from EM here.
        constant_term = params.resp * (params.E_log_gamma - params.E_gamma)
        constant_term = 1 + np.sum(constant_term, axis=0) / resp_sum
        for i in range(self.n_components):
            optimal_df = newton(func = self.dof_first_deriv, x0 = params.df_[i],
                                 fprime = self.dof_second_deriv,
                                 fprime2 = self.dof_third_deriv,
                                 args = ([constant_term[i]]),  maxiter=self.max_iter,
                                 full_output = False, disp=False, tol=1e-3)
            #It may occasionally happen that newton does not converge, usually
            #if the user has set a very small value for max_iter, which is used both
            #for the maximum number of iterations per restart AND for the max
            #number of iterations per newton raphson optimization, or because
            #df is going to infinity, because the distribution is very close to 
            #normal. If it doesn't converge, keep the last estimated value.
            if math.isnan(optimal_df) == False:
                params.df_[i] = optimal_df
            #DF should never be less than 1.
            if params.df_[i] < 1:
                params.df_[i] = 1.0
            #To avoid very large numbers in the df terms of the
            #variational lower bound, we also prevent the degrees of freedom from
            #going very high. Numbers >> than 100 essentially behave the same as
            #a normal distribution, so self.max_df defaults to 100 but user CAN
            #set a different value.
            if params.df_[i] > self.max_df:
                params.df_[i] = self.max_df
        return params


    # First derivative w/r/t df. This is used to 
    #optimize the input value (dof) via the Newton-Raphson algorithm using the scipy.optimize.
    #newton function (see self.optimize.df).
    def dof_first_deriv(self, dof, constant_term):
        clipped_dof = np.clip(dof, a_min=1e-9, a_max=None)
        return constant_term - digamma(dof * 0.5) + np.log(0.5 * clipped_dof)

    #Second derivative w/r/t df. This is used to
    #optimize the input value (dof) via the Newton-Raphson algorithm using the
    #scipy.optimize.newton function (see self.optimize_df).
    def dof_second_deriv(self, dof, constant_term):
        return -0.5 * polygamma(1, 0.5 * dof) + 1 / dof


    #Third derivative of the complete data log likelihood w/r/t df. This is used to optimize
    #the input value (dof) via the Newton-Raphson algorithm using the scipy.optimize.newton
    #function (see self.optimize_df). 
    def dof_third_deriv(self, dof, constant_term):
        return -0.25 * polygamma(2, 0.5 * dof) - 1 / (dof**2)



    #Calculates the squared mahalanobis distance for X to all components. Returns an
    #array of dim N x K for N datapoints, K mixture components. This is a pure python
    #version of the C extension which is used at present (and is significantly faster).
    #This pure python version is preserved for troubleshooting.
    def sq_maha_distance(self, X, loc_, scale_inv_cholesky_):
        sq_maha_dist = np.empty((X.shape[0], scale_inv_cholesky_.shape[2]))
        for i in range(sq_maha_dist.shape[1]):
            y = np.dot(X, scale_inv_cholesky_[:,:,i])
            y = y - np.dot(loc_[i,:], scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            sq_maha_dist[:,i] = np.sum(y**2, axis=1)
        return sq_maha_dist


    #Invert a stack of cholesky decompositions of a scale or precision matrix.
    #The inputs chole_array and inv_chole_array must both be of size D x D x K,
    #where D is dimensionality of data and K is number of components. inv_chole_array
    #will be populated with the output and returned.
    def get_scale_inv_cholesky(self, chole_array, inv_chole_array):
        for i in range(chole_array.shape[2]):
            inv_chole_array[:,:,i] = solve_triangular(chole_array[:,:,i],
                    np.eye(chole_array.shape[0]), lower=True).T
        return inv_chole_array


    #Calculates log p(X | theta) where theta is the current set of parameters but does
    #not apply mixture weights.
    #The function returns an array of dim N x K for N datapoints, K mixture components.
    #It expects to receive the squared mahalanobis distance and the model 
    #parameters (since during model fitting these are still being updated).
    def get_loglikelihood(self, X, sq_maha_dist, df_, scale_cholesky_, mix_weights_):
        sq_maha_dist = 1 + sq_maha_dist / df_[np.newaxis,:]
        
        #THe rest of this is just the calculations for log probability of X for the
        #student's t distributions described by the input parameters broken up
        #into three convenient chunks that we sum on the last line.
        sq_maha_dist = -0.5*(df_[np.newaxis,:] + X.shape[1]) * np.log(sq_maha_dist)
        
        const_term = loggamma(0.5*(df_ + X.shape[1])) - loggamma(0.5*df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(df_) + np.log(np.pi))
        
        scale_logdet = [np.sum(np.log(np.diag(scale_cholesky_[:,:,i])))
                        for i in range(self.n_components)]
        scale_logdet = np.asarray(scale_logdet)
        return -scale_logdet[np.newaxis,:] + const_term[np.newaxis,:] + sq_maha_dist



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

    #At the end of fitting, with a variational mixture, the user may sometimes find that
    #some of the components were unnecessary and can be killed off. This function
    #enables the user to remove empty clusters, which may be cheaper than re-fitting
    #the model with a smaller number of clusters.
    #INPUTS
    #empty_cluster_threshold    --  The number of datapoints below which a cluster
    #                               is considered empty.
    #X                          --  The training dataset. Note that this SHOULD be
    #                               the training dataset -- using a different dataset
    #                               to decide which clusters are empty could give you
    #                               very strange results!
    def purge_empty_clusters(self, X, empty_cluster_threshold = 1):
        cluster_probs = self.predict_proba(X)
        cluster_assignments = np.zeros_like(cluster_probs)
        cluster_assignments[np.arange(cluster_probs.shape[0]),
                np.argmax(cluster_probs, axis=1)] = 1
        cluster_assignments = np.sum(cluster_assignments, axis=0)
        scale, location, df, mix_weights, scale_chole = [], [], [], [], []
        for i in range(self.n_components):
            if cluster_assignments[i] >= empty_cluster_threshold:
                scale.append(self.scale_[:,:,i])
                location.append(self.location_[i,:])
                df.append(self.df_[i])
                mix_weights.append(self.mix_weights_[i])
                scale_chole.append(self.scale_cholesky_[:,:,i])
        if len(mix_weights) == 0:
            raise ValueError("You cannot purge all clusters! Try setting a lower value "
                    "for empty_cluster_threshold.")
        self.n_components = len(mix_weights)
        self.scale_ = np.stack(scale, axis=-1)
        self.location_ = np.stack(location, axis=0)
        self.df_ = np.asarray(df)
        self.mix_weights_ = np.asarray(mix_weights)
        self.mix_weights_ = self.mix_weights_ / np.sum(self.mix_weights_)
        self.scale_cholesky_ = np.stack(scale_chole, axis=-1)
        self.scale_inv_cholesky_ = np.empty_like(self.scale_cholesky_)
        self.scale_inv_cholesky_ = self.get_scale_inv_cholesky(self.scale_cholesky_,
                            self.scale_inv_cholesky_)
        

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
    @property
    def location(self):
        self.check_model()
        return self.location_

    #Setter for the location attribute.
    @location.setter
    def location(self, user_assigned_location):
        self.location_ = user_assigned_location

    #Gets the scale matrices for the fitted mixture model.
    @property
    def scale(self):
        self.check_model()
        return self.scale_

    #Setter for the scale attribute.
    @scale.setter
    def scale(self, user_assigned_scale):
        self.scale_ = user_assigned_scale

    #Gets the mixture weights for a fitted model.
    @property
    def mix_weights(self):
        self.check_model()
        return self.mix_weights_

    #Setter for the mix weights.
    @mix_weights.setter
    def mix_weights(self, user_assigned_weights):
        self.mix_weights_ = user_assigned_weights

    #Gets the degrees of freedom for the fitted mixture model.
    @property
    def degrees_of_freedom(self):
        self.check_model()
        return self.df_

    #Setter for the degrees of freedom.
    @degrees_of_freedom.setter
    def degrees_of_freedom(self, user_assigned_df):
        self.df_ = user_assigned_df


    #Returns log p(X | theta) + log mix_weights. This is called by other class
    #functions which check before calling it that the model has been fitted.
    def get_weighted_loglik(self, X):
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        squaredMahaDistance(X, self.location_, self.scale_inv_cholesky_,
                        sq_maha_dist)
        #sq_maha_dist = self.sq_maha_distance(X, self.location_, self.scale_inv_cholesky_)
        
        loglik = self.get_loglikelihood(X, sq_maha_dist, self.df_, self.scale_cholesky_,
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
        num_parameters = self.n_components - 1 + self.n_components * self.location_.shape[1]
        num_parameters += 0.5 * self.scale_.shape[0] * (self.scale_.shape[1] + 1) * self.scale_.shape[2]
        if self.fixed_df:
            return num_parameters
        else:
            return num_parameters + self.df_.shape[0]


    #Samples from the fitted model with a user-supplied random seed. (It is important not to
    #use the random seed saved as self.random_state because the user may want to easily update
    #the random seed when sampling, depending on their needs.)
    #We sample from the multinomial distribution described by the mixture weights to
    #determine the number of datapoints per component. Next, we sample from the
    #chisquare distribution (a chisquare is a special case of a gamma, and remember
    #that student's t distributions can be described as an infinite scale mixture
    #of Gaussians). Finally, we sample from a standard normal and shift using
    #the location and the sample from the chisquare distribution.
    def sample(self, num_samples = 1, random_seed = 123):
        if num_samples < 1:
            raise ValueError("You can't generate less than one sample!")
        self.check_model()
        rng = np.random.RandomState(random_seed)
        samples_per_component = rng.multinomial(n=num_samples, pvals=self.mix_weights_)
        sample_data = []
        for i in range(self.n_components):
            if np.isinf(self.df_[i]):
                x = 1.0
            else:
                x = rng.chisquare(self.df_[i], size=samples_per_component[i])
            comp_sample = rng.multivariate_normal(np.zeros(self.location_.shape[1]),
                            self.scale_[:,:,i], size=samples_per_component[i])
            sample_data.append(self.location_[i,:] + comp_sample / np.sqrt(x)[:,np.newaxis])
        return np.vstack(sample_data)

