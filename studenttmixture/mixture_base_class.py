"""The baseclass for the EMStudentMixture and VariationalStudentMixture classes.

Author: Jonathan Parkinson <jlparkinson1@gmail.com>
License: MIT
"""

from abc import ABCMeta
import numpy as np
import numpy as np, math
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, digamma, polygamma
from scipy.optimize import newton
from optimized_mstep_functions import squaredMahaDistance


class MixtureBaseClass(metaclass=ABCMeta):
    """This class serves as a base class for the other mixture classes,
    uniting functions used by all classes in a single location for
    ease of maintenance. Since training functions are specific to
    either the variational or EM fitting approaches, the functions which
    are shared are those used for checking user inputs to ensure they 
    are acceptable and those used by fully trained models.
    """

    def __init__(self):
        """Constructor. Each subclass will override."""
        pass
    
    #The first group of functions stored under the base class check the 
    #user's inputs for training and prediction for validity. There are some
    #additional checks unique to the Variational class which are handled
    #by that class separately.
    
    #Function to check the user specified model parameters for validity.
    def check_user_params(self, n_components, tol, reg_covar, 
            max_iter, n_init, df, random_state,
            init_type):
        """Check the user specified model parameters for validity.
        Raises an appropriate ValueError if input parameters are not
        valid.

        Args:
            n_components (int): The number of mixture components for the model
            tol (float): The threshold at which convergence is determined to
                have been attained when fitting the model.
            reg_covar (float): A regularization value added to the diagonal of
                the covariance matrices for numerical stability.
        max_iter (int): The maximum number of iterations for fitting on
                a given restart.
        n_init (int): The number of restarts when fitting (since the fitting
                algorithms can converge on a local maximum).
        df (int): The degrees of freedom used as a starting point when fitting.
        random_state (int): Seed for the random number generator.
        init_type (str): Either 'kmeans' or 'k++' -- determines how cluster
                centers are initiated.

        Raises:
            ValueError: An error indicating which input(s) were invalid (wrong type,
            unacceptable value -- e.g. negative value for df, etc.)
        """

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
        if init_type not in ["k++", "kmeans"]:
            raise ValueError("init_type must be one of either 'k++' or 'kmeans'.")
    

    def check_inputs(self, X):
        """Checks whether inputs to self.predict have the correct dimensionality.
        This is used by a fitted model, not during fitting.

        Args:
            X (np.ndarray): The N x M input data array. 
        
        Returns:
            x (np.ndarray) The input x array, reshaped to be 2d (if it was 1d),
                otherwise unchanged.
        
        Raises:
            ValueError: A ValueError is raised if the input array is not a numpy array,
                is not of type np.float64, or has incorrect dimensionality (does
                not match the data used for fitting).
        """

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


    def check_model(self):
        """Checks whether model has been fitted; since all parameters are
        set at the same time, we can simply check one of them.
        Does not return anything, merely raises a ValueError if model
        not fitted yet.

        Raises:
            ValueError: Raises an error if model not fitted.
        """

        if self.df_ is None:
            raise ValueError("The model has not been successfully fitted yet.")
    

    def check_fitting_data(self, X):
        """Check data supplied for fitting to ensure it meets basic criteria.
        criteria. We require that N > 2*D and N > 3*n_components. If the user
        supplied a 1d input array, we reshape it to 2d -- this enables us to
        cluster 1d input arrays without forcing the user to reshape them -- a
        little bit more user-friendly than scikitlearn's classes, which will just raise
        a value error if you input a 1d array as X.

        Args:
            X (np.ndarray): Data for fitting.
        
        Returns:
            x (np.ndarray): Input data, reshaped as 2d (if originally 1d), otherwise 
                unchanged.
        
        Raises:
            ValueError: An error is raised if the input array is not a numpy array,
                has too few datapoints, is not of type float64 or is a 3d or
                greater array.
        """

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




    def get_loglikelihood(self, X, sq_maha_dist, df_, scale_cholesky_, mix_weights_):
        """Calculates log p(X | theta) where theta is the current set of parameters but does
        not apply mixture weights.
        The function returns an array of dim N x K for N datapoints, K mixture components.
        It expects to receive the squared mahalanobis distance and the model 
        parameters (since during model fitting these are still being updated).
        Note that the Variational class uses this function only for trained models,
        whereas EM uses it during training.

        Args:
            X (np.ndarray): Input fitting data, dimensions N x M.
            sq_maha_dist (np.ndarray): The squared mahalanobis distance (dimensions N x K
                for K components.)
            df_ (np.ndarray): The degrees of freedom for each component; dimensions K
            scale_cholesky_ (np.ndarray): The cholesky decompositions of the scale matrices.
                Dimensions = M x M x K.
            mix_weights_ (np.ndarray): The mixture weights for the K components.

        Returns:
            The log-likelihood of the data given the input parameters (a float).
        """

        sq_maha_dist = 1 + sq_maha_dist / df_[np.newaxis,:]
        
        #THe rest of this is just the calculations for log probability of X for the
        #student's t distributions described by the input parameters broken up
        #into three convenient chunks that we sum on the last line.
        sq_maha_dist = -0.5*(df_[np.newaxis,:] + X.shape[1]) * np.log(sq_maha_dist)
        
        const_term = gammaln(0.5*(df_ + X.shape[1])) - gammaln(0.5*df_)
        const_term = const_term - 0.5*X.shape[1]*(np.log(df_) + np.log(np.pi))
        
        scale_logdet = [np.sum(np.log(np.diag(scale_cholesky_[:,:,i])))
                        for i in range(self.n_components)]
        scale_logdet = np.asarray(scale_logdet)
        return -scale_logdet[np.newaxis,:] + const_term[np.newaxis,:] + sq_maha_dist


    #The functions below are all called by models that have already been fitted and 
    #are shared between the EM and Variational classes.

    def predict(self, X):
        """Returns a categorical component assignment for each sample in the input. It calls
        predict_proba which performs a model fit check, then assigns each datapoint to
        the component with the largest likelihood for that datapoint.

        Args:
            X (np.ndarray): The input data; a 2d numpy array.

        Returns:
            The categorical assignment for each input datapoint; a 1d numpy array.
        """

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Returns the probability that each sample belongs to each component. Used by
        self.predict.

        Args:
            X (np.ndarray): The input data; an N x M array.

        Returns:
            The probability that each datapoint belongs to each component;
                an N x K array.
        """
        self.check_model()
        x = self.check_inputs(X)
        probs = self.get_component_probabilities(x)
        return probs


    def score(self, X, perform_model_checks = True):
        """Returns the average log likelihood (i.e. averaged over all datapoints). Calls
        self.score_samples to do the actual calculation. Useful for AIC, BIC. It has
        the option to not perform model checks since it is called by AIC and BIC which
        perform model checks before calling score.

        Args:
            X (np.ndarray): The input data; an N x M array.
        
        Returns:
            The average log-likelihood across all datapoints (a float).
        """
        return np.mean(self.score_samples(X, perform_model_checks))

    def score_samples(self, X, perform_model_checks = True):
        """Returns the per sample log likelihood. Useful if fitting a class conditional classifier
        with a mixture for each class.

        Args:
            X (np.ndarray): The input data, a 2d array of shape N x M.
            perform_model_checks (bool): Whether to check whether the model has been fitted and
                check the inputs for validity. Defaults to True.

        Returns:
            The per sample log likelihood (a numpy array).
        """
        if perform_model_checks:
            self.check_model()
            X = self.check_inputs(X)
        return logsumexp(self.get_weighted_loglik(X), axis=1)
        
    def fit_predict(self, X):
        """Simultaneously fits and makes predictions for the input dataset.
        Args:
            X (np.ndarray): A 2d numpy array of shape N x M.

        Returns:

        """
        self.fit(X)
        return self.predict(X)

    @property
    def location(self):
        """Get the locations for the fitted mixture model. self.check_model ensures
        model has been fitted and will raise a value error if it hasn't.
        """
        self.check_model()
        return self.location_

    #Setter for the location attribute.
    @location.setter
    def location(self, user_assigned_location):
        """Set the location attribute."""
        self.location_ = user_assigned_location

    @property
    def scale(self):
        """Get the scale matrices for the fitted mixture model."""
        self.check_model()
        return self.scale_

    @scale.setter
    def scale(self, user_assigned_scale):
        """Set the scale attribute."""
        self.scale_ = user_assigned_scale

    @property
    def mix_weights(self):
        """Get mixture weights for a fitted model."""
        self.check_model()
        return self.mix_weights_

    @mix_weights.setter
    def mix_weights(self, user_assigned_weights):
        """Set mixture weights for a fitted model."""
        self.mix_weights_ = user_assigned_weights

    @property
    def degrees_of_freedom(self):
        """Get degrees of freedom for a fitted mixture model."""
        self.check_model()
        return self.df_

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, user_assigned_df):
        """Set degrees of freedom for a fitted mixture model."""
        self.df_ = user_assigned_df



    def get_weighted_loglik(self, X):
        """Returns log p(X | theta) + log mix_weights. This is called by other class
        functions which check before calling it that the model has been fitted.
        
        Args:
            X (np.ndarray): A 2d numpy array of shape N x M.

        Returns:
            loglik (np.ndarray): The log-likelihood of each datapoint for each cluster.
        """
        
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        squaredMahaDistance(X, self.location_, self.scale_inv_cholesky_, 
                sq_maha_dist)
        
        loglik = self.get_loglikelihood(X, sq_maha_dist, self.df_, self.scale_cholesky_,
                        self.mix_weights_)
        return loglik + np.log(self.mix_weights_)[np.newaxis,:]


    def get_component_probabilities(self, X):
        """Returns the probability that the input data belongs to each component. Used
        for making predictions. This is called by other class functions which check before
        calling it that the model has been fitted.

        Args:
            X (np.ndarray): A 2d N x M array of input data.

        Returns:
            loglik (np.ndarray): The probability that each input datapoint belongs to
                each component.
        """
        weighted_loglik = self.get_weighted_loglik(X)
        with np.errstate(under="ignore"):
            loglik = weighted_loglik - logsumexp(weighted_loglik, axis=1)[:,np.newaxis]
        return np.exp(loglik)


    def get_num_parameters(self):
        """Gets the number of parameters (useful for AIC & BIC calculations). Note that df is only
        treated as a parameter if df is not fixed. This function is only used by AIC and BIC
        which check whether the model has been fitted first so no need to check here.
        
        Returns:
            num_parameters (int): The number of parameters for the model.
        """

        num_parameters = self.n_components - 1 + self.n_components * self.location_.shape[1]
        num_parameters += 0.5 * self.scale_.shape[0] * (self.scale_.shape[1] + 1) * self.scale_.shape[2]
        if self.fixed_df:
            return num_parameters
        else:
            return num_parameters + self.df_.shape[0]


    def sample(self, num_samples = 1, random_seed = 123):
        """Samples from the fitted model with a user-supplied random seed. (It is important not to
        use the random seed saved as self.random_state because the user may want to easily update
        the random seed when sampling, depending on their needs.)
        
        Args:
            num_samples (int): The number of samples to draw.
            random_seed (int): The random seed for random number generation.

        Returns:
            sample_data (np.ndarray): A 2d N x M array of N samples.

        Raises:
            ValueError: Raises an error if the user asks for less than one sample.
        """
        if num_samples < 1:
            raise ValueError("You can't generate less than one sample!")
        self.check_model()
        rng = np.random.RandomState(random_seed)
        #We sample from the multinomial distribution described by the mixture weights to
        #determine the number of datapoints per component.
        samples_per_component = rng.multinomial(n=num_samples, pvals=self.mix_weights_)
        sample_data = []
        #Next, we sample from the 
        #chisquare distribution (a chisquare is a special case of a gamma, and
        #that student's t distributions can be described as an infinite scale mixture
        #of Gaussians). Finally, we sample from a standard normal and shift using
        #the location and the sample from the chisquare distribution.
        for i in range(self.n_components):
            if np.isinf(self.df_[i]):
                x = 1.0
            else:
                x = rng.chisquare(self.df_[i], size=samples_per_component[i]) / self.df_[i]
            comp_sample = rng.multivariate_normal(np.zeros(self.location_.shape[1]),
                            self.scale_[:,:,i], size=samples_per_component[i])
            sample_data.append(self.location_[i,:] + comp_sample / np.sqrt(x)[:,np.newaxis])
        return np.vstack(sample_data)
    
