"""Describes the VariationalMixHyperparams class, used to store 
user-specified hyperparameters for the VariationalStudentMix class
in a convenient container.

Author: Jonathan Parkinson <jlparkinson1@gmail.com>
License: MIT
"""
import numpy as np

class VariationalMixHyperparams():
    """Stores all model hyperpameters in one convenient bundle.
    The hyperparameters are needed during fitting and are not modified at any time.
    Initially they are set to values passed in by the user, which for loc_prior and scale_inv_prior
    may be None, indicating the user wants to initialize those two based on the data (an empirical
    Bayes approach). If they are None then they are initialized here.

    Attributes:
        loc_prior (np.ndarray): The prior for the locations of the mixture components.
        S0 (np.ndarray): An M x M x K array (for M features, K components)
            that is the prior for the inverse of the scale matrices.
        kappa0 (float): The weight concentration prior. Large values cause
            model fitting to prefer many components, small values few components.
        eta0 (float): The diagonal value for the covariance matrix of the 
            prior for the location of the components of the mixture.
        wishart_v0 (float): The degrees of freedom parameter for the Wishart
            prior on the scale matrices.
    """

    def __init__(self, X, loc_prior, scale_inv_prior, weight_concentration_prior = 1.0,
                 wishart_v0 = 1.0, mean_covariance_prior = 1e-3, n_components = 1):
        """Constructor for VariationalMixHyperparams.

        Args:
            X (np.ndarray): The N x M training data array, for N datapoints with M features.
        loc_prior (np.ndarray): The prior for the locations of the mixture components.
        scale_inv_prior (np.ndarray): An M x M x K array (for M features, K components)
            that is the prior for the inverse of the scale matrices.
        weight_concentration_prior (float): The weight concentration prior. 
            Large values cause model fitting to prefer many components, 
            small values few components. Defaults to 1.0.
        wishart_v0 (float): The degrees of freedom parameter for the Wishart
            prior on the scale matrices.
        mean_covariance_prior (float): The diagonal value for the covariance matrix of the 
            prior for the location of the components of the mixture.
        n_components (int): The number of mixture components.

        Raises:
            ValueError: If the prior values supplied by caller are invalid (e.g.
                the loc_prior is not a numpy array).
        """
        self.loc_prior = loc_prior
        self.S0 = scale_inv_prior
        self.kappa0 = weight_concentration_prior
        self.eta0 = mean_covariance_prior
        self.wishart_v0 = wishart_v0

        #Check the user specified hyperparameters and if needed update loc_prior and scale_prior,
        #which the user is allowed to set to None, to data driven values.
        #Default for the location prior is the mean of the data.
        if self.loc_prior is None:
            self.loc_prior = np.mean(X, axis=0)
        #If the user DID pass in a loc_prior, make sure it's valid.
        else:
            if isinstance(self.loc_prior, np.ndarray) == False:
                raise ValueError("Mean prior must be a numpy array!")
            if len(self.loc_prior.shape) != 1:
                raise ValueError("Mean prior must be a 1d numpy array!")
            if self.loc_prior.shape[0] != X.shape[1]:
                raise ValueError("The length of mean_prior must match the "
                        "dimensionality of the training data!")

        if self.S0 is None:
            #Default to a vague empirical prior if nothing else is supplied.
            self.S0 = np.diag(1 / np.diag(np.cov(X, rowvar=False)))
        #Otherwise, check to make sure the prior the user chose was valid.
        else:
            if isinstance(self.S0, np.ndarray) == False:
                raise ValueError("Scale prior must be a numpy array!")
            if len(self.S0.shape) != 2:
                raise ValueError("Scale prior must be a 2d numpy array!")
            if self.S0.shape[0] != X.shape[1] or self.S0.shape[1] != X.shape[1]:
                raise ValueError("The shape of scale_prior must match the "
                        "dimensionality of the training data!")

        if weight_concentration_prior is None:
            self.kappa0 = 1 / n_components

        if self.wishart_v0 is None:
            self.wishart_v0 = X.shape[1]
        elif self.wishart_v0 < X.shape[1]:
            raise ValueError("To ensure numerical stability, the dof parameter of the Wishart prior for the "
                    "scale matrices should not be less than the number of features in the input.")
