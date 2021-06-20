import numpy as np

#A simple helper class to store all model hyperpameters in one convenient bundle.
#The hyperparameters are needed during fitting and are not modified at any time.
#Initially they are set to values passed in by the user, which for loc_prior and scale_inv_prior
#may be None, indicating the user wants to initialize those two based on the data (an empirical
#Bayes approach).
#Either way, when fitting begins, VariationalStudentMixture.fit() will call
#VariationalMixHyperparams.initialize() to check the stored hyperparameter values,
#ensure they are valid and set loc_prior and scale_inv_prior to data driven defaults if the
#user did in fact set them to None.
class VariationalMixHyperparams():

    def __init__(self, X, loc_prior, scale_inv_prior, weight_concentration_prior = 1.0,
                 wishart_v0 = 1.0, mean_covariance_prior = 1e-3, n_components = 1):
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
