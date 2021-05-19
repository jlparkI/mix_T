import numpy as np

#A simple helper class to store all model hyperpameters in one convenient bundle.
#The hyperparameters are needed during fitting and are not modified at any time.
class VariationalMixHyperparams():

    def __init__(self, loc_prior, scale_inv_prior, degrees_of_freedom_prior,
                    weight_concentration_prior, wishart_v0, mean_covariance_prior,
                    dirichlet_alpha_m_prior):
        self.loc = loc_prior
        self.wishart_scale_inv = scale_inv_prior
        self.dof = degrees_of_freedom_prior
        self.mean_cov_prior = mean_covariance_prior
        self.alpha0 = weight_concentration_prior
        self.alpha_m = dirichlet_alpha_m_prior
        self.wishart_v0 = wishart_v0

    #Check the user hyperparameters to make sure they are sensible. If any of them
    #are None, calculate useful defaults using the input data.
    def check_hyperparameters(self, X):
        if self.mean_prior is None:
            self.mean_prior = np.mean(X, axis=0)
        else:
            if isinstance(self.mean_prior, np.ndarray) == False:
                raise ValueError("Mean prior must be a numpy array!")
            if len(self.mean_prior.shape) != 1:
                raise ValueError("Mean prior must be a 1d numpy array!")
            if self.mean_prior.shape[0] != X.shape[1]:
                raise ValueError("The length of mean_prior must match the "
                        "dimensionality of the training data!")
        if self.scale_prior is None:
            x_centered = X - np.mean(X, axis=0)[np.newaxis,:]
            self.scale_prior = np.matmul(x_centered.T, x_centered)
        else:
            if isinstance(self.scale_prior, np.ndarray) == False:
                raise ValueError("Scale prior must be a numpy array!")
            if len(self.scale_prior.shape) != 2:
                raise ValueError("Scale prior must be a 2d numpy array!")
            if self.scale_prior.shape[0] != X.shape[1] or self.scale_prior.shape[1] != X.shape[1]:
                raise ValueError("The shape of scale_prior must match the "
                        "dimensionality of the training data!")
        if self.dof_prior is None:
            self.dof_prior = X.shape[1]
        else:
            try:
                self.dof_prior = float(self.dof_prior)
            except:
                raise ValueError("Degrees of freedom prior must be a float!")
            if self.dof_prior < 1:
                raise ValueError("Degrees of freedom prior must be >= 1!")
        if self.weight_conc_prior is None:
            self.weight_conc_prior = 1 / self.max_components
        else:
            try:
                self.weight_conc_prior = float(self.weight_conc_prior)
            except:
                raise ValueError("Weight concentration prior must be a float!")
            if self.weight_conc_prior <= 0:
                raise ValueError("Weight concentration prior must be > 0!")

