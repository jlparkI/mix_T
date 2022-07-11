"""Describes the ParameterBundle class used for storing parameters in the VariationalStudentMixture.

Author: Jonathan Parkinson <jlparkinson1@gmail.com>
License: MIT
"""

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.cluster import KMeans


class ParameterBundle():
    """A helper class to store all the model parameters that are updated during training.
    This includes expectations that are required to calculate the lower bound. 
    It is used exclusively by the VariationalStudentMixture class. Because
    there are a lot of steps involved in calculating the variational lower bound, it
    is easier to store everything required in this class so that all the required information
    can be passed as a "bundle". At the end
    of training, the parameters needed to make predictions are transferred to the 
    main model object and the object of this class is no longer required.

    Attributes:
        resp (np.ndarray): The responsibility of each component for each datapoint.
            An N x K numpy array for N datapoints, K components.
        a_nm (np.ndarray): The shape parameter of the Gamma distribution that supplies
            the hidden variable. Required for calculation of the lower bound.
        b_nm (np.ndarray): The scale parameter of the Gamma distribution that supplies
            the hidden variable. Required for calculation of the lower bound.
        E_gamma (np.ndarray): The expectation for the hidden variable in the formulation
            of a student's t distribution as a Gaussian scale mixture.
        E_log_gamma (np.ndarray): The expectation for the log of the hidden variable 
            in the formulation of a student's t distribution as a Gaussian scale mixture.
        E_log_weights (np.ndarray): The expectation of the log of the mixture weights.

        #Updated hyperparameters.
        wishart_vm (np.ndarray): The updated hyperparameter of the Wishart prior.
            Shape is K for K components.
        self.kappa_m (np.ndarray): The updated hyperparameter of the dirichlet 
            prior. Shape is K for K components.
        eta_m (np.ndarray): The updated mean covariance prior. Shape is K
            for K components.
    """


    def __init__(self, X, n_components, start_df, random_state, init_type):
        """Constructor for the ParameterBundle.

        Args:
            X (np.ndarray): The input data matrix of shape N x M for N
                datapoints, M features.
            n_components (int): The number of mixture components.
            start_df (float): The starting value for degrees of freedom
                for all clusters.
            random_state (int): The random seed for the random number
                generator.
            init_type (str): Must be one of 'kmeans', 'k++'. Determines
                how cluster centers are initialized. kmeans is better,
                k++ is slightly faster.
        """
        self.loc_, self.scale_, self.scale_inv_chole_, self.scale_chole_ = \
                   self.initialize_params(X, n_components, random_state, init_type)
        self.df_ = np.full(shape=n_components, fill_value = start_df)
        #The following values are all expectations required for calculating the lower
        #bound and for parameter updates. They are all set to None because the
        #VariationalStudentMixture class will take responsibility for assigning initial
        #values to each using a simple maximum likelihood procedure, then they will
        #be further updated during fitting.
        
        self.resp = None
        self.a_nm, self.b_nm = None, None
        self.E_gamma = None
        self.E_log_gamma = None
        self.E_log_weights = None
        self.wishart_vm = None
        self.kappa_m = None
        self.eta_m = None

        
        
    def initialize_params(self, X, n_components, random_seed, init_type):
        """Initializes the model parameters. The location (analogous to mean of a Gaussian)
        is initialized using either kmeans or k++ as indicated by user. The scale is
        always initialized using a broad value (the covariance of the full dataset + reg_covar)
        for all components, while df is set to the starting value.
    
        One very important note: For the variational model, unlike the EM model, we primarily
        track the precision matrix (i.e. the inverse of the scale) for convenience. This
        means that on initialization we obtain scale_inv rather than scale.
        
        Args:
            X (np.ndarray): An N x M input data array for N datapoints, M features.
            n_components (int): The number of mixture components.
            random_seed (int): The random seed for the random number generator.
            init_type (str): One of 'kmeans', 'k++'; 'kmeans' is better,
                'k++' is slightly faster.

        Returns:
            loc_ (np.ndarray): The initial cluster centers which will be optimized
                during fitting. This is a numpy array of shape K x M for K components,
                M features.
            scale_ (np.ndarray): Initial scale matrices (analogous to covariance of
                a Gaussian). This is a numpy array of shape M x M x K for M features,
                K components. These are initially set to a very broad default.
            scale_inv_cholesky_ (np.ndarray): The inverse of the cholesky decomposition
                of the scale matrices. Same shape as scale_.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale
                matrices; same shape as scale_.
        """
        if init_type == "kmeans":
            loc_ = self.kmeans_initialization(X, random_seed, n_components)
        else:
            loc_ = self.kplusplus_initialization(X, random_seed, n_components)

        #Set all scale matrices to a broad default -- the covariance of the data.
        default_scale_matrix = np.cov(X, rowvar=False)

        #For 1-d data, ensure default scale matrix has correct shape.
        if len(default_scale_matrix.shape) < 2:
            default_scale_matrix = default_scale_matrix.reshape(-1,1)
        scale_cholesky_ = np.linalg.cholesky(default_scale_matrix)
        scale_inv_cholesky_ = solve_triangular(scale_cholesky_,
                            np.eye(scale_cholesky_.shape[0]), lower=True).T
        scale_inv_cholesky_ = [scale_inv_cholesky_ for i in range(n_components)]
        scale_inv_cholesky_ = np.stack(scale_inv_cholesky_, axis=-1)
        scale_ = [default_scale_matrix for i in range(n_components)]
        scale_ = np.stack(scale_, axis=-1)
        scale_cholesky_ = [scale_cholesky_ for i in range(n_components)]
        scale_cholesky_ = np.stack(scale_cholesky_, axis=-1)

        return loc_, scale_, scale_inv_cholesky_, scale_cholesky_



    def kplusplus_initialization(self, X, random_seed, n_components):
        """The first option for initializing loc_ is k++, a modified version of the
        kmeans++ algorithm.
        On each iteration until we have reached n_components,
        a new cluster center is chosen randomly with a probability for each datapoint
        inversely proportional to its smallest distance to any existing cluster center.
        This tends to ensure that starting cluster centers are widely spread out.

        Args:
            X (np.ndarray): The input training data.
            random_seed (int): The random seed for the random number generator.

        Returns:
            loc_ (np.ndarray): The selected cluster centers.
        """
        np.random.seed(random_seed)
        loc_ = [X[np.random.randint(0, X.shape[0]-1), :]]
        dist_arr_list = []
        for i in range(1, n_components):
            dist_arr = np.sum((X - loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            loc_.append(X[next_center_id[0],:])
        return np.stack(loc_)

    def kmeans_initialization(self, X, random_state, n_components):
        """The second alternative for initializing loc_is kmeans clustering. It takes as
        input the data X (NxD for N datapoints, D dimensions) and as a starting point
        loc_ returned by self.kplusplus_initialization (KxD for K components, D dimensions).
        Starting with version 0.0.2.2, we are using sklearn's KMeans for clustering --
        it's simpler than rolling our own and theirs performs very well.

        Args:
            X (np.ndarray): The input data.
            random_state (int): The random seed for the random number generator.

        Returns:
            km.cluster_centers_ (np.ndarray): A numpy array of shape (n_components,)
                with the center of each cluster; these will be refined during fitting.
        """
        km = KMeans(n_clusters = n_components, n_init=3,
                random_state = random_state).fit(X)
        return km.cluster_centers_
    
