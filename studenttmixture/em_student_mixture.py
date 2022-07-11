"""Finite mixture of Student's t-distributions fit using EM.

Author: Jonathan Parkinson <jlparkinson1@gmail.com>
License: MIT
"""
import numpy as np, math
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, digamma, polygamma
from scipy.optimize import newton
from optimized_mstep_functions import squaredMahaDistance, EM_Mstep_Optimized_Calc
from sklearn.cluster import KMeans
from .mixture_base_class import MixtureBaseClass



class EMStudentMixture(MixtureBaseClass):
    """A finite student's t mixture model fitted using the EM algorithm.

    Attributes:
        start_df_ (float): The starting value for degrees of freedom.
        fixed_df (bool): If True, degrees of freedom are fixed and are not optimized.
        n_components (int): The number of mixture components.
        tol (float): The threshold at which the fitting process is determined to have
            converged.
        reg_covar (float): A small constant added to the diagonal of scale matrices
            for numerical stability; provides regularization.
        max_iter (int): The maximum number of iterations per restart during fitting.
        init_type (str): The procedure for initializing the cluster centers; one of
            either 'kmeans' or 'k++'.
        n_init (int): The number of restarts (since fitting can converge on a local
            maximum).
        random_state (int): The random seed for random number generator initialization.
        verbose (bool): If True, print updates throughout fitting.
        mix_weights_ (np.ndarray): The mixture weights; a 1d numpy array of shape K
            for K components.
        location_ (np.ndarray): The cluster centers; corresponds to the mean values
            in a Gaussian mixture model. A 2d numpy array of shape K x M (for K
            components, M input dimensions.)
        scale_ (np.ndarray): The component scale matrices; corresponds to the covariance
            matrices of a Gaussian mixture model. A 3d numpy array of shape M x M x K
            for M input dimensions, K components.
        scale_cholesky_ (np.ndarray): The cholesky decompositions of the scale matrices;
            same shape as the scale_ attribute.
        scale_inv_cholesky_ (np.ndarray): The cholesky decomposition of the inverse of
            the scale matrices; same shape as scale_ attribute.
        converged_ (bool): Indicates whether the model converged during fitting.
        n_iter_ (int): The number of iterations completed during fitting.
        df_ (np.ndarray): The degrees of freedom for each mixture component; a 1d
            array of shape K for K components.
    """

    def __init__(self, n_components = 2, tol=1e-5,
            reg_covar=1e-06, max_iter=1000, n_init=1,
            df = 4.0, fixed_df = True, random_state=123, verbose=False,
            init_type = "kmeans"):
        """Constructor for EMStudentMixture.

        Args:
            n_components (int): The number of mixture components. Defaults to 2.
            tol (float): A threshold below which fitting is determined to have
                converged. Defaults to 1e-5.
            reg_covar (float): A regularization parameter added to the diagonal
                of the scale matrices for numerical stability. Defaults to 1e-6,
                which is a good value in general.
            max_iter (int): The maximum number of iterations per restart. Defaults
                to 1000.
            n_init (int): The number of restarts (since a restart may converge on
                a local maximum). Defaults to 1.
            df (float): The starting value for degrees of freedom for all mixture
                components. Defaults to 4.0.
            fixed_df (bool): If True, df_ remains fixed at the starting value and
                is not optimized. Defaults to True.
            random_state (int): The seed for the random number generator for
                initializing the model.
            verbose (bool): If True, print updates throughout fitting. Defaults to
                False.
            init_type (str): One of 'kmeans', 'k++'. Determines how cluster centers
                are initialized. 'kmeans' provides better performance and is the
                default; 'k++' may be slightly faster.
        """
        super().__init__()
        self.check_user_params(n_components, tol, reg_covar, max_iter, n_init, df, random_state,
                init_type)
        self.start_df_ = float(df)
        self.fixed_df = fixed_df
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_type = init_type
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.mix_weights_ = None
        self.location_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.df_ = None




    def fit(self, X):
        """Fit model using the parameters the user selected when creating the model object.
        Creates multiple restarts by calling the fitting_restart function n_init times.
        
        Args:
            X (np.ndarray): The raw data for fitting. This must be either a 1d array, in which case
                self.check_fitting_data will reshape it to a 2d 1-column array, or
                a 2d array where each column is a feature and each row a datapoint.
        """
        x = self.check_fitting_data(X)
        best_lower_bound = -np.inf
        #We use self.n_init restarts and save the best result. More restarts = better 
        #chance to find the best possible solution, but also higher cost.
        for i in range(self.n_init):
            #Increment random state so that each random initialization is different from the
            #rest but so that the overall chain is reproducible.
            lower_bound, convergence, loc_, scale_, scale_inv_cholesky_, mix_weights_,\
                    df_, scale_cholesky_ = self.fitting_restart(x, self.random_state + i)
            if self.verbose:
                print("Restart %s now complete"%i)
            if convergence == False:
                print("Restart %s did not converge!"%(i+1))
            #If this is the best lower bound we've seen so far, update our saved
            #parameters.
            elif lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                self.df_, self.location_, self.scale_ = df_, loc_, scale_
                self.scale_inv_cholesky_ = scale_inv_cholesky_
                self.scale_cholesky_ = scale_cholesky_
                self.mix_weights_ = mix_weights_
                self.converged_ = True
        if self.converged_ == False:
            print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")
    

    def fitting_restart(self, X, random_state):
        """A single fitting restart.
        
        Args:
            X (np.ndarray): The raw data. Must be a 2d array where each column is a feature and
                each row is a datapoint. The caller (self.fit) ensures this is true.
            random_state (int): The seed for the random number generator.
        
        Returns:
            current_bound (float): The lower bound for the current fitting iteration. The caller (self.fit)
                keeps the set of parameters that have the best associated lower bound.
            convergence (bool): A boolean indicating convergence or lack thereof.
            loc_ (np.ndarray): The locations (analogous to means for a Gaussian) of the components.
                Shape is K x M for K components, M dimensions.
            scale_ (np.ndarray): The scale matrices (analogous to covariance for a Gaussian).
                Shape is M x M x K for M dimensions, K components.
            scale_inv_cholesky_ (np.ndarray): The cholesky decomposition of the 
                inverse of the scale matrices. Shape is M x M x K for D dimensions, 
                K components.
            mix_weights_ (np.ndarray): The mixture weights. SHape is K for K components.
            df_ (np.ndarray): The degrees of freedom. Shape is K for K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale matrices.
                Shape is M x M x K for M dimensions, K components.
        """
        df_ = np.full((self.n_components), self.start_df_, dtype=np.float64)
        loc_, scale_, mix_weights_, scale_cholesky_, scale_inv_cholesky_ = \
                self.initialize_params(X, random_state, self.init_type)
        lower_bound, convergence = -np.inf, False
        sq_maha_dist = np.empty((X.shape[0], self.n_components))
        #For each iteration, we run the E step calculations then the M step
        #calculations, update the lower bound then check for convergence.
        for i in range(self.max_iter):
            resp, E_gamma, current_bound = self.Estep(X, df_, loc_, scale_inv_cholesky_, 
                                scale_cholesky_, mix_weights_, sq_maha_dist)
            
            mix_weights_, loc_, scale_, scale_cholesky_, scale_inv_cholesky_,\
                            df_ = self.Mstep(X, resp, E_gamma, scale_, 
                                scale_cholesky_, df_, scale_inv_cholesky_)
            change = current_bound - lower_bound
            if abs(change) < self.tol:
                convergence = True
                break
            lower_bound = current_bound
            if self.verbose:
                print("Change in lower bound: %s"%change)
                print("Actual lower bound: %s" % current_bound)
        return current_bound, convergence, loc_, scale_, scale_inv_cholesky_,\
                mix_weights_, df_, scale_cholesky_



    def Estep(self, X, df_, loc_, scale_inv_cholesky_, scale_cholesky_, mix_weights_,
            sq_maha_dist):
        """Update the "hidden variables" in the mixture description.
        
        Args:
            X (np.ndarray): The input data. A 2d numpy array of shape N x M.
            df_ (np.ndarray): The degrees of freedom. Shape is K for K components.
            loc_ (np.ndarray): The current values of the locations of the components.
                Shape is K x D for K components, D dimensions.
            scale_inv_cholesky_ (np.ndarray): The cholesky decomposition of the 
                inverse of the scale matrices. Shape is M x M x K for M dimensions, 
                K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale matrices.
                Shape is M x M x K for M dimensions, K components.
            mix_weights_ (np.ndarray: The mixture weights. Shape is K for K components.
            sq_maha_dist (np.ndarray): The squared mahalanobis distance for each datapoint
                for each component. A 2d N x K numpy array.

        Returns:
            resp (np.ndarray): The responsibilities of each component for each datapoint.
                Shape is N x K (N datapoints, K components).
            E_gamma (np.ndarray): The ML estimate of the "hidden variable" described by 
                a gamma distribution in the formulation of the student's t-distribution
                as a scale mixture of normals. 
            lower_bound (np.ndarray): The lower bound (to determine whether fit has converged).
            """
        #We use the C extension to calculate squared mahalanobis distance and pass
        #it the array (sq_maha_dist) in which we would like to store the output.
        squaredMahaDistance(X, loc_, scale_inv_cholesky_, sq_maha_dist)
        
        loglik = self.get_loglikelihood(X, sq_maha_dist, df_, 
                scale_cholesky_, mix_weights_)

        weighted_log_prob = loglik + np.log(np.clip(mix_weights_,
                                        a_min=1e-9, a_max=None))[np.newaxis,:]
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            resp = np.exp(weighted_log_prob - log_prob_norm[:, np.newaxis])
        E_gamma = (df_[np.newaxis,:] + X.shape[1]) / (df_[np.newaxis,:] + sq_maha_dist)
        lower_bound = np.mean(log_prob_norm)
        return resp, E_gamma, lower_bound



    def Mstep(self, X, resp, E_gamma, scale_, scale_cholesky_, df_,
        scale_inv_cholesky_):
        """The M-step in mixture fitting. Updates the component parameters using the "hidden variable"
        values calculated in the E-step.
    
        Args:
            X (np.ndarray): The input data. A numpy array of shape N x M for N datapoints,
                M features.
            resp (np.ndarray): The responsibilities of each component for each datapoint.
                Shape is N x K (N datapoints, K components).
            E_gamma (np.ndarray): The ML estimate of the "hidden variable" described by 
                a gamma distribution in the formulation of the student's t-distribution
                as a scale mixture of normals. 
            scale_ (np.ndarray): The scale matrices (analogous to covariance for a Gaussian).
                Shape is M x M x K for M dimensions, K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale matrices.
                Shape is M x M x K for M dimensions, K components.
            df_ (np.ndarray): The degrees of freedom. Shape is K for K components.
            scale_inv_cholesky_ (np.ndarray): The cholesky decomposition of the inverse of the scale matrices.
                Shape is M x M x K for M dimensions, K components.
        
        Returns:
            mix_weights_ (np.ndarray): The mixture weights. SHape is K for K components.
            loc_ (np.ndarray): The locations (analogous to means for a Gaussian) of the components.
                Shape is K x D for K components, D dimensions.
            scale_ (np.ndarray): The scale matrices (analogous to covariance for a Gaussian).
                Shape is D x D x K for D dimensions, K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale matrices.
                Shape is D x D x K for D dimensions, K components.
            scale_inv_cholesky_ (np.ndarray): The cholesky decomposition of the inverse of the scale matrices.
                Shape is D x D x K for D dimensions, K components.
            df_ (np.ndarray): The degrees of freedom. Shape is K for K components.
        """
        mix_weights_ = np.mean(resp, axis=0)
        ru = resp * E_gamma
        loc_ = np.dot(ru.T, X)
        resp_sum = np.sum(ru, axis=0) + 10 * np.finfo(resp.dtype).eps
        loc_ = loc_ / resp_sum[:,np.newaxis]
        
        #This call to the Cython extension updates the scale, scale cholesky
        #and scale inv cholesky matrices and is faster than the pure
        #Python implementation for a large number of clusters or dimensions.
        EM_Mstep_Optimized_Calc(X, ru, scale_, scale_cholesky_,
                        scale_inv_cholesky_, loc_, resp_sum, self.reg_covar)
        if self.fixed_df == False:
            df_ = self.optimize_df(X, resp, E_gamma, df_)
        return mix_weights_, loc_, scale_, scale_cholesky_, scale_inv_cholesky_, df_



    def optimize_df(self, X, resp, E_gamma, df_):
        """Optimizes the df parameter using Newton Raphson.
        
        Args:
            X (np.ndarray): The input data, a numpy array of shape N x M
                for N datapoints, M features.
            resp (np.ndarray): The responsibility of each cluster for each
                datapoint. An N x K numpy array for K components.
            E_gamma (np.ndarray): The ML estimate of the "hidden variable" described by 
                a gamma distribution in the formulation of the student's t-distribution
                as a scale mixture of normals. 
            df_ (np.ndarray): The current estimate of degrees of freedom.

        Returns:
            df_ (np.ndarray): The updated estimate for degrees of freedom.
        """
        #First calculate the constant term of the degrees of freedom optimization
        #expression so that it does not need to be recalculated on each iteration.
        df_x_dim = 0.5 * (df_ + X.shape[1])
        resp_sum = np.sum(resp, axis=0)
        ru_sum = np.sum(resp * (np.log(E_gamma) - E_gamma), axis=0)
        constant_term = 1.0 + (ru_sum / resp_sum) + digamma(df_x_dim) - \
                    np.log(df_x_dim)
        for i in range(self.n_components):
            optimal_df = newton(func = self.dof_first_deriv, x0 = df_[i],
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
                df_[i] = optimal_df
            #DF should never be less than 1 but can go arbitrarily high.
            if df_[i] < 1:
                df_[i] = 1.0
        return df_


    def dof_first_deriv(self, dof, constant_term):
        """First derivative of the complete data log likelihood w/r/t df. This is used to 
        optimize the input value (dof) via the Newton-Raphson algorithm using the scipy.optimize.
        newton function (see self.optimize_df).
        """
        clipped_dof = np.clip(dof, a_min=1e-9, a_max=None)
        return constant_term - digamma(dof * 0.5) + np.log(0.5 * clipped_dof)

    def dof_second_deriv(self, dof, constant_term):
        """Second derivative of the complete data log likelihood w/r/t df. This is used to
        optimize the input value (dof) via the Newton-Raphson algorithm using the
        scipy.optimize.newton function (see self.optimize_df).
        """
        return -0.5 * polygamma(1, 0.5 * dof) + 1 / dof


    def dof_third_deriv(self, dof, constant_term):
        """Third derivative of the complete data log likelihood w/r/t df. This is used to optimize
        the input value (dof) via the Newton-Raphson algorithm using the scipy.optimize.newton
        function (see self.optimize_df). 
        """
        return -0.25 * polygamma(2, 0.5 * dof) - 1 / (dof**2)



    def sq_maha_distance(self, X, loc_, scale_inv_cholesky_):
        """Calculates the squared mahalanobis distance for X to all components. Returns an
        array of dim N x K for N datapoints, K mixture components.
        This is slower than the C extension so it is primarily used for testing
        purposes.

        Args:
            X (np.ndarray): A 2d numpy array containing the input data of shape
                N x M for N datapoints, M features.
            loc_ (np.ndarray): A 2d numpy array of shape K x M for K components,
                M features.
            scale_inv_cholesky_ (np.ndarray): The inverse of the cholesky 
                decomposition of the scale matrices.

        Returns:
            sq_maha_dist (np.ndarray): An N x K for N datapoints, K components
                numpy array containing the squared mahalanobis distance for
                each datapoint to each cluster.
        """
        sq_maha_dist = np.empty((X.shape[0], scale_inv_cholesky_.shape[2]))
        for i in range(sq_maha_dist.shape[1]):
            y = np.dot(X, scale_inv_cholesky_[:,:,i])
            y = y - np.dot(loc_[i,:], scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            sq_maha_dist[:,i] = np.sum(y**2, axis=1)
        return sq_maha_dist


    def get_scale_inv_cholesky(self, scale_cholesky_, scale_inv_cholesky_):
        """Gets the inverse of the cholesky decomposition of the scale matrix.

        Args:
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the 
                scale matrices, of shape M x M x K for M features, K components.
            scale_inv_cholesky_ (np.ndarray): The inverse of the cholesky
                decomposition of the scale matrices. Same shape as scale_cholesky_.
                This will be overwritten here.

        Returns:
            scale_inv_cholesky_ (np.ndarray): The input scale_inv_cholesky_
                overwritten with the updated inverse of the cholesky decomposition
                of the scale matrices.
        """
        for i in range(scale_cholesky_.shape[2]):
            scale_inv_cholesky_[:,:,i] = solve_triangular(scale_cholesky_[:,:,i],
                    np.eye(scale_cholesky_.shape[0]), lower=True).T
        return scale_inv_cholesky_


    

    def initialize_params(self, X, random_seed, init_type):
        """Initializes the model parameters. Two approaches are available for initializing
        the location (analogous to mean of a Gaussian): k++ and kmeans. THe scale is
        always initialized using a broad value (the covariance of the full dataset + reg_covar)
        for all components, while mixture_weights are always initialized to be equal for all
        components, and df is set to the starting value. All initialized parameters are 
        returned to caller.

        Args:
            X (np.ndarray): The input data; a 2d numpy array of shape N x M
                for N datapoints, M features.
            random_seed (int): The random seed for the random number generator.
            init_type (str): One of 'kmeans', 'k++'. 'kmeans' is better but slightly
                slower.

        Returns:
            loc_ (np.ndarray): The initial cluster centers which will be optimized
                during fitting. This is a numpy array of shape K x M for K components,
                M features.
            scale_ (np.ndarray): Initial scale matrices (analogous to covariance of
                a Gaussian). This is a numpy array of shape M x M x K for M features,
                K components. These are initially set to a very broad default.
            mix_weights_ (np.ndarray): The mixture weights. This is a numpy array
                of shape K for K components.
            scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale
                matrices; same shape as scale_.
            scale_inv_cholesky_ (np.ndarray): The inverse of the cholesky decomposition
                of the scale matrices. Same shape as scale_.
        """
        if init_type == "kmeans":
            loc_ = self.kmeans_initialization(X, random_seed)
        else:
            loc_ = self.kplusplus_initialization(X, random_seed)

        mix_weights_ = np.empty(self.n_components)
        mix_weights_.fill(1/self.n_components)

        #Set all scale matrices to a broad default -- the covariance of the data.
        default_scale_matrix = np.cov(X, rowvar=False)
        default_scale_matrix.flat[::X.shape[1] + 1] += self.reg_covar

        #For 1-d data, ensure default scale matrix has correct shape.
        if len(default_scale_matrix.shape) < 2:
            default_scale_matrix = default_scale_matrix.reshape(-1,1)
        scale_ = np.stack([default_scale_matrix for i in range(self.n_components)],
                        axis=-1)
        scale_cholesky_ = [np.linalg.cholesky(scale_[:,:,i]) for i in range(self.n_components)]
        scale_cholesky_ = np.stack(scale_cholesky_, axis=-1)
        scale_inv_cholesky_ = np.empty_like(scale_cholesky_)
        scale_inv_cholesky_ = self.get_scale_inv_cholesky(scale_cholesky_,
                            scale_inv_cholesky_)
        return loc_, scale_, mix_weights_, scale_cholesky_, scale_inv_cholesky_



    def kplusplus_initialization(self, X, random_seed):
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
        for i in range(1, self.n_components):
            dist_arr = np.sum((X - loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            loc_.append(X[next_center_id[0],:])
        return np.stack(loc_)



    def kmeans_initialization(self, X, random_state):
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
        km = KMeans(n_clusters = self.n_components, n_init=3,
                random_state = random_state).fit(X)
        return km.cluster_centers_


    #The remaining functions are called only for trained models. They are kept separate from the base
    #class because the variational approach does not require BIC or AIC calculations; 
    #these are EM-specific.


    def aic(self, X):
        """Returns the Akaike information criterion (AIC) for the input dataset.
        Useful in selecting the number of components. AIC places heavier
        weight on model performance than BIC and less weight on penalizing
        a large number of parameters.

        Args:
            X (np.ndarray): The input data, a 2d numpy array with N datapoints, 
                M dimensions.

        Returns:
            aic (float): The Akaike information criterion for the data. Lower is
                better.
        """
        self.check_model()
        x = self.check_inputs(X)
        n_params = self.get_num_parameters()
        score = self.score(x, perform_model_checks = False)
        return 2 * n_params - 2 * score * X.shape[0]


    def bic(self, X):
        """Returns the Bayes information criterion (BIC) for the input dataset.
        Useful in selecting the number of components, more heavily penalizes
        n_components than AIC.

        Args:
            X (np.ndarray): The input data, a 2d numpy array with N datapoints, 
                M dimensions.

        Returns:
            bic (float): The Bayes information criterion for the data. Lower is
                better.
        """
        self.check_model()
        x = self.check_inputs(X)
        score = self.score(x, perform_model_checks = False)
        n_params = self.get_num_parameters()
        return n_params * np.log(x.shape[0]) - 2 * score * x.shape[0]

