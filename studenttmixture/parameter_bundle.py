import numpy as np
from scipy.linalg import solve_triangular
from sklearn.cluster import KMeans

#A helper class to store all the model parameters that are updated during training.
#This includes expectations that are required to calculate the lower bound. Because
#there are a lot of steps involved in calculating the variational lower bound, it
#is easier to store everything required in this class so that all the required information
#can be passed as a "bundle". At the end
#of training, the parameters needed to make predictions are transferred to the 
#main model object and the object of this class is no longer required.

#Important note on nomenclature: Note that
#E in the following refers to expectation, i.e. E_log_mixweights = <log mix_weights>.
class ParameterBundle():

    def __init__(self, X, n_components, start_df, random_state, init_type):
        
        self.loc_, self.scale_, self.scale_inv_chole_, self.scale_chole_ = \
                   self.initialize_params(X, n_components, random_state, init_type)
        self.df_ = np.full(shape=n_components, fill_value = start_df)
        #The following values are all expectations required for calculating the lower
        #bound and for parameter updates. They are all set to None because the
        #VariationalStudentMixture class will take responsibility for assigning initial
        #values to each using a simple maximum likelihood procedure, then they will
        #be further updated during fitting.
        
        #Resp is the responsibility of each component for each datapoint.
        self.resp = None
        
        #a_nm and b_nm are the parameters of the Gamma distribution that supplies the hidden
        #variable (below). Required for calculation of the variational lower bound.
        self.a_nm, self.b_nm = None, None
        
        #E_gamma is the the expectation for the hidden variable in the formulation
        #of a student's t distribution as a Gaussian scale mixture.
        self.E_gamma = None

        #E_log_gamma is the expectation for the log of the hidden variable in the formulation
        #of a student's t distribution as a Gaussian scale mixture.
        self.E_log_gamma = None
        
        #The expectation of the log of the mixture weights.
        self.E_log_weights = None

        #Updated hyperparameters.
        self.wishart_vm = None
        self.kappa_m = None
        self.eta_m = None

        
        
    #Initializes the model parameters. The location (analogous to mean of a Gaussian)
    #is initialized using k++. The scale is
    #always initialized using a broad value (the covariance of the full dataset + reg_covar)
    #for all components, while df is set to the starting value.
    #
    #One very important note: For the variational model, unlike the EM model, we primarily
    #track the precision matrix (i.e. the inverse of the scale) for convenience. This
    #means that on initialization we obtain scale_inv rather than scale.
    def initialize_params(self, X, n_components, random_seed, init_type):
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



    #The first option for initializing loc_ is k++, a modified version of the
    #kmeans++ algorithm.
    #On each iteration (until we have reached n_components,
    #a new cluster center is chosen randomly with a probability for each datapoint
    #inversely proportional to its smallest distance to any existing cluster center.
    #This tends to ensure that starting cluster centers are widely spread out. This
    #algorithm originated with Arthur and Vassilvitskii (2007).
    def kplusplus_initialization(self, X, random_seed, n_components):
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

    #The second alternative for initializing loc_is kmeans clustering. It takes as
    #input the data X (NxD for N datapoints, D dimensions) and as a starting point
    #loc_ returned by self.kplusplus_initialization (KxD for K components, D dimensions).
    #Starting with version 0.0.2.2, we are using sklearn's KMeans for clustering --
    #it's simpler than rolling our own and theirs performs very well.
    def kmeans_initialization(self, X, random_state, n_components):
        km = KMeans(n_clusters = n_components, n_init=3,
                random_state = random_state).fit(X)
        return km.cluster_centers_
    
