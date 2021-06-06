import numpy as np

#A helper class to store all the model parameters that are updated during training.
#This includes expectations that are required to calculate the lower bound. Because
#there are a lot of steps involved in calculating the variational lower bound, it
#is easier to store everything required in this class so that all the required information
#can be passed as a "bundle". At the end
#of training, the parameters needed to make predictions are transferred to the 
#main model object and the object of this class is no longer required.
class ParameterBundle():

    def __init__(self, X, n_components, start_df, random_state):
        self.loc_, self.scale_, self.scale_chole_, self.scale_inv_chole_,\
                self.mix_weights_, self.df_ = self.initialize_params(X, n_components, 
                                                start_df, random_state)
        #The parameters can all be initialized using the data. The expectation
        #values will be initialized on the first fitting pass. E refers to
        #expectation, i.e. E_log_mixweights = <log mix_weights>.
        
        #E_gamma is the the expectation for the hidden variable in the formulation
        #of a student's t distribution as a Gaussian scale mixture.
        self.E_gamma = None

        #E_log_gamma is the expectation for the log of the hidden variable in the formulation
        #of a student's t distribution as a Gaussian scale mixture.
        self.E_log_gamma = None

        #a_nm and b_nm are the parameters of the Gamma distribution that supplies the hidden
        #variable from above. Required for calculation of the variational lower bound.
        self.a_nm = None
        self.b_nm = None
        
        #Resp is the responsibility of each component for each datapoint (the expectation
        #for s_nm using Bishop's notation).
        self.E_resp = None
        
        #The expectation of the log of the mixture weights.
        self.E_log_mixweights = None
        
        #The sum of the responsibilities for each component across all datapoints. Shape is dim K.
        self.Nk = None

        #The expectation of the outer product of the location of each component.
        self.E_loc_outer_prod = None

        #Matrix used for construction of the expected scale and location values.
        self.R_adj_scale = None

        #The expectation for the squared mahalanobis distance (w/r/t scale & location).
        self.E_sq_maha_dist = None

        
        
     
    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center to get the starting locations.
    def initialize_params(self, X, max_components, start_df, random_state):
        np.random.seed(random_state)
        df_ = np.full((max_components), start_df, dtype=np.float64)
        loc_ = [X[np.random.randint(0, X.shape[0]-1), :]]
        mix_weights_ = np.empty(max_components)
        mix_weights_.fill(1/max_components)
        dist_arr_list = []
        for i in range(1, max_components):
            dist_arr = np.sum((X - loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            loc_.append(X[next_center_id[0],:])

        loc_ = np.stack(loc_)
        #For initialization, set all covariance matrices to I.
        scale_ = [np.eye(X.shape[1]) for i in range(max_components)]
        scale_ = np.stack(scale_, axis=-1)
        scale_cholesky_ = np.copy(scale_)
        scale_inv_cholesky_ = np.copy(scale_)
        return loc_, scale_, scale_cholesky_, scale_inv_cholesky_, mix_weights_, df_

        self.loc_, self.scale_, self.scale_cholesky_, self.scale_inv_cholesky_,\
                self.mix_weights_, self.df_ = self.initialize_params(X, n_components, 
                                                start_df, random_state)
        
