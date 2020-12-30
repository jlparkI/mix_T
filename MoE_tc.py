import numpy as np, random

class MoE_TC():

    def __init__(self, tol=0.001,
            max_iter=300, n_init=1):
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init

        #Learned parameters optimized during a fit.
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_ = None
        self.precisions_cholesky_ = None
        self.converged_ = False
        self.lower_bound_ = np.inf
        self.n_iter_ = 0


    #Function to check whether the input has the correct dimensionality.
    def check_inputs(self, X):
        #Check first whether model has been fitted.
        self.check_model()
        if X.shape[1] != self.covariances_.shape[1]:
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.covariances_.shape[0])

    #Function to check whether the model has been fitted yet.
    def check_model(self):
        if self.means_ is None:
            raise ValueError("The model has not been fitted yet.")
        if self.converged == False:
            raise ValueError("Model fitting did not converge; no "
                    "predictions can be generated.")


    #Check data supplied for fitting to make sure it meets basic
    #criteria. We require that N > 2*D and N > 3*n_components.
    def check_fitting_data(self, X):
        if X.shape[0] <= 2*X.shape[1]:
            raise ValueError("Too few datapoints for dataset
            dimensionality. You should have at least 2 datapoints per
            dimension (preferably more).")
        if X.shape[0] <= 3*self.n_components:
            raise ValueError("Too few datapoints for number of components
            in mixture. You should have at least 3 datapoints per mixture
            component (preferably more).")
        if self.n_components < 1 or self.n_components > 100:
            raise ValueError("Too many or too few components. This class will only
            fit models that have at least 1 component and fewer than 100.")

    #Function for fitting a model using the parameters the user
    #selected when creating the model object.
    def fit(self, X):
        self.check_fitting_data(X)
        self.initialize_params(X)


    def Estep(self, X):
        pass

    def Mstep(self, x):
        pass


    def mahalanobis_distance(self, X):

    def initialize_params(self, X):
        self.means_ = [X[random.randint(0, X.shape[0]-1), :]]
        if self.n_components == 1:
            return
        dist_arr_list = []
        for i in range(1, self.n_components):
            dist_arr = np.sum((X - self.means_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.concatenate(dist_arr_list, axis=1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            self.means_.append(X[next_center_id,:])

        self.means_ = np.stack(self.means_)
        assignments = np.argmin(distmat, axis=1)
        #For initialization, set all covariance matrices to I.
        self.covariances_ = [np.eye(X.shape[0]) for i in range(self.n_components_)]
        self.covariances_ = np.stack(self.covariances_, axis=-1)
        self.precisions_ = np.copy(self.covariances_)
        self.precisions_cholesky_ = np.copy(self.covariances_)

    def get_log_prob(self, X):

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs


    def predict_proba(self, X):
        self.check_inputs(X)
        return np.exp(self.get_log_prob(X))




