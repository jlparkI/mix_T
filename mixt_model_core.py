import numpy as np, random
from scipy.linalg import solve_triangular, cholesky
from scipy.special import loggamma, logsumexp

class StudentTMix_Model_Core():

    def __init__(self, n_components = 2, tol=0.001,
            reg_covar=1e-06, max_iter=300, n_init=1,
            fixed_df = None, random_state=None):
        #Learned parameters optimized during a fit.
        self.mix_weights_ = None
        self.df_ = None
        self.loc_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.converged_ = False


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
        if self.loc_ is None:
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
        if self.max_iter < 1:
            raise ValueError("There must be at least one iteration to "
                    "fit the model.")

    #Function for fitting a model using the parameters the user
    #selected when creating the model object.
    def fit(self, X):
        self.check_fitting_data(X)
        self.initialize_params(X)
        lower_bound = -np.inf
        for i in range(self.max_iter):
            resp, z, current_bound = self.Estep(X)
            self.Mstep(X, resp, z)
            change = current_bound - lower_bound
            if abs(change) < self.tol:
                self.converged = True
                if change < 0:
                    print("Error! Lower bound DECREASED!")
                break
        if self.converged == False:
            print("Fit did not converge! Try increasing max_iter or tol or check "
                "data for possible issues.")


    def Estep(self, X):
        maha_dist = self.maha_distance(X)
        logprobs = self.get_log_prob(X, maha_dist)
        logprob_norm = logsumexp(logprobs, axis=1)
        logprobs = logprobs - logprobnorm[:,np.newaxis]
        z = (self.df_ + X.shape[1])[np.newaxis,:]
        z = z / (self.df_[np.newaxis,:] + maha_dist)
        return np.exp(logprobs), z, np.mean(logprobnorm)


    def Mstep(self, X, resp, z):
        self.mix_weights_ = np.sum(resp, axis=0) / resp.shape[0]
        weights = resp*z
        self.loc_ = np.sum(weights[:,np.newaxis,:] * X[:,:,np.newaxis], axis=0).T
        self.loc_ = self.loc_ / np.sum(weights, axis=0)[np.newaxis,:]
        scale_mats = np.empty((X.shape[1], X.shape[1], self.n_components))
        scale_chol = np.empty((X.shape[1], X.shape[1], self.n_components))
        resp_sum = np.sum(resp, axis=0)
        for i in range(self.n_components):
            scaled_x = X - self.loc[i,:]
            scale_mats[:,:,i] = np.dot(weights[:,i] * scaled_x.T, scaled_x) / resp_sum[i]
            scale_mats[::X.shape[1] + 1] += self.reg_covar
            scale_chol.append(cholesky(scale_mats[:,:,i]))
        if self.fixed_df is None:
            self.update_df(X, resp, z)

    def update_df(self, X, resp, z):
        pass

    def maha_distance(self, X):
        maha_dist = []
        for i in range(self.n_components):
            y = X - self.loc_[i,:]
            y = solve_triangular(self.covar_cholesky[:,:,i], y.T).T
            maha_dist.append(np.sum(y**2, axis=1))
        return np.stack(maha_dist, axis=-1)


    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center for the purposes of calculating covariance.
    def initialize_params(self, X):
        self.loc_ = [X[random.randint(0, X.shape[0]-1), :]]
        if self.n_components == 1:
            return
        dist_arr_list = []
        for i in range(1, self.n_components):
            dist_arr = np.sum((X - self.loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.concatenate(dist_arr_list, axis=1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            self.loc_.append(X[next_center_id,:])

        self.loc_ = np.stack(self.loc_)
        assignments = np.argmin(distmat, axis=1)
        #For initialization, set all covariance matrices to I.
        self.scale_ = [np.eye(X.shape[0]) for i in range(self.n_components_)]
        self.scale_ = np.stack(self.scale_, axis=-1)
        self.scale_cholesky_ = np.copy(self.scale_)
        self.df_ = np.empty(self.n_components)
        #If the user does not supply a fixed df, start with an arbitrary
        #large value, then find better values during fitting.
        if self.fixed_df is not None:
            self.df_.fill(self.fixed_df)
        else:
            self.df_.fill(50.0)


    def get_log_prob(self, X, precalc_dist = None):
        if precalc_dist is None:
            maha_dist = 1 + self.maha_distance(X) / self.df_[np.newaxis,:]
        else:
            maha_dist = 1 + precalc_dist / self.df_[np.newaxis,:]
        maha_dist = -0.5*(self.df_ + X.shape[1])[np.newaxis,:]*np.log(maha_dist)
        const_term = loggamma(0.5*(self.df_ + X.shape[1]))
        const_term -= loggamma(0.5*self.df_)
        const_term -= 0.5*X.shape[1]*(self.df_ + np.pi)
        scale_det = [np.sum(np.diag(self.scale_cholesky_[:,:,i]))
                        for i in self.n_components]
        scale_det = -0.5*np.asarray(scale_det)
        return scale_det[np.newaxis,:] + const_term[np.newaxis,:] + maha_dist

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        self.check_inputs(X)
        return np.exp(self.get_log_prob(X))


    def sample(self, n_samples):
        self.check_model()


    def score(self, X):
        pass


    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


    def get_params(self, X):
        pass


    def aic(self, X):
        pass


    def bic(self, X):
        pass
