import numpy as np, random
from scipy.linalg import solve_triangular, cholesky
from scipy.special import gammaln, digamma, polygamma, logsumexp


#Note that we require the user to
#specify the degrees of freedom. You CAN using the EM algorithm do MAP estimation
#over location, scale and df simultaneously BUT df is a nuisance because 1) it adds
#an extra (frequently unnecessary) parameter, 2) it must be obtained using an
#iterative algorithm within the M-step, so it slows things down considerably.
#Moreover, (see et al.), nu=4 is a good value for most cases
#where you need a student T because you are worried about outliers. Large values
#for nu are basically an MVN anyway, while nu=1 is more heavy-tailed than you
#usually need in practice. So, we require user to specify nu and default to 4.

class StudentTMixture():

    def __init__(self, n_components = 2, tol=0.001,
            reg_covar=1e-06, max_iter=500, n_init=1,
            df = 4, random_state=None):
        #General model parameters specified by user.
        self.df_ = float(df)
        self.n_components=n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        #Learned parameters optimized during a fit.
        self.mix_weights_ = None
        self.loc_ = None
        self.scale_ = None
        self.scale_cholesky_ = None
        self.scale_inv_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0


    #Function to check whether the input has the correct dimensionality.
    def check_inputs(self, X):
        #Check first whether model has been fitted.
        self.check_model()
        if X.shape[1] != self.scale_.shape[1]:
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.scale_.shape[0])

    #Function to check whether the model has been fitted yet.
    def check_model(self):
        if self.loc_ is None:
            raise ValueError("The model has not been fitted yet.")
        if self.converged_ == False:
            raise ValueError("Model fitting did not converge; no "
                    "predictions can be generated.")


    #Check data supplied for fitting to make sure it meets basic
    #criteria. We require that N > 2*D and N > 3*n_components.
    def check_fitting_data(self, X):
        if X.shape[0] <= 2*X.shape[1]:
            raise ValueError("Too few datapoints for dataset "
            "dimensionality. You should have at least 2 datapoints per "
            "dimension (preferably more).")
        if X.shape[0] <= 3*self.n_components:
            raise ValueError("Too few datapoints for number of components "
            "in mixture. You should have at least 3 datapoints per mixture "
            "component (preferably more).")
        if self.n_components < 1 or self.n_components > 100:
            raise ValueError("Too many or too few components. This class will only "
            "fit models that have at least 1 component and fewer than 100.")
        if self.max_iter < 1:
            raise ValueError("There must be at least one iteration to "
                    "fit the model.")
        if self.df_ < 1:
            raise ValueError("Degrees of freedom must be greater than or equal to "
                    "1.")
        if self.df_ > 1000:
            raise ValueError("Degrees of freedom must be < 1000; values > 30 "
                    "will give results very similar to a Gaussian mixture, "
                    "values as large as the one you have entered are indistinguishable "
                    "from a Gaussian mixture. DF = 4 is suggested as a good default.")

    #Function for fitting a model using the parameters the user
    #selected when creating the model object.
    def fit(self, X):
        self.check_fitting_data(X)
        self.initialize_params(X)
        lower_bound = -np.inf
        for i in range(self.max_iter):
            resp, z, logprobs = self.Estep(X)
            current_bound = self.get_lower_bound(X, logprobs)
            self.Mstep(X, resp, z)
            change = lower_bound - current_bound
            if change < 0:
                print("%s          %s"%(current_bound, lower_bound))
            if abs(change) < self.tol:
                self.converged_ = True
                break
            lower_bound = current_bound
        if self.converged_ == False:
            print("Fit did not converge! Try increasing max_iter or tol or check "
                "data for possible issues.")

    #The e-step in mixture fitting. Calculates responsibilities for each datapoint
    #and the lower bound.
    def Estep(self, X):
        maha_dist = self.maha_distance(X)
        logprobs = self.get_log_prob(X, maha_dist)
        logprob_norm = logsumexp(logprobs, axis=1)
        with np.errstate(under="ignore"):
            logprobs = logprobs - logprob_norm[:,np.newaxis]
        z = (self.df_ + X.shape[1]) / (self.df_ + maha_dist)
        return np.exp(logprobs), z, logprobs


    def Mstep(self, X, resp, z):
        self.mix_weights_ = np.mean(resp, axis=0)
        weights = resp*z
        self.loc_ = np.dot(weights.T, X)
        self.loc_ = self.loc_ / np.sum(weights, axis=0)[:,np.newaxis]
        resp_sum = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
        for i in range(self.n_components):
            scaled_x = X - self.loc_[i,:][np.newaxis,:]
            self.scale_[:,:,i] = np.dot(weights[:,i]*scaled_x.T,
                            scaled_x) / resp_sum[i]
            self.scale_[:,:,i].flat[::X.shape[1] + 1] += self.reg_covar
            self.scale_cholesky_[:,:,i] = cholesky(self.scale_[:,:,i], lower=True)
        self.get_scale_inv_cholesky()

    def maha_distance(self, X):
        maha_dist = []
        for i in range(self.n_components):
            y = np.dot(X, self.scale_inv_cholesky_[:,:,i])
            y -= np.dot(self.loc_[i,:], self.scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            maha_dist.append(np.sum(y**2, axis=1))
        return np.stack(maha_dist, axis=-1)

    #Gets the inverse of the cholesky decomposition of the scale matrix,
    #(don't use np.linalg.inv! triangular_solver is better)
    def get_scale_inv_cholesky(self):
        for i in range(self.n_components):
            self.scale_inv_cholesky_[:,:,i] = solve_triangular(self.scale_cholesky_[:,:,i].T,
                    np.eye(self.scale_cholesky_.shape[0]), lower=True).T

    #We initialize parameters using a modified kmeans++ algorithm, whereby
    #cluster centers are chosen and each datapoint is given a hard
    #assignment to the cluster
    #with the closest center for the purposes of calculating covariance.
    def initialize_params(self, X):
        self.loc_ = [X[random.randint(0, X.shape[0]-1), :]]
        self.mix_weights_ = np.empty(self.n_components)
        self.mix_weights_.fill(1/self.n_components)
        dist_arr_list = []
        for i in range(1, self.n_components):
            dist_arr = np.sum((X - self.loc_[i-1])**2, axis=1)
            dist_arr_list.append(dist_arr)
            distmat = np.stack(dist_arr_list, axis=-1)
            min_dist = np.min(distmat, axis=1)
            min_dist = min_dist / np.sum(min_dist)
            next_center_id = np.random.choice(distmat.shape[0], size=1, p=min_dist)
            self.loc_.append(X[next_center_id[0],:])

        self.loc_ = np.stack(self.loc_)
        assignments = np.argmin(distmat, axis=1)
        #For initialization, set all covariance matrices to I.
        self.scale_ = [np.eye(X.shape[1]) for i in range(self.n_components)]
        self.scale_ = np.stack(self.scale_, axis=-1)
        self.scale_cholesky_ = np.copy(self.scale_)
        self.scale_inv_cholesky_ = np.copy(self.scale_)

    #Lower bound on the log likelihood.
    def get_log_prob(self, X, precalc_dist = None):
        if precalc_dist is None:
            maha_dist = 1 + self.maha_distance(X) / self.df_
        else:
            maha_dist = 1 + precalc_dist / self.df_
        maha_dist = -0.5*(self.df_ + X.shape[1])*np.log(maha_dist)
        const_term = gammaln(0.5*(self.df_ + X.shape[1])) - gammaln(0.5*self.df_)
        const_term -= 0.5*X.shape[1]*(np.log(self.df_) + np.log(np.pi))
        scale_det = [np.sum(np.log(np.diag(self.scale_cholesky_[:,:,i])))
                        for i in range(self.n_components)]
        scale_det = -np.asarray(scale_det)
        return scale_det[np.newaxis,:] + const_term + maha_dist + np.log(self.mix_weights_[np.newaxis,:])


    def get_lower_bound(self, X, precalc_logprob=None):
        logprob = precalc_logprob
        if logprob is None:
            logprob = self.get_log_prob(X)
        loglik = logprob*self.mix_weights_[np.newaxis,:]
        return np.mean(loglik)


    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        self.check_inputs(X)
        logprob = self.get_log_prob(X)
        logprob = logprob - logsumexp(logprob, axis=1)[:,np.newaxis]
        return np.exp(logprob)


    def sample(self, n_samples):
        self.check_model()

    #Returns the average log likelihood (i.e. averaged over all datapoints).
    def score(self, X):
        self.check_inputs(X)
        return np.mean(self.get_log_prob(X))
        
    #Simultaneously fits and makes predictions for the input dataset.
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    #A handy function for toy 2d datasets -- generates coordinates for
    #elipses describing each t-distribution in the mixture.
    def get_ellipses(self, mag_factor=1):
        if self.scale_[:,:,0].shape[0] > 2:
            raise ValueError("This function will only generate elipse coordinates for "
                    "2d datasets.")
            return
        eigvals, eigvecs = [], []
        elipses = []
        base_coords = np.linspace(0, 2*np.pi, 250)
        polar_coords = np.empty((250,2))
        polar_coords[:,0] = np.cos(base_coords)
        polar_coords[:,1] = np.sin(base_coords)
        for i in range(self.n_components):
            val, vec = np.eigh(self.scale_[:,:,i])
            eigvals.append(val)
            eigvecs.append(vec)
            
    #Returns the Akaike information criterion (AIC) for the input dataset.
    #Useful in selecting the number of components.
    def aic(self, X):
        self.check_inputs(X)
        #Note that we treat each component as having two parameters since df is
        #fixed.
        n_params = 2*self.n_components
        loglik = np.sum(self.get_log_prob(X))
        return 2*n_params - 2*loglik

    #Returns the Bayes information criterion (BIC) for the input dataset.
    #Useful in selecting the number of components, more heavily penalizes
    #n_components than AIC.
    def bic(self, X):
        self.check_inputs(X)
        #Note that we treat each component as having two parameters since df is
        #fixed.
        n_params = 2*self.n_components
        loglik = np.sum(self.get_log_prob(X))
        return n_params*np.log(n_params) - 2*loglik
