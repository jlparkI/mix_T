import numpy as np, random
import mixt_core
from importlib import reload
reload(mixt_core)
from mixt_core import StudentMixtureModelCore 


class StudentMixture():

    def __init__(self, n_components = 2, tol=1e-5,
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

        #The model fit parameters (if a model has been fitted) are stored in the model
        #core. If we do multiple restarts we can create multiple core objects and store
        #the best one.
        self.model_core = None

    #Function to check whether the input has the correct dimensionality.
    def check_inputs(self, X):
        #Check first whether model has been fitted. If not, model_core will be None.
        self.check_model()
        if X.shape[1] != self.model_core.get_data_dim():
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.model_core.get_data_dim())

    #Function to check whether the model has been fitted yet.
    def check_model(self):
        if self.model_core is None:
            raise ValueError("The model has not been successfully fitted yet.")


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
        best_lower_bound = -np.inf
        best_model = None
        for i in range(self.n_init):
            model_core = StudentMixtureModelCore()
            lower_bound = model_core.fit(X, self.df_, self.tol, 
                    self.n_components, self.reg_covar, self.max_iter)
            if model_core.check_modelcore_convergence() == False:
                print("Fit did not converge! Try increasing max_iter or tol or check "
                        "data for possible issues.")
            elif lower_bound > best_lower_bound:
                best_model = model_core
                best_lower_bound = lower_bound
            if best_model is None:
                print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")
                self.model_core = None
            else:
                self.model_core = best_model



    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


    def predict_proba(self, X):
        self.check_model()
        self.check_inputs(X)
        logprob = self.get_logprob_tdist(X)
        logprob = logprob - logsumexp(logprob, axis=1)[:,np.newaxis]
        return np.exp(logprob)


    def sample(self, n_samples):
        self.check_model()


    #Returns the average log likelihood (i.e. averaged over all datapoints).
    def score(self, X):
        self.check_model()
        self.check_inputs(X)
        
    #Simultaneously fits and makes predictions for the input dataset.
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    #Gets the locations for the fitted mixture model.
    def get_cluster_centers(self):
        self.check_model()
        return self.model_core.get_location()

    #Gets the scale matrices for the fitted mixture model.
    def get_cluster_scales(self):
        self.check_model()
        return self.model_core.get_scale()


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
        self.check_model()
        self.check_inputs(X)
        #Note that we treat each component as having two parameters if df is fixed.
        n_params = self.model_core.get_num_parameters()
        score = self.score(X)
        return 2*n_params - 2*score

    #Returns the Bayes information criterion (BIC) for the input dataset.
    #Useful in selecting the number of components, more heavily penalizes
    #n_components than AIC.
    def bic(self, X):
        self.check_model()
        self.check_inputs(X)
