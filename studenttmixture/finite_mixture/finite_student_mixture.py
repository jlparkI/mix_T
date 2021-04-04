import numpy as np
from .finite_model_core import FiniteModelCore 


class FiniteStudentMixture():

    def __init__(self, n_components = 2, tol=1e-3,
            reg_covar=1e-06, max_iter=500, n_init=1,
            df = 4.0, fixed_df = True, random_state=123, verbose=False):
        self.check_user_params(n_components, tol, reg_covar, max_iter, n_init, df, random_state)
        #General model parameters specified by user.
        self.start_df_ = float(df)
        self.fixed_df = fixed_df
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        #The model fit parameters (if a model has been fitted) are stored in the model
        #core. If we do multiple restarts we can create multiple core objects and store
        #the best one.
        self.model_core = None
        self.verbose = verbose


    #Function to check the user specified model parameters for validity.
    def check_user_params(self, n_components, tol, reg_covar, max_iter, n_init, df, random_state):
        try:
            n_components = int(n_components)
            tol = float(tol)
            n_init = int(n_init)
            random_state = int(random_state)
            max_iter = int(max_iter)
            reg_covar = float(reg_covar)
        except:
            raise ValueError("n_components, tol, max_iter, n_init, reg_covar and random state should be numeric.")
        if df > 1000:
            raise ValueError("Very large values for dof will give results essentially identical to a Gaussian mixture."
                    "DF = 4 is suggested as a good default. If fixed_df is False, the df will be "
                             "optimized.")
        if df < 1:
            raise ValueError("Inappropriate starting value for df!")
        if max_iter < 1:
            raise ValueError("Inappropriate value for the maximum number of iterations! Must be >= 1.")
        if n_init < 1:
            raise ValueError("Inappropriate value for the number of restarts! Must be >= 1.")
        if tol <= 0:
            raise ValueError("Inappropriate value for tol! Must be greater than 0.")
        if n_components < 1:
            raise ValueError("Inappropriate value for n_components! Must be >= 1.")
        if reg_covar < 0:
            raise ValueError("Reg covar must be >= 0.")


    #Function to check whether the input has the correct dimensionality.
    def check_inputs(self, X):
        if isinstance(X, np.ndarray) == False:
            raise ValueError("X must be a numpy array.")
        #Check first whether model has been fitted. If not, model_core will be None.
        self.check_model()
        if X.dtype != "float64":
            raise ValueError("The input array should be of type float64.")
        if len(X.shape) > 2:
            raise ValueError("Only 1d or 2d arrays are accepted as input.")
        x = X
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if x.shape[1] != self.model_core.get_data_dim():
            raise ValueError("Dimension of data passed does not match "
                    "dimension of data used to fit the model! The "
                    "data used to fit the model has D=%s"
                    %self.model_core.get_data_dim())
        return x


    #Function to check whether the model has been fitted yet.
    def check_model(self):
        if self.model_core is None:
            raise ValueError("The model has not been successfully fitted yet.")


    #Check data supplied for fitting to make sure it meets basic
    #criteria. We require that N > 2*D and N > 3*n_components.
    def check_fitting_data(self, X):
        if isinstance(X, np.ndarray) == False:
            raise ValueError("X must be a numpy array.")
        if X.dtype != "float64":
            raise ValueError("The input array should be of type float64.")
        if len(X.shape) > 2:
            raise ValueError("This class only accepts 1d or 2d arrays as inputs.")
        if len(X.shape) == 1:
            x = X.reshape(-1,1)
        else:
            x = X
        if x.shape[0] <= 2*x.shape[1]:
            raise ValueError("Too few datapoints for dataset "
            "dimensionality. You should have at least 2 datapoints per "
            "dimension (preferably more).")
        if x.shape[0] <= 3*self.n_components:
            raise ValueError("Too few datapoints for number of components "
            "in mixture. You should have at least 3 datapoints per mixture "
            "component (preferably more).")
        return x


    #Function for fitting a model using the parameters the user
    #selected when creating the model object.
    def fit(self, X):
        x = self.check_fitting_data(X)
        best_lower_bound = -np.inf
        self.model_core = None
        model_cores = []
        for i in range(self.n_init):
            #Increment random state so that each random initialization is different from the
            #rest but so that the overall chain is reproducible.
            model_core = FiniteModelCore(self.random_state + i, self.fixed_df)
            lower_bound = model_core.fit(x, self.start_df_, self.tol,
                    self.n_components, self.reg_covar, self.max_iter, self.verbose)
            model_cores.append(model_core)
            if self.verbose:
                print("Restart %s now complete"%i)
            if model_core.check_modelcore_convergence() == False:
                print("Restart %s did not converge!"%(i+1))
            elif lower_bound > best_lower_bound:
                self.model_core = model_core
                best_lower_bound = lower_bound
        if self.model_core is None:
            print("The model did not converge on any of the restarts! Try increasing max_iter or "
                        "tol or check data for possible issues.")


    #Returns a categorical component assignment for each sample in the input.
    def predict(self, X):
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    #Returns the probability that each sample belongs to each component.
    def predict_proba(self, X):
        self.check_model()
        x = self.check_inputs(X)
        probs = self.model_core.get_component_probabilities(x)
        return probs


    #Returns the average log likelihood (i.e. averaged over all datapoints).
    def score(self, X, run_model_checks=True):
        if run_model_checks:
            self.check_model()
            X = self.check_inputs(X)
        return self.model_core.score(X)

    #Returns the per sample log likelihood. Useful if fitting a class conditional classifier
    #with a mixture for each class.
    def score_samples(self, X, run_model_checks=True):
        if run_model_checks:
            self.check_model()
            X = self.check_inputs(X)
        return self.model_core.score_samples(X)
        
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

    #Gets the mixture weights for a fitted model.
    def get_weights(self):
        self.check_model()
        return self.model_core.get_mix_weights()

    #Gets the degrees of freedom for the fitted mixture model.
    def get_df(self):
        self.check_model()
        return self.model_core.get_df()

    #Returns the Akaike information criterion (AIC) for the input dataset.
    #Useful in selecting the number of components.
    def aic(self, X):
        self.check_model()
        x = self.check_inputs(X)
        n_params = self.model_core.get_num_parameters()
        score = self.score(x, run_model_checks = False)
        return 2 * n_params - 2 * score * X.shape[0]

    #Returns the Bayes information criterion (BIC) for the input dataset.
    #Useful in selecting the number of components, more heavily penalizes
    #n_components than AIC.
    def bic(self, X):
        self.check_model()
        x = self.check_inputs(X)
        score = self.score(x, run_model_checks = False)
        n_params = self.model_core.get_num_parameters()
        return 2 * n_params * np.log(x.shape[0]) - 2 * score * x.shape[0]
