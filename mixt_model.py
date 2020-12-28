import numpy as np

class StudentTMixture():

    def __init__(self, n_components = 2, tol=0.001,
            reg_covar=1e-06, max_iter=300, n_init=1,
            fixed_df = False):
        self.fixed_df = fixed_df
        self.n_components=n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        

