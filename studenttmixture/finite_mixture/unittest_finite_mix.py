import unittest, numpy as np, scipy
from scipy import stats, spatial
from finite_student_mixture import FiniteStudentMixture


class TestCoreProbabilityFunctions(unittest.TestCase):


    #Test the full EM fitting procedure using an artificial dataset. Ensure
    #the model converges on locations roughly equivalent to those used to generate
    #the toy dataset. This test uses an artificial dataset composed of student's
    #t distributions.
    def test_em_fit_students_t(self):
        print("*********************")
        #Arbitrary scale matrices...
        true_cov = [np.asarray([[0.025, 0.0075, 0.00175],
                            [0.0075, 0.0070, 0.00135],
                            [0.00175, 0.00135, 0.00043]]),
                    np.asarray([[1.2, 0.1, 0.42],
                            [0.1, 0.5, 0.0035],
                            [0.42, 0.0035, 0.35]]),
                    np.asarray([[1.6, 0.0, 0.0],
                            [0.0, 4.5, 0.0],
                            [0.0, 0.0, 3.2]])]
        true_cov = np.stack(true_cov, axis=-1)
        #Arbitrary locations...
        true_loc = np.asarray([[-2.5,3.6,1.2],
                            [3.2,-5.2,-2.1],[4.5,3.6,7.2]])
        
        #Generate datapoints in 3 t distributions with df 4 specified by the
        #arbitrary locations and scale matrices we selected above.
        np.random.seed(123)
        samples = [scipy.stats.multivariate_t.rvs(true_loc[i,:], true_cov[:,:,i],
                    df=4, size=500) for i in range(3)]
        samples = np.vstack(samples)
        #We set a low value for tol to get the fit as "tight" as possible.
        FiniteMix = FiniteStudentMixture(fixed_df=False, random_state=123,
                n_components=3, max_iter=1500, tol=1e-7, init_type="k++")
        FiniteMix.fit(samples)

        #Retrieve the fit parameters and sort them so they can be compared to
        #the true ones.
        fit_loc = FiniteMix.location
        idx = np.argsort(fit_loc[:,0])
        fit_loc = fit_loc[idx,:]
        fit_cov = FiniteMix.scale[:,:,idx]
        fit_df = FiniteMix.df[idx]
        fit_mix_weights = FiniteMix.mix_weights

        print("For a three cluster toy dataset in a 3d space...")
        #For location, distance from fit location to true location should be 
        #less than 1% of the norm of the true location.
        location_distance = np.linalg.norm(fit_loc - true_loc, axis=1)
        true_loc_norm = np.linalg.norm(true_loc, axis=1)
        location_outcome = np.max(location_distance / true_loc_norm) < 0.02

        print("Is the distance from the fit locations to the true locations "
                "< 2 percent of the norm of the true locations? %s"%location_outcome)
        
        mix_weight_abs_error = np.max(np.abs(fit_mix_weights - 0.33))
        mix_weight_outcome = mix_weight_abs_error < 0.02
        print("Are the mixture weights all between 0.31 and 0.35? %s"%mix_weight_outcome)
        
        fit_df_error = np.max(np.abs(fit_df - 4))
        fit_df_outcome = fit_df_error < 0.5
        print("Are the estimated degrees of freedom all between 3.5 and 4.5? %s"%fit_df_outcome)
        
        #Last but not least, we check the covariance matrices. This is a little
        #tricky but to try to keep this simple we use Herdin (2005)'s approach
        #to estimating distance and require that distance be smaller than a
        #threshold.
        cov_mat_dist = []
        for i in range(true_loc.shape[0]):
            cov_mat_sim = (np.trace(np.dot(true_cov[:,:,i], fit_cov[:,:,i])) / 
                    (np.linalg.norm(true_cov[:,:,i]) * np.linalg.norm(fit_cov[:,:,i])))
            cov_mat_dist.append(1 - cov_mat_sim)
        cov_outcome = np.max(cov_mat_dist) < 0.01
        print("Is the covariance matrix distance < 0.01? %s"%cov_outcome)

        self.assertTrue(mix_weight_outcome)
        self.assertTrue(location_outcome)
        self.assertTrue(fit_df_outcome)
        self.assertTrue(cov_outcome)

        print('\n')
   


    #Generate an arbitrary t-distribution using scipy's t-distribution function,
    #sample from it and ensure the probabilities calculated by FiniteStudentMixture
    #are identical.
    def test_log_likelihood_calculation(self):
        print("*********************")
        FiniteMix = FiniteStudentMixture()
        #An arbitrary scale matrix...
        covmat = np.asarray([[0.025, 0.0075, 0.00175],
                            [0.0075, 0.0070, 0.00135],
                            [0.00175, 0.00135, 0.00043]])
        chole_covmat = np.linalg.cholesky(covmat).reshape((3,3,1))
        chole_inv_cov = np.empty((3,3,1))
        chole_inv_cov = FiniteMix.get_scale_inv_cholesky(chole_covmat, chole_inv_cov)
        #An arbitrary location...
        loc = np.asarray([[0.156,-0.324,0.456]])
        covmat = covmat.reshape((3,3,1))
        #Set up FiniteMix using the parameters of the distribution.
        FiniteMix.loc_ = loc
        FiniteMix.scale_ = covmat
        FiniteMix.scale_cholesky_ = chole_covmat
        FiniteMix.scale_inv_cholesky_ = chole_inv_cov
        FiniteMix.df_ = np.asarray([4.0])
        FiniteMix.mix_weights_ = np.asarray([1.0])
        FiniteMix.converged_ = True
        FiniteMix.n_components = 1

        #Generate a few hundred samples from scipy's multivariate t.
        rv = scipy.stats.multivariate_t(loc.flatten(), covmat.reshape((3,3)),
                df=4)
        samples = rv.rvs(size=300)
        true_loglik = rv.logpdf(samples)

        sq_maha_dist = FiniteMix.sq_maha_distance(samples, FiniteMix.loc_, FiniteMix.scale_inv_cholesky_)
        loglik = FiniteMix.get_loglikelihood(samples, sq_maha_dist, FiniteMix.df_,
                    FiniteMix.scale_cholesky_, FiniteMix.mix_weights_).flatten()
        outcome = np.allclose(true_loglik, loglik)
        print("Does scipy's multivariate t logpdf match "
                "the FiniteMixture get_loglikelihood function? %s"%outcome)
        self.assertTrue(outcome)

        print('\n')

    #Test the squared mahalanobis distance calculation by generating a set
    #of random datapoints and measuring squared mahalanobis distance to a
    #prespecified location & scale matrix distribution, then compare the
    #result with scipy's mahalanobis function. For unittesting this package
    #computational expense is not a key concern so we use inefficient approaches
    #(np.linalg.inv, Python for loop over a numpy array etc) for simplicity.
    def test_sq_maha_distance(self):
        print("*********************")
        np.random.seed(123)
        X = np.random.uniform(low=-10,high=10,size=(250,3))
        FiniteMix = FiniteStudentMixture()
        #Arbitrary scale matrices...
        covmat1 = np.asarray([[0.025, 0.0075, 0.00175],
                            [0.0075, 0.0070, 0.00135],
                            [0.00175, 0.00135, 0.00043]])
        covmat2 = np.asarray([[1.2, 0.1, 0.42],
                            [0.1, 0.5, 0.0035],
                            [0.42, 0.0035, 0.35]])
        chole_covmat1 = np.linalg.cholesky(covmat1)
        chole_covmat2 = np.linalg.cholesky(covmat2)
        chole_covmat = np.stack([chole_covmat1, chole_covmat2], axis=-1)
        chole_inv_cov = np.empty((3,3,2))
        chole_inv_cov = FiniteMix.get_scale_inv_cholesky(chole_covmat, chole_inv_cov)
        #Arbitrary locations...
        loc = np.asarray([[0.156,-0.324,0.456],[-2.5,3.6,1.2]])
        scale_inv1 = np.linalg.inv(covmat1)
        scale_inv2 = np.linalg.inv(covmat2)
        scale_inv = np.stack([scale_inv1, scale_inv2], axis=-1)

        finite_mix_dist = FiniteMix.vectorized_sq_maha_distance(X, loc, chole_inv_cov)
        true_dist = np.empty((X.shape[0], 2))
        for i in range(X.shape[0]):
            true_dist[i,0] = scipy.spatial.distance.mahalanobis(X[i,:], loc[0,:],
                            scale_inv1)**2
            true_dist[i,1] = scipy.spatial.distance.mahalanobis(X[i,:], loc[1,:],
                            scale_inv2)**2
        outcome = np.allclose(finite_mix_dist, true_dist)
        print("Does scipy's mahalanobis match "
                "the FiniteMixture vectorized_sq_maha_distance function? %s"%outcome)
        self.assertTrue(outcome)
        print('\n')


if __name__ == "__main__":
    unittest.main()
