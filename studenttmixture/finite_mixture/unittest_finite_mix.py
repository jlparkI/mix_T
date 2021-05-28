import unittest, numpy as np, scipy
from scipy import stats, spatial
from finite_student_mixture import FiniteStudentMixture


class TestCoreProbabilityFunctions(unittest.TestCase):


    #Test the squared mahalanobis distance calculation by generating a set
    #of random datapoints and measuring squared mahalanobis distance to a
    #prespecified location & scale matrix distribution, then compare the
    #result with scipy's mahalanobis function. For unittesting this package
    #computational expense is not a key concern so we use inefficient approaches
    #(np.linalg.inv, Python for loop over a numpy array etc) for simplicity.
    def test_sq_maha_distance(self):
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



    #Generate an arbitrary t-distribution using scipy's t-distribution function,
    #sample from it and ensure the probabilities calculated by FiniteStudentMixture
    #are identical.
    def test_log_likelihood_calculation(self):
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





if __name__ == "__main__":
    unittest.main()
