import unittest, numpy as np, scipy
from scipy import stats
from finite_student_mixture import FiniteStudentMixture


class TestCoreProbabilityFunctions(unittest.TestCase):



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
        print("np.allclose result for scipy's multivariate t logpdf vs "
                "the FiniteMixture get_loglikelihood function: %s"%np.allclose(true_loglik, loglik))
        self.assertTrue(np.allclose(true_loglik, loglik))





if __name__ == "__main__":
    unittest.main()
