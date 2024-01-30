"""Unit tests for the variational student mixture."""
import unittest, numpy as np, scipy, sys
from scipy import stats, spatial
import studenttmixture
from studenttmixture.variational_student_mixture import VariationalStudentMixture


class TestVariationalMixture(unittest.TestCase):


    def test_em_fit_students_t(self):
        """Test the full fitting procedure using an artificial dataset. Ensure
        the model converges on locations roughly equivalent to those used to generate
        the toy dataset. This test uses an artificial dataset composed of student's
        t distributions."""
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
        VarMix = VariationalStudentMixture(fixed_df=False, random_state=123,
                n_components=3, max_iter=4000, tol=1e-7)
        VarMix.fit(samples)

        #Retrieve the fit parameters and sort them so they can be compared to
        #the true ones.
        fit_loc = VarMix.location
        idx = np.argsort(fit_loc[:,0])
        fit_loc = fit_loc[idx,:]
        fit_cov = VarMix.scale[:,:,idx]
        fit_df = VarMix.degrees_of_freedom[idx]
        fit_mix_weights = VarMix.mix_weights

        print("For a three cluster toy dataset in a 3d space...")
        #For location, distance from fit location to true location should be 
        #less than 1% of the norm of the true location.
        location_distance = np.linalg.norm(fit_loc - true_loc, axis=1)
        true_loc_norm = np.linalg.norm(true_loc, axis=1)
        location_outcome = np.max(location_distance / true_loc_norm) < 0.02

        print("Is distance from fit locations to true locations "
                "< 2 percent of the norm of true locations? %s"%location_outcome)
        
        mix_weight_abs_error = np.max(np.abs(fit_mix_weights - 0.33))
        mix_weight_outcome = mix_weight_abs_error < 0.02
        print("Are the mixture weights all between 0.31 and 0.35? %s"%mix_weight_outcome)
        
        fit_df_error = np.max(np.abs(fit_df - 4))
        fit_df_outcome = fit_df_error < 2
        print("Are the estimated degrees of freedom all between 2 and 6? %s"%fit_df_outcome)
        
        #Next we check the covariance matrices. This is a little
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
   


    def test_log_likelihood_calculation(self):
        """Generate an arbitrary t-distribution using scipy's t-distribution function,
        sample from it and ensure the probabilities calculated by VariationalStudentMixture
        are identical."""
        print("*********************")
        VarMix = VariationalStudentMixture()
        #An arbitrary scale matrix...
        covmat = np.asarray([[0.025, 0.0075, 0.00175],
                            [0.0075, 0.0070, 0.00135],
                            [0.00175, 0.00135, 0.00043]])
        chole_covmat = np.linalg.cholesky(covmat).reshape((3,3,1))
        chole_inv_cov = np.empty((3,3,1))
        chole_inv_cov = VarMix.get_scale_inv_cholesky(chole_covmat, chole_inv_cov)

        #An arbitrary location...
        loc = np.asarray([[0.156,-0.324,0.456]])
        covmat = covmat.reshape((3,3,1))

        #Set up VarMix using the parameters of the distribution.
        VarMix.loc_ = loc
        VarMix.scale_ = covmat
        VarMix.scale_cholesky_ = chole_covmat
        VarMix.scale_inv_cholesky_ = chole_inv_cov
        VarMix.df_ = np.asarray([4.0])
        VarMix.mix_weights_ = np.asarray([1.0])
        VarMix.converged_ = True
        VarMix.n_components = 1

        #Generate a few hundred samples from scipy's multivariate t.
        rv = scipy.stats.multivariate_t(loc.flatten(), covmat.reshape((3,3)),
                df=4)
        samples = rv.rvs(size=300)
        true_loglik = rv.logpdf(samples)

        sq_maha_dist = VarMix.sq_maha_distance(samples, VarMix.loc_, VarMix.scale_inv_cholesky_)
        loglik = VarMix.get_loglikelihood(samples, sq_maha_dist, VarMix.df_,
                    VarMix.scale_cholesky_, VarMix.mix_weights_).flatten()
        outcome = np.allclose(true_loglik, loglik)
        print("Does scipy's multivariate t logpdf match "
                "the VariationalStudentMixture get_loglikelihood function? %s"%outcome)
        self.assertTrue(outcome)

        print('\n')


    def test_sampling_students_t(self):
        """Test the sampling procedure by sampling from a prespecified distribution,
        then fitting a single component model. If all is well, the model fit's
        parameters should be very similar to those used for sampling."""
        print("*********************")
        #Arbitrary scale matrix...
        true_cov = [np.asarray([[1.6, 0.0, 0.0],
                            [0.0, 4.5, 0.0],
                            [0.0, 0.0, 3.2]])]
        true_cov = np.stack(true_cov, axis=-1)
        #Arbitrary locations...
        true_loc = np.asarray([[4.5,3.6,7.2]])
        TrueVarMix = VariationalStudentMixture()
        chole_covmat = np.linalg.cholesky(true_cov[:,:,0])
        chole_covmat = np.stack([chole_covmat], axis=-1)
        chole_inv_cov = np.empty((3,3,2))
        chole_inv_cov = TrueVarMix.get_scale_inv_cholesky(chole_covmat, chole_inv_cov)

        #Set up VarMix using the parameters of the distribution.
        TrueVarMix.location_ = true_loc
        TrueVarMix.scale_ = true_cov
        TrueVarMix.scale_cholesky_ = chole_covmat
        TrueVarMix.scale_inv_cholesky_ = chole_inv_cov
        TrueVarMix.df_ = np.asarray([4.0])
        TrueVarMix.mix_weights_ = np.asarray([1.0])
        TrueVarMix.converged_ = True
        TrueVarMix.n_components = 1
        x = TrueVarMix.sample(num_samples=500, random_seed=123)
        
        FittedVarMix = VariationalStudentMixture(fixed_df=False, random_state=123,
                n_components=1, max_iter=1500, tol=1e-7)
        FittedVarMix.fit(x)

        #Retrieve the fit parameters. They should be very close to the true ones.
        fit_loc = FittedVarMix.location.flatten()
        fit_cov = FittedVarMix.scale[:,:,0]
        fit_df = FittedVarMix.degrees_of_freedom[0]
        
        print("If 500 samples are drawn from a VariationalStudentMixture with known "
        "arbitrary location...")
        #For location, distance from fit location to true location should be 
        #less than 1% of the norm of the true location.
        location_distance = np.linalg.norm(fit_loc - true_loc, axis=1)[0]
        true_loc_norm = np.linalg.norm(true_loc, axis=1)[0]
        location_outcome = (location_distance / true_loc_norm) < 0.02

        print("Is distance from fit location to true location "
                "< 2 percent of the norm of true location? %s"%location_outcome)
        
        fit_df_error = np.max(np.abs(fit_df - 4))
        fit_df_outcome = fit_df_error < 0.5
        print("Are the estimated degrees of freedom between 3.5 and 4.5? %s"%fit_df_outcome)
        
        #Last but not least, we check the covariance matrices. This is a little
        #tricky but to try to keep this simple we use Herdin (2005)'s approach
        #to estimating distance and require that distance be smaller than a
        cov_mat_sim = (np.trace(np.dot(true_cov[:,:,0], fit_cov)) / 
                    (np.linalg.norm(true_cov[:,:,0]) * np.linalg.norm(fit_cov)))
        cov_mat_dist = 1 - cov_mat_sim
        cov_outcome = (cov_mat_dist) < 0.01
        print("Is the covariance matrix distance < 0.01? %s"%cov_outcome)

        self.assertTrue(location_outcome)
        self.assertTrue(fit_df_outcome)
        self.assertTrue(cov_outcome)

        print('\n')




if __name__ == "__main__":
    unittest.main()
