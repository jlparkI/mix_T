import unittest, numpy as np, scipy, sys, time
from scipy import stats, spatial
import studenttmixture
from studenttmixture.em_student_mixture import EMStudentMixture
from optimized_mstep_functions import squaredMahaDistance, EM_Mstep_Optimized_Calc 

class TestSqMahaDistExtension(unittest.TestCase):
    
    #Test the squared mahalanobis distance calculation by generating a set
    #of random datapoints and measuring squared mahalanobis distance to a
    #prespecified location & scale matrix distribution, then compare the
    #result with scipy's mahalanobis function.
    def test_sq_maha_distance(self):
        print("*********************")
        np.random.seed(123)
        X = np.random.uniform(low=-10,high=10,size=(1000,3))
        FiniteMix = EMStudentMixture()
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
        
        extension_dist = np.zeros((X.shape[0], 2))
        true_dist = np.empty((X.shape[0], 2))
        #We store the minimum time required for squaredMahaDistance on 100 repeats
        #and do the same for the pure Python alternative.
        min_time = np.inf
        for i in range(20):
            start = time.time()
            squaredMahaDistance(X, loc, chole_inv_cov, extension_dist)
            end = time.time()
            if end - start < min_time:
                min_time = end - start
        print("The min time for the C extension is %s"%min_time)
        min_time = np.inf
        for i in range(20):
            start = time.time()
            sqdist = FiniteMix.sq_maha_distance(X, loc, chole_inv_cov)
            end = time.time()
            if end - start < min_time:
                min_time = end - start
        print("The min time for the pure python version is %s"%min_time)
        
        for i in range(X.shape[0]):
            true_dist[i,0] = scipy.spatial.distance.mahalanobis(X[i,:], loc[0,:],
                            scale_inv1)**2
            true_dist[i,1] = scipy.spatial.distance.mahalanobis(X[i,:], loc[1,:],
                            scale_inv2)**2
        outcome = np.allclose(extension_dist, true_dist)
        print("Does scipy's mahalanobis match "
                "the C extension-calculated distance? %s"%outcome)
        self.assertTrue(outcome)
        print('\n')


    #Test the EM_Mstep scale and cholesky matrix updates for speed and accuracy.
    def test_EM_Mstep_calcs(self):
        print("*********************")
        np.random.seed(123)
        dims = 5
        ncomps = 10
        X = np.random.uniform(low=-10,high=10,size=(1000,dims))
        FiniteMix = EMStudentMixture(n_components=10)
        loc_, scale_, mix_weights_, scale_cholesky_, \
            scale_inv_cholesky_, = FiniteMix.initialize_params(X, 123, "kmeans")
        sq_maha_dist = np.empty((X.shape[0], ncomps))
        df = np.full((ncomps), fill_value=4.0)
        resp, E_gamma, lower_bound = FiniteMix.Estep(X, df, loc_, scale_inv_cholesky_,
                            scale_cholesky_, mix_weights_, sq_maha_dist)
        min_python_time = 100
        for i in range(20):
            start_t = time.time()
            tru_weights, tru_loc, tru_scale, tru_scale_chol, tru_scale_inv_chol, tru_df = \
                FiniteMix.Mstep(X, resp, E_gamma, scale_, scale_cholesky_,
                        df, scale_inv_cholesky_)
            end_t = time.time()
            if end_t - start_t < min_python_time:
                min_python_time = end_t - start_t
       
        min_cython_time = 100
        for i in range(20):
            start_t = time.time()
            ru = resp * E_gamma
            resp_sum = np.sum(ru, axis=0) + 10 * np.finfo(resp.dtype).eps
            c_ext_scale = np.empty_like(scale_)
            c_ext_scale_cholesky = np.empty_like(scale_cholesky_)
            c_ext_scale_inv_chol = np.empty_like(scale_inv_cholesky_)
            reg_covar = 1e-6
        
            EM_Mstep_Optimized_Calc(X, ru, c_ext_scale, c_ext_scale_cholesky,
                        c_ext_scale_inv_chol, tru_loc, resp_sum, reg_covar) 
            end_t = time.time()
            if end_t - start_t < min_cython_time:
                min_cython_time = end_t - start_t
        
        outcome1 = np.allclose(c_ext_scale, tru_scale)
        outcome2 = np.allclose(c_ext_scale_cholesky, tru_scale_chol)
        outcome3 = np.allclose(tru_scale_inv_chol, c_ext_scale_inv_chol)
    
        print("min python time: %s"%min_python_time)
        print("min cython time: %s"%min_cython_time)
        
        print("Does the C extension correctly calculate the "
            "scale matrices? %s"%outcome1)
        print("Does the C extension correctly calculate the "
            "scale cholesky decompositions %s"%outcome2)
        print("Does the C extension correctly calculate the "
            "inverse cholesky decompositions? %s"%outcome3)
        self.assertTrue(outcome1)
        self.assertTrue(outcome2)
        self.assertTrue(outcome3)
        print('\n')


if __name__ == "__main__":
    unittest.main()

