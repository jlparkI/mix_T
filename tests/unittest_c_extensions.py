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
        X = np.random.uniform(low=-10,high=10,size=(1000,3))
        FiniteMix = EMStudentMixture()
        cov1 = 

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


if __name__ == "__main__":
    unittest.main()

