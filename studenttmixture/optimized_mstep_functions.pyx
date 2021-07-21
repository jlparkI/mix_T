#Finite mixture of Student's t-distributions fit using EM.

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>
#License: MIT

import numpy as np
cimport numpy as cnp
cimport cython
from scipy.linalg.cython_blas cimport dgemm

ctypedef cnp.float64_t FLOAT64
ctypedef cnp.uint64_t  INT64


#This function implements squared mahalanobis distance calculations.
#Note that caller must check that all arrays supplied by user are contiguous.
#cpdef squaredMahaDistance(double[:,:] X, 
#                        double[:,:] loc_, 
#                        double [:,:,:] scale_inv_cholesky_,
#                        double[:,:] sq_maha_dist,
#                        double [:,:] dotProd):
cpdef squaredMahaDistance(cnp.ndarray[FLOAT64, ndim=2] X,
                        cnp.ndarray[FLOAT64, ndim=2] loc_,
                        cnp.ndarray[FLOAT64, ndim=3] scale_inv_cholesky_,
                        cnp.ndarray[FLOAT64, ndim=2] sq_maha_dist,
                        cnp.ndarray[FLOAT64, ndim=2] dotProd):
    cdef int i
    i = 0
    cdef cnp.ndarray[FLOAT64, ndim=2] temp_arr

    temp_arr = scale_inv_cholesky_[:,:,i]
    
    fast_matmul(X, temp_arr, dotProd)
#    for i in range(n_components):
        #y = np.dot(X, scale_inv_cholesky_[:,:,i])
        #y = y - np.dot(loc_[i,:], scale_inv_cholesky_[:,:,i])[np.newaxis,:]
        #sq_maha_dist[:,i] = np.sum(y**2, axis=1)



cpdef fast_matmul(double[:,::1] A, double[:,::1] B, double[:,::1] C):
    cdef int lda, ldb, ldc
    cdef int m, n, k
    cdef double alpha = 1.0
    cdef double beta = 0.0

    n = A.shape[0]
    m = B.shape[1]
    k = B.shape[0]
    lda = k
    ldb = m
    ldc = m

    dgemm("N", "N", &m, &n, &k, &alpha, &B[0,0], &ldb, &A[0,0],
                &lda, &beta, &C[0,0], &ldc)
    
