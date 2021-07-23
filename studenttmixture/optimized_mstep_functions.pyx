#Finite mixture of Student's t-distributions fit using EM.

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>
#License: MIT

import numpy as np
cimport numpy as cnp
cimport cython
from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_lapack cimport dpotrf, dtrtrs

ctypedef cnp.float64_t FLOAT64
ctypedef cnp.uint64_t  INT64

#This function implements the M-step calculations for the EM mixture. 
@cython.cdivision(True)
@cython.wraparound(False)
def EM_Mstep_Optimized_Calc(cnp.ndarray[FLOAT64, ndim=2] X, 
                       double [:,:] ru,
                       cnp.ndarray[FLOAT64, ndim=3] scale_,
                       cnp.ndarray[FLOAT64, ndim=3] scale_cholesky_,
                       cnp.ndarray[FLOAT64, ndim=3] scale_inv_cholesky_,
                       double [:,:] loc_,
                       cnp.ndarray[FLOAT64, ndim=1] resp_sum,
                       double reg_covar):
    cdef int i, j, k, info
    cdef int n_components = ru.shape[1]
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef int lda = dims
    cdef int ldb = dims
    cdef int ncols = dims
    cdef cnp.ndarray[FLOAT64, ndim=2] solution_mat = np.eye(dims)
    cdef cnp.ndarray[FLOAT64, ndim=2] scaled_x = np.empty((N, dims))
    cdef cnp.ndarray[FLOAT64, ndim=2] temp_arr = np.empty((dims, dims))
    cdef cnp.ndarray[FLOAT64, ndim=2] temp_chole_arr = np.empty((dims, dims))
    cdef cnp.ndarray[FLOAT64, ndim=2] scaled_x_transpose = np.empty((dims, N))

    for k in range(n_components):
        for i in range(N):
            for j in range(dims):
                scaled_x[i,j] = X[i,j] - loc_[k,j]
                scaled_x_transpose[j,i] = scaled_x[i,j] * ru[i,k]
        fast_matmul(scaled_x_transpose, scaled_x, temp_arr)
        for i in range(dims):
            for j in range(dims):
                temp_arr[i,j] = temp_arr[i,j] / resp_sum[k]
            temp_arr[i,i] += reg_covar
        for i in range(dims):
            for j in range(dims):
                scale_[i,j,k] = temp_arr[i,j]

        info = chole_decomp(temp_arr, dims, lda)
        if info != 0:
            raise ValueError("Error on cholesky decomposition!")
        for i in range(dims):
            for j in range(i+1):
                scale_cholesky_[i,j,k] = temp_arr[j,i]
            for j in range(i+1, dims):
                scale_cholesky_[i,j,k] = 0
        
        info = invert_chole(temp_arr, solution_mat,
                        dims, ncols, lda, ldb)
        for i in range(dims):
            for j in range(dims):
                scale_inv_cholesky_[i,j,k] = solution_mat[i,j]
                solution_mat[i,j] = 0
            solution_mat[i,i] = 1

cdef chole_decomp(double [:,::1] scalemat,
                int dims, int lda):
    cdef int info
    dpotrf("L", &dims, &scalemat[0,0], &lda, &info)
    return info

cdef invert_chole(double [:,::1] cholemat, double [:,::1] identity,
                int dims, int ncols, 
                int lda, int ldb):
    cdef int info
    dtrtrs("L", "N", "N", &dims, &ncols, &cholemat[0,0],
            &lda, &identity[0,0], &ldb, &info)
    return info


#This function implements squared mahalanobis distance calculations.
#Note that we haven't been able to completely get rid of Python interactions here --
#isolating the matrix multiplication in its own function is just very convenient --
#but we still get a huge speedup on a Python / numpy implementation.
@cython.cdivision(True)
@cython.wraparound(False)
cpdef squaredMahaDistance(cnp.ndarray[FLOAT64, ndim=2] X,
                        cnp.ndarray[FLOAT64, ndim=2] loc_,
                        cnp.ndarray[FLOAT64, ndim=3] scale_inv_cholesky_,
                        cnp.ndarray[FLOAT64, ndim=2] sq_maha_dist):
    cdef int i, n_components, j, k, N, D
    i = 0
    cdef cnp.ndarray[FLOAT64, ndim=2] temp_arr = np.empty((X.shape[1], 
                                                            X.shape[1]))
    cdef cnp.ndarray[FLOAT64, ndim=2] y = np.empty((X.shape[0], X.shape[1]))
    cdef cnp.ndarray[FLOAT64, ndim=2] dotProd = np.empty((X.shape[0], X.shape[1]))

    n_components = scale_inv_cholesky_.shape[2]
    N = X.shape[0]
    D = X.shape[1]

    for i in range(n_components):
        for j in range(N):
            for k in range(D):
                y[j,k] = X[j,k] - loc_[i,k]
        #The copying adds cost but is much less expensive than the multiplication,
        #and converts the slice to a C-contiguous array.
        for j in range(D):
            for k in range(D):
                temp_arr[j,k] = scale_inv_cholesky_[j,k,i]
        fast_matmul(y, temp_arr, dotProd)
        for j in range(N):
            sq_maha_dist[j,i] = 0
            for k in range(D):
                sq_maha_dist[j,i] += dotProd[j,k]**2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef fast_matmul(double[:,::1] A, double[:,::1] B, double[:,::1] C):
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


