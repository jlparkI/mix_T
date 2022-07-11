"""A moderately faster version of the M-step calculations for
the EM mixture and of the squared mahalanobis distance
calculations for all classes.

Author: Jonathan Parkinson <jlparkinson1@gmail.com>
License: MIT
"""

import numpy as np
cimport numpy as cnp
cimport cython
from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_lapack cimport dpotrf, dtrtrs

ctypedef cnp.float64_t FLOAT64
ctypedef cnp.uint64_t  INT64

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
    """Implements the M-step calculations for the EM mixture.
    Note that we overwrite the input arrays so nothing is returned.

    Args:
        X (np.ndarray): The N x M input for N datapoints, M features.
        ru (double): A memory view to a numpy array that results
            from multiplying resp by E_gamma (the component responsibilities
            by the max-likelihood estimate of the "hidden variable" described by 
                a gamma distribution in the formulation of the student's t-distribution
        scale_ (np.ndarray): The scale matrices for the mixture components. A numpy
            array of shape M x M x K for M features, K components.
        scale_cholesky_ (np.ndarray): The cholesky decomposition of the scale
            matrices. Same shape as scale_.
        scale_inv_cholesky_ (np.ndarray): The inverse of the cholesky decomposition
            of the scale matrices. Same shape as scale_.
        loc_ (double): A memory view into a numpy array containing the current
            cluster locations (analogous to means for Gaussians).
        resp_sum (np.ndarray): The sum of the responsibilities across all datapoints.
        reg_covar (double): A small regularization constant added to the diagonal
            of the scale matrices for numerical stability.
    """
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
    """Performs the cholesky decomposition of scalemat in place
    and returns the error code info.

    Args:
        scalemat (double): A memory view to a square np.float64 numpy array.
        dims (int): The size of scalemat, which must be of size dims x dims.
        lda (int): The leading dimension of scalemat. Should match dims.
    
    Returns:
        info (int): An error code. 0 = success, anything else = fatal error.
    """
    cdef int info
    dpotrf("L", &dims, &scalemat[0,0], &lda, &info)
    return info

cdef invert_chole(double [:,::1] cholemat, double [:,::1] identity,
                int dims, int ncols, 
                int lda, int ldb):
    """Inverts the cholesky decomposition of a scale matrix.

    Args:
        cholemat (double): A memory view to a square np.float64 
            numpy array. Should be of size dims x dims.
        identity (double): A memory view to a square identity 
            matrix of the same size as cholemat.
        dims (int): The size of cholemat and identity. Both should
            be of size dims x dims.
        lda (int): The leading dimension of cholemat. Should match dims.
        ldb (int): The leading dimension of identity. Should match
            dims.

    Returns:
        info (int): 0 = success, anything else = failure.
    """
    cdef int info
    dtrtrs("L", "N", "N", &dims, &ncols, &cholemat[0,0],
            &lda, &identity[0,0], &ldb, &info)
    return info


@cython.cdivision(True)
@cython.wraparound(False)
cpdef squaredMahaDistance(cnp.ndarray[FLOAT64, ndim=2] X,
                        cnp.ndarray[FLOAT64, ndim=2] loc_,
                        cnp.ndarray[FLOAT64, ndim=3] scale_inv_cholesky_,
                        cnp.ndarray[FLOAT64, ndim=2] sq_maha_dist):
    """This function implements squared mahalanobis distance calculations.
    Note that we haven't been able to completely get rid of Python interactions here --
    isolating the matrix multiplication in its own function is just very convenient --
    but we still get a decent speedup on a Python / numpy implementation.

    Args:
        X (np.ndarray): An N x M numpy array with N datapoints, M features.
        loc_ (np.ndarray): A K x M numpy array for K components, M features.
            Stores the locations of mixture components (analogous to means
            of Gaussians).
        scale_inv_cholesky (np.ndarray): The inverse of the cholesky decomposition
            of the scale matrices. An M x M x K numpy array.
        sq_maha_dist (np.ndarray): Stores the squared mahalanobis distance for
            the X-data. The results will be written here. An N x K numpy array
            for K components, N datapoints.
    """

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


cdef fast_matmul(double[:,::1] A, double[:,::1] B, double[:,::1] C):
    """Wraps the dgemm function. This is (very slightly) faster than pure Python
    matrix multiplication via Numpy because we lose a little overhead.

    Args:
        A (double): The first matrix in the operation A.T B.
        B (double): The second matrix in the operation A.T B.
        C (double): The matrix where results are stored.
    """
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


