'''Finite mixture of Student's t-distributions fit using EM.'''

#Author: Jonathan Parkinson <jlparkinson1@gmail.com>
#License: MIT

import numpy as np
cimport numpy as cnp
cimport cython
from scipy.linalg.cython_blas import dgemm

ctypedef cnp.float64_t FLOAT64
ctypedef cnp.uint64_t  INT64

#This file contains Cython implementations of the M-step calculations
#for both EM and variational algorithms. While it is definitely possible
#to write a C API extension for the M-step as we did for the squared
#mahalanobis distance calculation, Cython is a little more convenient for this
#purpose because 1) it is easier to access the LAPACK and BLAS routines via 
#Cython using scipy's wrappers than
#via the C API and 2) it improves ease of maintenance.
#Most of the speed benefit we get lies in avoiding the inefficiency of 
#Python loops over n_components, which does not require extensive changes.

#This function implements the M-step calculations for the EM mixture. 
#TODO: Some parts of this have more Python interactions than they should -- optimize
#using memory views
@cython.wraparound(False)
def EM_Mstep_Optimized_Calc(cnp.ndarray[FLOAT64, ndim=2] X, np.ndarray[FLOAT64, ndim=2] ru,
                       cnp.ndarray[FLOAT64, ndim=3] scale_,
                       cnp.ndarray[FLOAT64, ndim=3] scale_cholesky_,
                       cnp.ndarray[FLOAT64, ndim=3] scale_inv_cholesky_,
                       cnp.ndarray[FLOAT64, ndim=2] loc_,
                       cnp.ndarray[FLOAT64, ndim=1] resp_sum,
                       double reg_covar):
    cdef int i, j, k
    cdef int n_components = ru.shape[1]
    cdef int N = X.shape[0]
    cdef int dims = X.shape[1]
    cdef cnp.ndarray[FLOAT64, ndim=2] identity_mat = np.eye(dims)
    cdef cnp.ndarray[FLOAT64, ndim=2] scaled_x = np.empty((N, dims))

    for k in range(dims):
        scaled_x = X - loc_[k,:]
        scale_[:,:,k] = np.dot((ru[:,k:k+1] * scaled_x).T,
                            scaled_x) / resp_sum[k]
        scale_[:,:,k].flat[::X.shape[1] + 1] += reg_covar
        scale_cholesky_[:,:,k] = np.linalg.cholesky(scale_[:,:,k])
        scale_inv_cholesky_[:,:,k] = solve_triangular(scale_cholesky_[:,:,k],
                                            identity_mat, lower=True).T

       

#This function implements squared mahalanobis distance calculations.
#Note that caller must check that all arrays supplied by user are contiguous.
cpdef squaredMahaDistance(double[:,::1] X, 
                        double[:,:] loc_, 
                        double [:,::1,:] scale_inv_cholesky_,
                        double[:,::1] sq_maha_dist) nogil:
    cdef int i, n_components, ldx, ldloc, ld_invchole, ldmaha_dist, ld_y
    cdef int N, dims, dims2
    cdef double alpha = 1.0
    cdef double beta = 0.0
    ldx, dims, dims2, ld_invchole = X.shape[0], X.shape[0], X.shape[0], X.shape[0]
    n_components = loc_.shape[0]
    N = X.shape[1]

    cdef np.ndarray[FLOAT64, ndim=2] y = np.empty((X.shape[0], X.shape[1]), double,
                                    order="F")
    dgemm("N", "N", &N, &dims, &dims2, &alpha, &X[0,0], &ldx, &scale_inv_cholesky_[0,0,i],
                &ld_invchole, &beta, &y[0,0], &ld_y)
    i = 0
#    for i in range(n_components):
        #y = np.dot(X, scale_inv_cholesky_[:,:,i])
        #y = y - np.dot(loc_[i,:], scale_inv_cholesky_[:,:,i])[np.newaxis,:]
        #sq_maha_dist[:,i] = np.sum(y**2, axis=1)

    return y

