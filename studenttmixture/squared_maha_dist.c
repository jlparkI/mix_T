#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


static PyObject *squaredMahaDistance(PyObject *self, PyObject *args);


//Boilerplate required for setuptools.
static PyMethodDef sqMahaMethods[] = {
    {"squaredMahaDistance", 
    squaredMahaDistance,
    METH_VARARGS,
    "Calculate the squared mahalanobis distance using input numpy arrays"},
    {NULL, NULL, 0, NULL}
};

//More boilerplate required for setuptools
PyMODINIT_FUNC PyInit_squaredMahaDistance(void)
{
    PyObject *module;
    static struct PyModuleDef sqMahaDistDef = {
        PyModuleDef_HEAD_INIT,
        "squaredMahaDistance",
        "Calculate the squared mahalanobis distance using input numpy arrays",
        -1,
        sqMahaMethods,
        NULL,NULL,NULL,NULL
    };
    module = PyModule_Create(&sqMahaDistDef);
    if (!module) return NULL;

    import_array();
    return module;
};


//The actual extension (this is a single-function extension).
//EXPECTED ARGUMENTS
//x             the raw data; must be a 2d np.float64 numpy array
//location      the component locations; must be a 2d np.float64 numpy array
//chole_decomp  The inverse of the cholesky decomposition of the scale matrices;
//              must be a 3d np.float64 numpy array
//maha_distance The array in which the output of this function will be stored;
//              must be a 2d np.float64 array.
//
//The function returns None -- the results are written to the maha_distance
//input array. Much of this function is dedicated to checking that the inputs
//are valid, checking array bounds etc. Some of this is redundant because
//the caller checks the data passed to it by the user, but it's a lot better
//to be safe than sorry...
static PyObject *squaredMahaDistance(PyObject *self, PyObject *args){
    PyArrayObject *x, *location, *chole_decomp, *maha_distance;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &x, &PyArray_Type, 
                &location, &PyArray_Type, &chole_decomp,
                &PyArray_Type, &maha_distance)){
        PyErr_SetString(PyExc_ValueError,
                "squaredMahaDistance (C extension) expects four arguments: "
                "a numpy array of the data, a numpy array of the component locations, "
                "a numpy array of the inverse of the cholesky "
                "decomposition of the scale matrix, and a numpy array in which "
                "the squared mahalanobis distance will be stored. "
                "This function has received incorrect "
                "arguments.");
        return NULL;
    }
    if (x->nd != 2 || x->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString(PyExc_ValueError,
                "The data array input to squaredMahaDistance must be 2d and "
                "of type double.");
        return NULL;
    }
    if (location->nd != 2 || location->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString(PyExc_ValueError,
                "The location array input to squaredMahaDistance must be 2d and "
                "of type double.");
        return NULL; 
    }
    if (chole_decomp->nd != 3 || 
            chole_decomp->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString(PyExc_ValueError,
                "The inverse cholesky decomposition argument to squaredMahaDistance "
                "must be 3d and of type double.");
        return NULL;
    }
    if (maha_distance->nd != 2 || maha_distance->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString(PyExc_ValueError,
                "The maha_distance argument to squaredMahaDistance must be 2d and "
                "of type double.");
        return NULL; 
    }
    //Check that all dimensions match.
    if (maha_distance->dimensions[1] != chole_decomp->dimensions[2] || 
            location->dimensions[0] != chole_decomp->dimensions[2]){
        PyErr_SetString(PyExc_ValueError,
                "The number of components is inconsistent between the chole_decomp, "
                "maha_distance and location passed to squaredMahaDistance.");
        return NULL; 
    }
    if (maha_distance->dimensions[0] != x->dimensions[0]){
        PyErr_SetString(PyExc_ValueError,
                "The number of datapoints is inconsistent between the x "
                "and maha_distance passed to squaredMahaDistance.");
        return NULL; 
    }
    if (x->dimensions[1] != location->dimensions[1] || x->dimensions[1] != 
            chole_decomp->dimensions[1]){
        PyErr_SetString(PyExc_ValueError,
                "The number of dimensions is inconsistent between the x, location "
                "and chole_decomp passed to squaredMahaDistance.");
        return NULL; 
    }
    if (chole_decomp->dimensions[0] != chole_decomp->dimensions[1]){
        PyErr_SetString(PyExc_ValueError,
                "chole_decomp passed to squaredMahaDistance should be a D x D x k "
                "array, instead it is a m x n x k array.");
        return NULL;     
    }

    //Check that all input arrays are contiguous.
    if (!(PyArray_FLAGS(x) & NPY_ARRAY_C_CONTIGUOUS)){
        PyErr_SetString(PyExc_ValueError,
                "The x input to squaredMahaDistance is not contiguous.");
        return NULL;     
    }
    if (!(PyArray_FLAGS(location) & NPY_ARRAY_C_CONTIGUOUS)){
        PyErr_SetString(PyExc_ValueError,
                "The location input to squaredMahaDistance is not contiguous.");
        return NULL;     
    }
    if (!(PyArray_FLAGS(chole_decomp) & NPY_ARRAY_C_CONTIGUOUS)){
        PyErr_SetString(PyExc_ValueError,
                "The chole_decomp input to squaredMahaDistance is not contiguous.");
        return NULL;     
    }
    if (!(PyArray_FLAGS(maha_distance) & NPY_ARRAY_C_CONTIGUOUS)){
        PyErr_SetString(PyExc_ValueError,
                "The maha_distance input to squaredMahaDistance is not contiguous.");
        return NULL;     
    }
    //////////////////////////////////////////////
    //If we made it here, all the inputs are valid, we can proceed.
    //x_adjusted is a temporary array for storing x - location 
    //for each datapoint and cluster.
    double *x_adjusted = (double*)malloc(x->dimensions[1] * sizeof(double));

    if (x_adjusted == NULL){
        PyErr_SetString(PyExc_ValueError, "C extension cannot allocate memory!");
        return NULL;
    }
    int i, j, k, m;
    double dotProductTerm;
    double *currentMahaElement;

    //Loop over COMPONENTS (k), DATAPOINTS (i), DIMENSIONS (j).
    //m is an inner loop over DIMENSIONS when multiplying (x[i,:] - location[k,:])
    //against the cholesky decomposition of the scale matrix. Note that because
    //the cholesky decomposition is inverted it is here upper triangular.
    for (k=0; k < location->dimensions[0]; k++){
        for (i=0; i < x->dimensions[0]; i++){
            currentMahaElement = (double *)PyArray_GETPTR2(maha_distance, i, k);
            *currentMahaElement = 0;
            for (j=0; j < x->dimensions[1]; j++){
                x_adjusted[j] = *((double *)PyArray_GETPTR2(x, i, j)) - 
                        *((double *)PyArray_GETPTR2(location, k, j));
            }
            for (j=0; j < x->dimensions[1]; j++){
                dotProductTerm = 0;
                for (m=0; m < j+1; m++){
                    dotProductTerm += *((double *)PyArray_GETPTR3(chole_decomp,
                                    m, j, k)) * x_adjusted[m];
                }
                *currentMahaElement += dotProductTerm * dotProductTerm;
            }
        }
    }
    free(x_adjusted);

    //If returning Py_None, Py_None must be INCREFed.
    Py_INCREF(Py_None);
    return Py_None;
}
