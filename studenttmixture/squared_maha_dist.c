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

static PyObject *squaredMahaDistance(PyObject *self, PyObject *args){
    PyArrayObject *x, *location, *lower_chole_decomp, *maha_distance;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &x, &PyArray_Type, 
                &location, &PyArray_Type, &lower_chole_decomp,
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
    if (lower_chole_decomp->nd != 3 || 
            lower_chole_decomp->descr->type_num != PyArray_DOUBLE){
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
    double *x_adjusted = (double*)malloc(x->dimensions[1] * sizeof(double));
    if (x_adjusted == NULL){
        PyErr_SetString(PyExc_ValueError, "C extension cannot allocate memory!");
        return NULL;
    }
    int i, j, k, m;
    double dotProductTerm;
    double *currentMahaElement;


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
                    dotProductTerm += *((double *)PyArray_GETPTR3(lower_chole_decomp,
                                    m, j, k)) * x_adjusted[m];
                }
                *currentMahaElement += dotProductTerm * dotProductTerm;
            }
        }
    }
    free(x_adjusted);

    Py_INCREF(Py_None);
    return Py_None;
}
