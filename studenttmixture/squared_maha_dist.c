#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


//Boilerplate required for setuptools.
static PyMethodDef sqMahaMethods[] = {
{ 
    "squaredMahaDistance", 
    sqMahaDistance,
    METH_VARARGS,
    "Calculate the squared mahalanobis distance using input numpy arrays",
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

//The actual function -- will write this soon...
static PyObject *squaredMahaDistance(PyObject *self, PyObject *args){
    //Parse arguments, make sure the inputs are numpy arrays with expected
    //shapes and data types,
    //if not, kick this back to the Python interpreter with a description
    //of the problem.
    PyArrayObject *x, *location, *lower_chole_decomp;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &x, &location, &lower_chole_decomp){
        PyErr_SetString("squaredMahaDistance (C extension) expects three arguments: "
                "a numpy array of the data, a numpy array of the component locations, "
                "and a numpy array of the inverse of the cholesky "
                "decomposition of the scale matrix. This function has received incorrect "
                "arguments.");
        return NULL;
    }
    if (x->nd != 2 || x->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString("The data array input to squaredMahaDistance must be 2d and "
                "of type double.");
        return NULL;
    }
    if (location->nd != 2 || location->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString("The location array input to squaredMahaDistance must be 2d and "
                "of type double.");
        return NULL;
        
    }
    if (lower_chole_decomp->nd != 3 || lower_chole_decomp->descr->type_num != PyArray_DOUBLE){
        PyErr_SetString("The inverse cholesky decomposition argument to squaredMahaDistance "
                "must be 3d and of type double.");
        return NULL;
    }
    //If we are here, the inputs have valid shapes and types. We now construct an array
    //to store the output. This array will be returned so we will not decref.
    int dimensions[2];
    dimensions[0] = x->dimensions[0];
    dimensions[1] = location->dimensions[0];
    PyObject *mahaDistance;
    int i, j, k;

    mahaDistance = (PyArrayObject *)PyArray_FromDims(2, dimensions, PyArray_DOUBLE);

    
}
