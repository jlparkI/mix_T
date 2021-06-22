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

}
