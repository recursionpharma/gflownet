#define PY_SSIZE_T_CLEAN
#include "./main.h"
#include "structmember.h"
#include <Python.h>

static void DegreeView_dealloc(DegreeView *self) {
    Py_XDECREF(self->graph);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DegreeView_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    DegreeView *self;
    self = (DegreeView *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->graph = NULL;
    }
    return (PyObject *)self;
}

static int DegreeView_init(DegreeView *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"graph", NULL};
    PyObject *graph = NULL, *tmp;
    int node_id = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &graph, &node_id))
        return -1;

    if (graph) {
        tmp = (PyObject *)self->graph;
        Py_INCREF(graph);
        self->graph = (Graph *)graph;
        Py_XDECREF(tmp);
    }
    return 0;
}

static PyMemberDef DegreeView_members[] = {
    {NULL} /* Sentinel */
};

static PyObject *DegreeView_getitem(PyObject *_self, PyObject *k) {
    DegreeView *self = (DegreeView *)_self;
    int index = PyLong_AsLong(k);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyLong_FromLong(self->graph->degrees[index]);
}

static PyMappingMethods DegreeView_mapmeth = {
    .mp_subscript = DegreeView_getitem,
};

static PyMethodDef DegreeView_methods[] = {
    {NULL} /* Sentinel */
};

PyTypeObject DegreeViewType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_C.DegreeView",
    .tp_doc = PyDoc_STR("DegreeView object"),
    .tp_basicsize = sizeof(DegreeView),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = DegreeView_new,
    .tp_init = (initproc)DegreeView_init,
    .tp_dealloc = (destructor)DegreeView_dealloc,
    .tp_members = DegreeView_members,
    .tp_methods = DegreeView_methods,
    .tp_as_mapping = &DegreeView_mapmeth,
};