#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include "./main.h"

static void
GraphDef_dealloc(GraphDef *self)
{
    Py_XDECREF(self->node_values);
    Py_XDECREF(self->node_keypos);
    Py_XDECREF(self->edge_values);
    Py_XDECREF(self->edge_keypos);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
GraphDef_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    GraphDef *self;
    self = (GraphDef *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->node_values = NULL;
        self->node_keypos = NULL;
        self->edge_values = NULL;
        self->edge_keypos = NULL;
    }
    return (PyObject *)self;
}

static int
GraphDef_init(GraphDef *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"node_values", "edge_values", NULL};
    PyObject *node_values = NULL, *edge_values = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &node_values, &edge_values))
        return -1;

    if (node_values && edge_values)
    {
        if (!(PyDict_Check(node_values) && PyDict_Check(edge_values)))
        {
            PyErr_SetString(PyExc_TypeError, "node_values and edge_values must be dicts");
            return -1;
        }
        SELF_SET(node_values);
        SELF_SET(edge_values);
        self->node_keypos = PyDict_New();
        // We want to create a dict, node_keypos = {k: i for i, k in enumerate(sorted(node_values.keys()))}
        // First get the keys
        PyObject *keys = PyDict_Keys(node_values);
        // Then sort them
        PyObject *sorted_keys = PySequence_List(keys);
        Py_DECREF(keys);
        PyList_Sort(sorted_keys);
        // Then create the dict
        Py_ssize_t len = PyList_Size(sorted_keys);
        for (Py_ssize_t i = 0; i < len; i++)
        {
            PyObject *k = PyList_GetItem(sorted_keys, i);
            PyDict_SetItem(self->node_keypos, k, PyLong_FromSsize_t(i));
        }
        Py_DECREF(sorted_keys);
        // Repeat for edge_keypos
        self->edge_keypos = PyDict_New();
        keys = PyDict_Keys(edge_values);
        sorted_keys = PySequence_List(keys);
        Py_DECREF(keys);
        PyList_Sort(sorted_keys);
        len = PyList_Size(sorted_keys);
        for (Py_ssize_t i = 0; i < len; i++)
        {
            PyObject *k = PyList_GetItem(sorted_keys, i);
            PyDict_SetItem(self->edge_keypos, k, PyLong_FromSsize_t(i));
        }
        return 0;
    }
    return -1;
}

static PyMemberDef GraphDef_members[] = {
    {NULL} /* Sentinel */
};

static PyMethodDef GraphDef_methods[] = {
    {NULL} /* Sentinel */
};

PyTypeObject GraphDefType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "_C.GraphDef",
    .tp_doc = PyDoc_STR("GraphDef object"),
    .tp_basicsize = sizeof(GraphDef),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = GraphDef_new,
    .tp_init = (initproc)GraphDef_init,
    .tp_dealloc = (destructor)GraphDef_dealloc,
    .tp_members = GraphDef_members,
    .tp_methods = GraphDef_methods,
};