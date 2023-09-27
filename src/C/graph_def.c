#define PY_SSIZE_T_CLEAN
#include "./main.h"
#include "structmember.h"
#include <Python.h>

static void GraphDef_dealloc(GraphDef *self) {
    Py_XDECREF(self->node_values);
    Py_XDECREF(self->node_keypos);
    Py_XDECREF(self->node_poskey);
    Py_XDECREF(self->edge_values);
    Py_XDECREF(self->edge_keypos);
    Py_XDECREF(self->edge_poskey);
    free(self->node_attr_offsets);
    free(self->edge_attr_offsets);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *GraphDef_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    GraphDef *self;
    self = (GraphDef *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->node_values = NULL;
        self->node_keypos = NULL;
        self->node_poskey = NULL;
        self->edge_values = NULL;
        self->edge_keypos = NULL;
        self->edge_poskey = NULL;
        self->node_attr_offsets = NULL;
        self->edge_attr_offsets = NULL;
    }
    return (PyObject *)self;
}

static int GraphDef_init(GraphDef *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"node_values", "edge_values", NULL};
    PyObject *node_values = NULL, *edge_values = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &node_values, &edge_values))
        return -1;

    if (!(node_values && edge_values)) {
        return -1;
    }
    if (!(PyDict_Check(node_values) && PyDict_Check(edge_values))) {
        PyErr_SetString(PyExc_TypeError, "GraphDef: node_values and edge_values must be dicts");
        return -1;
    }
    PyObject *v_list = PyDict_GetItemString(node_values, "v");
    if (!v_list) {
        PyErr_SetString(PyExc_TypeError, "GraphDef: node_values must contain 'v'");
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
    self->node_attr_offsets = malloc(sizeof(int) * (len + 1));
    self->node_attr_offsets[0] = 0;
    // TODO: here this works because 'v' is last in the list, if it's not
    // node_logit_offsets[i] != node_attr_offsets[i] - i, which we're using a lot as an assumption
    // throughout this and mol_graph_to_Data
    Py_ssize_t offset = 0;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *k = PyList_GetItem(sorted_keys, i);
        PyDict_SetItem(self->node_keypos, k, PyLong_FromSsize_t(i));
        PyObject *vals = PyDict_GetItem(node_values, k);
        if (!PyList_Check(vals)) {
            PyErr_SetString(PyExc_TypeError, "GraphDef: node_values must be a Dict[str, List[Any]]");
            return -1;
        }
        offset += PyList_Size(vals);
        self->node_attr_offsets[i + 1] = offset;
        if (i == len - 1 && strcmp(PyUnicode_AsUTF8(k), "v") != 0) {
            PyErr_SetString(PyExc_TypeError, "GraphDef: 'v' must be the last sorted key in current implementation");
            return -1;
        }
    }
    self->num_node_dim = offset + 1;         // + 1 for the empty graph
    self->num_settable_node_attrs = len - 1; // 'v' is not settable by setnodeattr but only by addnode
    self->num_new_node_values = PyList_Size(v_list);
    self->num_node_attr_logits = offset - (len - 1) - self->num_new_node_values; // 'v' is not settable
    self->node_poskey = sorted_keys;

    // Repeat for edge_keypos
    self->edge_keypos = PyDict_New();
    keys = PyDict_Keys(edge_values);
    sorted_keys = PySequence_List(keys);
    Py_DECREF(keys);
    PyList_Sort(sorted_keys);
    len = PyList_Size(sorted_keys);
    self->edge_attr_offsets = malloc(sizeof(int) * (len + 1));
    self->edge_attr_offsets[0] = 0;
    offset = 0;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *k = PyList_GetItem(sorted_keys, i);
        PyDict_SetItem(self->edge_keypos, k, PyLong_FromSsize_t(i));
        PyObject *vals = PyDict_GetItem(edge_values, k);
        if (!PyList_Check(vals)) {
            PyErr_SetString(PyExc_TypeError, "GraphDef: edge_values must be a Dict[str, List[Any]]");
            return -1;
        }
        offset += PyList_Size(vals);
        self->edge_attr_offsets[i + 1] = offset;
    }
    self->num_edge_dim = offset;
    self->num_settable_edge_attrs = len;
    self->num_edge_attr_logits = offset - len;
    self->edge_poskey = sorted_keys;

    return 0;
}

PyObject *GraphDef___getstate__(GraphDef *self, PyObject *args) {
    return PyTuple_Pack(1, Py_BuildValue("OO", self->node_values, self->edge_values));
}

PyObject *GraphDef___setstate__(GraphDef *self, PyObject *state) {
    // new() was just called on self
    GraphDef_init(self, PyTuple_GetItem(state, 0), NULL);
    Py_RETURN_NONE;
}

static PyMemberDef GraphDef_members[] = {
    {NULL} /* Sentinel */
};

static PyMethodDef GraphDef_methods[] = {
    {"__getstate__", (PyCFunction)GraphDef___getstate__, METH_NOARGS, "Pickle the Custom object"},
    {"__setstate__", (PyCFunction)GraphDef___setstate__, METH_O, "Un-pickle the Custom object"},
    {NULL} /* Sentinel */
};

PyTypeObject GraphDefType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "gflownet._C.GraphDef",
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