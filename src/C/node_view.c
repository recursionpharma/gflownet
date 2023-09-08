#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include "./main.h"

static void
NodeView_dealloc(NodeView *self)
{
    Py_XDECREF(self->graph);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
NodeView_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    NodeView *self;
    self = (NodeView *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->graph = NULL;
    }
    return (PyObject *)self;
}

static int
NodeView_init(NodeView *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"graph", "index", NULL};
    PyObject *graph = NULL, *tmp;
    int index = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &graph, &index))
        return -1;

    if (graph)
    {
        tmp = (PyObject *)self->graph;
        Py_INCREF(graph);
        self->graph = (Graph *)graph;
        Py_XDECREF(tmp);
        self->index = index;
    }
    return 0;
}

static PyMemberDef NodeView_members[] = {
    {NULL} /* Sentinel */
};

static int
NodeView_setitem(PyObject *_self, PyObject *k, PyObject *v)
{
    NodeView *self = (NodeView *)_self;
    if (self->index < 0)
    {
        PyErr_SetString(PyExc_KeyError, "Cannot assign to a node");
        return -1;
    }
    PyObject *r = Graph_setnodeattr(self->graph, self->index, k, v);
    return r == NULL ? -1 : 0;
}

static PyObject *
NodeView_getitem(PyObject *_self, PyObject *k)
{
    NodeView *self = (NodeView *)_self;
    if (self->index < 0)
    {
        PyObject *args = PyTuple_Pack(2, self->graph, k);
        PyObject *res = PyObject_CallObject((PyObject *)&NodeViewType, args);
        Py_DECREF(args);
        return res;
    }
    return Graph_getnodeattr(self->graph, self->index, k);
}

static int NodeView_contains(PyObject *_self, PyObject *v)
{
    NodeView *self = (NodeView *)_self;
    if (self->index < 0)
    {
        // Graph_containsnode
        if (!PyLong_Check(v))
        {
            PyErr_SetString(PyExc_TypeError, "NodeView.__contains__ only accepts integers");
            return -1;
        }
        int index = PyLong_AsLong(v);
        for (int i = 0; i < self->graph->num_nodes; i++)
        {
            if (self->graph->nodes[i] == index)
            {
                return 1;
            }
        }
        return 0;
    }
    // Graph_nodehasattr
    PyObject *attr = Graph_getnodeattr(self->graph, self->index, v);
    if (attr == NULL)
    {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static PyMappingMethods NodeView_mapmeth = {
    .mp_subscript = NodeView_getitem,
    .mp_ass_subscript = NodeView_setitem,
};

static PySequenceMethods NodeView_seqmeth = {
    .sq_contains = NodeView_contains,
};

static PyMethodDef NodeView_methods[] = {
    {NULL} /* Sentinel */
};

PyTypeObject NodeViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "_C.NodeView",
    .tp_doc = PyDoc_STR("Constrained NodeView object"),
    .tp_basicsize = sizeof(NodeView),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = NodeView_new,
    .tp_init = (initproc)NodeView_init,
    .tp_dealloc = (destructor)NodeView_dealloc,
    .tp_members = NodeView_members,
    .tp_methods = NodeView_methods,
    .tp_as_mapping = &NodeView_mapmeth,
    .tp_as_sequence = &NodeView_seqmeth,
};