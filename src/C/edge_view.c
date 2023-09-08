#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include "./main.h"

static void
EdgeView_dealloc(EdgeView *self)
{
    Py_XDECREF(self->graph);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
EdgeView_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    EdgeView *self;
    self = (EdgeView *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->graph = NULL;
    }
    return (PyObject *)self;
}

static int
EdgeView_init(EdgeView *self, PyObject *args, PyObject *kwds)
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
    printf("New EdgeView %p %p %d\n", self, self->graph, self->index);
    return 0;
}

static PyMemberDef EdgeView_members[] = {
    {NULL} /* Sentinel */
};

static int
EdgeView_setitem(PyObject *_self, PyObject *k, PyObject *v)
{
    EdgeView *self = (EdgeView *)_self;
    printf("EdgeView.__setitem__ %p %p %d\n", self, self->graph, self->index);
    if (self->index < 0)
    {
        PyErr_SetString(PyExc_KeyError, "Cannot assign to a node");
        return -1;
    }
    PyObject *r = Graph_setedgeattr(self->graph, self->index, k, v);
    return r == NULL ? -1 : 0;
}

static PyObject *
EdgeView_getitem(PyObject *_self, PyObject *k)
{
    EdgeView *self = (EdgeView *)_self;
    printf("EdgeView.__getitem__ %p %p %d ", self, self->graph, self->index);
    PyObject_Print(k, stdout, 0);
    puts("");
    if (self->index < 0)
    {
        int edge_idx = -1;
        int u, v;
        if (!PyTuple_Check(k) || PyTuple_Size(k) != 2)
        {
            PyErr_SetString(PyExc_KeyError, "EdgeView key must be a tuple of length 2");
            return NULL;
        }
        u = PyLong_AsLong(PyTuple_GetItem(k, 0));
        v = PyLong_AsLong(PyTuple_GetItem(k, 1));
        if (u > v)
        {
            int tmp = u;
            u = v;
            v = tmp;
        }
        for (int i = 0; i < self->graph->num_edges; i++)
        {
            if (self->graph->edges[2 * i] == u && self->graph->edges[2 * i + 1] == v)
            {
                edge_idx = i;
                break;
            }
        }
        PyObject *idx = PyLong_FromLong(edge_idx);
        PyObject *args = PyTuple_Pack(2, self->graph, idx);
        PyObject *res = PyObject_CallObject((PyObject *)&EdgeViewType, args);
        Py_DECREF(args);
        Py_DECREF(idx);
        return res;
    }
    return Graph_getedgeattr(self->graph, self->index, k);
}

static PyMappingMethods EdgeView_seqmeth = {
    .mp_subscript = EdgeView_getitem,
    .mp_ass_subscript = EdgeView_setitem,
};

static PyMethodDef EdgeView_methods[] = {
    {NULL} /* Sentinel */
};

PyTypeObject EdgeViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "_C.EdgeView",
    .tp_doc = PyDoc_STR("Constrained EdgeView object"),
    .tp_basicsize = sizeof(EdgeView),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = EdgeView_new,
    .tp_init = (initproc)EdgeView_init,
    .tp_dealloc = (destructor)EdgeView_dealloc,
    .tp_members = EdgeView_members,
    .tp_methods = EdgeView_methods,
    .tp_as_mapping = &EdgeView_seqmeth,
};