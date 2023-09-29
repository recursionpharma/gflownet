#define PY_SSIZE_T_CLEAN
#include "./main.h"
#include "structmember.h"
#include <Python.h>

static void NodeView_dealloc(NodeView *self) {
    Py_XDECREF(self->graph);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *NodeView_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    NodeView *self;
    self = (NodeView *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->graph = NULL;
    }
    return (PyObject *)self;
}

static int NodeView_init(NodeView *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"graph", "index", NULL};
    PyObject *graph = NULL, *tmp;
    int node_id = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &graph, &node_id))
        return -1;

    if (graph) {
        tmp = (PyObject *)self->graph;
        Py_INCREF(graph);
        self->graph = (Graph *)graph;
        Py_XDECREF(tmp);
        self->index = node_id;
        if (node_id >= 0) {
            int node_exists = 0;
            for (int i = 0; i < self->graph->num_nodes; i++) {
                if (self->graph->nodes[i] == node_id) {
                    node_exists = 1;
                }
            }
            if (!node_exists) {
                PyErr_SetString(PyExc_KeyError, "Trying to create a view with a node that does not exist");
                return -1;
            }
        }
    }
    return 0;
}

static PyMemberDef NodeView_members[] = {
    {NULL} /* Sentinel */
};

static int NodeView_setitem(PyObject *_self, PyObject *k, PyObject *v) {
    NodeView *self = (NodeView *)_self;
    if (self->index < 0) {
        PyErr_SetString(PyExc_KeyError, "Cannot assign to a node");
        return -1;
    }
    PyObject *r = Graph_setnodeattr(self->graph, self->index, k, v);
    Py_DECREF(r);
    return r == NULL ? -1 : 0;
}

static PyObject *NodeView_getitem(PyObject *_self, PyObject *k) {
    NodeView *self = (NodeView *)_self;
    if (self->index < 0) {
        PyObject *args = PyTuple_Pack(2, self->graph, k);
        PyObject *res = PyObject_CallObject((PyObject *)&NodeViewType, args);
        Py_DECREF(args);
        return res;
    }
    return Graph_getnodeattr(self->graph, self->index, k);
}

static int NodeView_contains(PyObject *_self, PyObject *v) {
    NodeView *self = (NodeView *)_self;
    if (self->index < 0) {
        // Graph_containsnode
        if (!PyLong_Check(v)) {
            PyErr_SetString(PyExc_TypeError, "NodeView.__contains__ only accepts integers");
            return -1;
        }
        int index = PyLong_AsLong(v);
        for (int i = 0; i < self->graph->num_nodes; i++) {
            if (self->graph->nodes[i] == index) {
                return 1;
            }
        }
        return 0;
    }
    // Graph_nodehasattr
    PyObject *attr = Graph_getnodeattr(self->graph, self->index, v);
    if (attr == NULL) {
        PyErr_Clear();
        return 0;
    }
    Py_DECREF(attr);
    return 1;
}

static Py_ssize_t NodeView_len(PyObject *_self) {
    NodeView *self = (NodeView *)_self;
    if (self->index != -1) {
        // This is inefficient, much like a lot of this codebase... but it's in C. For our graph sizes
        // it's not a big deal (and unsurprisingly, it's fast)
        Py_ssize_t num_attrs = 0;
        for (int i = 0; i < self->graph->num_node_attrs; i++) {
            if (self->graph->node_attrs[3 * i] == self->index) {
                num_attrs++;
            }
        }
        return num_attrs;
    }
    return self->graph->num_nodes;
}

PyObject *NodeView_iter(PyObject *_self) {
    NodeView *self = (NodeView *)_self;
    if (self->index != -1) {
        PyErr_SetString(PyExc_TypeError, "A bound NodeView is not iterable");
        return NULL;
    }
    Py_INCREF(_self); // We have to return a new reference, not a borrowed one
    return _self;
}

PyObject *NodeView_iternext(PyObject *_self) {
    NodeView *self = (NodeView *)_self;
    self->index++;
    if (self->index >= self->graph->num_nodes) {
        return NULL;
    }
    return PyLong_FromLong(self->graph->nodes[self->index]);
}

PyObject *NodeView_get(PyObject *_self, PyObject *args) {
    PyObject *key;
    PyObject *default_value = Py_None;
    if (!PyArg_ParseTuple(args, "O|O", &key, &default_value)) {
        return NULL;
    }
    PyObject *res = NodeView_getitem(_self, key);
    if (res == NULL) {
        PyErr_Clear();
        Py_INCREF(default_value);
        return default_value;
    }
    return res;
}

static PyMappingMethods NodeView_mapmeth = {
    .mp_subscript = NodeView_getitem,
    .mp_ass_subscript = NodeView_setitem,
};

static PySequenceMethods NodeView_seqmeth = {
    .sq_contains = NodeView_contains,
    .sq_length = NodeView_len,
};

static PyMethodDef NodeView_methods[] = {
    {"get", (PyCFunction)NodeView_get, METH_VARARGS, "dict-like get"}, {NULL} /* Sentinel */
};

PyTypeObject NodeViewType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_C.NodeView",
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
    .tp_iter = NodeView_iter,
    .tp_iternext = NodeView_iternext,
};