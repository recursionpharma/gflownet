#include "./main.h"
#include "structmember.h"

static void EdgeView_dealloc(EdgeView *self) {
    Py_XDECREF(self->graph);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *EdgeView_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    EdgeView *self;
    self = (EdgeView *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->graph = NULL;
    }
    return (PyObject *)self;
}

static int EdgeView_init(EdgeView *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"graph", "index", NULL};
    PyObject *graph = NULL, *tmp;
    int index = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &graph, &index))
        return -1;

    if (graph) {
        tmp = (PyObject *)self->graph;
        Py_INCREF(graph);
        self->graph = (Graph *)graph;
        Py_XDECREF(tmp);
        self->index = index;
    }
    return 0;
}

static PyMemberDef EdgeView_members[] = {
    {NULL} /* Sentinel */
};

static int EdgeView_setitem(PyObject *_self, PyObject *k, PyObject *v) {
    EdgeView *self = (EdgeView *)_self;
    if (self->index < 0) {
        PyErr_SetString(PyExc_KeyError, "Cannot assign to a node");
        return -1;
    }
    PyObject *r = Graph_setedgeattr(self->graph, self->index, k, v);
    return r == NULL ? -1 : 0;
}

int get_edge_index_from_pos(Graph *g, int u_pos, int v_pos) {
    for (int i = 0; i < g->num_edges; i++) {
        if ((g->edges[2 * i] == u_pos && g->edges[2 * i + 1] == v_pos) ||
            (g->edges[2 * i] == v_pos && g->edges[2 * i + 1] == u_pos)) {
            return i;
        }
    }
    return -2;
}

int get_edge_index(Graph *g, int u, int v) {
    if (u > v) {
        int tmp = u;
        u = v;
        v = tmp;
    }
    int u_pos = -1;
    int v_pos = -1;
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i] == u) {
            u_pos = i;
        } else if (g->nodes[i] == v) {
            v_pos = i;
        }
        if (u_pos >= 0 && v_pos >= 0) {
            break;
        }
    }
    return get_edge_index_from_pos(g, u_pos, v_pos);
}

int get_edge_index_py(Graph *g, PyObject *k) {
    if (!PyTuple_Check(k) || PyTuple_Size(k) != 2) {
        PyErr_SetString(PyExc_KeyError, "EdgeView key must be a tuple of length 2");
        return -1;
    }
    int u = PyLong_AsLong(PyTuple_GetItem(k, 0));
    int v = PyLong_AsLong(PyTuple_GetItem(k, 1));
    return get_edge_index(g, u, v);
}

static PyObject *EdgeView_getitem(PyObject *_self, PyObject *k) {
    EdgeView *self = (EdgeView *)_self;
    if (self->index < 0) {
        int edge_idx = get_edge_index_py(self->graph, k);
        // printf("EdgeView_getitem: %d %d\n", self->index, edge_idx);
        // PyObject_Print(k, stdout, 0);
        // printf("\n");
        if (edge_idx < 0) {
            if (edge_idx == -2)
                PyErr_SetString(PyExc_KeyError, "Edge not found");
            return NULL;
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

static int EdgeView_contains(PyObject *_self, PyObject *v) {
    EdgeView *self = (EdgeView *)_self;
    if (self->index < 0) {
        int index = get_edge_index_py(self->graph,
                                      v); // Returns -2 if not found, -1 on error
        if (index == -1) {
            return -1; // There was an error
        }
        return index >= 0;
    }
    PyObject *attr = Graph_getedgeattr(self->graph, self->index, v);
    if (attr == NULL) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static PyMappingMethods EdgeView_mapmeth = {
    .mp_subscript = EdgeView_getitem,
    .mp_ass_subscript = EdgeView_setitem,
};

static PySequenceMethods EdgeView_seqmeth = {
    .sq_contains = EdgeView_contains,
};

static PyMethodDef EdgeView_methods[] = {
    {NULL} /* Sentinel */
};

PyTypeObject EdgeViewType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_C.EdgeView",
    .tp_doc = PyDoc_STR("Constrained EdgeView object"),
    .tp_basicsize = sizeof(EdgeView),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = EdgeView_new,
    .tp_init = (initproc)EdgeView_init,
    .tp_dealloc = (destructor)EdgeView_dealloc,
    .tp_members = EdgeView_members,
    .tp_methods = EdgeView_methods,
    .tp_as_mapping = &EdgeView_mapmeth,
    .tp_as_sequence = &EdgeView_seqmeth,
};