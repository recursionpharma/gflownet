#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include "main.h"

static void
Graph_dealloc(Graph *self)
{
    Py_XDECREF(self->graph_def);
    if (self->num_edges > 0)
    {
        free(self->edges);
        free(self->edge_attrs);
    }
    if (self->num_nodes > 0)
    {
        free(self->nodes);
        free(self->node_attrs);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Graph_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Graph *self;
    self = (Graph *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->graph_def = Py_None;
        Py_INCREF(Py_None);
        self->num_nodes = 0;
        self->num_edges = 0;
        self->nodes = self->edges = self->node_attrs = self->edge_attrs = NULL;
    }
    return (PyObject *)self;
}

static int
Graph_init(Graph *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"graph_def", NULL};
    PyObject *graph_def = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
                                     &graph_def))
        return -1;

    if (graph_def)
    {
        SELF_SET(graph_def);
    }
    return 0;
}

static PyMemberDef Custom_members[] = {
    {"graph_def", T_OBJECT_EX, offsetof(Graph, graph_def), 0,
     "node values"},
    {NULL} /* Sentinel */
};

static PyObject *
Graph_add_node(Graph *self, PyObject *args, PyObject *kwds)
{
    PyObject *node = NULL;
    if (!PyArg_ParseTuple(args, "O", &node))
        return NULL;
    if (node)
    {
        if (!PyLong_Check(node))
        {
            PyErr_SetString(PyExc_TypeError, "node must be an int");
            return NULL;
        }
        int node_id = PyLong_AsLong(node);
        for (int i = 0; i < self->num_nodes; i++)
        {
            if (self->nodes[i] == node_id)
            {
                PyErr_SetString(PyExc_KeyError, "node already exists");
                return NULL;
            }
        }
        self->num_nodes++;
        self->nodes = realloc(self->nodes, self->num_nodes * sizeof(int));
        self->nodes[self->num_nodes - 1] = node_id;
        Py_ssize_t num_attrs;
        if (kwds == NULL || (num_attrs = PyDict_Size(kwds)) == 0)
        {
            Py_RETURN_NONE;
        }
        // Now we need to add the node attributes
        // First check if the attributes are found in the GraphDef
        GraphDef *gt = (GraphDef *)self->graph_def;
        // for k in kwds:
        //   assert k in gt.node_values and kwds[k] in gt.node_values[k]
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        int node_attr_pos = self->num_node_attrs;
        self->num_node_attrs += num_attrs;
        self->node_attrs = realloc(self->node_attrs, self->num_node_attrs * 3 * sizeof(int));

        while (PyDict_Next(kwds, &pos, &key, &value))
        {
            PyObject *node_values = PyDict_GetItem(gt->node_values, key);
            if (node_values == NULL)
            {
                PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
                return NULL;
            }
            Py_ssize_t value_idx = PySequence_Index(node_values, value);
            if (value_idx == -1)
            {
                PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
                return NULL;
            }
            int attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, key));
            self->node_attrs[node_attr_pos * 3] = node_id;
            self->node_attrs[node_attr_pos * 3 + 1] = attr_index;
            self->node_attrs[node_attr_pos * 3 + 2] = value_idx;
            node_attr_pos++;
        }
    }
    Py_RETURN_NONE;
}

static PyObject *
Graph_add_edge(Graph *self, PyObject *args, PyObject *kwds)
{
    PyObject *u = NULL, *v = NULL;
    if (!PyArg_ParseTuple(args, "OO", &u, &v))
        return NULL;
    if (u && v)
    {
        if (!PyLong_Check(u) || !PyLong_Check(v))
        {
            PyErr_SetString(PyExc_TypeError, "u, v must be ints");
            return NULL;
        }
        int u_id = -PyLong_AsLong(u);
        int v_id = -PyLong_AsLong(v);
        if (u_id < v_id)
        {
            int tmp = u_id;
            u_id = v_id;
            v_id = tmp;
        }
        for (int i = 0; i < self->num_nodes; i++)
        {
            if (self->nodes[i] == -u_id)
                u_id = -u_id;
            if (self->nodes[i] == -v_id)
                v_id = -v_id;
        }
        if (u_id < 0 || v_id < 0)
        {
            PyErr_SetString(PyExc_KeyError, "u, v must refer to existing nodes");
            return NULL;
        }
        for (int i = 0; i < self->num_edges; i++)
        {
            if (self->edges[i * 2] == u_id && self->edges[i * 2 + 1] == v_id)
            {
                PyErr_SetString(PyExc_KeyError, "edge already exists");
                return NULL;
            }
        }
        self->num_edges++;
        self->edges = realloc(self->edges, self->num_edges * sizeof(int) * 2);
        self->edges[self->num_edges * 2 - 2] = u_id;
        self->edges[self->num_edges * 2 - 1] = v_id;
        Py_ssize_t num_attrs;
        if (kwds == NULL || (num_attrs = PyDict_Size(kwds)) == 0)
        {
            Py_RETURN_NONE;
        }
        int edge_id = self->num_edges - 1;
        // First check if the attributes are found in the GraphDef
        GraphDef *gt = (GraphDef *)self->graph_def;
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        int edge_attr_pos = self->num_edge_attrs;
        self->num_edge_attrs += num_attrs;
        self->edge_attrs = realloc(self->edge_attrs, self->num_edge_attrs * 3 * sizeof(int));

        while (PyDict_Next(kwds, &pos, &key, &value))
        {
            PyObject *edge_values = PyDict_GetItem(gt->edge_values, key);
            if (edge_values == NULL)
            {
                PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
                return NULL;
            }
            Py_ssize_t value_idx = PySequence_Index(edge_values, value);
            if (value_idx == -1)
            {
                PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
                return NULL;
            }
            int attr_index = PyLong_AsLong(PyDict_GetItem(gt->edge_keypos, key));
            self->edge_attrs[edge_attr_pos * 3] = edge_id;
            self->edge_attrs[edge_attr_pos * 3 + 1] = attr_index;
            self->edge_attrs[edge_attr_pos * 3 + 2] = value_idx;
            edge_attr_pos++;
        }
    }
    Py_RETURN_NONE;
}

static int
NodeView_setitem(PyObject *self, PyObject *k, PyObject *v)
{
    PyObject_Print(k, stdout, 0);
    PyObject_Print(v, stdout, 0);
    return 0;
}

static PyMappingMethods Custom_seqmeth = {
    .mp_ass_subscript = NodeView_setitem,
};

static PyMethodDef Custom_methods[] = {
    {"add_node", (PyCFunction)Graph_add_node, METH_VARARGS | METH_KEYWORDS, "Add a node"},
    {"add_edge", (PyCFunction)Graph_add_edge, METH_VARARGS | METH_KEYWORDS, "Add an edge"},
    {NULL} /* Sentinel */
};

static PyObject *
Graph_getnodes(Graph *self, void *closure)
{
    // Return a new NodeView
    PyObject *args = PyTuple_Pack(1, self);
    PyObject *obj = PyObject_CallObject((PyObject *)&NodeViewType, args);
    Py_DECREF(args);
    return obj;
}

static PyObject *
Graph_getedges(Graph *self, void *closure)
{
    printf("new edgeview \n");
    // Return a new EdgeView
    PyObject *args = PyTuple_Pack(1, self);
    PyObject *obj = PyObject_CallObject((PyObject *)&EdgeViewType, args);
    Py_DECREF(args);
    return obj;
}

static PyGetSetDef Custom_getsetters[] = {
    {"nodes", (getter)Graph_getnodes, NULL, "nodes", NULL},
    {"edges", (getter)Graph_getedges, NULL, "edges", NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject GraphType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "_C.Graph",
    .tp_doc = PyDoc_STR("Constrained Graph object"),
    .tp_basicsize = sizeof(Graph),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Graph_new,
    .tp_init = (initproc)Graph_init,
    .tp_dealloc = (destructor)Graph_dealloc,
    .tp_members = Custom_members,
    .tp_methods = Custom_methods,
    .tp_getset = Custom_getsetters,
    .tp_as_mapping = &Custom_seqmeth,
};

PyObject *Graph_getnodeattr(Graph *self, int index, PyObject *k)
{
    GraphDef *gt = (GraphDef *)self->graph_def;
    // printf("Graph_getnodeattr %p %d %p\n", self, index, gt->node_values);
    PyObject *value_list = PyDict_GetItem(gt->node_values, k);
    if (value_list == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "key not found");
        return NULL;
    }
    long attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, k));
    int true_node_index = -1;
    for (int i = 0; i < self->num_nodes; i++)
    {
        // printf("%d %d %d\n", self->num_nodes, self->nodes[i], index);
        if (self->nodes[i] == index)
        {
            true_node_index = i;
            break;
        }
    }
    if (true_node_index == -1)
    {
        PyErr_SetString(PyExc_KeyError, "node not found");
        return NULL;
    }

    for (int i = 0; i < self->num_node_attrs; i++)
    {
        printf("%d %d %d\n", self->node_attrs[i * 3], self->node_attrs[i * 3 + 1], self->node_attrs[i * 3 + 2]);
        if (self->node_attrs[i * 3] == index && self->node_attrs[i * 3 + 1] == attr_index)
        {
            return PyList_GetItem(value_list, self->node_attrs[i * 3 + 2]);
        }
    }
    PyErr_SetString(PyExc_KeyError, "attribute not set for this node");
    return NULL;
}

PyObject *Graph_setnodeattr(Graph *self, int index, PyObject *k, PyObject *v)
{
    GraphDef *gt = (GraphDef *)self->graph_def;
    int true_node_index = -1;
    for (int i = 0; i < self->num_nodes; i++)
    {
        if (self->nodes[i] == index)
        {
            true_node_index = i;
            break;
        }
    }
    if (true_node_index == -1)
    {
        PyErr_SetString(PyExc_KeyError, "node not found");
        return NULL;
    }
    PyObject *node_values = PyDict_GetItem(gt->node_values, k);
    if (node_values == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
        return NULL;
    }
    Py_ssize_t value_idx = PySequence_Index(node_values, v);
    if (value_idx == -1)
    {
        PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
        return NULL;
    }
    int attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, k));
    for (int i = 0; i < self->num_node_attrs; i++)
    {
        if (self->node_attrs[i * 3] == index && self->node_attrs[i * 3 + 1] == attr_index)
        {
            self->node_attrs[i * 3 + 2] = value_idx;
            Py_RETURN_NONE;
        }
    }
    // Could not find the attribute, add it
    int new_idx = self->num_node_attrs;
    self->num_node_attrs++;
    self->node_attrs = realloc(self->node_attrs, self->num_node_attrs * 3 * sizeof(int));
    self->node_attrs[new_idx * 3] = index;
    self->node_attrs[new_idx * 3 + 1] = attr_index;
    self->node_attrs[new_idx * 3 + 2] = value_idx;
    printf("setnodeattr %d %d %d resulted in realloc\n", index, attr_index, value_idx);
    Py_RETURN_NONE;
}

PyObject *Graph_getedgeattr(Graph *self, int index, PyObject *k)
{
    GraphDef *gt = (GraphDef *)self->graph_def;
    printf("Graph_getedgeattr %p %d %p\n", self, index, gt->edge_values);
    PyObject *value_list = PyDict_GetItem(gt->edge_values, k);
    if (value_list == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "key not found");
        return NULL;
    }
    long attr_index = PyLong_AsLong(PyDict_GetItem(gt->edge_keypos, k));
    if (index > self->num_edges)
    {
        // Should never happen, index is computed by us in EdgeView_getitem, not by the user!
        PyErr_SetString(PyExc_KeyError, "edge not found [but this should never happen!]");
        return NULL;
    }

    for (int i = 0; i < self->num_edge_attrs; i++)
    {
        printf("%d %d %d\n", self->edge_attrs[i * 3], self->edge_attrs[i * 3 + 1], self->edge_attrs[i * 3 + 2]);
        if (self->edge_attrs[i * 3] == index && self->edge_attrs[i * 3 + 1] == attr_index)
        {
            return PyList_GetItem(value_list, self->edge_attrs[i * 3 + 2]);
        }
    }
    PyErr_SetString(PyExc_KeyError, "attribute not set for this node");
    return NULL;
}
PyObject *Graph_setedgeattr(Graph *self, int index, PyObject *k, PyObject *v)
{
    printf("setedgeattr %d %p %p\n", index, k, v);
    PyObject_Print(k, stdout, 0);
    printf(" ");
    PyObject_Print(v, stdout, 0);
    puts("");
    GraphDef *gt = (GraphDef *)self->graph_def;
    PyObject *edge_values = PyDict_GetItem(gt->edge_values, k);
    if (edge_values == NULL)
    {
        PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
        return NULL;
    }
    Py_ssize_t value_idx = PySequence_Index(edge_values, v);
    if (value_idx == -1)
    {
        PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
        return NULL;
    }
    int attr_index = PyLong_AsLong(PyDict_GetItem(gt->edge_keypos, k));
    for (int i = 0; i < self->num_edge_attrs; i++)
    {
        if (self->edge_attrs[i * 3] == index && self->edge_attrs[i * 3 + 1] == attr_index)
        {
            self->edge_attrs[i * 3 + 2] = value_idx;
            Py_RETURN_NONE;
        }
    }
    // Could not find the attribute, add it
    int new_idx = self->num_edge_attrs;
    self->num_edge_attrs++;
    self->edge_attrs = realloc(self->edge_attrs, self->num_edge_attrs * 3 * sizeof(int));
    self->edge_attrs[new_idx * 3] = index;
    self->edge_attrs[new_idx * 3 + 1] = attr_index;
    self->edge_attrs[new_idx * 3 + 2] = value_idx;
    printf("setedgeattr %d %d %d resulted in realloc\n", index, attr_index, value_idx);
    Py_RETURN_NONE;
}

static PyObject *spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0)
    {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);
}

static PyMethodDef SpamMethods[] = {
    {"system", spam_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "_C",  /* name of module */
    "doc", /* module documentation, may be NULL */
    -1,    /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
    SpamMethods};

PyMODINIT_FUNC
PyInit__C(void)
{
    PyObject *m;

    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    SpamError = PyErr_NewException("_C.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0)
    {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }
    printf("SpamError %p %p\n", SpamError, m);
    PyTypeObject *types[] = {
        &GraphType, &GraphDefType,
        &NodeViewType, &EdgeViewType};
    char *names[] = {"Graph", "GraphDef", "NodeView", "EdgeView"};
    for (int i = 0; i < sizeof(types) / sizeof(PyTypeObject *); i++)
    {
        if (PyType_Ready(types[i]) < 0)
        {
            printf("Could not ready %s\n", names[i]);
            Py_DECREF(m);
            return NULL;
        }
        Py_XINCREF(types[i]);
        PyObject_Print((PyObject *)types[i], stdout, 0);
        printf(" %s\n", names[i]);
        if (PyModule_AddObject(m, names[i], (PyObject *)types[i]) < 0)
        {
            printf("Could not add %s\n", names[i]);
            Py_XDECREF(types[i]);
            Py_DECREF(m);
            return NULL;
        }
    }

    return m;
}