#include "main.h"
#include "structmember.h"
#include <stdio.h>

// #define GRAPH_DEBUG
#ifdef GRAPH_DEBUG
void *__real_malloc(size_t size);
void __real_free(void *ptr);
void *__real_realloc(void *ptr, size_t size);

long total_mem_alloc = 0;
long cnt = 0;
long things_alloc = 0;
/* build with LDFLAGS="-Wl,--wrap,malloc -Wl,--wrap,free -Wl,--wrap,realloc" python setup.py build_ext --inplace
 * __wrap_malloc - malloc wrapper function
 */
void *__wrap_malloc(size_t size) {
    void *ptr = __real_malloc(size + sizeof(long) * 2);
    void *ptr2 = ptr + size;
    *(long *)ptr2 = size;
    *(long *)(ptr2 + 8) = 0x14285700deadbeef;
    // printf("malloc(%d) = %p\n", size, ptr);
    total_mem_alloc += size;
    things_alloc++;
    if (cnt++ % 10000 == 0) {
        printf("total_mem_alloc: %ld [%ld]\n", total_mem_alloc, things_alloc);
    }
    return ptr;
}

/*
 * __wrap_free - free wrapper function
 */
void __wrap_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    int size = -8;
    void *ptr2 = ptr;
    while (*(long *)ptr2 != 0x14285700deadbeef) {
        ptr2++;
        size++;
    }
    if (size != *(long *)(ptr2 - 8)) {
        printf("size mismatch?: %d != %ld\n", size, *(long *)(ptr2 - 8));
        abort();
    }
    __real_free(ptr);
    // printf("free(%p)\n", ptr);
    total_mem_alloc -= size;
    things_alloc--;
}

void *__wrap_realloc(void *ptr, size_t new_size) {
    if (ptr == NULL) {
        return __wrap_malloc(new_size);
    }
    int size = -8;
    void *ptr2 = ptr;
    while (*(long *)ptr2 != 0x14285700deadbeef) {
        ptr2++;
        size++;
    }
    *(long *)ptr2 = 0; // erase marker
    if (size != *(long *)(ptr2 - 8)) {
        printf("size mismatch?: %d != %ld\n", size, *(long *)(ptr2 - 8));
        abort();
    }
    void *newptr = __real_realloc(ptr, new_size + sizeof(long) * 2);
    void *ptr3 = newptr + new_size;
    *(long *)ptr3 = new_size;
    *(long *)(ptr3 + 8) = 0x14285700deadbeef;
    total_mem_alloc -= size;
    total_mem_alloc += new_size;
    return newptr;
}

#endif

static void Graph_dealloc(Graph *self) {
    Py_XDECREF(self->graph_def);
    free(self->nodes);
    free(self->node_attrs);
    free(self->edges);
    free(self->edge_attrs);
    free(self->degrees);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Graph_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Graph *self;
    self = (Graph *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->graph_def = Py_None;
        Py_INCREF(Py_None);
        self->num_nodes = 0;
        self->num_edges = 0;
        self->nodes = self->edges = self->node_attrs = self->edge_attrs = self->degrees = NULL;
        self->num_node_attrs = 0;
        self->num_edge_attrs = 0;
    }
    return (PyObject *)self;
}

static int Graph_init(Graph *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"graph_def", NULL};
    PyObject *graph_def = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &graph_def))
        return -1;
    if (PyObject_TypeCheck(graph_def, &GraphDefType) == 0) {
        PyErr_SetString(PyExc_TypeError, "Graph: graph_def must be a GraphDef");
        return -1;
    }
    if (graph_def) {
        SELF_SET(graph_def);
    }
    return 0;
}

static PyMemberDef Graph_members[] = {
    {"graph_def", T_OBJECT_EX, offsetof(Graph, graph_def), 0, "node values"}, {NULL} /* Sentinel */
};

int graph_get_node_pos(Graph *g, int node_id) {
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i] == node_id) {
            return i;
        }
    }
    return -1;
}

static PyObject *Graph_add_node(Graph *self, PyObject *args, PyObject *kwds) {
    PyObject *node = NULL;
    if (!PyArg_ParseTuple(args, "O", &node))
        return NULL;
    if (node) {
        if (!PyLong_Check(node)) {
            PyErr_SetString(PyExc_TypeError, "node must be an int");
            return NULL;
        }
        int node_id = PyLong_AsLong(node);
        if (graph_get_node_pos(self, node_id) >= 0) {
            PyErr_SetString(PyExc_KeyError, "node already exists");
            return NULL;
        }
        int node_pos = self->num_nodes;
        self->num_nodes++;
        self->nodes = realloc(self->nodes, self->num_nodes * sizeof(int));
        self->nodes[node_pos] = node_id;
        self->degrees = realloc(self->degrees, self->num_nodes * sizeof(int));
        self->degrees[node_pos] = 0;
        Py_ssize_t num_attrs;
        if (kwds == NULL || (num_attrs = PyDict_Size(kwds)) == 0) {
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

        while (PyDict_Next(kwds, &pos, &key, &value)) {
            PyObject *node_values = PyDict_GetItem(gt->node_values, key);
            if (node_values == NULL) {
                PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
                return NULL;
            }
            Py_ssize_t value_idx = PySequence_Index(node_values, value);
            if (value_idx == -1) {
                PyObject *repr = PyObject_Repr(value);
                PyObject *key_repr = PyObject_Repr(key);
                PyErr_Format(PyExc_KeyError, "value %s not found in GraphDef for key %s", PyUnicode_AsUTF8(repr),
                             PyUnicode_AsUTF8(key_repr));
                Py_DECREF(repr);
                // PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
                return NULL;
            }
            int attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, key));
            self->node_attrs[node_attr_pos * 3] = node_pos;
            self->node_attrs[node_attr_pos * 3 + 1] = attr_index;
            self->node_attrs[node_attr_pos * 3 + 2] = value_idx;
            node_attr_pos++;
        }
    }
    Py_RETURN_NONE;
}

static PyObject *Graph_add_edge(Graph *self, PyObject *args, PyObject *kwds) {
    PyObject *u = NULL, *v = NULL;
    if (!PyArg_ParseTuple(args, "OO", &u, &v))
        return NULL;
    if (u && v) {
        if (!PyLong_Check(u) || !PyLong_Check(v)) {
            PyErr_SetString(PyExc_TypeError, "u, v must be ints");
            return NULL;
        }
        int u_id = -PyLong_AsLong(u);
        int v_id = -PyLong_AsLong(v);
        int u_pos = -1;
        int v_pos = -1;
        if (u_id < v_id) {
            int tmp = u_id;
            u_id = v_id;
            v_id = tmp;
        }
        for (int i = 0; i < self->num_nodes; i++) {
            if (self->nodes[i] == -u_id) {
                u_id = -u_id;
                u_pos = i;
            } else if (self->nodes[i] == -v_id) {
                v_id = -v_id;
                v_pos = i;
            }
            if (u_pos >= 0 && v_pos >= 0) {
                break;
            }
        }
        if (u_id < 0 || v_id < 0) {
            PyErr_SetString(PyExc_KeyError, "u, v must refer to existing nodes");
            return NULL;
        }
        if (u_id == v_id){
            PyErr_SetString(PyExc_KeyError, "u, v must be different nodes");
            return NULL;
        }
        for (int i = 0; i < self->num_edges; i++) {
            if (self->edges[i * 2] == u_pos && self->edges[i * 2 + 1] == v_pos) {
                PyErr_SetString(PyExc_KeyError, "edge already exists");
                return NULL;
            }
        }
        self->num_edges++;
        self->edges = realloc(self->edges, self->num_edges * sizeof(int) * 2);
        self->edges[self->num_edges * 2 - 2] = u_pos;
        self->edges[self->num_edges * 2 - 1] = v_pos;
        self->degrees[u_pos]++;
        self->degrees[v_pos]++;

        Py_ssize_t num_attrs;
        if (kwds == NULL || (num_attrs = PyDict_Size(kwds)) == 0) {
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

        while (PyDict_Next(kwds, &pos, &key, &value)) {
            PyObject *edge_values = PyDict_GetItem(gt->edge_values, key);
            if (edge_values == NULL) {
                PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
                return NULL;
            }
            Py_ssize_t value_idx = PySequence_Index(edge_values, value);
            if (value_idx == -1) {
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

PyObject *Graph_isdirected(Graph *self, PyObject *Py_UNUSED(ignored)) { Py_RETURN_FALSE; }

void bridge_dfs(int v, int parent, int *visited, int *tin, int *low, int *timer, int **adj, int *degrees, int *result,
                Graph *g) {
    visited[v] = 1;
    tin[v] = low[v] = (*timer)++;
    for (int i = 0; i < degrees[v]; i++) {
        int to = adj[v][i];
        if (to == parent)
            continue;
        if (visited[to]) {
            low[v] = mini(low[v], tin[to]);
        } else {
            bridge_dfs(to, v, visited, tin, low, timer, adj, degrees, result, g);
            low[v] = mini(low[v], low[to]);
            if (low[to] > tin[v]) {
                result[get_edge_index_from_pos(g, v, to)] = 1;
            }
        }
    }
}
PyObject *Graph_bridges(PyObject *_self, PyObject *args) {
    Graph *self = (Graph *)_self;
    // from:
    // https://cp-algorithms.com/graph/bridge-searching.html
    const int n = self->num_nodes; // number of nodes
    int *adj[n];
    for (int i = 0; i < n; i++) {
        adj[i] = malloc(self->degrees[i] * sizeof(int));
    }
    int is_bridge[self->num_edges];
    for (int i = 0; i < self->num_edges; i++) {
        is_bridge[i] = 0;
        *(adj[self->edges[i * 2]]) = self->edges[i * 2 + 1];
        *(adj[self->edges[i * 2 + 1]]) = self->edges[i * 2];
        // increase the pointer
        adj[self->edges[i * 2]]++;
        adj[self->edges[i * 2 + 1]]++;
    }
    // reset the pointers using the degrees
    for (int i = 0; i < n; i++) {
        adj[i] -= self->degrees[i];
    }
    int visited[n];
    int tin[n];
    int low[n];
    for (int i = 0; i < n; i++) {
        visited[i] = 0;
        tin[i] = -1;
        low[i] = -1;
    }

    int timer = 0;
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            bridge_dfs(i, -1, visited, tin, low, &timer, adj, self->degrees, is_bridge, self);
    }
    if (args == NULL) {
        // This function is being called from python, so we must return a list
        PyObject *result = PyList_New(0);
        for (int i = 0; i < self->num_edges; i++) {
            if (is_bridge[i]) {
                PyObject *u = PyLong_FromLong(self->nodes[self->edges[i * 2]]);
                PyObject *v = PyLong_FromLong(self->nodes[self->edges[i * 2 + 1]]);
                PyObject *t = PyTuple_Pack(2, u, v);
                PyList_Append(result, t);
                Py_DECREF(u);
                Py_DECREF(v);
                Py_DECREF(t);
            }
        }
        for (int i = 0; i < n; i++) {
            free(adj[i]);
        }
        return result;
    }
    // This function is being called from C, so args is actually a int* to store the result
    int *result = (int *)args;
    for (int i = 0; i < self->num_edges; i++) {
        result[i] = is_bridge[i];
    }
    for (int i = 0; i < n; i++) {
        free(adj[i]);
    }
    return 0; // success
}

PyObject *Graph_copy(PyObject *_self, PyObject *unused_args) {
    Graph *self = (Graph *)_self;
    PyObject *args = PyTuple_Pack(1, self->graph_def);
    Graph *obj = (Graph *)PyObject_CallObject((PyObject *)&GraphType, args);
    Py_DECREF(args);
    obj->num_nodes = self->num_nodes;
    obj->num_edges = self->num_edges;
    obj->nodes = malloc(self->num_nodes * sizeof(int));
    memcpy(obj->nodes, self->nodes, self->num_nodes * sizeof(int));
    obj->edges = malloc(self->num_edges * 2 * sizeof(int));
    memcpy(obj->edges, self->edges, self->num_edges * 2 * sizeof(int));
    obj->num_node_attrs = self->num_node_attrs;
    obj->num_edge_attrs = self->num_edge_attrs;
    obj->node_attrs = malloc(self->num_node_attrs * 3 * sizeof(int));
    memcpy(obj->node_attrs, self->node_attrs, self->num_node_attrs * 3 * sizeof(int));
    obj->edge_attrs = malloc(self->num_edge_attrs * 3 * sizeof(int));
    memcpy(obj->edge_attrs, self->edge_attrs, self->num_edge_attrs * 3 * sizeof(int));
    obj->degrees = malloc(self->num_nodes * sizeof(int));
    memcpy(obj->degrees, self->degrees, self->num_nodes * sizeof(int));
    return (PyObject *)obj;
}

PyObject *Graph_has_edge(PyObject *_self, PyObject *args) {
    int u, v;
    if (!PyArg_ParseTuple(args, "ii", &u, &v))
        return NULL;
    if (get_edge_index((Graph *)_self, u, v) >= 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

void _Graph_check(Graph *g) {
#ifdef GRAPH_DEBUG
    for (int i = 0; i < g->num_node_attrs; i++) {
        if (g->node_attrs[i * 3] >= g->num_nodes) {
            printf("Invalid node attr pointer %d %d\n", g->node_attrs[i * 3], g->num_nodes);
            abort();
        }
    }
    for (int i = 0; i < g->num_edge_attrs; i++) {
        if (g->edge_attrs[i * 3] >= g->num_edges) {
            printf("Invalid edge attr pointer %d %d\n", g->edge_attrs[i * 3], g->num_edges);
            abort();
        }
    }
#endif
}

PyObject *Graph_remove_edge(PyObject *, PyObject *); // fwd decl

PyObject *Graph_remove_node(PyObject *_self, PyObject *args) {
    Graph *self = (Graph *)_self;
    int u;
    if (!PyArg_ParseTuple(args, "i", &u))
        return NULL;
    int pos = graph_get_node_pos(self, u);
    if (pos < 0) {
        PyErr_SetString(PyExc_KeyError, "node not found");
        return NULL;
    }
    // Check if any edge has this node
    for (int i = 0; i < self->num_edges; i++) {
        if (self->edges[i * 2] == pos || self->edges[i * 2 + 1] == pos) {
            PyObject *u = PyLong_FromLong(self->nodes[self->edges[i * 2]]);
            PyObject *v = PyLong_FromLong(self->nodes[self->edges[i * 2 + 1]]);
            PyObject *rm_args = PyTuple_Pack(2, u, v);
            // if so remove it
            PyObject *rm_res = Graph_remove_edge(_self, rm_args);
            Py_DECREF(u);
            Py_DECREF(v);
            Py_DECREF(rm_args);
            if (rm_res == NULL) {
                return NULL;
            }
            Py_DECREF(rm_res);
            i--;
        }
    }
    // Remove the node
    // self->nodes contains node "names", so we just pop one element from the array
    int *old_nodes = self->nodes;
    self->nodes = malloc((self->num_nodes - 1) * sizeof(int));
    memcpy(self->nodes, old_nodes, pos * sizeof(int));
    memcpy(self->nodes + pos, old_nodes + pos + 1, (self->num_nodes - pos - 1) * sizeof(int));
    free(old_nodes);
    int *old_degrees = self->degrees;
    self->degrees = malloc((self->num_nodes - 1) * sizeof(int));
    memcpy(self->degrees, old_degrees, pos * sizeof(int));
    memcpy(self->degrees + pos, old_degrees + pos + 1, (self->num_nodes - pos - 1) * sizeof(int));
    free(old_degrees);
    self->num_nodes--;
    // Remove the node attributes
    int *old_node_attrs = self->node_attrs;
    int num_rem = 0;
    // first find out how many attributes we need to remove
    for (int i = 0; i < self->num_node_attrs; i++) {
        if (self->node_attrs[i * 3] == pos) {
            num_rem++;
        }
    }
    if (num_rem) {
        // now remove them
        self->node_attrs = malloc((self->num_node_attrs - num_rem) * 3 * sizeof(int));
        int i, j = 0;
        for (i = 0; i < self->num_node_attrs; i++) {
            if (old_node_attrs[i * 3] == pos) {
                continue;
            }
            int old_node_index = old_node_attrs[i * 3];
            self->node_attrs[j * 3] = old_node_index;
            self->node_attrs[j * 3 + 1] = old_node_attrs[i * 3 + 1];
            self->node_attrs[j * 3 + 2] = old_node_attrs[i * 3 + 2];
            j++;
        }
        self->num_node_attrs -= num_rem;
        free(old_node_attrs);
    }
    // since we removed the node at pos, all other node indices past that must be decremented
    for (int i = 0; i < self->num_node_attrs; i++) {
        if (self->node_attrs[i * 3] > pos) {
            self->node_attrs[i * 3]--;
        }
    }
    // We must also relabel the edges
    for (int i = 0; i < self->num_edges; i++) {
        if (self->edges[i * 2] > pos) {
            self->edges[i * 2]--;
        }
        if (self->edges[i * 2 + 1] > pos) {
            self->edges[i * 2 + 1]--;
        }
    }
    _Graph_check(self);
    Py_RETURN_NONE;
}

PyObject *Graph_remove_edge(PyObject *_self, PyObject *args) {
    Graph *self = (Graph *)_self;
    int u, v;
    if (!PyArg_ParseTuple(args, "ii", &u, &v))
        return NULL;
    int edge_index = get_edge_index((Graph *)_self, u, v);
    if (edge_index < 0) {
        PyErr_SetString(PyExc_KeyError, "edge not found");
        return NULL;
    }
    // Decrease degree
    self->degrees[self->edges[edge_index * 2]]--;
    self->degrees[self->edges[edge_index * 2 + 1]]--;
    // Remove the edge
    int *old_edges = self->edges;
    self->edges = malloc((self->num_edges - 1) * 2 * sizeof(int));
    memcpy(self->edges, old_edges, edge_index * 2 * sizeof(int));
    memcpy(self->edges + edge_index * 2, old_edges + edge_index * 2 + 2,
           (self->num_edges - edge_index - 1) * 2 * sizeof(int));
    free(old_edges);
    self->num_edges--;
    // Remove the edge attributes
    int num_rem = 0;
    for (int i = 0; i < self->num_edge_attrs; i++) {
        if (self->edge_attrs[i * 3] == edge_index) {
            num_rem++;
        }
    }
    if (num_rem) {
        int *old_edge_attrs = self->edge_attrs;
        self->edge_attrs = malloc((self->num_edge_attrs - num_rem) * 3 * sizeof(int));
        int i, j = 0;
        for (i = 0; i < self->num_edge_attrs; i++) {
            int old_edge_index = old_edge_attrs[i * 3];
            if (old_edge_index == edge_index) {
                continue;
            }
            self->edge_attrs[j * 3] = old_edge_index;
            self->edge_attrs[j * 3 + 1] = old_edge_attrs[i * 3 + 1];
            self->edge_attrs[j * 3 + 2] = old_edge_attrs[i * 3 + 2];
            j++;
        }

        self->num_edge_attrs -= num_rem;
        free(old_edge_attrs);
    }
    // since we removed the edge at edge_index, all other edge indices past that must be decremented
    for (int i = 0; i < self->num_edge_attrs; i++) {
        if (self->edge_attrs[i * 3] > edge_index) {
            self->edge_attrs[i * 3]--;
        }
    }
    _Graph_check(self);
    Py_RETURN_NONE;
}

PyObject *Graph_relabel(PyObject *_self, PyObject *args) {
    PyObject *mapping = NULL;
    if (!PyArg_ParseTuple(args, "O", &mapping))
        return NULL;
    if (!PyDict_Check(mapping)) {
        PyErr_SetString(PyExc_TypeError, "mapping must be a dict");
        return NULL;
    }
    Graph *new = (Graph *)Graph_copy(_self, NULL); // new ref
    for (int i = 0; i < new->num_nodes; i++) {
        PyObject *node = PyLong_FromLong(new->nodes[i]);
        PyObject *new_node = PyDict_GetItem(mapping, node); // ref is borrowed
        Py_DECREF(node);
        if (new_node == NULL) {
            PyErr_Format(PyExc_KeyError, "node %d not found in mapping", new->nodes[i]);
            return NULL;
        }
        if (!PyLong_Check(new_node)) {
            PyErr_SetString(PyExc_TypeError, "mapping must be a dict of ints");
            return NULL;
        }
        new->nodes[i] = PyLong_AsLong(new_node);
    }
    _Graph_check((Graph *)_self);
    return (PyObject *)new;
}

PyObject *Graph_inspect(PyObject *_self, PyObject *args) {
    Graph *self = (Graph *)_self;
    char buffer[4096];
    memset(buffer, 0, 4096);
    int cap = 4096;
    int ptr = 0;

    ptr += snprintf(buffer, cap - ptr, "Node labels:\n  ");
    for (int i = 0; i < self->num_nodes; i++) {
        printf("%d ", self->nodes[i]);
    }
    ptr += snprintf(buffer + ptr, cap - ptr, "\n");
    ptr += snprintf(buffer + ptr, cap - ptr, "Edges:\n");
    for (int i = 0; i < self->num_edges; i++) {
        ptr += snprintf(buffer + ptr, cap - ptr, "  %d %d (%d %d)\n", self->nodes[self->edges[i * 2]], self->nodes[self->edges[i * 2 + 1]],
               self->edges[i * 2], self->edges[i * 2 + 1]);
    }
    ptr += snprintf(buffer + ptr, cap - ptr, "Node attributes:\n");
    for (int i = 0; i < self->num_node_attrs; i++) {
        ptr += snprintf(buffer + ptr, cap - ptr, "  %d %d %d\n", self->node_attrs[i * 3], self->node_attrs[i * 3 + 1], self->node_attrs[i * 3 + 2]);
    }
    ptr += snprintf(buffer + ptr, cap - ptr, "Edge attributes:\n");
    for (int i = 0; i < self->num_edge_attrs; i++) {
        ptr += snprintf(buffer + ptr, cap - ptr, "  %d %d %d\n", self->edge_attrs[i * 3], self->edge_attrs[i * 3 + 1], self->edge_attrs[i * 3 + 2]);
    }
    ptr += snprintf(buffer + ptr, cap - ptr, "Degrees:\n  ");
    for (int i = 0; i < self->num_nodes; i++) {
        ptr += snprintf(buffer + ptr, cap - ptr, "%d ", self->degrees[i]);
    }
    ptr += snprintf(buffer + ptr, cap - ptr, "\n\n");
    PyObject *res = PyUnicode_FromString(buffer);
    return res;
}

PyObject *Graph_getstate(PyObject *_self, PyObject *args) {
    Graph *self = (Graph *)_self;
    int num_bytes = (4 + self->num_nodes * 2 + self->num_edges * 2 + self->num_node_attrs * 3 + self->num_edge_attrs * 3) * sizeof(int);
    // Create a new bytes object
    PyObject *state = PyBytes_FromStringAndSize(NULL, num_bytes);
    char *buf = PyBytes_AS_STRING(state);
    int* ptr = (int*)buf;
    ptr[0] = self->num_nodes;
    ptr[1] = self->num_edges;
    ptr[2] = self->num_node_attrs;
    ptr[3] = self->num_edge_attrs;
    memcpy(ptr + 4, self->nodes, self->num_nodes * sizeof(int));
    memcpy(ptr + 4 + self->num_nodes, self->edges, self->num_edges * 2 * sizeof(int));
    memcpy(ptr + 4 + self->num_nodes + self->num_edges * 2, self->node_attrs, self->num_node_attrs * 3 * sizeof(int));
    memcpy(ptr + 4 + self->num_nodes + self->num_edges * 2 + self->num_node_attrs * 3,
           self->edge_attrs, self->num_edge_attrs * 3 * sizeof(int));
    memcpy(ptr + 4 + self->num_nodes + self->num_edges * 2 + self->num_node_attrs * 3 + self->num_edge_attrs * 3,
           self->degrees, self->num_nodes * sizeof(int));
    // Return (bytes, GraphDef)
    PyObject *res = PyTuple_Pack(2, state, self->graph_def);
    Py_DECREF(state);
    return res;
}
PyObject *Graph_setstate(PyObject *_self, PyObject *args) {
    PyObject* state = PyTuple_GetItem(args, 0);
    PyObject* graph_def = PyTuple_GetItem(args, 1);
    if (!PyBytes_Check(state)) {
        PyErr_SetString(PyExc_TypeError, "state must be a bytes object");
        return NULL;
    }
    if (PyObject_TypeCheck(graph_def, &GraphDefType) == 0) {
        PyErr_SetString(PyExc_TypeError, "graph_def must be a GraphDef");
        return NULL;
    }
    Graph *self = (Graph *)_self;
    Py_XDECREF(self->graph_def);
    self->graph_def = graph_def;
    Py_XINCREF(graph_def);
    char *buf = PyBytes_AS_STRING(state);
    int* ptr = (int*)buf;
    self->num_nodes = ptr[0];
    self->num_edges = ptr[1];
    self->num_node_attrs = ptr[2];
    self->num_edge_attrs = ptr[3];
    self->nodes = malloc(self->num_nodes * sizeof(int));
    self->edges = malloc(self->num_edges * 2 * sizeof(int));
    self->node_attrs = malloc(self->num_node_attrs * 3 * sizeof(int));
    self->edge_attrs = malloc(self->num_edge_attrs * 3 * sizeof(int));
    self->degrees = malloc(self->num_nodes * sizeof(int));
    memcpy(self->nodes, ptr + 4, self->num_nodes * sizeof(int));
    memcpy(self->edges, ptr + 4 + self->num_nodes, self->num_edges * 2 * sizeof(int));
    memcpy(self->node_attrs, ptr + 4 + self->num_nodes + self->num_edges * 2, self->num_node_attrs * 3 * sizeof(int));
    memcpy(self->edge_attrs, ptr + 4 + self->num_nodes + self->num_edges * 2 + self->num_node_attrs * 3,
           self->num_edge_attrs * 3 * sizeof(int));
    memcpy(self->degrees, ptr + 4 + self->num_nodes + self->num_edges * 2 + self->num_node_attrs * 3 + self->num_edge_attrs * 3,
           self->num_nodes * sizeof(int));
    Py_RETURN_NONE;
}


int _NodeView_eq(NodeView *self, NodeView *other);
int _EdgeView_eq(EdgeView *self, EdgeView *other);

inline int _fast_Node_eq(int i, int j, int* nodeval_cache, int N, int M){
    for (int k = 0; k < M; k++){
        //printf("nodeeq i=%d j=%d k=%d M=%d: %d == %d\n", i, j, k, M, nodeval_cache[i * M + k], nodeval_cache[(j + N) * M + k]);
        if (nodeval_cache[i * M + k] != nodeval_cache[(j + N) * M + k])
            return 0;
    }
    return 1;
}

int _Graph_iso_search(Graph* a, Graph* b, int* assignment, int depth, int* nodeval_cache, int* a_edgeidx_cache){
    /* Python version:
def search(graph, subgraph, assignments, depth):
    # Make sure that every edge between assigned vertices in the subgraph is also an
    # edge in the graph.
    for u, v in subgraph.edges:
        if u < depth and v < depth:
            x, y = assignments[u], assignments[v]
            if not (
                graph.has_edge(x, y) and
                graph.nodes[x] == subgraph.nodes[u] and graph.nodes[y] == subgraph.nodes[v] and
                graph.edges[x, y] == subgraph.edges[u, v]):
                return False

    # If all the vertices in the subgraph are assigned, then we are done.
    if depth == len(subgraph):
        return True

    for j in range(len(subgraph)): #possible_assignments[i]:
        if assignments[depth] == -1:
            assignments[depth] = j
            if search(graph, subgraph, assignments, depth+1):
                return True

            assignments[depth] = -1    */
    /*
    for (int i=0;i<a->num_nodes*2;i++){
        if (i == a->num_nodes){
            printf("|| ");
        }
        printf("%d ", assignment[i]);
    }
    puts("");*/
    for (int i = 0; i < b->num_edges; i++){
        int u = b->edges[i * 2];
        int v = b->edges[i * 2 + 1];
        if (u < depth && v < depth){
            int x = assignment[u];
            int y = assignment[v];
            int x_y = a_edgeidx_cache[x * a->num_nodes + y]; //  get_edge_index(a, a->nodes[x], a->nodes[y]);
            if (x_y < 0)
                return 0;
            //NodeView nv_x = {.graph = a, .index = x};
            //NodeView nv_y = {.graph = a, .index = y};
            //NodeView nv_u = {.graph = b, .index = u};
            //NodeView nv_v = {.graph = b, .index = v};
            //if (!_NodeView_eq(&nv_x, &nv_u) || !_NodeView_eq(&nv_y, &nv_v))
            if (!_fast_Node_eq(x, u, nodeval_cache, a->num_nodes, 1 + ((GraphDef*)a->graph_def)->num_settable_node_attrs) ||
                !_fast_Node_eq(y, v, nodeval_cache, a->num_nodes, 1 + ((GraphDef*)a->graph_def)->num_settable_node_attrs))
                return 0;
            EdgeView ev_x_y = {.graph = a, .index = x_y};
            EdgeView ev_u_v = {.graph = b, .index = i};
            if (!_EdgeView_eq(&ev_x_y, &ev_u_v))
                return 0;
        }
    }

    if (depth == b->num_nodes){
        // Found a valid assignment
        return 1;
    }
    for (int j = 0; j < b->num_nodes; j++){
        //printf("Trying %d = %d\n", depth, j);
        if (assignment[b->num_nodes + j] != -1)
            continue;
        //NodeView nv_a = {.graph = a, .index = j};
        //NodeView nv_b = {.graph = b, .index = depth};
        if (a->degrees[j] == b->degrees[depth]){
            if (!_fast_Node_eq(j, depth, nodeval_cache, a->num_nodes, 1 + ((GraphDef*)a->graph_def)->num_settable_node_attrs)){
                continue;
            }
            /*if (!_NodeView_eq(&nv_a, &nv_b)){
                continue;
            }*/
            assignment[depth] = j;
            assignment[j + b->num_nodes] = depth;
            if (_Graph_iso_search(a, b, assignment, depth + 1, nodeval_cache, a_edgeidx_cache)){
                return 1;
            }
            assignment[depth] = -1;
            assignment[j + b->num_nodes] = -1;
        }
    }
    return 0;
}

PyObject *Graph_is_isomorphic(PyObject *_self, PyObject *args) {
    Graph *self = (Graph *)_self;
    PyObject *other = NULL;
    if (!PyArg_ParseTuple(args, "O", &other))
        return NULL;
    if (PyObject_TypeCheck(other, &GraphType) == 0) {
        PyErr_SetString(PyExc_TypeError, "other must be a Graph");
        return NULL;
    }
    Graph *other_graph = (Graph *)other;
    if (self->graph_def != other_graph->graph_def) {
        Py_RETURN_FALSE;
    }
    if (self->num_nodes != other_graph->num_nodes || self->num_edges != other_graph->num_edges) {
        Py_RETURN_FALSE;
    }
    int assignments[self->num_nodes * 2];
    for (int i = 0; i < self->num_nodes * 2; i++) {
        assignments[i] = -1;
    }
    int num_attrs = ((GraphDef *)self->graph_def)->num_settable_node_attrs + 1;
    int nodeval_cache[self->num_nodes * num_attrs * 2];
    memset(nodeval_cache, 0, self->num_nodes * num_attrs * 2 * sizeof(int));
    for (int j = 0; j < self->num_node_attrs; j++){
        int node_attr_index = self->node_attrs[j * 3];
        int attr_index = self->node_attrs[j * 3 + 1];
        int value_index = self->node_attrs[j * 3 + 2];
        nodeval_cache[node_attr_index * num_attrs + attr_index] = value_index;
    }
    for (int j = 0; j < other_graph->num_node_attrs; j++){
        int node_attr_index = other_graph->node_attrs[j * 3];
        int attr_index = other_graph->node_attrs[j * 3 + 1];
        int value_index = other_graph->node_attrs[j * 3 + 2];
        nodeval_cache[(node_attr_index + self->num_nodes) * num_attrs + attr_index] = value_index;
    }
    // Q: Is something wrong with the above? Are we setting the values correctly?
    // A: Yes, we are setting the values correctly. The nodeval_cache is a 3D array, where the first dimension is the node index, the second dimension is the attribute index, and the third dimension is the value index.
    //    The first num_nodes * num_attrs entries are for the first graph, and the next num_nodes * num_attrs entries are for the second graph.

    int a_edgeidx_cache[self->num_nodes * self->num_nodes];
    memset(a_edgeidx_cache, -1, self->num_nodes * self->num_nodes * sizeof(int));
    for (int i = 0; i < self->num_edges; i++){
        a_edgeidx_cache[self->edges[i * 2] * self->num_nodes + self->edges[i * 2 + 1]] = i;
    }
    if (_Graph_iso_search(self, other_graph, assignments, 0, nodeval_cache, a_edgeidx_cache)){
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}


static PyMethodDef Graph_methods[] = {
    {"add_node", (PyCFunction)Graph_add_node, METH_VARARGS | METH_KEYWORDS, "Add a node"},
    {"add_edge", (PyCFunction)Graph_add_edge, METH_VARARGS | METH_KEYWORDS, "Add an edge"},
    {"has_edge", (PyCFunction)Graph_has_edge, METH_VARARGS, "Check if an edge is present"},
    {"remove_node", (PyCFunction)Graph_remove_node, METH_VARARGS, "Remove a node"},
    {"remove_edge", (PyCFunction)Graph_remove_edge, METH_VARARGS, "Remove an edge"},
    {"is_directed", (PyCFunction)Graph_isdirected, METH_NOARGS, "Is the graph directed?"},
    {"is_multigraph", (PyCFunction)Graph_isdirected, METH_NOARGS, "Is the graph a multigraph?"},
    {"bridges", (PyCFunction)Graph_bridges, METH_NOARGS, "Find the bridges of the graph"},
    {"copy", (PyCFunction)Graph_copy, METH_VARARGS, "Copy the graph"},
    {"relabel_nodes", (PyCFunction)Graph_relabel, METH_VARARGS, "relabel the graph"},
    {"is_isomorphic", (PyCFunction)Graph_is_isomorphic, METH_VARARGS, "isomorphism test"},
    {"__deepcopy__", (PyCFunction)Graph_copy, METH_VARARGS, "Copy the graph"},
    {"__getstate__", (PyCFunction)Graph_getstate, METH_NOARGS, "pickling"},
    {"__setstate__", (PyCFunction)Graph_setstate, METH_O, "unpickling"},
    {"_inspect", (PyCFunction)Graph_inspect, METH_VARARGS, "Print the graph's information"},
    {NULL} /* Sentinel */
};

static PyObject *Graph_getnodes(Graph *self, void *closure) {
    // Return a new NodeView
    PyObject *args = PyTuple_Pack(1, self);
    PyObject *obj = PyObject_CallObject((PyObject *)&NodeViewType, args); // new ref
    Py_DECREF(args);
    return obj;
}

static PyObject *Graph_getedges(Graph *self, void *closure) {
    // Return a new EdgeView
    PyObject *args = PyTuple_Pack(1, self);
    PyObject *obj = PyObject_CallObject((PyObject *)&EdgeViewType, args);
    Py_DECREF(args);
    return obj;
}

static PyObject *Graph_getdegree(Graph *self, void *closure) {
    PyObject *args = PyTuple_Pack(1, self);
    PyObject *obj = PyObject_CallObject((PyObject *)&DegreeViewType, args);
    Py_DECREF(args);
    return obj;
}

static PyGetSetDef Graph_getsetters[] = {
    {"nodes", (getter)Graph_getnodes, NULL, "nodes", NULL},
    {"edges", (getter)Graph_getedges, NULL, "edges", NULL},
    {"degree", (getter)Graph_getdegree, NULL, "degree", NULL},
    {NULL} /* Sentinel */
};

static PyObject *Graph_contains(Graph *self, PyObject *v) {
    if (!PyLong_Check(v)) {
        PyErr_SetString(PyExc_TypeError, "Graph.__contains__ only accepts integers");
        return NULL;
    }
    int node_id = PyLong_AsLong(v);
    int pos = graph_get_node_pos(self, node_id);
    if (pos >= 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static Py_ssize_t Graph_len(Graph *self) { return self->num_nodes; }

static PySequenceMethods Graph_seqmeth = {
    .sq_contains = (objobjproc)Graph_contains,
    .sq_length = (lenfunc)Graph_len,
};

static PyObject *Graph_iter(PyObject *self) {
    PyObject *args = PyTuple_Pack(1, self);
    PyObject *obj = PyObject_CallObject((PyObject *)&NodeViewType, args); // new ref
    Py_DECREF(args);
    return obj;
}

static PyObject *Graph_getitem(PyObject *self, PyObject *key) {
    PyObject *args = PyTuple_Pack(2, self, key);
    PyObject *obj = PyObject_CallObject((PyObject *)&NodeViewType, args); // new ref
    Py_DECREF(args);
    return obj;
}
static PyMappingMethods Graph_mapmeth = {
    .mp_subscript = Graph_getitem,
};

PyTypeObject GraphType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "gflownet._C.Graph",
    .tp_doc = PyDoc_STR("Constrained Graph object"),
    .tp_basicsize = sizeof(Graph),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Graph_new,
    .tp_init = (initproc)Graph_init,
    .tp_dealloc = (destructor)Graph_dealloc,
    .tp_members = Graph_members,
    .tp_methods = Graph_methods,
    .tp_iter = Graph_iter,
    .tp_getset = Graph_getsetters,
    .tp_as_sequence = &Graph_seqmeth,
    .tp_as_mapping = &Graph_mapmeth,
};

PyObject *Graph_getnodeattr(Graph *self, int index, PyObject *k) {
    GraphDef *gt = (GraphDef *)self->graph_def;
    PyObject *value_list = PyDict_GetItem(gt->node_values, k); // borrowed ref
    if (value_list == NULL) {
        PyErr_SetString(PyExc_KeyError, "key not found");
        return NULL;
    }
    long attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, k));
    int true_node_index = -1;
    for (int i = 0; i < self->num_nodes; i++) {
        if (self->nodes[i] == index) {
            true_node_index = i;
            break;
        }
    }
    if (true_node_index == -1) {
        PyErr_SetString(PyExc_KeyError, "node not found");
        return NULL;
    }

    for (int i = 0; i < self->num_node_attrs; i++) {
        if (self->node_attrs[i * 3] == true_node_index && self->node_attrs[i * 3 + 1] == attr_index) {
            // borrowed ref so we have to increase its refcnt because we are returning it
            PyObject *r = PyList_GetItem(value_list, self->node_attrs[i * 3 + 2]);
            Py_INCREF(r);
            return r;
        }
    }
    PyErr_SetString(PyExc_KeyError, "attribute not set for this node");
    return NULL;
}

PyObject *Graph_setnodeattr(Graph *self, int index, PyObject *k, PyObject *v) {
    GraphDef *gt = (GraphDef *)self->graph_def;
    int true_node_index = -1;
    for (int i = 0; i < self->num_nodes; i++) {
        if (self->nodes[i] == index) {
            true_node_index = i;
            break;
        }
    }
    if (true_node_index == -1) {
        PyErr_SetString(PyExc_KeyError, "node not found");
        return NULL;
    }
    PyObject *node_values = PyDict_GetItem(gt->node_values, k);
    if (node_values == NULL) {
        PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
        return NULL;
    }
    if (v == NULL) {
        // this means we have to delete g.nodes[index][k]
        int attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, k));
        for (int i = 0; i < self->num_node_attrs; i++) {
            if (self->node_attrs[i * 3] == true_node_index && self->node_attrs[i * 3 + 1] == attr_index) {
                // found the attribute, remove it
                int *old_node_attrs = self->node_attrs;
                self->node_attrs = malloc((self->num_node_attrs - 1) * 3 * sizeof(int));
                memcpy(self->node_attrs, old_node_attrs, i * 3 * sizeof(int));
                memcpy(self->node_attrs + i * 3, old_node_attrs + (i + 1) * 3,
                       (self->num_node_attrs - i - 1) * 3 * sizeof(int));
                self->num_node_attrs--;
                free(old_node_attrs);
                Py_RETURN_NONE;
            }
        }
        PyErr_SetString(PyExc_KeyError, "trying to delete a key that's not set");
        return NULL;
    }
    Py_ssize_t value_idx = PySequence_Index(node_values, v);
    if (value_idx == -1) {
        PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
        return NULL;
    }
    int attr_index = PyLong_AsLong(PyDict_GetItem(gt->node_keypos, k));
    for (int i = 0; i < self->num_node_attrs; i++) {
        if (self->node_attrs[i * 3] == true_node_index && self->node_attrs[i * 3 + 1] == attr_index) {
            self->node_attrs[i * 3 + 2] = value_idx;
            Py_RETURN_NONE;
        }
    }
    // Could not find the attribute, add it
    int new_idx = self->num_node_attrs;
    self->num_node_attrs++;
    self->node_attrs = realloc(self->node_attrs, self->num_node_attrs * 3 * sizeof(int));
    self->node_attrs[new_idx * 3] = true_node_index;
    self->node_attrs[new_idx * 3 + 1] = attr_index;
    self->node_attrs[new_idx * 3 + 2] = value_idx;
    Py_RETURN_NONE;
}

PyObject *Graph_getedgeattr(Graph *self, int index, PyObject *k) {
    GraphDef *gt = (GraphDef *)self->graph_def;
    PyObject *value_list = PyDict_GetItem(gt->edge_values, k); // borrowed ref
    if (value_list == NULL) {
        PyErr_SetString(PyExc_KeyError, "key not found");
        return NULL;
    }
    long attr_index = PyLong_AsLong(PyDict_GetItem(gt->edge_keypos, k));
    if (index > self->num_edges) {
        // Should never happen, index is computed by us in EdgeView_getitem, not
        // by the user!
        PyErr_SetString(PyExc_KeyError, "edge not found [but this should never happen!]");
        return NULL;
    }

    for (int i = 0; i < self->num_edge_attrs; i++) {
        if (self->edge_attrs[i * 3] == index && self->edge_attrs[i * 3 + 1] == attr_index) {
            // borrowed ref so we have to increase its refcnt because we are returning it
            PyObject *r = PyList_GetItem(value_list, self->edge_attrs[i * 3 + 2]);
            Py_INCREF(r);
            return r;
        }
    }
    PyErr_SetString(PyExc_KeyError, "attribute not set for this node");
    return NULL;
}
PyObject *Graph_setedgeattr(Graph *self, int index, PyObject *k, PyObject *v) {
    GraphDef *gt = (GraphDef *)self->graph_def;
    PyObject *edge_values = PyDict_GetItem(gt->edge_values, k);
    if (edge_values == NULL) {
        PyErr_SetString(PyExc_KeyError, "key not found in GraphDef");
        return NULL;
    }
    if (v == 0) {
        // this means we have to delete g.edges[index][k]
        int attr_index = PyLong_AsLong(PyDict_GetItem(gt->edge_keypos, k));
        for (int i = 0; i < self->num_edge_attrs; i++) {
            if (self->edge_attrs[i * 3] == index && self->edge_attrs[i * 3 + 1] == attr_index) {
                // found the attribute, remove it
                int *old_edge_attrs = self->edge_attrs;
                self->edge_attrs = malloc((self->num_edge_attrs - 1) * 3 * sizeof(int));
                memcpy(self->edge_attrs, old_edge_attrs, i * 3 * sizeof(int));
                memcpy(self->edge_attrs + i * 3, old_edge_attrs + (i + 1) * 3,
                       (self->num_edge_attrs - i - 1) * 3 * sizeof(int));
                self->num_edge_attrs--;
                free(old_edge_attrs);
                Py_RETURN_NONE;
            }
        }
        PyErr_SetString(PyExc_KeyError, "trying to delete a key that's not set");
        return NULL;
    }
    Py_ssize_t value_idx = PySequence_Index(edge_values, v);
    if (value_idx == -1) {
        PyErr_SetString(PyExc_KeyError, "value not found in GraphDef");
        return NULL;
    }
    int attr_index = PyLong_AsLong(PyDict_GetItem(gt->edge_keypos, k));
    for (int i = 0; i < self->num_edge_attrs; i++) {
        if (self->edge_attrs[i * 3] == index && self->edge_attrs[i * 3 + 1] == attr_index) {
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
    Py_RETURN_NONE;
}

static PyMethodDef SpamMethods[] = {
    //{"print_count", print_count, METH_VARARGS, "Execute a shell command."},
    {"mol_graph_to_Data", mol_graph_to_Data, METH_VARARGS, "Convert a mol_graph to a Data object."},
    {"Data_collate", Data_collate, METH_VARARGS, "collate Data instances"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef spammodule = {PyModuleDef_HEAD_INIT, "gflownet._C", /* name of module */
                                        "doc",                       /* module documentation, may be NULL */
                                        -1,                          /* size of per-interpreter state of the module,
                                                                        or -1 if the module keeps state in global variables. */
                                        SpamMethods};
PyObject *SpamError;

PyMODINIT_FUNC PyInit__C(void) {
    PyObject *m;

    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    SpamError = PyErr_NewException("gflownet._C.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0) {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }
    PyTypeObject *types[] = {&GraphType, &GraphDefType, &NodeViewType, &EdgeViewType, &DegreeViewType, &DataType};
    char *names[] = {"Graph", "GraphDef", "NodeView", "EdgeView", "DegreeView", "Data"};
    for (int i = 0; i < (int)(sizeof(types) / sizeof(PyTypeObject *)); i++) {
        if (PyType_Ready(types[i]) < 0) {
            Py_DECREF(m);
            return NULL;
        }
        Py_XINCREF(types[i]);
        if (PyModule_AddObject(m, names[i], (PyObject *)types[i]) < 0) {
            Py_XDECREF(types[i]);
            Py_DECREF(m);
            return NULL;
        }
    }

    return m;
}
