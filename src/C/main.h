#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject *SpamError;

typedef struct {
    PyObject_HEAD;
    PyObject *node_values;  /* Dict[str, List[Any]] */
    PyObject *edge_values;  /* Dict[str, List[Any]] */
    PyObject *node_keypos;  /* Dict[str, int] */
    PyObject *edge_keypos;  /* Dict[str, int] */
    int *node_attr_offsets; /* List[int] */
    int *edge_attr_offsets; /* List[int] */
    int num_node_dim;
    int num_settable_node_attrs;
    int num_node_attr_logits;
    int num_new_node_values;
    int num_edge_dim;
    int num_settable_edge_attrs;
    int num_edge_attr_logits;
} GraphDef;

extern PyTypeObject GraphDefType;

typedef struct {
    PyObject_HEAD;
    PyObject *graph_def;
    int num_nodes;
    int num_edges;
    int num_node_attrs;
    int num_edge_attrs;
    int *nodes;      /* List[int] */
    int *edges;      /* List[Tuple[int, int]] */
    int *node_attrs; /* List[Tuple[nodeid, attrid, attrvalueidx]] */
    int *edge_attrs; /* List[Tuple[edgeid, attrid, attrvalueidx]] */
    int *degrees;    /* List[int] */
} Graph;

extern PyTypeObject GraphType;

PyObject *Graph_getnodeattr(Graph *self, int index, PyObject *k);
PyObject *Graph_setnodeattr(Graph *self, int index, PyObject *k, PyObject *v);
PyObject *Graph_getedgeattr(Graph *self, int index, PyObject *k);
PyObject *Graph_setedgeattr(Graph *self, int index, PyObject *k, PyObject *v);
PyObject *Graph_bridges(PyObject *self, PyObject *args);

int get_edge_index(Graph *g, int u, int v);
int get_edge_index_from_pos(Graph *g, int u_pos, int v_pos);

PyObject *mol_graph_to_Data(PyObject *self, PyObject *args);

typedef struct {
    PyObject_HEAD;
    Graph *graph;
    int index;
} NodeView;

extern PyTypeObject NodeViewType;

typedef struct {
    PyObject_HEAD;
    Graph *graph;
    int index;
} EdgeView;

extern PyTypeObject EdgeViewType;

#define SELF_SET(v)                                                                                                    \
    {                                                                                                                  \
        PyObject *tmp = self->v;                                                                                       \
        Py_INCREF(v);                                                                                                  \
        self->v = v;                                                                                                   \
        Py_XDECREF(tmp);                                                                                               \
    }

static inline int maxi(int a, int b) { return a > b ? a : b; }
static inline int mini(int a, int b) { return a < b ? a : b; }