#define PY_SSIZE_T_CLEAN
#include <Python.h>

PyObject *SpamError;

typedef struct {
    PyObject_HEAD;
    PyObject *node_values;  /* Dict[str, List[Any]] */
    PyObject *edge_values;  /* Dict[str, List[Any]] */
    PyObject *node_keypos;  /* Dict[str, int] */
    PyObject *edge_keypos;  /* Dict[str, int] */
    PyObject *node_poskey;  /* List[str] */
    PyObject *edge_poskey;  /* List[str] */
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

typedef struct NodeView {
    PyObject_HEAD;
    Graph *graph;
    int index;
} NodeView;

extern PyTypeObject NodeViewType;

typedef struct EdgeView {
    PyObject_HEAD;
    Graph *graph;
    int index;
} EdgeView;

extern PyTypeObject EdgeViewType;

typedef struct DegreeView {
    PyObject_HEAD;
    Graph *graph;
} DegreeView;

extern PyTypeObject DegreeViewType;

/* This is a specialized Data instance that only knows how to hold onto 2d tensors
   that are either float32 or int64 (for reasons of compatibility with torch_geometric) */
typedef struct Data {
    PyObject_HEAD;
    PyObject *bytes;
    GraphDef *graph_def;
    int *shapes;
    int *is_float;
    int *offsets; // in bytes
    int num_matrices;
    const char **names;
} Data;

extern PyTypeObject DataType;

#define SELF_SET(v)                                                                                                    \
    {                                                                                                                  \
        PyObject *tmp = self->v;                                                                                       \
        Py_INCREF(v);                                                                                                  \
        self->v = v;                                                                                                   \
        Py_XDECREF(tmp);                                                                                               \
    }

// when we want to do f(x) where x is a new reference that we want to decref after the call
// e.g. PyLong_AsLong(PyObject_GetAttrString(x, "id")), GetAttrString returns a new reference
#define borrow_new_and_call(f, x)                                                                                      \
    ({                                                                                                                 \
        PyObject *tmp = x;                                                                                             \
        __auto_type rval = f(tmp);                                                                                     \
        Py_DECREF(tmp);                                                                                                \
        rval;                                                                                                          \
    })

static inline int maxi(int a, int b) { return a > b ? a : b; }
static inline int mini(int a, int b) { return a < b ? a : b; }

PyObject *Graph_getnodeattr(Graph *self, int index, PyObject *k);
PyObject *Graph_setnodeattr(Graph *self, int index, PyObject *k, PyObject *v);
PyObject *Graph_getedgeattr(Graph *self, int index, PyObject *k);
PyObject *Graph_setedgeattr(Graph *self, int index, PyObject *k, PyObject *v);
PyObject *Graph_bridges(PyObject *self, PyObject *args);

int get_edge_index(Graph *g, int u, int v);
int get_edge_index_from_pos(Graph *g, int u_pos, int v_pos);

PyObject *Data_collate(PyObject *self, PyObject *args);
void Data_init_C(Data *self, PyObject *bytes, PyObject *graph_def, int shapes[][2], int *is_float, int num_matrices,
                 const char **names);

PyObject *mol_graph_to_Data(PyObject *self, PyObject *args);