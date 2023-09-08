static PyObject *SpamError;

typedef struct
{
    PyObject_HEAD;
    PyObject *node_values; /* Dict[str, List[Any]] */
    PyObject *edge_values; /* Dict[str, List[Any]] */
    PyObject *node_keypos; /* Dict[str, int] */
    PyObject *edge_keypos; /* Dict[str, int] */
} GraphDef;

extern PyTypeObject GraphDefType;

typedef struct
{
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
} Graph;

PyObject *Graph_getnodeattr(Graph *self, int index, PyObject *k);
PyObject *Graph_setnodeattr(Graph *self, int index, PyObject *k, PyObject *v);
PyObject *Graph_getedgeattr(Graph *self, int index, PyObject *k);
PyObject *Graph_setedgeattr(Graph *self, int index, PyObject *k, PyObject *v);

typedef struct
{
    PyObject_HEAD;
    Graph *graph;
    int index;
} NodeView;

extern PyTypeObject NodeViewType;

typedef struct
{
    PyObject_HEAD;
    Graph *graph;
    int index;
} EdgeView;

extern PyTypeObject EdgeViewType;

#define SELF_SET(v)              \
    {                            \
        PyObject *tmp = self->v; \
        Py_INCREF(v);            \
        self->v = v;             \
        Py_XDECREF(tmp);         \
    }