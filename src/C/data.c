#define PY_SSIZE_T_CLEAN
#include "main.h"
#include "structmember.h"
#include <Python.h>

static void Data_dealloc(Data *self) {
    Py_XDECREF(self->bytes);
    Py_XDECREF(self->graph_def);
    free(self->shapes);
    free(self->is_float);
    free(self->offsets);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *torch_module = NULL;
static PyObject *torch_gd_module = NULL;
static void _check_torch() {
    if (torch_module == NULL) {
        torch_module = PyImport_ImportModule("torch");
        torch_gd_module = PyImport_ImportModule("torch_geometric.data");
    }
}

static PyObject *gbe_module = NULL;
static PyObject *Stop, *AddNode, *SetNodeAttr, *AddEdge, *SetEdgeAttr, *RemoveNode, *RemoveNodeAttr, *RemoveEdge,
    *RemoveEdgeAttr, *GraphAction, *GraphActionType;
void _check_gbe() {
    if (gbe_module == NULL) {
        gbe_module = PyImport_ImportModule("gflownet.envs.graph_building_env");
        if (gbe_module == NULL) {
            PyErr_SetString(PyExc_ImportError, "Could not import gflownet.envs.graph_building_env");
            return;
        }
        GraphActionType = PyObject_GetAttrString(gbe_module, "GraphActionType");
        GraphAction = PyObject_GetAttrString(gbe_module, "GraphAction");
        Stop = PyObject_GetAttrString(GraphActionType, "Stop");
        AddNode = PyObject_GetAttrString(GraphActionType, "AddNode");
        SetNodeAttr = PyObject_GetAttrString(GraphActionType, "SetNodeAttr");
        AddEdge = PyObject_GetAttrString(GraphActionType, "AddEdge");
        SetEdgeAttr = PyObject_GetAttrString(GraphActionType, "SetEdgeAttr");
        RemoveNode = PyObject_GetAttrString(GraphActionType, "RemoveNode");
        RemoveNodeAttr = PyObject_GetAttrString(GraphActionType, "RemoveNodeAttr");
        RemoveEdge = PyObject_GetAttrString(GraphActionType, "RemoveEdge");
        RemoveEdgeAttr = PyObject_GetAttrString(GraphActionType, "RemoveEdgeAttr");
    }
}

static PyObject *Data_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Data *self;
    self = (Data *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->bytes = NULL;
        self->graph_def = NULL;
        self->shapes = NULL;
        self->is_float = NULL;
        self->num_matrices = 0;
        self->names = NULL;
        self->offsets = NULL;
    }
    return (PyObject *)self;
}

static int Data_init(Data *self, PyObject *args, PyObject *kwds) {
    if (args != NULL && 0) {
        PyErr_SetString(PyExc_KeyError, "Trying to initialize a Data object from Python. Use *_graph_to_Data instead.");
        return -1;
    }
    return 0;
}

void *Data_ptr_and_shape(Data *self, char *name, int *n, int *m) {
    for (int i = 0; i < self->num_matrices; i++) {
        if (strcmp(self->names[i], name) == 0) {
            *n = self->shapes[i * 2 + 0];
            *m = self->shapes[i * 2 + 1];
            return PyByteArray_AsString(self->bytes) + self->offsets[i];
        }
    }
    return (void *)0xdeadbeef;
}

void Data_init_C(Data *self, PyObject *bytes, PyObject *graph_def, int shapes[][2], int *is_float, int num_matrices,
                 const char **names) {
    PyObject *tmp;
    tmp = (PyObject *)self->bytes;
    Py_INCREF(bytes);
    self->bytes = bytes;
    Py_XDECREF(tmp);

    tmp = (PyObject *)self->graph_def;
    Py_INCREF(graph_def);
    self->graph_def = (GraphDef *)graph_def;
    Py_XDECREF(tmp);

    self->shapes = malloc(sizeof(int) * num_matrices * 2);
    self->offsets = malloc(sizeof(int) * num_matrices);
    int offset_bytes = 0;
    for (int i = 0; i < num_matrices; i++) {
        self->shapes[i * 2 + 0] = shapes[i][0];
        self->shapes[i * 2 + 1] = shapes[i][1];
        self->offsets[i] = offset_bytes;
        offset_bytes += shapes[i][0] * shapes[i][1] * (is_float[i] ? 4 : 8);
    }
    self->is_float = malloc(sizeof(int) * num_matrices);
    memcpy(self->is_float, is_float, sizeof(int) * num_matrices);
    self->num_matrices = num_matrices;
    self->names = names;
}

PyObject *Data_mol_aidx_to_GraphAction(PyObject *_self, PyObject *args) {
    _check_gbe();
    int _t, row, col; // _t is unused, we have gt from python
    PyObject *gt = NULL;
    if (!PyArg_ParseTuple(args, "(iii)O", &_t, &row, &col, &gt)) {
        return NULL;
    }
    Data *self = (Data *)_self;
    // These are singletons, so we can just compare pointers!
    if (gt == Stop) {
        PyObject *args = PyTuple_Pack(1, Stop);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(args);
        return res;
    }
    if (gt == AddNode) {
        // return GraphAction(t, source=act_row, value=self.atom_attr_values["v"][act_col])
        PyObject *source = PyLong_FromLong(row);                                  // ref is new
        PyObject *vals = PyDict_GetItemString(self->graph_def->node_values, "v"); // ref is borrowed
        PyObject *value = PyList_GetItem(vals, col);                              // ref is borrowed
        PyObject *args = PyTuple_Pack(5, gt, source, Py_None, value, Py_None);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(source);
        Py_DECREF(args);
        return res;
    }
    if (gt == SetNodeAttr) {
        // attr, val = self.atom_attr_logit_map[act_col]
        // return GraphAction(t, source=act_row, attr=attr, value=val)
        PyObject *attr, *val;
        for (int i = 0; i < self->graph_def->num_settable_node_attrs; i++) {
            int start = self->graph_def->node_attr_offsets[i] - i; // - i because these are logits
            int end = self->graph_def->node_attr_offsets[i + 1] - (i + 1);
            if (start <= col && col < end) {
                attr = PyList_GetItem(self->graph_def->node_poskey, i);
                PyObject *vals = PyDict_GetItem(self->graph_def->node_values, attr);
                val = PyList_GetItem(vals, col - start + 1); // + 1 because the default is 0
                break;
            }
        }
        PyObject *source = PyLong_FromLong(row); // ref is new
        PyObject *args = PyTuple_Pack(5, gt, source, Py_None, val, attr);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(source);
        Py_DECREF(args);
        return res;
    }
    if (gt == AddEdge) {
        int n, m;
        long *non_edge_index = Data_ptr_and_shape(self, "non_edge_index", &n, &m);
        PyObject *u = PyLong_FromLong(non_edge_index[row]);
        PyObject *v = PyLong_FromLong(non_edge_index[row + m]);
        PyObject *args = PyTuple_Pack(3, AddEdge, u, v);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(u);
        Py_DECREF(v);
        Py_DECREF(args);
        return res;
    }
    if (gt == SetEdgeAttr) {
        // attr, val = self.bond_attr_logit_map[act_col]
        // return GraphAction(t, source=act_row, attr=attr, value=val)
        int n, m;
        long *edge_index = Data_ptr_and_shape(self, "edge_index", &n, &m);
        // edge_index should be (2, m), * 2 because edges are duplicated
        PyObject *u = PyLong_FromLong(edge_index[row * 2]);
        PyObject *v = PyLong_FromLong(edge_index[row * 2 + m]);
        PyObject *attr, *val = NULL;
        for (int i = 0; i < self->graph_def->num_settable_edge_attrs; i++) {
            int start = self->graph_def->edge_attr_offsets[i] - i; // - i because these are logits
            int end = self->graph_def->edge_attr_offsets[i + 1] - (i + 1);
            if (start <= col && col < end) {
                attr = PyList_GetItem(self->graph_def->edge_poskey, i);
                PyObject *vals = PyDict_GetItem(self->graph_def->edge_values, attr);
                val = PyList_GetItem(vals, col - start + 1); // + 1 because the default is 0
                break;
            }
        }
        if (val == NULL) {
            PyErr_SetString(PyExc_ValueError, "failed to find edge attr");
            return NULL;
        }
        PyObject *args = PyTuple_Pack(5, gt, u, v, val, attr);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(u);
        Py_DECREF(v);
        Py_DECREF(args);
        return res;
    }
    if (gt == RemoveNode) {
        PyObject *source = PyLong_FromLong(row); // ref is new
        PyObject *args = PyTuple_Pack(2, RemoveNode, source);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(source);
        Py_DECREF(args);
        return res;
    }
    if (gt == RemoveNodeAttr) {
        PyObject *source = PyLong_FromLong(row);                            // ref is new
        PyObject *attr = PyList_GetItem(self->graph_def->node_poskey, col); // this works because 'v' is last
        PyObject *args = PyTuple_Pack(5, RemoveNodeAttr, source, Py_None, Py_None, attr);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(source);
        Py_DECREF(args);
        return res;
    }
    if (gt == RemoveEdge) {
        int n, m;
        long *edge_index = Data_ptr_and_shape(self, "edge_index", &n, &m);
        // edge_index should be (2, m), * 2 because edges are duplicated
        PyObject *u = PyLong_FromLong(edge_index[row * 2 + 0]);
        PyObject *v = PyLong_FromLong(edge_index[row * 2 + m]);
        PyObject *vargs = PyTuple_Pack(3, RemoveEdge, u, v);
        PyObject *res = PyObject_CallObject(GraphAction, vargs);
        Py_DECREF(u);
        Py_DECREF(v);
        Py_DECREF(vargs);
        return res;
    }
    if (gt == RemoveEdgeAttr) {
        int n, m;
        long *edge_index = Data_ptr_and_shape(self, "edge_index", &n, &m);
        // edge_index should be (2, m), * 2 because edges are duplicated
        PyObject *u = PyLong_FromLong(edge_index[row * 2 + 0]);
        PyObject *v = PyLong_FromLong(edge_index[row * 2 + m]);
        PyObject *attr = PyList_GetItem(self->graph_def->edge_poskey, col);
        PyObject *args = PyTuple_Pack(5, RemoveEdgeAttr, u, v, Py_None, attr);
        PyObject *res = PyObject_CallObject(GraphAction, args);
        Py_DECREF(u);
        Py_DECREF(v);
        Py_DECREF(args);
        return res;
    }
    PyErr_SetString(PyExc_ValueError, "Unknown action type");
    return NULL;
}

PyObject *Data_mol_GraphAction_to_aidx(PyObject *_self, PyObject *args) {
    _check_gbe();

    PyObject *action = NULL;
    if (!PyArg_ParseTuple(args, "O", &action)) {
        return NULL;
    }
    Data *self = (Data *)_self;
    PyObject *action_type = PyObject_GetAttrString(action, "action"); // new ref
    // These are singletons, so we can just compare pointers!
    long row = 0, col = 0;
    if (action_type == Stop) {
        // return (0, 0)
    } else if (action_type == AddNode) {
        row = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        PyObject *val = PyObject_GetAttrString(action, "value");
        PyObject *vals = PyDict_GetItemString(self->graph_def->node_values, "v");
        col = PySequence_Index(vals, val);
        Py_DECREF(val);
    } else if (action_type == SetNodeAttr) {
        row = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        PyObject *attr = PyObject_GetAttrString(action, "attr");
        PyObject *val = PyObject_GetAttrString(action, "value");
        PyObject *vals = PyDict_GetItem(self->graph_def->node_values, attr);
        col = PySequence_Index(vals, val) - 1; // -1 because the default value is at index 0
        int attr_pos = PyLong_AsLong(PyDict_GetItem(self->graph_def->node_keypos, attr));
        col += self->graph_def->node_attr_offsets[attr_pos] - attr_pos;
        Py_DECREF(attr);
        Py_DECREF(val);
    } else if (action_type == AddEdge) {
        col = 0;
        int u = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        int v = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "target"));
        int n, m;
        long *non_edge_index = Data_ptr_and_shape(self, "non_edge_index", &n, &m);
        for (int i = 0; i < m; i++) {
            if ((non_edge_index[i] == u && non_edge_index[i + m] == v) ||
                (non_edge_index[i] == v && non_edge_index[i + m] == u)) {
                row = i;
                break;
            }
        }
    } else if (action_type == SetEdgeAttr) {
        int u = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        int v = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "target"));
        int n, m;
        long *edge_index = Data_ptr_and_shape(self, "edge_index", &n, &m);
        for (int i = 0; i < m; i++) {
            if ((edge_index[i] == u && edge_index[i + m] == v) || (edge_index[i] == v && edge_index[i + m] == u)) {
                row = i / 2; // edges are duplicated
                break;
            }
        }
        PyObject *attr = PyObject_GetAttrString(action, "attr");
        PyObject *val = PyObject_GetAttrString(action, "value");
        PyObject *vals = PyDict_GetItem(self->graph_def->edge_values, attr);
        col = PySequence_Index(vals, val) - 1; // -1 because the default value is at index 0
        int attr_pos = PyLong_AsLong(PyDict_GetItem(self->graph_def->edge_keypos, attr));
        col += self->graph_def->edge_attr_offsets[attr_pos] - attr_pos;
        Py_DECREF(attr);
        Py_DECREF(val);
    } else if (action_type == RemoveNode) {
        row = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        col = 0;
    } else if (action_type == RemoveNodeAttr) {
        row = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        PyObject *attr = PyObject_GetAttrString(action, "attr");
        col = PyLong_AsLong(PyDict_GetItem(self->graph_def->node_keypos, attr));
    } else if (action_type == RemoveEdge) {
        col = 0;
        int u = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        int v = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "target"));
        int n, m;
        long *edge_index = Data_ptr_and_shape(self, "edge_index", &n, &m);
        for (int i = 0; i < m; i++) {
            if ((edge_index[i] == u && edge_index[i + m] == v) || (edge_index[i] == v && edge_index[i + m] == u)) {
                row = i / 2; // edges are duplicated
                break;
            }
        }
    } else if (action_type == RemoveEdgeAttr) {
        int u = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "source"));
        int v = borrow_new_and_call(PyLong_AsLong, PyObject_GetAttrString(action, "target"));
        int n, m;
        long *edge_index = Data_ptr_and_shape(self, "edge_index", &n, &m);
        for (int i = 0; i < m; i++) {
            if ((edge_index[i] == u && edge_index[i + m] == v) || (edge_index[i] == v && edge_index[i + m] == u)) {
                row = i / 2; // edges are duplicated
                break;
            }
        }
        PyObject *attr = PyObject_GetAttrString(action, "attr");
        col = PyLong_AsLong(PyDict_GetItem(self->graph_def->edge_keypos, attr));
    } else {
        PyErr_SetString(PyExc_ValueError, "Unknown action type");
        return NULL;
    }
    PyObject *py_row = PyLong_FromLong(row);
    PyObject *py_col = PyLong_FromLong(col);
    PyObject *res = PyTuple_Pack(2, py_row, py_col);
    Py_DECREF(py_row);
    Py_DECREF(py_col);
    Py_DECREF(action_type);
    return res;
}

PyObject *Data_as_torch(PyObject *_self, PyObject *unused_args) {
    _check_torch();
    Data *self = (Data *)_self;
    PyObject *res = PyDict_New();
    PyObject *frombuffer = PyObject_GetAttrString(torch_module, "frombuffer");
    PyObject *empty = PyObject_GetAttrString(torch_module, "empty");
    PyObject *dtype_f32 = PyObject_GetAttrString(torch_module, "float32");
    PyObject *dtype_i64 = PyObject_GetAttrString(torch_module, "int64");
    PyObject *fb_args = PyTuple_Pack(1, self->bytes);
    PyObject *fb_kwargs = PyDict_New();
    int do_del_kw = 0;
    int offset = 0;
    for (int i = 0; i < self->num_matrices; i++) {
        int i_num_items = self->shapes[i * 2 + 0] * self->shapes[i * 2 + 1];
        PyObject *tensor;
        PyDict_SetItemString(fb_kwargs, "dtype", self->is_float[i] ? dtype_f32 : dtype_i64);
        if (i_num_items == 0) {
            if (do_del_kw) {
                PyDict_DelItemString(fb_kwargs, "offset");
                PyDict_DelItemString(fb_kwargs, "count");
                do_del_kw = 0;
            }
            PyObject *zero = PyLong_FromLong(0);
            PyObject *args = PyTuple_Pack(1, zero);
            tensor = PyObject_Call(empty, args, fb_kwargs);
            Py_DECREF(args);
            Py_DECREF(zero);
        } else {
            PyObject *py_offset = PyLong_FromLong(offset);
            PyObject *py_numi = PyLong_FromLong(i_num_items);
            PyDict_SetItemString(fb_kwargs, "offset", py_offset);
            PyDict_SetItemString(fb_kwargs, "count", py_numi);
            Py_DECREF(py_offset);
            Py_DECREF(py_numi);
            do_del_kw = 1;
            tensor = PyObject_Call(frombuffer, fb_args, fb_kwargs);
        }
        PyObject *reshaped_tensor =
            PyObject_CallMethod(tensor, "view", "ii", self->shapes[i * 2 + 0], self->shapes[i * 2 + 1]);
        PyDict_SetItemString(res, self->names[i], reshaped_tensor);
        Py_DECREF(tensor);
        Py_DECREF(reshaped_tensor);
        offset += i_num_items * (self->is_float[i] ? 4 : 8);
    }
    Py_DECREF(frombuffer);
    Py_DECREF(dtype_f32);
    Py_DECREF(dtype_i64);
    Py_DECREF(fb_args);
    Py_DECREF(fb_kwargs);

    PyObject *Data_cls = PyObject_GetAttrString(torch_gd_module, "Data"); // new ref
    PyObject *args = PyTuple_New(0);
    PyObject *Data_res = PyObject_Call(Data_cls, args, res);
    Py_DECREF(args);
    Py_DECREF(Data_cls);
    Py_DECREF(res);
    return Data_res;
}

static PyMemberDef Data_members[] = {
    {NULL} /* Sentinel */
};

static PyMethodDef Data_methods[] = {
    {"mol_aidx_to_GraphAction", (PyCFunction)Data_mol_aidx_to_GraphAction, METH_VARARGS, "mol_aidx_to_GraphAction"},
    {"mol_GraphAction_to_aidx", (PyCFunction)Data_mol_GraphAction_to_aidx, METH_VARARGS, "mol_GraphAction_to_aidx"},
    {"as_torch", (PyCFunction)Data_as_torch, METH_NOARGS, "to pyg data"},
    {NULL} /* Sentinel */
};

PyTypeObject DataType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "_C.Data",
    .tp_doc = PyDoc_STR("Constrained Data object"),
    .tp_basicsize = sizeof(Data),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Data_new,
    .tp_init = (initproc)Data_init,
    .tp_dealloc = (destructor)Data_dealloc,
    .tp_members = Data_members,
    .tp_methods = Data_methods,
};

#include <signal.h>

PyObject *Data_collate(PyObject *self, PyObject *args) {
    PyObject *graphs = NULL, *follow_batch = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &graphs, &follow_batch)) {
        return NULL;
    }
    if (!PyList_Check(graphs) || PyList_Size(graphs) < 1 || !PyObject_TypeCheck(PyList_GetItem(graphs, 0), &DataType)) {
        PyErr_SetString(PyExc_TypeError, "Data_collate expects a non-empty list of Data objects");
        return NULL;
    }
    _check_torch();
    PyObject *empty_cb = PyObject_GetAttrString(torch_module, "empty");
    PyObject *empty_cb_int64_arg = PyDict_New();
    PyDict_SetItemString(empty_cb_int64_arg, "dtype", PyObject_GetAttrString(torch_module, "int64"));
    int num_graphs = PyList_Size(graphs);
    // PyObject *batch = PyObject_CallMethod(torch_gd_module, "Batch", NULL);
    // Actually we want batch = Batch(_base_cls=graphs[0].__class__)
    PyObject *base_cls = PyObject_GetAttrString(PyList_GetItem(graphs, 0), "__class__");
    PyObject *batch = PyObject_CallMethod(torch_gd_module, "Batch", "O", base_cls);
    PyObject *slice_dict = PyDict_New();
    Data *first = (Data *)PyList_GetItem(graphs, 0);
    int index_of_x = 0;
    for (int i = 0; i < first->num_matrices; i++) {
        if (strcmp(first->names[i], "x") == 0) {
            index_of_x = i;
            break;
        }
    }
    for (int i = 0; i < first->num_matrices; i++) {
        PyObject *tensor = NULL;
        const char *name = first->names[i];
        int do_compute_batch = 0;
        if (strcmp(name, "x") == 0) {
            do_compute_batch = 1;
        } else if (follow_batch != NULL) {
            PyObject *py_name = PyUnicode_FromString(name);
            do_compute_batch = PySequence_Contains(follow_batch, py_name);
            Py_DECREF(py_name);
        }
        int item_size = (first->is_float[i] ? sizeof(float) : sizeof(long));
        // First let's count the total number of items in the batch
        int num_rows = 0;
        int cat_dim = 0, val_dim = 1;
        if (strstr(name, "index") != NULL) {
            cat_dim = 1;
            val_dim = 0;
        }
        int val_dim_size = first->shapes[i * 2 + val_dim];
        for (int j = 0; j < num_graphs; j++) {
            Data *data = (Data *)PyList_GetItem(graphs, j);
            // We're concatenating along the first dimension, so we should check that the second dimension is the same
            if (data->shapes[i * 2 + val_dim] != val_dim_size) {
                PyErr_Format(PyExc_TypeError,
                             "mol_Data_collate concatenates %s along dimension %d, but tensor has shape %d along "
                             "dimension %d",
                             name, cat_dim, data->shapes[i * 2 + val_dim], val_dim);
                return NULL;
            }
            num_rows += data->shapes[i * 2 + cat_dim];
        }
        // Now we allocate the tensor itself, its batch tensor & slices
        PyObject *py_num_rows = PyLong_FromLong(num_rows);
        PyObject *py_val_dim_size = PyLong_FromLong(val_dim_size);
        PyObject *py_shape = cat_dim == 0 ? PyTuple_Pack(2, py_num_rows, py_val_dim_size)
                                          : PyTuple_Pack(2, py_val_dim_size, py_num_rows);
        int tensor_shape_x = cat_dim == 0 ? num_rows : val_dim_size;
        int tensor_shape_y = cat_dim == 0 ? val_dim_size : num_rows;
        if (first->is_float[i]) {
            tensor = PyObject_Call(empty_cb, py_shape, NULL);
        } else {
            tensor = PyObject_Call(empty_cb, py_shape, empty_cb_int64_arg);
        }
        int _total_ni = val_dim_size * num_rows;
        Py_DECREF(py_shape);
        PyObject *batch_tensor = NULL;
        if (do_compute_batch) {
            py_shape = PyTuple_Pack(1, py_num_rows);
            batch_tensor = PyObject_Call(empty_cb, py_shape, empty_cb_int64_arg);
            Py_DECREF(py_shape);
        }
        PyObject *py_num_graphsp1 = PyLong_FromLong(num_graphs + 1);
        py_shape = PyTuple_Pack(1, py_num_graphsp1);
        PyObject *slice_tensor = PyObject_Call(empty_cb, py_shape, empty_cb_int64_arg);
        Py_DECREF(py_shape);
        Py_DECREF(py_num_rows);
        Py_DECREF(py_val_dim_size);
        Py_DECREF(py_num_graphsp1);
        PyObject *ptr = PyObject_CallMethod(tensor, "data_ptr", "");
        void *tensor_ptr = PyLong_AsVoidPtr(ptr);
        Py_DECREF(ptr);
        long *batch_tensor_ptr;
        if (do_compute_batch) {
            ptr = PyObject_CallMethod(batch_tensor, "data_ptr", "");
            batch_tensor_ptr = PyLong_AsVoidPtr(ptr);
            Py_DECREF(ptr);
        }
        ptr = PyObject_CallMethod(slice_tensor, "data_ptr", "");
        long *slice_ptr = PyLong_AsVoidPtr(ptr);
        Py_DECREF(ptr);
        // Now we copy the data from the individual Data objects to the batch
        int offset_bytes = 0;
        int offset_items = 0;
        int offset_rows = 0;
        slice_ptr[0] = 0;
        int value_increment = 0; // we need to increment edge indices across graphs
        int do_increment = strstr(name, "index") != NULL;
        for (int j = 0; j < num_graphs; j++) {
            Data *data = (Data *)PyList_GetItem(graphs, j); // borrowed ref
            int num_items_j = data->shapes[i * 2 + 0] * data->shapes[i * 2 + 1];
            if (cat_dim == 0) {
                // we're given a 2d matrix m, n and we want to fit it into a bigger matrix M, n
                void *dst = tensor_ptr;
                void *src = PyByteArray_AsString(data->bytes) + data->offsets[i];
                for (int u = 0; u < data->shapes[i * 2 + 0]; u++) {
                    for (int v = 0; v < data->shapes[i * 2 + 1]; v++) {
                        // dst[u + offset_rows, v] = src[u, v]
                        if (first->is_float[i]) {
                            ((float *)dst)[(u + offset_rows) * tensor_shape_y + v] =
                                ((float *)src)[u * data->shapes[i * 2 + 1] + v];
                        } else {
                            ((long *)dst)[(u + offset_rows) * tensor_shape_y + v] =
                                ((long *)src)[u * data->shapes[i * 2 + 1] + v] + value_increment;
                        }
                    }
                }
            } else {
                // we're given a 2d matrix m, n and we want to fit it into a bigger matrix m, N
                void *dst = tensor_ptr;
                void *src = PyByteArray_AsString(data->bytes) + data->offsets[i];
                for (int u = 0; u < data->shapes[i * 2 + 0]; u++) {
                    for (int v = 0; v < data->shapes[i * 2 + 1]; v++) {
                        // dst[u, v + offset_rows] = src[u, v]
                        if (first->is_float[i]) {
                            ((float *)dst)[u * tensor_shape_y + v + offset_rows] =
                                ((float *)src)[u * data->shapes[i * 2 + 1] + v];
                        } else {
                            ((long *)dst)[u * tensor_shape_y + v + offset_rows] =
                                ((long *)src)[u * data->shapes[i * 2 + 1] + v] + value_increment;
                        }
                    }
                }
            }
            if (do_compute_batch) {
                for (int k = 0; k < data->shapes[i * 2 + cat_dim]; k++) {
                    batch_tensor_ptr[k + offset_rows] = j;
                }
            }
            offset_rows += data->shapes[i * 2 + cat_dim];
            offset_items += num_items_j;
            offset_bytes += num_items_j * item_size;
            slice_ptr[j + 1] = offset_rows;
            if (do_increment) {
                value_increment += data->shapes[index_of_x * 2]; // increment by num_nodes
            }
        }
        PyObject_SetAttrString(batch, name, tensor);
        // for x, the batch is just 'batch', otherwise '%s_batch'
        if (strcmp(name, "x") == 0) {
            PyObject_SetAttrString(batch, "batch", batch_tensor);
        } else if (do_compute_batch) {
            char buf[100];
            sprintf(buf, "%s_batch", name);
            PyObject_SetAttrString(batch, buf, batch_tensor);
        }
        PyDict_SetItemString(slice_dict, name, slice_tensor);
        Py_DECREF(tensor);
        Py_XDECREF(batch_tensor);
        Py_DECREF(slice_tensor);
    }
    PyObject_SetAttrString(batch, "_slice_dict", slice_dict);
    Py_DECREF(empty_cb);
    Py_DECREF(empty_cb_int64_arg);
    Py_DECREF(slice_dict);

    return batch;
}
