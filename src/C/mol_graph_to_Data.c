#include "main.h"
#include <signal.h>

void memsetf(float *ptr, float value, size_t num) {
    for (size_t i = 0; i < num; i++) {
        ptr[i] = value;
    }
}

#define RETURN_C_DATA 1
static const char *mol_Data_names[] = {"x",
                                       "edge_index",
                                       "edge_attr",
                                       "non_edge_index",
                                       "stop_mask",
                                       "add_node_mask",
                                       "set_node_attr_mask",
                                       "add_edge_mask",
                                       "set_edge_attr_mask",
                                       "remove_node_mask",
                                       "remove_node_attr_mask",
                                       "remove_edge_mask",
                                       "remove_edge_attr_mask",
                                       "cond_info"};

static PyObject *torch_module = NULL;
static PyObject *torch_gd_module = NULL;
void _check_torch() {
    if (torch_module == NULL) {
        torch_module = PyImport_ImportModule("torch");
        torch_gd_module = PyImport_ImportModule("torch_geometric.data");
    }
}

PyObject *mol_graph_to_Data(PyObject *self, PyObject *args) {
    PyObject *mol_graph = NULL, *ctx = NULL, *_torch_module = NULL, *cond_info = NULL;
    if (PyArg_ParseTuple(args, "OOO|O", &mol_graph, &ctx, &_torch_module, &cond_info) == 0) {
        return NULL;
    }
    _check_torch();
    if (PyObject_TypeCheck(mol_graph, &GraphType) == 0) {
        PyErr_SetString(PyExc_TypeError, "mol_graph must be a Graph");
        return NULL;
    }
    if (cond_info == Py_None) {
        cond_info = NULL;
    }
    Graph *g = (Graph *)mol_graph;
    GraphDef *gd = (GraphDef *)g->graph_def;
    PyObject *atom_types = PyDict_GetItemString(gd->node_values, "v"); // borrowed ref
    int num_atom_types = PySequence_Size(atom_types);
    int atom_valences[num_atom_types];
    float used_valences[g->num_nodes];
    int max_valence[g->num_nodes];
    PyObject *_max_atom_valence = PyObject_GetAttrString(ctx, "_max_atom_valence"); // new ref
    for (int i = 0; i < num_atom_types; i++) {
        atom_valences[i] = PyLong_AsLong(PyDict_GetItem(_max_atom_valence, PyList_GetItem(atom_types, i))); // borrowed
    }
    Py_DECREF(_max_atom_valence);
    int v_val[g->num_nodes];
    int v_idx = PyLong_AsLong(PyDict_GetItemString(gd->node_keypos, "v"));
    int charge_val[g->num_nodes];
    int charge_idx = PyLong_AsLong(PyDict_GetItemString(gd->node_keypos, "charge"));
    int explH_val[g->num_nodes];
    int explH_idx = PyLong_AsLong(PyDict_GetItemString(gd->node_keypos, "expl_H"));
    int chi_val[g->num_nodes];
    int chi_idx = PyLong_AsLong(PyDict_GetItemString(gd->node_keypos, "chi"));
    int noImpl_val[g->num_nodes];
    int noImpl_idx = PyLong_AsLong(PyDict_GetItemString(gd->node_keypos, "no_impl"));
    int num_set_attribs[g->num_nodes];
    PyObject *N_str = PyUnicode_FromString("N");
    Py_ssize_t nitro_attr_value = PySequence_Index(PyDict_GetItemString(gd->node_values, "v"), N_str);
    Py_DECREF(N_str);
    for (int i = 0; i < g->num_nodes; i++) {
        used_valences[i] = max_valence[i] = v_val[i] = charge_val[i] = explH_val[i] = chi_val[i] = noImpl_val[i] =
            num_set_attribs[i] = 0;
    }
    for (int i = 0; i < g->num_node_attrs; i++) {
        int node_pos = g->node_attrs[3 * i];
        int attr_type = g->node_attrs[3 * i + 1];
        int attr_value = g->node_attrs[3 * i + 2];
        num_set_attribs[node_pos] += 1;
        if (attr_type == v_idx) {
            v_val[node_pos] = attr_value;
            max_valence[node_pos] += atom_valences[v_val[node_pos]];
        } else if (attr_type == charge_idx) {
            charge_val[node_pos] = attr_value;
            max_valence[node_pos] -= 1; // If we change the possible charge ranges from [0,1,-1] this won't work
        } else if (attr_type == explH_idx) {
            explH_val[node_pos] = attr_value;
            max_valence[node_pos] -= attr_value;
        } else if (attr_type == chi_idx) {
            chi_val[node_pos] = attr_value;
        } else if (attr_type == noImpl_idx) {
            noImpl_val[node_pos] = attr_value;
        }
    }

    // bonds are the only edge attributes
    char has_connecting_edge_attr_set[g->num_nodes];
    memset(has_connecting_edge_attr_set, 0, g->num_nodes);
    float bond_valence[] = {1, 2, 3, 1.5}; // single, double, triple, aromatic
    int bond_val[g->num_edges];
    memset(bond_val, 0, g->num_edges * sizeof(int));
    for (int i = 0; i < g->num_edge_attrs; i++) {
        int edge_pos = g->edge_attrs[3 * i];
        int attr_type = g->edge_attrs[3 * i + 1];
        int attr_value = g->edge_attrs[3 * i + 2];
        if (attr_type == 0) { // this should always be true, but whatever
            bond_val[edge_pos] = attr_value;
        }
        has_connecting_edge_attr_set[g->edges[2 * edge_pos]] = 1;
        has_connecting_edge_attr_set[g->edges[2 * edge_pos + 1]] = 1;
    }
    for (int i = 0; i < g->num_edges; i++) {
        int u = g->edges[2 * i];
        int v = g->edges[2 * i + 1];
        used_valences[u] += bond_valence[bond_val[i]];
        used_valences[v] += bond_valence[bond_val[i]];
    }
    // Correct for the valence of charge Nitro atoms
    for (int i = 0; i < g->num_nodes; i++) {
        if (v_val[i] == nitro_attr_value && charge_val[i] == 1) {
            max_valence[i] += 2;
        }
    }
    // Compute creatable edges
    char can_create_edge[g->num_nodes][g->num_nodes];
    for (int i = 0; i < g->num_nodes; i++) {
        for (int j = 0; j < g->num_nodes; j++) {
            can_create_edge[i][j] =
                (used_valences[i] + 1 <= max_valence[i]) && (used_valences[j] + 1 <= max_valence[j]);
        }
    }
    for (int i = 0; i < g->num_edges; i++) {
        int u = g->edges[2 * i];
        int v = g->edges[2 * i + 1];
        can_create_edge[u][v] = can_create_edge[v][u] = 0;
    }
    int num_creatable_edges = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        for (int j = i + 1; j < g->num_nodes; j++) {
            num_creatable_edges += can_create_edge[i][j];
        }
    }
    PyObject *max_nodes_py = PyObject_GetAttrString(ctx, "max_nodes"); // new ref
    int max_nodes = max_nodes_py == Py_None ? 1000 : PyLong_AsLong(max_nodes_py);
    Py_DECREF(max_nodes_py);

    PyObject *cond_info_shape = cond_info == NULL ? NULL : PyObject_CallMethod(cond_info, "size", NULL);
    int num_cond_info_dims = cond_info == NULL ? 0 : PyLong_AsLong(PyTuple_GetItem(cond_info_shape, 1));
    Py_XDECREF(cond_info_shape);
    int node_feat_shape = maxi(1, g->num_nodes);
    int is_float[] = {1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int shapes[14][2] = {
        {node_feat_shape, gd->num_node_dim},            // node_feat
        {2, g->num_edges * 2},                          // edge_index
        {2 * g->num_edges, gd->num_edge_dim},           // edge_feat
        {2, num_creatable_edges},                       // non_edge_index
        {1, 1},                                         // stop_mask
        {node_feat_shape, gd->num_new_node_values},     // add_node_mask
        {node_feat_shape, gd->num_node_attr_logits},    // set_node_attr_mask
        {num_creatable_edges, 1},                       // add_edge_mask
        {g->num_edges, gd->num_edge_attr_logits},       // set_edge_attr_mask
        {node_feat_shape, 1},                           // remove_node_mask
        {node_feat_shape, gd->num_settable_node_attrs}, // remove_node_attr_mask
        {g->num_edges, 1},                              // remove_edge_mask
        {g->num_edges, gd->num_settable_edge_attrs},    // remove_edge_attr_mask
        {1, num_cond_info_dims},                        // cond_info
    };
    int offsets[14];
    Py_ssize_t num_items = 0;
    for (int i = 0; i < 14; i++) {
        offsets[i] = num_items;
        // we need twice the space for longs
        num_items += shapes[i][0] * shapes[i][1] * (2 - is_float[i]);
    }
    // Allocate the memory for the Data object in a way we can return it to Python
    // (and let Python free it when it's done)
    PyObject *data = PyByteArray_FromStringAndSize(NULL, num_items * sizeof(float));
    int *dataptr = (int *)PyByteArray_AsString(data);
    memset(dataptr, 0, num_items * sizeof(float));
    if (PyErr_Occurred()) {
        raise(SIGINT);
        return NULL;
    }

    float *node_feat = (float *)(dataptr + offsets[0]);
    long *edge_index = (long *)(dataptr + offsets[1]);
    float *edge_feat = (float *)(dataptr + offsets[2]);
    long *non_edge_index = (long *)(dataptr + offsets[3]);
    float *stop_mask = (float *)(dataptr + offsets[4]);
    float *add_node_mask = (float *)(dataptr + offsets[5]);
    float *set_node_attr_mask = (float *)(dataptr + offsets[6]);
    float *add_edge_mask = (float *)(dataptr + offsets[7]);
    float *set_edge_attr_mask = (float *)(dataptr + offsets[8]);
    float *remove_node_mask = (float *)(dataptr + offsets[9]);
    float *remove_node_attr_mask = (float *)(dataptr + offsets[10]);
    float *remove_edge_mask = (float *)(dataptr + offsets[11]);
    float *remove_edge_attr_mask = (float *)(dataptr + offsets[12]);
    float *cond_info_data = (float *)(dataptr + offsets[13]);

    int bridges[g->num_edges];
    // Graph_brigdes' second argument is expected to be NULL when called by Python
    // we're using it to pass the bridges array instead
    Graph_bridges((PyObject *)g, (PyObject *)bridges);
    if (PyErr_Occurred()) {
        raise(SIGINT);
        return NULL;
    }

    // sorted attrs is 'charge', 'chi', 'expl_H', 'no_impl', 'v'
    int *_node_attrs[5] = {charge_val, chi_val, explH_val, noImpl_val, v_val};
    int *_edge_attrs[1] = {bond_val};
    if (g->num_nodes == 0) {
        node_feat[gd->num_node_dim - 1] = 1;
        memsetf(add_node_mask, 1, gd->num_new_node_values);
        remove_node_mask[0] = 1;
    }
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->degrees[i] <= 1 && !has_connecting_edge_attr_set[i] && num_set_attribs[i] == 1) {
            remove_node_mask[i] = 1;
        }
        int is_nitro = v_val[i] == nitro_attr_value;
        for (int j = 0; j < 5; j++) {
            int one_hot_idx = gd->node_attr_offsets[j] + _node_attrs[j][i];
            node_feat[i * gd->num_node_dim + one_hot_idx] = 1;
            int logit_slice_start = gd->node_attr_offsets[j] - j;
            int logit_slice_end = gd->node_attr_offsets[j + 1] - j - 1;
            if (j == v_idx)
                continue; // we cannot remove nor set 'v'
            if (_node_attrs[j][i] > 0) {
                // Special case: if the node is a positively charged Nitrogen, and the valence is maxed out (i.e. 5)
                // the agent cannot remove the charge, it has to remove the bonds (or bond attrs) making this valence
                // maxed out first.
                if (j == charge_idx && is_nitro && used_valences[i] >= max_valence[i] && charge_val[i] == 1)
                    continue;
                remove_node_attr_mask[i * gd->num_settable_node_attrs + j] = 1;
            } else {
                // Special case for N: adding charge to N increases its valence by 2, so we don't want to prevent that
                // action, even if the max_valence is "full" (for v=3)
                if (j == charge_idx && used_valences[i] >= (max_valence[i] + (is_nitro ? 2 : 0))) // charge
                    continue;
                if (j == explH_idx && used_valences[i] >= max_valence[i]) // expl_H
                    continue;
                // Following on the special case, we made it here, now if the node is not yet charged but already
                // has a valence of 3, the only charge we can add is a +1, which is the 0th logit
                // (again this assumes charge ranges are [0,1,-1])
                if (j == charge_idx && is_nitro && used_valences[i] >= max_valence[i] && charge_val[i] == 0){
                    set_node_attr_mask[i * gd->num_node_attr_logits + logit_slice_start] = 1;
                    continue;
                }
                memsetf(set_node_attr_mask + i * gd->num_node_attr_logits + logit_slice_start, 1,
                        (logit_slice_end - logit_slice_start));
            }
        }
        if (used_valences[i] < max_valence[i] && g->num_nodes < max_nodes) {
            memsetf(add_node_mask + i * gd->num_new_node_values, 1, gd->num_new_node_values);
        }
    }
    if (PyErr_Occurred()) {
        raise(SIGINT);
        return NULL;
    }
    for (int i = 0; i < g->num_edges; i++) {
        if (bridges[i] == 0) {
            remove_edge_mask[i] = 1;
        }
        int j = 0; // well there's only the bond attr
        int one_hot_idx = gd->edge_attr_offsets[j] + _edge_attrs[j][i];
        edge_feat[2 * i * gd->num_edge_dim + one_hot_idx] = 1;
        edge_feat[(2 * i + 1) * gd->num_edge_dim + one_hot_idx] = 1;
        int logit_slice_start = gd->edge_attr_offsets[j] - j;
        if (_edge_attrs[j][i] > 0) {
            remove_edge_attr_mask[i * gd->num_settable_edge_attrs + j] = 1;
        } else {
            // TODO: we're not using aromatics here, instead single-double-triple is hardcoded
            // k starts at 1 because the default value (0) is the single bond
            for (int k = 1; k < 3; k++) {
                // bond_valence - 1 because the 0th value is the single bond which has valence 1, and we'd be "removing"
                // it to replace it with another bond
                if ((used_valences[g->edges[2 * i]] + bond_valence[k] - 1 > max_valence[g->edges[2 * i]]) ||
                    (used_valences[g->edges[2 * i + 1]] + bond_valence[k] - 1 > max_valence[g->edges[2 * i + 1]]))
                    continue;

                // use k - 1 because the default value (0) doesn't have an associated logit
                set_edge_attr_mask[i * gd->num_edge_attr_logits + logit_slice_start + k - 1] = 1;
            }
        }
        edge_index[2 * i] = g->edges[2 * i];
        edge_index[2 * i + 1] = g->edges[2 * i + 1];
        edge_index[2 * i + g->num_edges * 2] = g->edges[2 * i + 1];
        edge_index[2 * i + g->num_edges * 2 + 1] = g->edges[2 * i];
    }
    // already filtered out for valence and such
    memsetf(add_edge_mask, 1, num_creatable_edges);
    int non_edge_idx_idx = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        for (int j = i + 1; j < g->num_nodes; j++) {
            if (!can_create_edge[i][j])
                continue;
            non_edge_index[non_edge_idx_idx] = i;
            non_edge_index[num_creatable_edges + non_edge_idx_idx] = j;
            non_edge_idx_idx++;
        }
    }
    if (PyErr_Occurred()) {
        raise(SIGINT);
        return NULL;
    }
    if (cond_info != NULL) {
        PyObject *cpu_cond_info = PyObject_CallMethod(cond_info, "cpu", "");
        PyObject *cont_cond_info = PyObject_CallMethod(cpu_cond_info, "contiguous", "");
        PyObject *ptr = PyObject_CallMethod(cont_cond_info, "data_ptr", "");
        float *tensor_ptr = PyLong_AsVoidPtr(ptr);

        for (int i = 0; i < num_cond_info_dims; i++) {
            cond_info_data[i] = tensor_ptr[i];
        }
        Py_DECREF(ptr);
        Py_DECREF(cpu_cond_info);
        Py_DECREF(cont_cond_info);
    }

    if (PyErr_Occurred()) {
        PyErr_Print();
        raise(SIGINT);
        return NULL;
    }
    *stop_mask = g->num_nodes > 0 ? 1 : 0;
    if (RETURN_C_DATA) {
        // Data *data_obj = DataType.tp_new(&DataType, NULL, NULL);
        Data *data_obj = (Data *)PyObject_CallObject((PyObject *)&DataType, NULL);
        Data_init_C(data_obj, data, g->graph_def, shapes, is_float, 13 + (cond_info != NULL), mol_Data_names);
        Py_DECREF(data);
        return (PyObject *)data_obj;
    }
    // The following lines take about 80% of the runtime of the function on ~50 node graphs :"(
    PyObject *res = PyDict_New();
    PyObject *frombuffer = PyObject_GetAttrString(torch_module, "frombuffer");
    PyObject *empty = PyObject_GetAttrString(torch_module, "empty");
    PyObject *dtype_f32 = PyObject_GetAttrString(torch_module, "float32");
    PyObject *dtype_i64 = PyObject_GetAttrString(torch_module, "int64");
    PyObject *fb_args = PyTuple_Pack(1, data);
    PyObject *fb_kwargs = PyDict_New();
    int do_del_kw = 0;
    for (int i = 0; i < 13; i++) {
        int i_num_items = shapes[i][0] * shapes[i][1];
        PyObject *tensor;
        PyDict_SetItemString(fb_kwargs, "dtype", is_float[i] ? dtype_f32 : dtype_i64);
        if (i_num_items == 0) {
            if (do_del_kw) {
                PyDict_DelItemString(fb_kwargs, "offset");
                PyDict_DelItemString(fb_kwargs, "count");
                do_del_kw = 0;
            }
            PyObject *zero = PyLong_FromLong(0);
            tensor = PyObject_Call(empty, PyTuple_Pack(1, zero), fb_kwargs);
            Py_DECREF(zero);
        } else {
            PyObject *py_offset = PyLong_FromLong(offsets[i] * sizeof(float));
            PyObject *py_count = PyLong_FromLong(i_num_items);
            PyDict_SetItemString(fb_kwargs, "offset", py_offset);
            PyDict_SetItemString(fb_kwargs, "count", py_count);
            do_del_kw = 1;
            tensor = PyObject_Call(frombuffer, fb_args, fb_kwargs);
            Py_DECREF(py_offset);
            Py_DECREF(py_count);
        }
        PyObject *reshaped_tensor = PyObject_CallMethod(tensor, "view", "ii", shapes[i][0], shapes[i][1]);
        PyDict_SetItemString(res, mol_Data_names[i], reshaped_tensor);
        Py_DECREF(tensor);
        Py_DECREF(reshaped_tensor);
    }
    Py_DECREF(frombuffer);
    Py_DECREF(dtype_f32);
    Py_DECREF(dtype_i64);
    Py_DECREF(fb_args);
    Py_DECREF(fb_kwargs);
    Py_DECREF(data);
    return res;
}
