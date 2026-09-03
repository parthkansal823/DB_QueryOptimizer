// A C++ port of `app/optimizer/plan_tree.py::encode_plan_tree`.
//
// Why this one function, and not more. Measured on a four-table query with a
// ten-node plan, per candidate:
//
//     encode_plan_tree   0.0205 ms   45%   <- recursive walk, ported here
//     rest of featurize  0.0252 ms   55%   <- per-table dict slots
//     ------------------------------------
//     featurize total    0.0457 ms         x18 candidates = 0.82 ms
//
// The recursive walk is the only part with a shape a compiled language is
// better at: it visits every node several times over in Python, allocating a
// list of (node, depth) tuples and then re-scanning it six times. The rest is
// dictionary manipulation, where C gains almost nothing because it pays the
// same PyDict lookups.
//
// The honest context, which `docs/WRITEUP.md` states and this file should not
// hide: 0.82 ms sits inside a ~26 ms optimizer overhead on a ~110 ms query.
// Even making this function free is about 3% of the overhead and under 1% of
// end-to-end time. This is built because it was asked for and because it is the
// one place where the answer is not obviously "no" -- not because the profile
// demanded it.
//
// Correctness is the whole risk. A feature vector that differs from Python's in
// the sixth decimal is a silent divergence between what the model was trained on
// and what it is served, which is the exact failure class this project keeps
// finding. `tests/test_plan_tree_native.py` asserts bit-identical output on
// every plan in the execution log, and the Python wrapper falls back rather than
// guessing if the module is absent.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

// Operator families, mirroring plan_tree.OPERATOR_FAMILIES exactly. The Python
// membership test is `node_type in prefixes` -- an exact match against the
// tuple, not a prefix match, despite the variable's name.
struct Family {
    const char* key;
    const char* members[3];
};

const Family FAMILIES[] = {
    {"n_seq_scan", {"Seq Scan", nullptr, nullptr}},
    {"n_index_scan", {"Index Scan", "Index Only Scan", nullptr}},
    {"n_bitmap_scan", {"Bitmap Heap Scan", "Bitmap Index Scan", nullptr}},
    {"n_sort", {"Sort", "Incremental Sort", nullptr}},
    {"n_hash", {"Hash", nullptr, nullptr}},
    {"n_aggregate", {"Aggregate", "GroupAggregate", "HashAggregate"}},
    {"n_materialize", {"Materialize", "Memoize", nullptr}},
    {"n_gather", {"Gather", "Gather Merge", nullptr}},
};
const int N_FAMILIES = sizeof(FAMILIES) / sizeof(FAMILIES[0]);

// Every key in TREE_FEATURES, so the returned dict has the same shape as
// Python's even when a plan contributes nothing to a slot.
const char* SCALAR_FEATURES[] = {
    "tree_depth",
    "tree_leaves",
    "tree_bushiness",
    "log_max_est_rows",
    "log_sum_est_rows",
    "log_max_intermediate_bytes",
    "max_child_rows_amplification",
    "log_total_cost",
    "startup_cost_fraction",
    "deepest_join_depth",
    "log_max_join_est_rows",
    "join_rows_reduction",
    "cost_concentration",
    "log_max_hash_build_rows",
    "n_parallel_aware",
    "weighted_nestloop_depth",
    "log_root_width",
};
const int N_SCALARS = sizeof(SCALAR_FEATURES) / sizeof(SCALAR_FEATURES[0]);

// `float(node.get(key) or 0)` -- note the `or`, which turns a present-but-zero,
// a None and a missing key all into 0.0. Reproducing that exactly matters: an
// implementation that treated missing as "skip" would change every sum.
double get_number(PyObject* node, const char* key) {
    PyObject* value = PyDict_GetItemString(node, key);  // borrowed
    if (value == nullptr || value == Py_None) return 0.0;
    double out = PyFloat_AsDouble(value);
    if (out == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0.0;
    }
    return out;
}

bool truthy(PyObject* node, const char* key) {
    PyObject* value = PyDict_GetItemString(node, key);  // borrowed
    return value != nullptr && PyObject_IsTrue(value) == 1;
}

// The child list, or nullptr. Python's `node.get("Plans", [])` treats a missing
// key and an empty list identically, and so does this.
PyObject* children_of(PyObject* node) {
    PyObject* plans = PyDict_GetItemString(node, "Plans");  // borrowed
    if (plans == nullptr || !PyList_Check(plans)) return nullptr;
    return PyList_GET_SIZE(plans) > 0 ? plans : nullptr;
}

struct Visited {
    PyObject* node;
    long depth;
};

// One flat pass, matching Python's `list(_walk(raw_plan))` order: a node comes
// before its children, children in list order.
void walk(PyObject* node, long depth, std::vector<Visited>& out) {
    out.push_back({node, depth});
    PyObject* plans = children_of(node);
    if (plans == nullptr) return;
    Py_ssize_t n = PyList_GET_SIZE(plans);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* child = PyList_GET_ITEM(plans, i);  // borrowed
        if (PyDict_Check(child)) walk(child, depth + 1, out);
    }
}

bool set_double(PyObject* dict, const char* key, double value) {
    PyObject* obj = PyFloat_FromDouble(value);
    if (obj == nullptr) return false;
    int rc = PyDict_SetItemString(dict, key, obj);
    Py_DECREF(obj);
    return rc == 0;
}

PyObject* encode_plan_tree(PyObject* /*self*/, PyObject* args) {
    PyObject* raw_plan = nullptr;
    if (!PyArg_ParseTuple(args, "O", &raw_plan)) return nullptr;

    PyObject* features = PyDict_New();
    if (features == nullptr) return nullptr;

    for (int i = 0; i < N_SCALARS; i++) {
        if (!set_double(features, SCALAR_FEATURES[i], 0.0)) goto fail;
    }
    for (int i = 0; i < N_FAMILIES; i++) {
        if (!set_double(features, FAMILIES[i].key, 0.0)) goto fail;
    }

    // `if not raw_plan` in Python: None, a non-dict, or an empty dict all return
    // the zeroed feature set rather than raising.
    if (!PyDict_Check(raw_plan) || PyDict_Size(raw_plan) == 0) return features;

    {
        std::vector<Visited> nodes;
        nodes.reserve(32);
        walk(raw_plan, 0, nodes);

        long max_depth = 0;
        long n_leaves = 0;
        bool have_join_depth = false;
        long deepest_join_depth = 0;
        double max_est_rows = 0.0;
        double sum_est_rows = 0.0;
        double max_intermediate_bytes = 0.0;
        double balance_sum = 0.0;
        long balance_count = 0;
        bool have_amplification = false;
        double max_amplification = 0.0;
        double max_join_rows = 0.0;
        bool have_join_rows = false;
        double max_hash_build_rows = 0.0;
        double n_parallel_aware = 0.0;
        double weighted_nestloop_depth = 0.0;
        double max_child_cost = 0.0;
        bool have_child_cost = false;
        double family_counts[N_FAMILIES] = {0.0};

        for (size_t i = 0; i < nodes.size(); i++) {
            PyObject* node = nodes[i].node;
            const long depth = nodes[i].depth;
            if (depth > max_depth) max_depth = depth;

            PyObject* plans = children_of(node);
            if (plans == nullptr) n_leaves++;

            const double rows = get_number(node, "Plan Rows");
            const double width = get_number(node, "Plan Width");
            if (rows > max_est_rows) max_est_rows = rows;
            sum_est_rows += rows;
            const double bytes = rows * width;
            if (bytes > max_intermediate_bytes) max_intermediate_bytes = bytes;

            const bool is_join = PyDict_GetItemString(node, "Join Type") != nullptr;
            if (is_join) {
                if (!have_join_depth || depth > deepest_join_depth) {
                    deepest_join_depth = depth;
                    have_join_depth = true;
                }
                if (!have_join_rows || rows > max_join_rows) {
                    max_join_rows = rows;
                    have_join_rows = true;
                }
            }

            // The root's Total Cost is cumulative, so Python excludes index 0.
            if (i > 0) {
                const double cost = get_number(node, "Total Cost");
                if (!have_child_cost || cost > max_child_cost) {
                    max_child_cost = cost;
                    have_child_cost = true;
                }
            }

            PyObject* type_obj = PyDict_GetItemString(node, "Node Type");  // borrowed
            const char* node_type = nullptr;
            if (type_obj != nullptr && PyUnicode_Check(type_obj)) {
                node_type = PyUnicode_AsUTF8(type_obj);
            }
            if (node_type != nullptr) {
                for (int f = 0; f < N_FAMILIES; f++) {
                    for (int m = 0; m < 3 && FAMILIES[f].members[m] != nullptr; m++) {
                        if (std::strcmp(node_type, FAMILIES[f].members[m]) == 0) {
                            family_counts[f] += 1.0;
                        }
                    }
                }
                if (std::strcmp(node_type, "Hash") == 0 && rows > max_hash_build_rows) {
                    max_hash_build_rows = rows;
                }
                if (std::strcmp(node_type, "Nested Loop") == 0) {
                    weighted_nestloop_depth += static_cast<double>(depth + 1);
                }
            }

            if (truthy(node, "Parallel Aware")) n_parallel_aware += 1.0;

            if (plans != nullptr) {
                const Py_ssize_t n_children = PyList_GET_SIZE(plans);
                if (n_children == 2) {
                    const double a = get_number(PyList_GET_ITEM(plans, 0), "Plan Rows") + 1.0;
                    const double b = get_number(PyList_GET_ITEM(plans, 1), "Plan Rows") + 1.0;
                    balance_sum += std::min(a, b) / std::max(a, b);
                    balance_count++;
                }
                double child_rows = 1.0;
                for (Py_ssize_t c = 0; c < n_children; c++) {
                    child_rows += get_number(PyList_GET_ITEM(plans, c), "Plan Rows");
                }
                const double amplification = (rows + 1.0) / child_rows;
                if (!have_amplification || amplification > max_amplification) {
                    max_amplification = amplification;
                    have_amplification = true;
                }
            }
        }

        const double total_cost = get_number(raw_plan, "Total Cost");
        const double startup_cost = get_number(raw_plan, "Startup Cost");
        const double root_rows = get_number(raw_plan, "Plan Rows");

        bool ok = true;
        ok = ok && set_double(features, "tree_depth", static_cast<double>(max_depth));
        ok = ok && set_double(features, "tree_leaves", static_cast<double>(n_leaves));
        ok = ok && set_double(features, "tree_bushiness",
                              balance_count ? balance_sum / static_cast<double>(balance_count) : 0.0);
        ok = ok && set_double(features, "log_max_est_rows", std::log1p(max_est_rows));
        ok = ok && set_double(features, "log_sum_est_rows", std::log1p(sum_est_rows));
        ok = ok && set_double(features, "log_max_intermediate_bytes",
                              std::log1p(max_intermediate_bytes));
        ok = ok && set_double(features, "max_child_rows_amplification",
                              have_amplification ? std::min(max_amplification, 1e6) : 0.0);
        ok = ok && set_double(features, "log_total_cost", std::log1p(total_cost));
        ok = ok && set_double(features, "startup_cost_fraction",
                              total_cost > 0.0 ? startup_cost / total_cost : 0.0);
        ok = ok && set_double(features, "deepest_join_depth",
                              have_join_depth ? static_cast<double>(deepest_join_depth) : 0.0);
        ok = ok && set_double(features, "log_max_join_est_rows", std::log1p(max_join_rows));
        // `if largest else 1.0` -- a falsy check, so a largest of exactly 0.0
        // yields 1.0 rather than (root+1)/1.
        ok = ok && set_double(features, "join_rows_reduction",
                              max_join_rows != 0.0 ? (root_rows + 1.0) / (max_join_rows + 1.0) : 1.0);
        ok = ok && set_double(features, "cost_concentration",
                              (have_child_cost && total_cost > 0.0) ? max_child_cost / total_cost : 0.0);
        ok = ok && set_double(features, "log_max_hash_build_rows", std::log1p(max_hash_build_rows));
        ok = ok && set_double(features, "n_parallel_aware", n_parallel_aware);
        ok = ok && set_double(features, "weighted_nestloop_depth", weighted_nestloop_depth);
        ok = ok && set_double(features, "log_root_width",
                              std::log1p(get_number(raw_plan, "Plan Width")));
        for (int f = 0; f < N_FAMILIES; f++) {
            ok = ok && set_double(features, FAMILIES[f].key, family_counts[f]);
        }
        if (!ok) goto fail;
    }

    return features;

fail:
    Py_DECREF(features);
    return nullptr;
}

PyMethodDef METHODS[] = {
    {"encode_plan_tree", encode_plan_tree, METH_VARARGS,
     "Structural features for one plan tree. Matches "
     "app.optimizer.plan_tree.encode_plan_tree exactly."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT,
    "plan_tree_native",
    "Compiled plan-tree feature encoder.",
    -1,
    METHODS,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_plan_tree_native(void) { return PyModule_Create(&MODULE); }
