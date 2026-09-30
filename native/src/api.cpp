#include "../include/circuit/circuit.h"
#include "../include/circuit/gpu_index.hpp"
#include <new>

extern "C" {

    circuit_index_t circuit_load_index(const char* filepath, int device_id) {
    auto* idx = new (std::nothrow) circuit_index_opaque();
    if (!idx) {
        return nullptr;
    }

    if (!idx->load(filepath, device_id)) {
        idx->destroy();
        delete idx;
        return nullptr;
    }
    return idx;
    }

    int circuit_search(
        circuit_index_t index,
        const float* query,
        int k,
        float* out_scores,
        int* out_ids,
        circuit_trace_t* trace) {
            if (!index) {
                return -1;
            }
            return index->search(query, k, out_scores, out_ids, trace);
        }

    int circuit_search_device(
        circuit_index_t index,
        const float* dev_query,
        int k,
        float* dev_out_scores,
        int* dev_out_ids,
        circuit_trace_t* trace) {
            if (!index) {
                return -1;
            }
            return index->search_device(dev_query, k, dev_out_scores, dev_out_ids, trace);
        }
    
    void circuit_destroy_index(circuit_index_t index) {
        if (!index) return;
        index->destroy();
        delete index;
    }
}