#ifndef CIRCUIT_H
#define CIRCUIT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct circuit_index_opaque* circuit_index_t;

typedef struct {
    int top_k;
    float threshold;
    int use_cuda_graph;
} circuit_search_config_t;

typedef struct {
    float latency_us;
    int vectors_scored;
} circuit_trace_t;

// loading fill .circuit index directly into gpu vram via zero copy mmap
circuit_index_t circuit_load_index(const char* filepath, int device_id);

// search the index using cuda without python overhead (host pointers)
int circuit_search(
    circuit_index_t index,
    const float* query,   // [dim] on host
    int k,
    float* out_scores,     // [k] on host
    int* out_ids,          // [k] on host
    circuit_trace_t* trace
);

// search the index using cuda when tensors are on VRAM (in vram)

int circuit_search_device(
    circuit_index_t index,
    const float* dev_query,  // [dim] on vram
    int k,
    float* dev_out_scores,    // [k] on vram
    int* dev_out_ids,         // [k] on vram
    circuit_trace_t* trace
);

// purge the index and free gpu and mmap memory
void circuit_destroy_index(circuit_index_t index);

#ifdef __cplusplus
}
#endif

#endif // CIRCUIT_H


