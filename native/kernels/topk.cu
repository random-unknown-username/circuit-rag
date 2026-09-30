#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace circuit {

constexpr int MAX_K = 32;
constexpr int PADDED_K = 33; // stride of 33 elimaties 32 way shmem bank conflicts
constexpr int NUM_BLOCKS = 32;
constexpr int BLOCK_THREADS = 128;

static __device__ float g_stage1_scores[NUM_BLOCKS * MAX_K];
static __device__ int g_stage1_ids[NUM_BLOCKS * MAX_K];

__device__ inline void insert_sorted(float* vals, int* idxs, int K, float v, int id) {
    int pos = K - 1;
    while (pos > 0 && vals[pos - 1] < v) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        pos--;
    }
    vals[pos] = v;
    idxs[pos] = id;
}

__device__ inline void merge_topk(
    float* a_vals, int* a_ids,
    const float* b_vals, const int* b_ids,
    int K
) {
    float res_vals[MAX_K];
    int res_ids[MAX_K];
    int i = 0, j = 0, idx = 0;
    while (idx < K && (i < K || j < K)) {
        float va = (i < K) ? a_vals[i] : -1e9f;
        float vb = (j < K) ? b_vals[j] : -1e9f;
        if (va >= vb) {
            res_vals[idx] = va;
            res_ids[idx] = a_ids[i];
            i++;
        } else {
            res_vals[idx] = vb;
            res_ids[idx] = b_ids[j];
            j++;
        }
        idx++;
    }
    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        if (k < K) {
            a_vals[k] = res_vals[k];
            a_ids[k] = res_ids[k];
        }
    }
}

__global__ void select_topk_stage1_kernel(
    const float* __restrict__ scores,
    const int* __restrict__ original_ids,
    int N,
    int K
) { 
    float r_vals[MAX_K];
    int r_idxs[MAX_K];
    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        r_vals[k] = -1e9f;
        r_idxs[k] = -1;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float min_val = -1e9f;
    for (int i = tid; i < N; i += stride) {
        float s = scores[i];
        if (s > min_val) {
            int id = original_ids ? original_ids[i] : i;
            insert_sorted(r_vals, r_idxs, K, s, id);
            min_val = r_vals[K - 1];
        }
    }


    __shared__ float s_vals[BLOCK_THREADS][PADDED_K];
    __shared__ int s_ids[BLOCK_THREADS][PADDED_K];

    for (int k = 0; k < K; ++k) {
        s_vals[threadIdx.x][k] = r_vals[k];
        s_ids[threadIdx.x][k] = r_idxs[k];
    }
    __syncthreads();

    // parallel tree reduction across threads in block
    #pragma unroll
    for (int s = BLOCK_THREADS / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            merge_topk(
                s_vals[threadIdx.x], s_ids[threadIdx.x],
                s_vals[threadIdx.x + s], s_ids[threadIdx.x + s],
                K
            );
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            g_stage1_scores[blockIdx.x * K + k] = s_vals[0][k];
            g_stage1_ids[blockIdx.x * K + k] = s_ids[0][k];
        }
    }
}

__global__ void select_topk_stage2_kernel(
    int K,
    float* __restrict__ out_scores,
    int* __restrict__ out_ids
) {
    __shared__ float s_vals[NUM_BLOCKS][PADDED_K];
    __shared__ int s_ids[NUM_BLOCKS][PADDED_K];

    int t = threadIdx.x;
    if (t < NUM_BLOCKS) {
        for (int k = 0; k < K; ++k) {
            s_vals[t][k] = g_stage1_scores[t * K + k];
            s_ids[t][k] = g_stage1_ids[t * K + k];
        }
    } else {
        for (int k = 0; k < K; ++k) {
            s_vals[t][k] = -1e9f;
            s_ids[t][k] = -1;
        }
    }
    __syncwarp();

    // warp lvl parallel reduction across 32 candidates 
    #pragma unroll
    for (int s = NUM_BLOCKS / 2; s > 0; s /= 2) {
        if (t < s) {
            merge_topk(s_vals[t], s_ids[t], s_vals[t + s], s_ids[t + s], K);
        }
        __syncwarp();
    }

    if (t == 0) {
        for (int k = 0; k < K; ++k) {
            out_scores[k] = s_vals[0][k];
            out_ids[k] = s_ids[0][k];
        }
    }
}


void launch_select_topk(
    const float* scores,
    const int* original_ids,
    int N,
    int K,
    float* out_scores,
    int* out_ids,
    cudaStream_t stream = 0
) {
    if (K > MAX_K) K = MAX_K;
    select_topk_stage1_kernel<<<NUM_BLOCKS, BLOCK_THREADS, 0, stream>>>(
        scores, original_ids, N, K
    );
    select_topk_stage2_kernel<<<1, 32, 0, stream>>>(
        K, out_scores, out_ids
    );
}

} // namespace circuit