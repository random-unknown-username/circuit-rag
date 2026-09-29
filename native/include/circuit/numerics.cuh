#pragma once


#include <cuda_runtime.h>
#include <cmath>

namespace circuit {

    // fast wrap reduction
    __device__ __forceinline__ float warpReduceSum(float val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __device__ __force_inline__ float blockReduceSum(float val) {
        static __shared__ float shared[32]; // shared mem for 32 partial sums
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        val = warpReduceSum(val); // each warp performs partial reduction
        if (lane == 0) shared[wid] = val; // write reduced value to shared memory
        __syncthreads(); // wait for all partial reductions

        val = (threadIdx.x < (blockDim.x / 32.0f)) ? shared[lane] : 0.0f;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }
} //namespace circuit
