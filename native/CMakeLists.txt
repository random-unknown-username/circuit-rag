cmake_minimum_required(VERSION 3.20)
project(circuit_rag LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(CIRCUIT_SOURCES
    src/gpu_index.cpp
    src/api.cpp
    src/dispatch.cu
    kernels/root_bounds.cu
    kernels/parent_projection.cu
    kernels/child_bounds.cu
    kernels/dense_score.cu
    kernels/topk.cu
)

# building the shared lib libcircuit.so
add_library(circuit SHARED ${CIRCUIT_SOURCES})
set_target_properties(circuit PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(circuit PRIVATE CUDA::cudart CUDA::cublas)

# smol benchmark for circuit
add_executable(circuit_bench bin/circuit_bench.cpp)
target_link_libraries(circuit_bench PRIVATE circuit CUDA::cudart CUDA::cublas)
