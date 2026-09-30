#include "../include/circuit/circuit.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

int main(int argc, char** argv) {
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    std::cout << "CIRCUIT RAG cpp cuda native benchmark, without python overhead in account rn" << std::endl;

    const char* filepath = "data/scifact_index.circuit";
    if (argc > 1) {
        filepath = argv[1];
    }

    std::cout << "loading .circuit index from " << filepath << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    circuit_index_t index = circuit_load_index(filepath, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!index) {
        std::cerr << "error loading index" << std::endl;
        return 1;
    }
    std::cout << "index loaded & warmed up in " << std::fixed << std::setprecision(2) << load_ms << " ms!" << std::endl;

    int dim = 384;
    int k = 10;

    // use host mem for fast DMA
    float* h_query = nullptr;
    float* h_scores = nullptr;
    int* h_ids = nullptr;
    cudaMallocHost(&h_query, dim * sizeof(float));
    cudaMallocHost(&h_scores, k * sizeof(float));
    cudaMallocHost(&h_ids, k * sizeof(int));
    for (int d = 0; d < dim; ++d) h_query[d] = 0.05f;

    circuit_trace_t trace

    // warmup 
    std::cout << "[C++] Executing 50 warmup queries on GPU..." << std::endl;
    for (int i = 0; i < 50; ++i) {
        circuit_search(index, h_query, k, h_scores, h_ids, &trace);
    }

    // Benchmark loop: 500 queries
    const int num_queries = 500;
    std::cout << "[C++] Running " << num_queries << " pure C++ host-to-host queries..." << std::endl;

    std::vector<double> latencies_us;
    latencies_us.reserve(num_queries);

    // randomize query slightly per run to prevevnt cache hits
    td::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto total_bench_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; ++i) {
        for (int d = 0; d < dim; ++d) {
            h_query[d] += dist(rng) * 0.001f;
        }

        auto q_start = std::chrono::high_resolution_clock::now();
        circuit_search(index, h_query, k, h_scores, h_ids, &trace);
        auto q_end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(q_end - q_start).count();
        latencies_us.push_back(us);
    }
    auto total_bench_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_bench_end - total_bench_start).count();

    std::sort(latencies_us.begin(), latencies_us.end());
    double p50 = latencies_us[num_queries * 50 / 100];
    double p90 = latencies_us[num_queries * 90 / 100];
    double p99 = latencies_us[num_queries * 99 / 100];
    double avg = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / num_queries;
    double qps = (num_queries / total_ms) * 1000.0;

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "CUDA HARDWARE PERFORMANCE: HOST-TO-HOST (DMA + GRAPH)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  • P50 Host-to-Host Latency:  " << std::setw(8) << p50 << " us (" << p50/1000.0 << " ms)" << std::endl;
    std::cout << "  • P90 Host-to-Host Latency:  " << std::setw(8) << p90 << " us (" << p90/1000.0 << " ms)" << std::endl;
    std::cout << "  • P99 Host-to-Host Latency:  " << std::setw(8) << p99 << " us (" << p99/1000.0 << " ms)" << std::endl;
    std::cout << "  • Mean Latency:              " << std::setw(8) << avg << " us" << std::endl;
    std::cout << "  • Host-to-Host Throughput:   " << std::setw(8) << static_cast<int>(qps) << " queries/sec (Single Stream)" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // Benchmark Mode 2: in VRAM pipeline (zero DMA transfers, e.g. embedded RAG with GPU embedding model)
    std::cout << "\n[C++] Benchmarking in VRAM pipeline" << std::endl;
    float* d_query = nullptr;
    float* d_scores = nullptr;
    int* d_ids = nullptr;
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_scores, k * sizeof(float));
    cudaMalloc(&d_ids, k * sizeof(int));
    cudaMemcpy(d_query, h_query, dim * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<double> vram_latencies_us;
    vram_latencies_us.reserve(num_queries);

    auto vram_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; ++i) {
        auto q_start = std::chrono::high_resolution_clock::now();
        circuit_search_device(index, d_query, k, d_scores, d_ids, &trace);
        auto q_end = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(q_end - q_start).count();
        vram_latencies_us.push_back(us);
    }
    auto vram_end = std::chrono::high_resolution_clock::now();
    double vram_ms = std::chrono::duration<double, std::milli>(vram_end - vram_start).count();

    std::sort(vram_latencies_us.begin(), vram_latencies_us.end());
    double v_p50 = vram_latencies_us[num_queries * 50 / 100];
    double v_p90 = vram_latencies_us[num_queries * 90 / 100];
    double v_p99 = vram_latencies_us[num_queries * 99 / 100];
    double v_avg = std::accumulate(vram_latencies_us.begin(), vram_latencies_us.end(), 0.0) / num_queries;
    double v_qps = (num_queries / vram_ms) * 1000.0;

    std::cout << "\n=================================================================" << std::endl;
    std::cout << "CUDA IN-VRAM PIPELINE (EMBEDDED RAG / ZERO DMA)" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "  • P50 In-VRAM Latency:       " << std::setw(8) << v_p50 << " us (" << v_p50/1000.0 << " ms)" << std::endl;
    std::cout << "  • P90 In-VRAM Latency:       " << std::setw(8) << v_p90 << " us (" << v_p90/1000.0 << " ms)" << std::endl;
    std::cout << "  • P99 In-VRAM Latency:       " << std::setw(8) << v_p99 << " us (" << v_p99/1000.0 << " ms)" << std::endl;
    std::cout << "  • Mean Latency:              " << std::setw(8) << v_avg << " us" << std::endl;
    std::cout << "  • In-VRAM Throughput:        " << std::setw(8) << static_cast<int>(v_qps) << " queries/sec (Single Stream)" << std::endl;
    std::cout << "=================================================================" << std::endl;

    std::cout << "\nSample Top-10 Retrieval Results:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "  Rank " << std::setw(2) << (i + 1) << " | Original ID: " << std::setw(6) << h_ids[i]
                  << " | Similarity Score: " << std::setprecision(4) << h_scores[i] << std::endl;
    }
    std::cout << "=================================================================" << std::endl;

    circuit_destroy_index(index);
    cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_ids);
    cudaFreeHost(h_query);
    cudaFreeHost(h_scores);
    cudaFreeHost(h_ids);
    return 0;
}