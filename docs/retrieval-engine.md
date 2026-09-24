# CIRCUIT-LCE retrieval engine

Status: design contracts; implementation pending.

Python builds and trains the hierarchy. The final builder reconstructs every certificate from the encoded corpus and publishes a versioned artifact. Rust loads it, owns production query state and storage, and selects refinement, descent, or dense execution. C++ owns GPU contexts, GPU indexes, device buffers, streams/events, and kernel dispatch. Rust calls this executor through a narrow C ABI; CUDA implements its kernels.

The initial physical layout is leaf-contiguous canonical vectors plus root certificates, parent bases, child certificates, and leaf certificates. Keep stable external IDs separate from physical ordering. Store ownership and offsets explicitly; avoid implicit cross-language layout assumptions.

Root bounds include centroid quantization error. Child bounds include interval endpoints, residual radii, parent-center error, and finite-precision basis error. Per-vector refinement runs only after macro certification. Every eligible vector must be reachable or covered by a sound rejection.

Approximate seeds must be deduplicated, filtered, and scored under canonical semantics. Until K distinct eligible vectors have been scored, a finite K-th threshold is unavailable. A scheduler decision changes execution cost, never admissibility.

Use reusable device scratch buffers and grouped native calls. Kernel fusion, async execution, and caching are measurement-driven decisions. Profile dense fallback alongside certificates.

Deferred: distributed search, updates/deletes, service deployment, composite agent relevance, host/NVMe paging, and encoder replacement.

## Two integration paths

Research: Python/PyTorch → direct C++/CUDA extension → GPU. Reuse the native executor and kernels without requiring Rust or PyO3 for each experiment.

Production: optional Python/PyO3 → Rust control/storage plane → C ABI → C++ GPU execution plane → CUDA kernels. Develop the Rust plane independently against the shared artifact specification, then integrate once native kernel contracts stabilize.

Rust owns SegmentManager, CertificateStore, PageCache, IndexManifest, QueryPlan, StorageTier, CostModel, and Scheduler. C++ owns GpuContext, GpuIndex, KernelDispatcher, and CudaBuffers. Rust wrappers manage opaque handle lifetimes; they do not implement CUDA resource orchestration.

Planned C ABI operations: circuit_context_create, circuit_index_load, circuit_query, circuit_profile_query, and circuit_context_destroy. Signatures are intentionally deferred. Specify error propagation, thread safety, asynchronous completion, host tensor lifetimes, stream interoperability, and destruction behavior before implementing either binding.

Native library integration, CUDA graphs, and specialized execution facilities belong in C++. GPUDirect and OptiX remain optional future investigations, not requirements for the first experiment.
