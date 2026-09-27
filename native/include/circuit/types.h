#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace circuit {

    // config for hierarchy
    struct HierarchyConfig {
        int32_t num_roots; // no of root clusters
        int32_t childern_per_root; // no of children per root cluster
        int32_t rank; //subspace rank r
        int32_t dim; // vec dimension d
    };

    // root cert metadata
    struct RootCertificate {
        float center_norm; // ||c_B||
        float delta_plus_rho; // delta_b + delta_rho (cons radius)
    };

    // child cert metadata
    struct ChildCertificate {
        int32_t parent_idx; // parent root idx
        int32_t leaf_offset; // offset into contiguous leaf vec array
        int32_t leaf_size; // no of vecs in leaf
        float residual_radius; // rho_b == max ||(I - V_P V_P^T)(x - c_P)||
    };

} // namespace circuit
