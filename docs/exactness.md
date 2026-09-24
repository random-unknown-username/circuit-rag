# Exactness contract

Before implementing search, define the canonical stored vector representation, query representation, normalization, dot-product accumulation, and total result ordering. Exact relative to FP16 stored vectors is a different claim from exact relative to original FP32 embeddings. Tensor-core and flat-baseline arithmetic must not silently disagree.

For each block, the computed upper bound must dominate every allowed descendant score under the declared semantics. Outward storage rounding alone does not bound runtime arithmetic error. Account for dot products, norms, projections, center quantization, non-orthogonal stored bases, and threshold arithmetic. Specify behavior for NaN, infinity, zero norms, and overflow.

With arbitrary tie selection, a block whose upper bound equals the current K-th score can sometimes be discarded. For deterministic score/ID ordering, equality may contain a preferred ID. Use strict score rejection or a proven tie-aware rule. Tests must cover duplicate vectors and exact ties.

Filtered queries require seeds from eligible members. Certificates over a superset may remain conservative, but threshold construction must use the filtered population. Handle fewer than K eligible members explicitly.

Never publish learned interval endpoints as certificates. Recompute extrema and residual maxima over all actual descendants and include conservative error margins. Validate membership coverage, corpus fingerprints, dimensions, and versions before loading.

Numerical checks against flat search are necessary evidence, not a proof for every query. Derive the bound and its numerical error budget, then test adversarial edge cases and held-out queries.
