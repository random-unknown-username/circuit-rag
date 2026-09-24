# Extending the harness

Add source adapters in circuit-ingest, embedding adapters in circuit-embed, generation adapters in circuit-agent, and external agent protocols in circuit-integrations. Keep commands thin and call circuit-app workflows.

Before implementation, define versioned request/result types and fixtures for each boundary. Embeddings must declare dimensionality, normalization, and model identity. Retrieval must report backend, score semantics, IDs, source references, and trace availability. Generation must handle streaming, cancellation, citations, and provider errors. Tool execution must have explicit authorization and resource limits.

An adapter may depend on a provider SDK; memory domain types must not. Optional providers should not force GPU, network, or unrelated SDK dependencies into core workflows. Keep alternative implementations comparable with the same contract tests.
