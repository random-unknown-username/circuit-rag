# Architecture

Status: scaffold only.

The product is `circuit`, a local memory CLI. CIRCUIT-LCE is its retrieval engine.

CLI commands → circuit-app workflows → memory, ingest, embed, agent, storage, and retrieval interfaces. Optional agent integrations call the same application services. Presentation and protocol adapters never own retrieval or ingestion logic.

Ingest produces source-attributed chunks. Embedding adapters produce vectors with a model fingerprint. Storage keeps the content catalog and vector/index snapshots consistent. Search returns scored source references. Ask adds context assembly and cited generation; recall orders retrieved memories into a timeline. Chat uses explicit session persistence and writeback settings.

Rust owns the production control/storage plane. C++ owns GPU execution and resources. CUDA owns kernels. Python/PyTorch research calls the native extension directly. See [retrieval engine](retrieval-engine.md).

Keep providers replaceable through small contracts: source records, embedding batches, retrieval requests/results, generation streams, and tool invocations. Convert external SDK types at the adapter boundary. No provider SDK types in memory domain models. Share contract fixtures between alternative adapters.

A flat exact backend provides a reference and a planned CPU path; CUDA availability must be detected, never assumed. Product workflows and retrieval research can develop independently. Do not silently claim CIRCUIT-LCE ran when a fallback executed.

The additional circuit-app crate keeps CLI, future MCP, and structured stdio clients on the same workflows. circuit-integrations holds optional protocol adapters. Both are placeholders.

Shared model transport and configurable OpenAI-compatible protocol adapters live in circuit-providers; see [provider contracts](providers.md).
