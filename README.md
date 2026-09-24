# CIRCUIT

Memory for your terminal and your agents.

Feed it files, chats, repos, notes, or tool traces. Ask for the context you need without remembering which folder, conversation, or late-night experiment it came from.

The product is **circuit-cli**, with the planned command `circuit`. Underneath it, CIRCUIT-LCE is the exact retrieval engine we're building to avoid reading vectors that provably cannot affect the result.

**Current status:** this is a project scaffold. The commands below show the intended experience. There is no installable CLI, working index, or measured speedup yet. Source files, manifests, configs, and tests contain placeholders.

## The experience we're building

```sh
circuit init
circuit add ~/projects/my-project
circuit add ~/notes
circuit search "persistent buffer cache"
circuit ask "what did I change to fix that CUDA bug?"
circuit ask "what did I change to fix that CUDA bug?" --trace
circuit recall "CUDA experiments"
circuit chat
```

Search gives you memories with source locations. Ask builds an answer with citations. Recall puts related memories on a timeline. Chat brings that memory into an ongoing conversation.

`circuit demo` should take you through a small sample corpus, ingestion, search, a cited answer, a retrieval trace, and a comparison against flat search. Optional model downloads and generation requirements need to be explicit; a demo should explain what's missing rather than pretend it worked.

## Show the work

The trace should show regions rejected, leaves visited, per-vector certificates read, full vectors scored, bytes read, and retrieval latency. `circuit inspect` shows the memory and backend state. `circuit benchmark` compares real runs against flat exact search and optional ANN baselines.

Every number must come from the run. Estimated bytes and measured device traffic must have different labels. Exact top-K describes vector retrieval under the configured score; it does not guarantee an LLM answer is correct.

## Built to plug into your setup

Start with files and existing embeddings. Preserve source paths, external IDs, timestamps, and provenance. Keep embedding models, generation providers, parsers, and agent integrations behind small interfaces so changing one doesn't mean rebuilding the whole harness.

[OpenAI-compatible provider slots](docs/providers.md) support the planned choice of custom endpoints and separate generation/embedding models. Local and remote provider slots are reserved. Network use, credentials, and conversation writeback will be explicit settings. The core experience should remain useful without an answer-generation provider: ingest, search, recall, and inspect.

## Inside the repo

| Path | Purpose |
| --- | --- |
| `crates/circuit-cli/` | The product: commands and terminal output |
| `crates/circuit-app/` | Shared workflows for CLI and agent clients |
| `crates/circuit-memory/` | Memories, chunks, sources, metadata, namespaces |
| `crates/circuit-ingest/` | Files, repos, chats, parsing, chunking, incremental ingestion |
| `crates/circuit-providers/` | Configurable OpenAI-compatible endpoints, protocol adapters, transport |
| `crates/circuit-embed/` | Embedding providers and cache |
| `crates/circuit-agent/` | Context, generation, citations, conversations, tools |
| `crates/circuit-integrations/` | Optional agent protocols and portable import/export |
| `crates/circuit-core/` | Exact retrieval contracts |
| `crates/circuit-runtime/` | CIRCUIT-LCE planning and scheduling |
| `crates/circuit-storage/` | Content catalog, segments, indexes, pages, migrations |
| `crates/circuit-cuda/` | Rust wrapper around the native GPU executor |
| `crates/circuit-python/` | Future production Python bindings |
| `native/` | C++ GPU executor, direct PyTorch binding, CUDA kernels |
| `research/` | CIRCUIT-LCE learning, certificates, experiments, notebooks |
| `benchmarks/` | Retrieval, ingestion, GPU, and complete workflow measurements |
| `examples/` | Personal, codebase, agent memory, and RAG walkthroughs |
| `docs/` | Product contracts, architecture, and retrieval research |

Python/PyTorch for research and learning, Rust for the production control/storage plane, C++ for the GPU execution plane, CUDA for kernels.

The research path goes directly from Python/PyTorch to C++/CUDA. Production uses Rust and a small C ABI. Research stays independent until the kernels and formats stabilize.

## Where to start

Read the [planned CLI](docs/cli.md), [architecture](docs/architecture.md), and [adapter boundaries](docs/extending.md). The [research plan](docs/research.md) explains the engine's acceptance gates. The [file map](docs/file-map.md) lists every scaffold file.

We're leaving room to learn from open-source harnesses without copying their entire architecture. Record inspirations, keep attribution, and adapt them at a defined boundary. See [third-party policy](docs/third-party.md).

No license has been selected yet. No third-party implementation has been imported.
