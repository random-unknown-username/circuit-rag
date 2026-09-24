# Planned CLI

These commands are design targets, not working commands.

| Command | Purpose |
| --- | --- |
| init | Create or open a local memory workspace and report actual backend/provider availability |
| add | Ingest files, directories, chats, or explicitly requested URLs |
| search | Retrieve scored memories and source locations without generation |
| ask | Retrieve context and produce an answer with memory citations |
| chat | Run a memory-backed conversation with configurable writeback |
| recall | Present related memories as a provenance-based timeline |
| stats | Summarize memory counts and storage |
| inspect | Explain index, provider, backend, and last-query state |
| benchmark | Compare measured retrieval runs under documented conditions |
| demo | Run a small sample workflow with explicit prerequisites |
| remove | Remove a source from active memory and reconcile its index visibility |
| doctor | Diagnose configuration, model, storage, and backend availability |

Planned common flags: namespace/workspace selection, JSON output, and trace where meaningful. Search supports exact mode. Keep stdout machine-readable in JSON mode and progress on stderr. Define stable exit codes, output versions, and cancellation behavior before implementation.

Trace fields must distinguish corpus size, eligible size, bound evaluations, vectors scored, logical bytes, measured physical bytes, and timing boundaries. Missing measurements appear as unavailable. No synthetic demo performance figures.

Watch mode is deferred until incremental ingestion and deletion reconciliation are correct.
