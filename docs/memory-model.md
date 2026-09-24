# Memory model

A source produces versioned chunks with stable memory IDs, source spans, content hashes, timestamps, namespace, and provenance. Embeddings carry a model fingerprint distinct from source identity. The catalog stores retrievable content; the index stores vectors and certificates.

Re-ingestion must be idempotent. Updates and deletions create consistent visible snapshots, with tombstones respected by search and threshold seeding. Preserve the difference between source timestamps and ingest time. Generated summaries and extracted memories retain links to their original evidence.
