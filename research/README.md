# Research gates

01 geometry: frozen real embeddings, held-out queries, true top-K thresholds, true block support, practical safe bounds, and descendant-weighted exposure. Report population size, K, partition settings, average, tails, and per-query results. A block rejection percentage alone is insufficient.

02 hierarchy: optimize partition/basis geometry with the encoder frozen. Compare conventional partitions and certificate types. Reconstruct exact certificates before held-out evaluation. Compare oracle and realistic seed thresholds separately.

03 representation: optional encoder training, with retrieval-quality versus work curves and a predefined acceptable quality loss. Keep train, validation, and test queries separate.

04 GPU: all-resident corpus, flat exact baseline, conventional exact hierarchy, HN-like prefix baseline, and optional CAGRA and compressed baselines. Compare exactness separately from ANN recall. Record device/software identity, warmup, batch size, P50/P95/P99, actual bytes, kernel timings, scheduler/compaction cost, and total latency. Cost models must come from measurements.

05 storage: only after the earlier gates pass. Record physical page reads, cache conditions, transfer latency, and complete query time.

No results are included in this scaffold. The research brief's prior measurements and literature claims have not been independently reproduced here.
