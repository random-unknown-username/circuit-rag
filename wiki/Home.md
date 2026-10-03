# circuit-rag wiki

exact top-k vector search on the gpu. results are identical to brute force, the hierarchy only skips work it can prove is useless.

pages:

- [install](Install.md) building it, what u need
- [usage](Usage.md) `CircuitRAG`, raw index, save/load, reading the trace
- [how it works](How-it-works.md) the 5 kernels + why its exact
- [file format](File-format.md) the `.circuit` binary layout
- [native bench](Native-bench.md) the c++ `circuit_bench` binary
- [limits + faq](Limits-and-FAQ.md) what doesnt work yet, the 0% pruning thing

fastest way to see it work: `python tests/test_exact.py` then `python examples/real_proof.py`.
