# native bench

`native/bin/circuit_bench.cpp`: loads a `.circuit` file straight into vram through the c++ api (`libcircuit.so`) and times queries. no python in the loop, so this is the real kernel speed.

## run it

```bash
cmake -B build -S native
cmake --build build -j

# make an index to feed it
python - <<'EOF'
import numpy as np
from circuit_rag import build_circuit_index
X = np.random.randn(200_000, 128).astype("float32")
X /= np.linalg.norm(X, axis=1, keepdims=True)
build_circuit_index(X, num_roots=64, children_per_root=16, rank=8).save("bench.circuit")
EOF

./build/circuit_bench bench.circuit
```

the one arg is the path to the `.circuit` file.

## what it prints

p50 latency for host-to-host (query copied in, results copied back) and in-vram (data already on gpu), plus qps for both, then a sample top-10 so u can eyeball the ids + scores.

## my numbers

scifact index, rtx 5050 laptop:

| | p50 | qps |
|---|---|---|
| host to host | 38 us | ~24k |
| in vram | 37 us | ~26k |

## why the python path is slower

this is the c++ api. the python `circuit_search` adds per-call python overhead and one host sync to compute the seed threshold (it needs an `.item()`), which puts it at roughly cublas-gemv speed (~370us vs ~330us on 200k vectors). python-side speed isnt solved yet.
