# install

## what u need

- nvidia gpu
- cuda toolkit (`nvcc` on your path)
- python 3.10+
- pytorch built w cuda (the ext compiles against it)

tested on: rtx 5050 laptop (8gb), arch linux, cuda 13.3, python 3.14.

## build the python ext

```bash
git clone https://github.com/random-unknown-username/circuit-rag
cd circuit-rag
pip install -e . --no-build-isolation
```

`--no-build-isolation` matters. setup.py imports torch to find the cuda headers, and isolated builds dont have it.

takes about a minute. it compiles 5 kernels + the torch bindings into `circuit_cuda`, only for the gpu in your machine (setup.py reads the compute capability). building for a different gpu:

```bash
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation
```

## check it worked

```bash
python -c "import torch, circuit_cuda"      # should print nothing
python tests/test_exact.py           # exact: 25/25 queries match brute force
```

## build the c++ bench (optional)

```bash
cmake -B build -S native
cmake --build build -j
```

gives u `build/libcircuit.so` and `build/circuit_bench`. see [native bench](Native-bench.md).

## if it breaks

- `libc10.so: cannot open shared object file` on `import circuit_cuda`: import torch first (`import torch, circuit_cuda`), the ext links against torch libs and they only get loaded once torch is imported. `import circuit_rag` already does this for u
- `nvcc: command not found`: cuda toolkit isnt on PATH (arch: `/opt/cuda/bin`).
- no gpu: the ext wont build. `circuit_search` itself falls back to a plain cpu matmul if the ext isnt importable, but install needs nvcc.
