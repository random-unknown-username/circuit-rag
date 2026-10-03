import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

native_dir = os.path.abspath("native")

# only build for the gpu in this machine unless TORCH_CUDA_ARCH_LIST is set
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

setup(
    name="circuit-rag",
    version="0.1.0",
    description="Exact, certified GPU vector search for RAG",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="circuit_cuda",
            sources=[
                "native/bindings/torch_extension.cpp",
                "native/src/dispatch.cu",
                "native/kernels/root_bounds.cu",
                "native/kernels/parent_projection.cu",
                "native/kernels/child_bounds.cu",
                "native/kernels/dense_score.cu",
                "native/kernels/topk.cu",
            ],
            include_dirs=[os.path.join(native_dir, "include")],
            libraries=["cublas"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
