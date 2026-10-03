import json
import mmap
import os
import struct
from typing import Optional, Union, Dict, Any
import numpy as np
import torch

from circuit_rag.hierarchy.build import CircuitIndex

MAGIC_HEADER = b"CIRCUIT1" # 8bytes 1 stands for v1

def save_circuit_index(index: CircuitIndex, filepath: str) -> int:
    """
    saves a CircuitIndex to a single bin file .circuit
    aligned on 64byte boundaries for  zerocopy mmap
    """
    tensors = {
        "root_centers": (index.root_centers.cpu().contiguous().numpy(), "float32"),
        "root_radii": (index.root_radii.cpu().contiguous().numpy(), "float32"),
        "bases": (index.bases.cpu().contiguous().numpy(), "float32"),
        "child_intervals_l": (index.child_intervals_l.cpu().contiguous().numpy(), "float32"),
        "child_intervals_u": (index.child_intervals_u.cpu().contiguous().numpy(), "float32"),
        "child_radii": (index.child_radii.cpu().contiguous().numpy(), "float32"),
        "child_parent_ids": (index.child_parent_ids.cpu().contiguous().numpy(), "int32"),
        "leaf_offsets": (index.leaf_offsets.cpu().contiguous().numpy(), "int32"),
        "leaf_sizes": (index.leaf_sizes.cpu().contiguous().numpy(), "int32"),
        "leaf_vectors": (index.leaf_vectors.cpu().contiguous().numpy(), "float32"),
        "original_ids": (index.original_ids.cpu().contiguous().numpy(), "int32"),
    }

    meta: Dict[str, Any] = {
        "version": 1,
        "dim": int(index.dim),
        "rank": int(index.rank),
        "num_roots": int(index.num_roots),
        "num_total_children": int(index.num_total_children),
        "N": int(index.N),
        "tensor_meta": {},
    }

    # uses 8KB page aligned header block
    padded_header_len = 8192
    data_start = 8 + 4 + padded_header_len

    current_offset = data_start
    for name, (arr, dtype_str) in tensors.items():
        nbytes = arr.nbytes
        meta["tensor_meta"][name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "offset": current_offset,
            "nbytes": nbytes,
        }
        current_offset += (nbytes + 63) & ~63  # 64-byte alignment

    header_bytes = json.dumps(meta).encode("utf-8")
    actual_header_len = len(header_bytes)
    assert actual_header_len <= padded_header_len, "Header too large to fit in 8KB"

    # Write file
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<I", padded_header_len))
        f.write(header_bytes)
        # pad to data_start
        pad_size = data_start - f.tell()
        if pad_size > 0:
            f.write(b"\x00" * pad_size)

        for name in tensors:
            arr, _ = tensors[name]
            f.write(arr.tobytes())
            align_pad = ((arr.nbytes + 63) & ~63) - arr.nbytes
            if align_pad > 0:
                f.write(b"\x00" * align_pad)

    total_bytes = current_offset
    return total_bytes

def load_circuit_index(filepath: str, device: str = "cuda", use_mmap: bool = True) -> CircuitIndex:
    """
    loads a CircuitIndex from a .circuit bin
    sub 5ms instant idx loads
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")

    with open(filepath, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC_HEADER:
            raise ValueError(f"File {filepath} is not a valid CircuitIndex file.")
        header_len = struct.unpack("<I", f.read(4))[0]
        header_bytes = f.read(header_len)
        meta = json.loads(header_bytes.decode("utf-8").rstrip("\x00"))

        tensor = {}

        if use_mmap:
            with open(filepath, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
                for name, t_info in meta["tensor_meta"].items():
                    offset = t_info["offset"]
                    nbytes = t_info["nbytes"]
                    shape = tuple(t_info["shape"])
                    dtype = np.dtype(t_info["dtype"])
                    # zerocopy buffer
                    arr = np.ndarray(buffer=mm, dtype=dtype, offset=offset, shape=shape)
                    # convert to torch tensor
                    t = torch.from_numpy(arr)
                    if device == "cuda" and torch.cuda.is_available():
                        t = t.cuda()
                    tensor[name] = t
        else:
            with open(filepath, "rb") as f:
                for name, t_info in meta["tensor_meta"].items():
                    f.seek(t_info["offset"])
                    data = f.read(t_info["nbytes"])
                    dtype = np.dtype(t_info["dtype"])
                    arr = np.frombuffer(data, dtype=dtype).reshape(t_info["shape"])
                    t = torch.from_numpy(arr.copy())
                    if device == "cuda" and torch.cuda.is_available():
                        t = t.cuda()
                    tensor[name] = t

    idx = CircuitIndex(
        root_centers=tensor["root_centers"],
        root_radii=tensor["root_radii"],
        bases=tensor["bases"],
        child_intervals_l=tensor["child_intervals_l"],
        child_intervals_u=tensor["child_intervals_u"],
        child_radii=tensor["child_radii"],
        child_parent_ids=tensor["child_parent_ids"],
        leaf_offsets=tensor["leaf_offsets"],
        leaf_sizes=tensor["leaf_sizes"],
        leaf_vectors=tensor["leaf_vectors"],
        original_ids=tensor["original_ids"],
        dim=meta["dim"],
        rank=meta["rank"]
    )
    return idx