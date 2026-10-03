# file format (.circuit)

single binary file, written by `CircuitIndex.save`, read by `CircuitIndex.load` (python, mmap) and by the c++ loader in `native/src/gpu_index.cpp`.

```
bytes 0..7        magic "CIRCUIT1"
bytes 8..11       uint32 little endian: header length (always 8192)
bytes 12..8203    json header, zero padded to 8192
bytes 8204..      tensors, each starting on a 64 byte boundary, in the order below
```

## json header

```json
{
  "version": 1,
  "dim": 384,
  "rank": 8,
  "num_roots": 16,
  "num_total_children": 128,
  "N": 5183,
  "tensor_meta": {
    "root_centers": {"dtype": "float32", "shape": [16, 384], "offset": 8204, "nbytes": 24576},
    "...": "..."
  }
}
```

`offset` is an absolute byte offset into the file, `nbytes` is the unpadded size.

## tensors (in order)

| name | dtype | shape |
|---|---|---|
| root_centers | float32 | [num_roots, dim] |
| root_radii | float32 | [num_roots] |
| bases | float32 | [num_roots, dim, rank] |
| child_intervals_l | float32 | [num_total_children, rank] |
| child_intervals_u | float32 | [num_total_children, rank] |
| child_radii | float32 | [num_total_children] |
| child_parent_ids | int32 | [num_total_children] |
| leaf_offsets | int32 | [num_total_children] |
| leaf_sizes | int32 | [num_total_children] |
| leaf_vectors | float32 | [N, dim] |
| original_ids | int32 | [N] |

`leaf_vectors` is stored grouped by child (child `i` owns rows `leaf_offsets[i] ... leaf_offsets[i] + leaf_sizes[i]`), `original_ids[row]` maps back to the row number in the `X` u built from.

## why 64 byte alignment + mmap

so the tensors can be used straight out of the page cache with no parsing or copying on the cpu side. with `use_mmap=True` loading is basically just opening the file, the copy to vram (if device is cuda) is the only real cost.
