# Artifact format

Placeholder for the versioned on-disk specification.

Required fields: format version, build provenance, corpus fingerprint, embedding identity, dimensions, dtype, normalization, canonical arithmetic, tie policy, ID mapping, partition coverage, certificate type and precision, basis layout, numerical error budgets, offsets, lengths, alignment, and checksums.

Publish atomically after validation. Reject unsupported versions and inconsistent lengths. Keep canonical corpus input immutable. Hardware cost tables need a separate device/software identity and version; they do not establish mathematical correctness.
