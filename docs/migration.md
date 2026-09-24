# Adoption and migration

Planned first import: existing dense embeddings, stable external IDs, and a separate query matrix. Preserve the caller's score metric and normalization explicitly. Conversion to lower precision must be opt-in and must state which corpus defines exactness.

A build writes a new index artifact. It must not overwrite source embeddings. Verify corpus identity and compare exact search with the caller's canonical flat baseline before switching readers.

Version migrations write a separate destination and validate it before publication. No silent encoder replacement, automatic corpus download, or destructive in-place upgrade.

No importers, converters, or migration commands are implemented yet.
