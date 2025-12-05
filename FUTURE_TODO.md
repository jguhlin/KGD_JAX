# Future Improvements for KGD-JAX GRM Pipeline

- **JIT warmup & staging**: Pre-JIT `diag_G5` and `submatrix_G4` once (tiny warmup call) and reuse compiled functions to avoid cold-start cost on big runs.

- **Chunked Zarr streaming**: Drive QC and GRM directly from the Zarr store in SNP-chunks (e.g., 4–8k SNPs), so memory stays flat even for very large m; avoid loading full ref/alt arrays at once.

- **Mixed precision**: Keep counts in `int32`, do GRM arithmetic in `float32`, and accumulate diagonals in `float64` to halve bandwidth while preserving accuracy.

- **Per-chrom / block parallelism**: Split SNPs by chromosome and reduce per-chrom in parallel (threads or JAX pmap on CPU), summing partial numerators/denominators to speed tcrossprod-heavy parts.

- **Cache K(depth)**: Precompute `depth2K(depth)` per chunk during QC and reuse in both diag and off-diag paths, avoiding redundant K computations.

- **Matrix-free PCA/eigensolvers**: Provide a randomized SVD/Lanczos wrapper over `GRMOperator` (`G·v` only) for top PCs of G, avoiding dense G formation.

- **Block scheduling & dtype options**: For writing large GRM blocks to disk, schedule tiles to reuse SNP chunks (striped/Morton order) and allow float32 blocks for visualization to cut I/O and memory.

- **GPU guardrails**: Dynamically cap GRM block sizes based on available HBM and fall back to CPU for oversized tiles to prevent GPU OOMs.
