# KGD JAX

KGD JAX is a Python/JAX implementation of the core algorithms from the KGD project:

- Depth-aware genomic relationship matrices (`G5`) built in a matrix-free way.
- GBS/QC pipeline (sample and SNP filtering, allele frequencies).
- Population-genetic utilities (heterozygosity, Hardy–Weinberg, Fst, DAPC, Ne).
- Parentage verification routines using depth-aware mismatch expectations.

The implementation is designed to:

- Use JAX for numerically heavy operations.
- Avoid constructing full `n_ind × n_snp` or `n_ind × n_ind` matrices where possible.
- Interoperate with standard Python tools (pandas, matplotlib).

Most functionality is exposed via the `kgd-jax` command-line interface, which is wired
through `pixi` for reproducible environments.

