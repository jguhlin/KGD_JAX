# KGD Hypothesis Testing Log

## 2025-12-31
- Added CPU-only JAX preflight assertion for tests via `tests/jax_preflight.py` and imports in test modules.
- Fixed `popG` diagonal adjustment to avoid division by zero for single-individual populations; leave diagonal unchanged when pop size <= 1. (`kgd_jax/popgen.py`)
- Fixed `qc.run_qc` to handle cases where all samples or SNPs are filtered out (avoid reductions on empty arrays; return empty outputs). (`kgd_jax/qc.py`)
- Hardened `hw_pops` depth-adjusted x2* to avoid invalid multiply (mask non-finite ratios) and `fst_pairwise` mean/median to skip all-NaN vectors. (`kgd_jax/popgen.py`)
- Hardened `Nefromr2` to avoid divide-by-zero when r2 values align with 1/(beta*meanN); now returns NaN for undefined cases. (`kgd_jax/popgen.py`)
- `fst_gbs` now raises a ValueError when all genotypes are NaN (instead of emitting invalid divisions). (`kgd_jax/popgen.py`)
- `fst_pairwise` now errors early when all genotypes are NaN to avoid downstream invalid math. (`kgd_jax/popgen.py`)
- Avoided NaN-mean warnings in `mismatch_par_comb` and `mismatch_two_parents` by guarding nanmeans when all entries are NaN. (`kgd_jax/ped.py`)
- `best_parents_by_emm` now raises a clearer ValueError when all EMM values are NaN. (`kgd_jax/ped.py`)
