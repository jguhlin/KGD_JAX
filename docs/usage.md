# Command Line Interface

All CLI commands are accessible through pixi:

```bash
pixi run kgd-jax --help
```

## Core GRM and QC

- `diag`: compute the depth-aware `G5` diagonal.
- `inbreed`: inbreeding coefficients `F_hat = G5_diag - 1`.
- `run`: end-to-end pipeline from `.ra.tab` to a collection of CSV outputs
  (`G5diag`, `inbreed`, `het`, `hw`, `fst`, `dapc`, `ped` if pedigree is provided).

Example:

```bash
pixi run kgd-jax run data.ra.tab --prefix results/run1 --pop-file pops.csv --ped-file Ped-GBS.csv
```

## Population Genetics

- `het`: depth-aware heterozygosity summary.
- `hw`: Hardy–Weinberg diagnostics per population and SNP.
- `fst`: Fst across populations.
- `dapc`: DAPC on genotype PCs.
- `fit-depth`: fit depth-model parameters (e.g. beta-binomial alpha) to inbreeding targets.
- `snpselect`: select SNP pairs for LD/Ne analysis.

## Parentage and Merging

- `ped`: depth-aware parentage checks for recorded parents.
- `merge-ra`: merge RA samples according to a mapping file (technical replicates).

## Plotting Helpers

- `plot-fst`: Manhattan-style plot of Fst.
- `plot-hw`: HW disequilibrium vs MAF scatter.
- `plot-dapc`: scatter plot of DAPC discriminant coordinates.

