# KGD JAX

🚧 **IN DEVELOPMENT** — interfaces may change; verify outputs. Maintainer: **Joseph Guhlin**.

KGD core algorithms rewritten for JAX with a compact, chunked store (`*.kgd.zarr`) for fast I/O. Use pixi to keep environments reproducible.

## Quick start

```bash
echo "🚧 IN DEVELOPMENT"
pixi install            # set up the environment
pixi run kgd-jax --help # list CLI commands
```

## File formats

- **VCF**: standard variant call format with depth fields.
- **RA tab (`*.ra.tab`)**: KGD text layout, columns `CHROM POS sample1 sample2 ...`, sample cells are `ref,alt` read counts.
- **KGD-JAX store (`*.kgd.zarr`)**: compressed Zarr group storing `chrom`, `pos`, `sample_ids`, `ref`, `alt` (JAX-friendly, column-chunked).

## Common workflows

### 1) VCF → KGD-JAX store

Requires the original `vcf2ra.py` from the KGD project at `orig_kgd/vcf2ra.py` (same directory layout as upstream KGD).

```bash
echo "🚧 IN DEVELOPMENT"
pixi run kgd-jax vcf2kgd input.vcf --store data.kgd.zarr --keep-ra  # keep intermediate .ra.tab if you want to inspect it
```

What happens:
- `vcf2ra.py` is called to produce `input.vcf.ra.tab`.
- The RA tab is converted to `data.kgd.zarr` via `kgd_jax.io.write_ra_store`.
- The intermediate RA is deleted unless `--keep-ra` is set.

### 2) RA tab → KGD-JAX store (no VCF needed)

Use the in-repo I/O helpers directly:

```bash
echo "🚧 IN DEVELOPMENT"
pixi run python - <<'PY'
from kgd_jax import io

ra = io.read_ra_tab("data.ra.tab")
io.write_ra_store(ra, "data.kgd.zarr")
print("wrote data.kgd.zarr")
PY
```

### 3) KGD-JAX store → RA tab (round-trip check)

Useful for validating parity with the original text format:

```bash
echo "🚧 IN DEVELOPMENT"
pixi run python - <<'PY'
from pathlib import Path
from kgd_jax import io

store = "data.kgd.zarr"
out = Path("data.roundtrip.ra.tab")

ra = io.read_ra_store(store)
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w") as fh:
    header = ["CHROM", "POS"] + ra.sample_ids
    fh.write("\t".join(header) + "\n")
    for idx in range(ra.n_snp):
        row = [str(ra.chrom[idx]), str(ra.pos[idx])]
        for j in range(ra.n_ind):
            row.append(f"{int(ra.ref[j, idx])},{int(ra.alt[j, idx])}")
        fh.write("\t".join(row) + "\n")

print(f"wrote {out}")
PY
```

### 4) KGD-JAX store → GRM / metrics (Python snippet)

You can compute G5 diagonal (inbreeding) or other metrics straight from the Zarr store without rehydrating a `.ra.tab` file:

```bash
echo "🚧 IN DEVELOPMENT"
pixi run python - <<'PY'
import numpy as np
from kgd_jax import io, qc, grm

store = "data.kgd.zarr"
ra = io.read_ra_store(store)

qc_res = qc.run_qc(ra)
op = grm.build_grm_operator(
    depth=qc_res.depth,
    genon=qc_res.genon,
    p=qc_res.p,
    dmodel="bb",
    dparam=float("inf"),
)

G5_diag = np.asarray(op.diag_G5())
print("G5 diag shape:", G5_diag.shape)
print("first 5 values:", G5_diag[:5])
PY
```

## CLI command cheat sheet

All commands are launched with `pixi run kgd-jax <command> ...` (prefix with the 🚧 banner as above).

- `diag`: compute depth-aware G5 diagonal (inbreeding) from RA. Example: `pixi run kgd-jax diag data.ra.tab --out diag.csv`.
- `block`: small G4 block for selected samples without building the full GRM. Example: `--samples S1 S2 S3 --out block.csv`.
- `het`: heterozygosity summary (matrix-free). Example: `pixi run kgd-jax het data.ra.tab --out het.csv`.
- `hw`: Hardy–Weinberg diagnostics/LRT; optional populations file. Example: `pixi run kgd-jax hw data.ra.tab --pop-file pops.csv --out hw.csv`.
- `fst`: depth-aware Fst across populations. Example: `pixi run kgd-jax fst data.ra.tab --pop-file pops.csv --out fst.csv`.
- `dapc`: DAPC on genotype PCs. Example: `pixi run kgd-jax dapc data.ra.tab --pop-file pops.csv --out dapc.csv`.
- `fit-depth`: fit depth-model parameter (bb/modp) to target inbreeding. Example: `pixi run kgd-jax fit-depth data.ra.tab targets.csv --dmodel bb --out fit.csv`.
- `snpselect`: pick SNP pairs for LD/Ne analysis. Example: `pixi run kgd-jax snpselect data.ra.tab --out snp_pairs.csv`.
- `merge-ra`: merge RA samples by mapping file (technical replicates). Example: `pixi run kgd-jax merge-ra data.ra.tab map.csv --out merged.ra.tab`.
- `ped`: depth-aware parentage verification given pedigree CSV. Example: `pixi run kgd-jax ped data.ra.tab pedigree.csv --out ped_check.csv`.
- `ped-best`: best/second-best parent search across all candidates. Example: `pixi run kgd-jax ped-best data.ra.tab pedigree.csv --out ped_best.csv`.
- `plot-fst`, `plot-hw`, `plot-dapc`: plotting helpers for outputs of `fst`, `hw`, `dapc`. Example: `pixi run kgd-jax plot-fst fst.csv --out fst.png`.
- `simulate`: generate chip + RA + Zarr for cross-checks. Example: `pixi run kgd-jax simulate --n-ind 200 --n-snp 2000 --prefix sim_run`.
- `vcf2kgd`: VCF → RA (via `orig_kgd/vcf2ra.py`) → Zarr. Example: `pixi run kgd-jax vcf2kgd input.vcf --store data.kgd.zarr`.
- `run`: end-to-end QC + GRM diag + popgen outputs. Example: `pixi run kgd-jax run data.ra.tab --prefix results/run1`.

## Tips

- Keep the compressed `.kgd.zarr` under version control if it is small enough; otherwise regenerate from the source VCF/RA. The repo `.gitignore` already skips large stores by default.
- For end-to-end analysis on a tab file: `pixi run kgd-jax run data.ra.tab --prefix results/run1` (see `docs/usage.md` for options).
