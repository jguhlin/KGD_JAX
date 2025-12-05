from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import zarr


@dataclass
class RAData:
    """Reference/alternate read counts in a matrix-free friendly layout.

    Shapes follow the KGD conventions but oriented for fast JAX usage:
    - n_ind:   number of individuals / samples
    - n_snp:   number of SNPs
    - ref[i,s] / alt[i,s]: read counts
    """

    sample_ids: List[str]
    chrom: np.ndarray
    pos: np.ndarray
    ref: np.ndarray  # shape (n_ind, n_snp), int32
    alt: np.ndarray  # shape (n_ind, n_snp), int32

    @property
    def n_ind(self) -> int:
        return int(self.ref.shape[0])

    @property
    def n_snp(self) -> int:
        return int(self.ref.shape[1])


def read_ra_tab(path: str | Path) -> RAData:
    """Read a KGD-style .ra.tab file into RAData.

    The expected format matches orig_kgd/vcf2ra.py:
    - tab-delimited
    - columns: CHROM, POS, sample1, sample2, ...
    - each sample column is 'ref,alt'.
    """
    path = Path(path)
    df = pd.read_table(path, dtype=str)
    if df.shape[1] < 3:
        raise ValueError(f"RA file {path} must have at least CHROM, POS and one sample column")

    chrom = df.iloc[:, 0].to_numpy()
    pos = df.iloc[:, 1].to_numpy()
    sample_cols: Sequence[str] = list(df.columns[2:])
    n_snp = df.shape[0]
    n_ind = len(sample_cols)

    ref = np.zeros((n_ind, n_snp), dtype=np.int32)
    alt = np.zeros((n_ind, n_snp), dtype=np.int32)

    # Loop over samples (columns), vectorised over SNPs.
    for i, col in enumerate(sample_cols):
        split_vals = df[col].str.split(",", n=1, expand=True)
        if split_vals.shape[1] != 2:
            raise ValueError(f"Column {col} in {path} is not in 'ref,alt' format")
        ref[i, :] = split_vals.iloc[:, 0].astype("int32").to_numpy()
        alt[i, :] = split_vals.iloc[:, 1].astype("int32").to_numpy()

    return RAData(
        sample_ids=list(sample_cols),
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
    )


def write_ra_store(ra: RAData, store_path: str | Path) -> None:
    """Write RAData to a Zarr store for fast, JAX-friendly I/O.

    Layout:
      - chrom:      (n_snp,)
      - pos:        (n_snp,)
      - sample_ids: (n_ind,)
      - ref:        (n_ind, n_snp)
      - alt:        (n_ind, n_snp)

    Compression:
      - ref/alt use blosc+zstd with moderate compression,
        chunked primarily along the SNP dimension.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(store_path.as_posix(), mode="w")

    n_ind = ra.n_ind
    n_snp = ra.n_snp

    # Metadata / 1D arrays.
    sample_ids_arr = np.asarray(ra.sample_ids, dtype="U")
    g.create_dataset(
        "sample_ids",
        data=sample_ids_arr,
        compressor=None,
        overwrite=True,
    )
    g.create_dataset(
        "chrom",
        data=ra.chrom,
        chunks=(min(n_snp, 4096),),
        overwrite=True,
    )
    g.create_dataset(
        "pos",
        data=ra.pos,
        chunks=(min(n_snp, 4096),),
        overwrite=True,
    )

    # 2D count matrices: chunk along SNPs for column-wise access.
    snp_chunk = min(n_snp, 1024)
    g.create_dataset(
        "ref",
        data=ra.ref,
        chunks=(n_ind, snp_chunk),
        overwrite=True,
    )
    g.create_dataset(
        "alt",
        data=ra.alt,
        chunks=(n_ind, snp_chunk),
        overwrite=True,
    )


def read_ra_store(store_path: str | Path) -> RAData:
    """Read a Zarr RA store written by write_ra_store into RAData."""
    store_path = Path(store_path)
    g = zarr.open_group(store_path.as_posix(), mode="r")

    sample_ids = [str(s) for s in np.asarray(g["sample_ids"][:])]
    chrom = np.asarray(g["chrom"][:])
    pos = np.asarray(g["pos"][:])
    ref = np.asarray(g["ref"][:], dtype=np.int32)
    alt = np.asarray(g["alt"][:], dtype=np.int32)

    return RAData(
        sample_ids=sample_ids,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
    )


def select_samples(ra: RAData, sample_indices: Sequence[int]) -> RAData:
    """Return an RAData view restricted to a subset of samples."""
    idx = np.asarray(sample_indices, dtype=int)
    return RAData(
        sample_ids=[ra.sample_ids[i] for i in idx],
        chrom=ra.chrom,
        pos=ra.pos,
        ref=ra.ref[idx, :],
        alt=ra.alt[idx, :],
    )


def select_snps(ra: RAData, snp_indices: Sequence[int]) -> RAData:
    """Return an RAData view restricted to a subset of SNPs."""
    idx = np.asarray(snp_indices, dtype=int)
    return RAData(
        sample_ids=ra.sample_ids,
        chrom=ra.chrom[idx],
        pos=ra.pos[idx],
        ref=ra.ref[:, idx],
        alt=ra.alt[:, idx],
    )
