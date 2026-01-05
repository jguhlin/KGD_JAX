from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import zarr


@dataclass
class RAData:
    """Reference/alternate read counts in a matrix-free friendly layout."""

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


class LazyRAData:
    """Provider for RA data that loads/generates chunks on demand."""

    def __init__(
        self,
        sample_ids: List[str],
        chrom: np.ndarray,
        pos: np.ndarray,
        ref_provider: Any,
        alt_provider: Any,
    ):
        self.sample_ids = sample_ids
        self.chrom = chrom
        self.pos = pos
        self.ref_provider = ref_provider
        self.alt_provider = alt_provider
        self.n_ind = len(sample_ids)
        self.n_snp = len(chrom)

    def get_chunk(
        self, start_snp: int, end_snp: int, start_ind: int = 0, end_ind: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if end_ind is None:
            end_ind = self.n_ind
        ref = self.ref_provider[start_ind:end_ind, start_snp:end_snp]
        alt = self.alt_provider[start_ind:end_ind, start_snp:end_snp]
        return np.asarray(ref), np.asarray(alt)


def create_lazy_ra_store(store_path: str | Path) -> LazyRAData:
    """Open a Zarr store for lazy access."""
    store_path = Path(store_path)
    g = zarr.open_group(store_path.as_posix(), mode="r")
    sample_ids = [str(s) for s in np.asarray(g["sample_ids"][:])]
    chrom = np.asarray(g["chrom"][:])
    pos = np.asarray(g["pos"][:])
    return LazyRAData(sample_ids, chrom, pos, g["ref"], g["alt"])


class PRNGProvider:
    """Deterministic PRNG provider for genotypes."""

    def __init__(self, n_ind: int, n_snp: int, seed: int = 42, is_ref: bool = True):
        self.n_ind = n_ind
        self.n_snp = n_snp
        self.seed = seed
        self.is_ref = is_ref

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Indexing must be [ind_slice, snp_slice]")

        ind_slice, snp_slice = key
        start_ind, stop_ind, step_ind = ind_slice.indices(self.n_ind)
        start_snp, stop_snp, step_snp = snp_slice.indices(self.n_snp)

        shape = (stop_ind - start_ind, stop_snp - start_snp)
        res = np.zeros(shape, dtype=np.int32)
        
        # Uniqueness per individual AND SNP.
        # We use a combined seed for the individual+SNP block.
        for i in range(shape[0]):
            global_ind = start_ind + i
            # Seed depends on individual AND start_snp to ensure determinism across chunks
            rng = np.random.default_rng(self.seed + global_ind * 1000000 + start_snp + (1 if self.is_ref else 2))
            res[i, :] = rng.integers(0, 21, size=shape[1])
            
        return res


def create_lazy_simulation(n_ind: int, n_snp: int, seed: int = 42) -> LazyRAData:
    sample_ids = [f"S{i}" for i in range(n_ind)]
    chrom = np.array(["1"] * n_snp)
    pos = np.arange(n_snp)
    return LazyRAData(
        sample_ids,
        chrom,
        pos,
        PRNGProvider(n_ind, n_snp, seed, is_ref=True),
        PRNGProvider(n_ind, n_snp, seed, is_ref=False),
    )


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
