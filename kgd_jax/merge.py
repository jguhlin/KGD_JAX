from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .io import RAData


@dataclass
class MergeMapping:
    """Mapping from original samples to merged sample IDs."""

    sample_ids: np.ndarray  # original sample IDs
    merge_ids: np.ndarray   # same length, new group IDs


def load_merge_mapping(csv_path: str) -> MergeMapping:
    """Load a sample→merge mapping from CSV with columns sample_id, merge_id."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "sample_id" in cols:
        sample_col = cols["sample_id"]
    elif "sample" in cols:
        sample_col = cols["sample"]
    else:
        raise ValueError("Merge mapping must have 'sample_id' or 'sample' column.")

    if "merge_id" in cols:
        merge_col = cols["merge_id"]
    elif "group" in cols:
        merge_col = cols["group"]
    else:
        raise ValueError("Merge mapping must have 'merge_id' or 'group' column.")

    sample_ids = df[sample_col].astype(str).to_numpy()
    merge_ids = df[merge_col].astype(str).to_numpy()
    return MergeMapping(sample_ids=sample_ids, merge_ids=merge_ids)


def merge_ra_samples(ra: RAData, mapping: MergeMapping) -> RAData:
    """Merge RA samples according to a mapping (technical replicates).

    - All samples in the mapping must appear in ra.sample_ids.
    - Samples not present in the mapping are kept as their own group.
    """
    sample_ids = np.array(ra.sample_ids, dtype=str)
    map_dict: Dict[str, str] = {
        s: g for s, g in zip(mapping.sample_ids, mapping.merge_ids)
    }

    # Determine group for each sample; default to its own ID if not mapped.
    group_ids = np.array(
        [map_dict.get(s, s) for s in sample_ids], dtype=str
    )

    uniq_groups, inv = np.unique(group_ids, return_inverse=True)
    n_groups = uniq_groups.shape[0]
    n_snp = ra.ref.shape[1]

    ref_merged = np.zeros((n_groups, n_snp), dtype=np.int32)
    alt_merged = np.zeros((n_groups, n_snp), dtype=np.int32)

    # Sum ref/alt counts per group.
    for g in range(n_groups):
        mask = inv == g
        if not np.any(mask):
            continue
        ref_merged[g, :] = ra.ref[mask, :].sum(axis=0)
        alt_merged[g, :] = ra.alt[mask, :].sum(axis=0)

    return RAData(
        sample_ids=list(uniq_groups),
        chrom=ra.chrom,
        pos=ra.pos,
        ref=ref_merged,
        alt=alt_merged,
    )

