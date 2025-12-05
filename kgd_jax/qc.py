from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .io import RAData


@dataclass
class QCPipelineResult:
    """Outputs of the ingestion + QC stage."""

    # masks into the original RA matrices
    keep_ind: np.ndarray  # shape (n_ind,), bool
    keep_snp: np.ndarray  # shape (n_snp,), bool

    # depth and genotypes after QC, JAX arrays
    depth: jnp.ndarray  # (n_ind_keep, n_snp_keep), float32
    genon: jnp.ndarray  # (n_ind_keep, n_snp_keep), float32 with nan for missing

    # allele frequencies after QC, 1D over SNPs
    p: jnp.ndarray  # (n_snp_keep,), float32


def alleles_to_depth_genon(
    ref: np.ndarray,
    alt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Implement alleles2g (Section 2.2 of KGD_CORE) in numpy.

    Returns:
        depth: (n_ind, n_snp) float32
        genon: (n_ind, n_snp) float32 with NaN for missing (depth == 0).
    """
    ref = ref.astype(np.float32, copy=False)
    alt = alt.astype(np.float32, copy=False)
    depth = ref + alt

    # Hard genotype calls based on ref / depth, mirroring KGD alleles2g:
    #   g_cont = ref / depth  (in [0,1])
    #   g = trunc(2 * g_cont - 1) + 1  in {0,1,2}
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_ref = np.where(depth > 0, ref / depth, np.nan)
        g = np.floor(2.0 * frac_ref - 1.0) + 1.0
        g[depth == 0] = np.nan

    return depth.astype(np.float32), g.astype(np.float32)


def calcp_alleles(depth_ref: np.ndarray, depth_alt: np.ndarray) -> np.ndarray:
    """Allele-frequency estimator using read counts (pmethod='A')."""
    num = depth_ref.sum(axis=0, dtype=np.float64)
    den = (depth_ref + depth_alt).sum(axis=0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = num / den
    p[~np.isfinite(p)] = np.nan
    return p.astype(np.float32)


def calcp_genotypes(genon: np.ndarray) -> np.ndarray:
    """Allele-frequency estimator using hard calls (pmethod='G')."""
    mask = ~np.isnan(genon)
    num = np.nansum(genon, axis=0, dtype=np.float64)
    den = 2.0 * mask.sum(axis=0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = num / den
    p[~np.isfinite(p)] = np.nan
    return p.astype(np.float32)


def run_qc(
    ra: RAData,
    sampdepth_thresh: float = 0.01,
    snpdepth_thresh: float = 0.01,
    maf_thresh: float = 1e-9,
    pmethod: str = "A",
) -> QCPipelineResult:
    """Approximate GBSsummary + calcp from KGD in numpy, output JAX arrays.

    This mirrors the core filtering logic:
    - remove samples with max depth 0 or 1, or mean depth < sampdepth_thresh
    - compute allele frequencies p
    - remove SNPs with MAF < maf_thresh or mean depth < snpdepth_thresh
    - iterate until stable
    """
    ref = ra.ref.astype(np.float32, copy=False)
    alt = ra.alt.astype(np.float32, copy=False)

    n_ind, n_snp = ref.shape
    keep_ind = np.ones(n_ind, dtype=bool)
    keep_snp = np.ones(n_snp, dtype=bool)

    changed = True
    while changed:
        changed = False

        ref_sub = ref[keep_ind][:, keep_snp]
        alt_sub = alt[keep_ind][:, keep_snp]
        depth_sub = ref_sub + alt_sub

        # Sample QC
        sampdepth_max = depth_sub.max(axis=1)
        sampdepth_mean = depth_sub.mean(axis=1)

        drop_ind = (sampdepth_max == 0) | (sampdepth_max == 1) | (
            sampdepth_mean < sampdepth_thresh
        )
        if drop_ind.any():
            changed = True
            drop_idx = np.where(keep_ind)[0][drop_ind]
            keep_ind[drop_idx] = False
            continue  # recompute on updated masks

        # SNP QC
        if pmethod.upper() == "A":
            p_sub = calcp_alleles(ref_sub, alt_sub)
        else:
            _, genon_sub = alleles_to_depth_genon(ref_sub, alt_sub)
            p_sub = calcp_genotypes(genon_sub)

        maf = np.minimum(p_sub, 1.0 - p_sub)
        snpdepth = depth_sub.mean(axis=0)
        drop_snp = (maf < maf_thresh) | (snpdepth < snpdepth_thresh) | np.isnan(maf)
        if drop_snp.any():
            changed = True
            drop_idx = np.where(keep_snp)[0][drop_snp]
            keep_snp[drop_idx] = False

    # Final depth / genotypes / p on kept entries.
    ref_final = ref[keep_ind][:, keep_snp]
    alt_final = alt[keep_ind][:, keep_snp]
    depth_final_np, genon_final_np = alleles_to_depth_genon(ref_final, alt_final)

    if pmethod.upper() == "A":
        p_final_np = calcp_alleles(ref_final, alt_final)
    else:
        p_final_np = calcp_genotypes(genon_final_np)

    depth_jax = jnp.asarray(depth_final_np, dtype=jnp.float32)
    genon_jax = jnp.asarray(genon_final_np, dtype=jnp.float32)
    p_jax = jnp.asarray(p_final_np, dtype=jnp.float32)

    return QCPipelineResult(
        keep_ind=keep_ind,
        keep_snp=keep_snp,
        depth=depth_jax,
        genon=genon_jax,
        p=p_jax,
    )
