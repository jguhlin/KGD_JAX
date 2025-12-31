from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .grm import GRMOperator, Depth2KFn


@dataclass
class PairMismatchResult:
    mmrate: float
    ncompare: int
    exp_mmrate: float


def mismatch_par_pair(
    depth: np.ndarray,
    genon: np.ndarray,
    p: np.ndarray,
    depth2K: Depth2KFn,
    i: int,
    j: int,
    mindepth_mm: float = 1.0,
) -> PairMismatchResult:
    """Single-parent mismatch stats (port of mismatch.par for one offspring-parent pair).

    Args:
        depth: depth matrix (n_ind, n_snp).
        genon: genotype matrix (n_ind, n_snp) with NaN for missing.
        p: allele frequencies per SNP (n_snp,).
        depth2K: depth-to-K function (e.g. from make_depth2K).
        i: offspring index (row in depth/genon).
        j: parent index (row in depth/genon).
        mindepth_mm: minimum depth for both individuals to use a SNP.
    """
    depth_i = depth[i, :]
    depth_j = depth[j, :]

    mask = (depth_i >= mindepth_mm) & (depth_j >= mindepth_mm)
    ncompare = int(mask.sum())
    if ncompare == 0:
        return PairMismatchResult(mmrate=np.nan, ncompare=0, exp_mmrate=np.nan)

    pi = genon[i, mask] / 2.0
    pj = genon[j, mask] / 2.0

    Ko = np.asarray(depth2K(depth_i[mask]), dtype=np.float64)
    Kp = np.asarray(depth2K(depth_j[mask]), dtype=np.float64)

    nmismatch = int(np.sum(np.abs(pi - pj) == 1.0))

    ptemp = p[mask].astype(np.float64)
    P = ptemp * (1.0 - ptemp)

    expmm = np.full(ptemp.shape, np.nan, dtype=np.float64)

    ug = pi == 1.0
    if np.any(ug):
        num = P[ug] * (ptemp[ug] * Kp[ug] + (1.0 - ptemp[ug]) * Ko[ug] + Kp[ug] * Ko[ug])
        den = ptemp[ug] ** 2 + 2.0 * P[ug] * Ko[ug]
        with np.errstate(divide="ignore", invalid="ignore"):
            expmm[ug] = num / den

    ug = pi == 0.5
    if np.any(ug):
        expmm[ug] = 0.0

    ug = pi == 0.0
    if np.any(ug):
        num = P[ug] * (ptemp[ug] * Ko[ug] + (1.0 - ptemp[ug]) * Kp[ug] + Kp[ug] * Ko[ug])
        den = (1.0 - ptemp[ug]) ** 2 + 2.0 * P[ug] * Ko[ug]
        with np.errstate(divide="ignore", invalid="ignore"):
            expmm[ug] = num / den

    exp_mmrate = float(np.nanmean(expmm))
    mmrate = float(nmismatch / float(ncompare))
    return PairMismatchResult(mmrate=mmrate, ncompare=ncompare, exp_mmrate=exp_mmrate)


@dataclass
class MatrixMismatchResult:
    """Mismatch results for multiple offspring × parent combinations."""

    mmrate: np.ndarray  # shape (n_offspring, n_parents)
    ncompare: np.ndarray  # same shape
    exp_mmrate: np.ndarray  # same shape


@dataclass
class TwoParentMismatchResult:
    mmrate: np.ndarray  # shape (n_offspring,)
    ncompare: np.ndarray
    exp_mmrate: np.ndarray


@dataclass
class BestParentResult:
    """Best and second-best parents for each offspring."""

    offspring_idx: np.ndarray
    best_parent_idx: np.ndarray
    second_parent_idx: np.ndarray
    rel_best: np.ndarray
    rel_second: np.ndarray
    rel_parents: np.ndarray  # relatedness between best and second parent


def mismatch_par_comb(
    depth: np.ndarray,
    genon: np.ndarray,
    p: np.ndarray,
    depth2K: Depth2KFn,
    offspring_idx: Sequence[int],
    parent_idx: Sequence[int],
    snpsubset: Optional[Sequence[int]] = None,
    mindepth_mm: float = 1.0,
) -> MatrixMismatchResult:
    """All offspring × parent combinations (analogue of mismatch.par.comb).

    Args:
        depth: depth matrix (n_ind, n_snp).
        genon: genotype matrix (n_ind, n_snp) with NaN for missing.
        p: allele frequencies vector (n_snp,).
        depth2K: depth-to-K function.
        offspring_idx: indices of offspring rows.
        parent_idx: indices of parent rows.
        snpsubset: optional SNP indices to use.
        mindepth_mm: minimum depth for mismatch calculations.
    """
    depth = np.asarray(depth)
    genon = np.asarray(genon)
    p = np.asarray(p)

    offspring_idx = np.asarray(offspring_idx, dtype=int)
    parent_idx = np.asarray(parent_idx, dtype=int)
    n_off = offspring_idx.size
    n_par = parent_idx.size
    n_snp = depth.shape[1]

    if snpsubset is None:
        snps = np.arange(n_snp, dtype=int)
    else:
        snps = np.asarray(snpsubset, dtype=int)

    mmrate = np.full((n_off, n_par), np.nan, dtype=float)
    ncompare = np.zeros((n_off, n_par), dtype=int)
    expmm_all = np.full((n_off, n_par), np.nan, dtype=float)

    depth_par = depth[parent_idx, :]
    Kp_all = np.asarray(depth2K(depth_par), dtype=float)

    for io, o in enumerate(offspring_idx):
        depth_i = depth[o, :]
        usnp = snps[depth_i[snps] >= mindepth_mm]
        if usnp.size == 0:
            continue

        pi = genon[o, usnp] / 2.0
        pj = genon[parent_idx[:, None], usnp] / 2.0  # (n_par, n_snp_sel)

        Ko = np.asarray(depth2K(depth_i[usnp]), dtype=float)
        Kp = Kp_all[:, usnp]

        # Observed mismatches and comparisons.
        mismatch_mask = np.abs(pj - pi[None, :]) == 1.0
        nmismatch = mismatch_mask.sum(axis=1)
        ncmp = np.sum(~np.isnan(pj), axis=1)

        ptemp = p[usnp].astype(float)
        P = ptemp * (1.0 - ptemp)

        expmm = np.full((n_par, usnp.size), np.nan, dtype=float)

        # pi == 1
        ug = np.where(pi == 1.0)[0]
        if ug.size > 0:
            num = (
                P[ug]
                * (
                    ptemp[ug] * Kp[:, ug]
                    + (1.0 - ptemp[ug]) * Ko[ug][None, :]
                    + Kp[:, ug] * Ko[ug][None, :]
                )
            )
            den = ptemp[ug] ** 2 + 2.0 * P[ug] * Ko[ug]
            with np.errstate(divide="ignore", invalid="ignore"):
                expmm[:, ug] = num / den

        # pi == 0.5
        ug = np.where(pi == 0.5)[0]
        if ug.size > 0:
            expmm[:, ug] = 0.0

        # pi == 0
        ug = np.where(pi == 0.0)[0]
        if ug.size > 0:
            num = (
                P[ug]
                * (
                    ptemp[ug] * Ko[ug][None, :]
                    + (1.0 - ptemp[ug]) * Kp[:, ug]
                    + Kp[:, ug] * Ko[ug][None, :]
                )
            )
            den = (1.0 - ptemp[ug]) ** 2 + 2.0 * P[ug] * Ko[ug]
            with np.errstate(divide="ignore", invalid="ignore"):
                expmm[:, ug] = num / den

        # Mask SNPs with low depth / NA genotypes.
        depth_par_usnp = depth_par[:, usnp]
        expmm[np.isnan(pj) | (depth_par_usnp < mindepth_mm)] = np.nan

        valid = np.sum(np.isfinite(expmm), axis=1)
        exp_mmrate = np.full(n_par, np.nan, dtype=float)
        with np.errstate(invalid="ignore"):
            if np.any(valid > 0):
                exp_mmrate[valid > 0] = np.nanmean(expmm[valid > 0], axis=1)

        mmrate[io, :] = nmismatch / np.where(ncmp == 0, np.nan, ncmp)
        ncompare[io, :] = ncmp
        expmm_all[io, :] = exp_mmrate

    return MatrixMismatchResult(mmrate=mmrate, ncompare=ncompare, exp_mmrate=expmm_all)


def mismatch_two_parents(
    depth: np.ndarray,
    genon: np.ndarray,
    p: np.ndarray,
    depth2K: Depth2KFn,
    offspring_idx: Sequence[int],
    parent1_idx: Sequence[int],
    parent2_idx: Sequence[int],
    snpsubset: Optional[Sequence[int]] = None,
    mindepth_mm: float = 1.0,
    doublemm: bool = False,
) -> TwoParentMismatchResult:
    """Mismatch stats for offspring vs parent pairs (mismatch.2par analogue)."""
    depth = np.asarray(depth)
    genon = np.asarray(genon)
    p = np.asarray(p)

    offspring_idx = np.asarray(offspring_idx, dtype=int)
    parent1_idx = np.asarray(parent1_idx, dtype=int)
    parent2_idx = np.asarray(parent2_idx, dtype=int)
    n_off = offspring_idx.size

    n_snp = depth.shape[1]
    if snpsubset is None:
        snps = np.arange(n_snp, dtype=int)
    else:
        snps = np.asarray(snpsubset, dtype=int)

    mmrate = np.full(n_off, np.nan, dtype=float)
    ncompare = np.zeros(n_off, dtype=int)
    exp_mmrate = np.full(n_off, np.nan, dtype=float)

    for io in range(n_off):
        o = offspring_idx[io]
        p1 = parent1_idx[io]
        p2 = parent2_idx[io]

        depth_i = depth[o, :]
        depth_j = depth[p1, :]
        depth_k = depth[p2, :]

        usnp = snps[np.minimum.reduce([depth_i[snps], depth_j[snps], depth_k[snps]]) >= mindepth_mm]
        if usnp.size == 0:
            continue

        pi = genon[o, usnp] / 2.0
        pj = genon[p1, usnp] / 2.0
        pk = genon[p2, usnp] / 2.0

        Ko = np.asarray(depth2K(depth_i[usnp]), dtype=float)
        Kf = np.asarray(depth2K(depth_j[usnp]), dtype=float)
        Km = np.asarray(depth2K(depth_k[usnp]), dtype=float)

        ptemp = p[usnp].astype(float)
        P = ptemp * (1.0 - ptemp)

        expmm = np.full(usnp.size, np.nan, dtype=float)

        # Formulas follow mismatch.2par (Dodds et al. 2019), ported literally.
        if not doublemm:
            ug = np.where(pi == 1.0)[0]
            if ug.size > 0:
                num = (
                    ptemp[ug] ** 2
                    * P[ug]
                    * (Km[ug] + Kf[ug])
                    * (1.0 + Ko[ug])
                    + P[ug] ** 2
                    * (
                        2 * Ko[ug]
                        + Km[ug]
                        + Kf[ug]
                        - Kf[ug] * Km[ug]
                        + 2 * Km[ug] * Ko[ug]
                        + 2 * Kf[ug] * Ko[ug]
                        - 2 * Kf[ug] * Km[ug] * Ko[ug]
                    )
                    + 2 * P[ug] * (1.0 - ptemp[ug]) ** 2 * Ko[ug]
                )
                den = ptemp[ug] ** 2 + 2.0 * P[ug] * Ko[ug]
                with np.errstate(divide="ignore", invalid="ignore"):
                    expmm[ug] = num / den

            ug = np.where(pi == 0.5)[0]
            if ug.size > 0:
                expmm[ug] = ((1.0 - 2.0 * P[ug]) * (Km[ug] + Kf[ug]) + 4.0 * P[ug] * Kf[ug] * Km[ug]) / 2.0

            ug = np.where(pi == 0.0)[0]
            if ug.size > 0:
                num = (
                    2.0 * ptemp[ug] ** 2 * P[ug] * Ko[ug]
                    + P[ug] ** 2
                    * (
                        2 * Ko[ug]
                        + Km[ug]
                        + Kf[ug]
                        - Kf[ug] * Km[ug]
                        + 2 * Km[ug] * Ko[ug]
                        + 2 * Kf[ug] * Ko[ug]
                        - 2 * Kf[ug] * Km[ug] * Ko[ug]
                    )
                    + P[ug] * (1.0 - ptemp[ug]) ** 2 * (Km[ug] + Kf[ug]) * (1.0 + Ko[ug])
                )
                den = (1.0 - ptemp[ug]) ** 2 + 2.0 * P[ug] * Ko[ug]
                with np.errstate(divide="ignore", invalid="ignore"):
                    expmm[ug] = num / den
        else:
            ug = np.where(pi == 1.0)[0]
            if ug.size > 0:
                num = (
                    ptemp[ug] ** 2
                    * P[ug]
                    * (Km[ug] + Kf[ug])
                    * (1.0 + Ko[ug])
                    + P[ug] ** 2
                    * (
                        2 * Ko[ug]
                        + Km[ug]
                        + Kf[ug]
                        + 2 * Km[ug] * Ko[ug]
                        + 2 * Kf[ug] * Ko[ug]
                    )
                    + 2.0
                    * P[ug]
                    * (1.0 - ptemp[ug]) ** 2
                    * Ko[ug]
                    * (1.0 + Km[ug] + Kf[ug])
                )
                den = ptemp[ug] ** 2 + 2.0 * P[ug] * Ko[ug]
                with np.errstate(divide="ignore", invalid="ignore"):
                    expmm[ug] = num / den

            ug = np.where(pi == 0.5)[0]
            if ug.size > 0:
                expmm[ug] = ((1.0 - 2.0 * P[ug]) * (Km[ug] + Kf[ug]) + 4.0 * P[ug] * Kf[ug] * Km[ug]) / 2.0

            ug = np.where(pi == 0.0)[0]
            if ug.size > 0:
                num = (
                    2.0 * ptemp[ug] ** 2 * P[ug] * Ko[ug] * (1.0 + Km[ug] + Kf[ug])
                    + P[ug] ** 2
                    * (
                        2 * Ko[ug]
                        + Km[ug]
                        + Kf[ug]
                        + 2 * Km[ug] * Ko[ug]
                        + 2 * Kf[ug] * Ko[ug]
                    )
                    + P[ug] * (1.0 - ptemp[ug]) ** 2 * (Km[ug] + Kf[ug]) * (1.0 + Ko[ug])
                )
                den = (1.0 - ptemp[ug]) ** 2 + 2.0 * P[ug] * Ko[ug]
                with np.errstate(divide="ignore", invalid="ignore"):
                    expmm[ug] = num / den

        if np.any(np.isfinite(expmm)):
            with np.errstate(invalid="ignore"):
                exp_mmrate[io] = float(np.nanmean(expmm))

        # Observed mismatches.
        mismatch_mask = (np.abs(pi - pj) == 1.0) | (np.abs(pi - pk) == 1.0) | (
            (pj == pk) & (pj != 0.5) & (pi == 0.5)
        )
        nm = int(np.sum(mismatch_mask))
        if doublemm:
            nm += int(np.sum((np.abs(pi - pj) == 1.0) & (pj == pk)))

        mmrate[io] = nm / float(usnp.size)
        ncompare[io] = usnp.size

    return TwoParentMismatchResult(mmrate=mmrate, ncompare=ncompare, exp_mmrate=exp_mmrate)


def best_parents_by_relatedness(
    grm_op: GRMOperator,
    offspring_idx: Sequence[int],
    parent_idx: Sequence[int],
    depth_min: float = 0.0,
    depth_max: float = np.inf,
) -> BestParentResult:
    """Select best and second-best parents per offspring using G4 relatedness.

    This mirrors the 'rel' branch of bestmatch() in GBSPedAssign.R, but here we
    operate on index sets and a GRMOperator instead of R's G matrix directly.
    """
    offspring_idx = np.asarray(offspring_idx, dtype=int)
    parent_idx = np.asarray(parent_idx, dtype=int)

    if offspring_idx.size == 0 or parent_idx.size == 0:
        n = offspring_idx.size
        nan = np.full(n, np.nan)
        return BestParentResult(
            offspring_idx=offspring_idx,
            best_parent_idx=np.full(n, -1),
            second_parent_idx=np.full(n, -1),
            rel_best=nan,
            rel_second=nan,
            rel_parents=nan,
        )

    # Compute G4 block between offspring and candidate parents.
    G_block = np.asarray(
        grm_op.submatrix_G4(offspring_idx, parent_idx, depth_min=depth_min, depth_max=depth_max),
        dtype=float,
    )

    # Avoid self-parenting when offspring are also in parent_idx.
    # If they overlap in indices, set those entries to -inf.
    for i_row, oi in enumerate(offspring_idx):
        overlaps = np.where(parent_idx == oi)[0]
        if overlaps.size > 0:
            G_block[i_row, overlaps] = -np.inf

    # Best parent = argmax, second best = argmax after masking best.
    best_pos = np.argmax(G_block, axis=1)
    rel_best = G_block[np.arange(G_block.shape[0]), best_pos]

    # Mask out best parent and recompute argmax for second parent.
    G_temp = G_block.copy()
    G_temp[np.arange(G_temp.shape[0]), best_pos] = -np.inf
    second_pos = np.argmax(G_temp, axis=1)
    rel_second = G_temp[np.arange(G_temp.shape[0]), second_pos]

    # Parent-parent relatedness: between the two chosen parents.
    # We can approximate using a symmetric block from GRMOperator.
    if parent_idx.size > 1:
        G_par = np.asarray(grm_op.submatrix_G4(parent_idx, parent_idx), dtype=float)
        rel_parents = G_par[best_pos, second_pos]
    else:
        rel_parents = np.full_like(rel_best, np.nan)

    return BestParentResult(
        offspring_idx=offspring_idx,
        best_parent_idx=parent_idx[best_pos],
        second_parent_idx=parent_idx[second_pos],
        rel_best=rel_best,
        rel_second=rel_second,
        rel_parents=rel_parents,
    )


def best_parents_by_emm(
    depth: np.ndarray,
    genon: np.ndarray,
    p: np.ndarray,
    depth2K: Depth2KFn,
    offspring_idx: Sequence[int],
    parent_idx: Sequence[int],
    snpsubset: Optional[Sequence[int]] = None,
    mindepth_mm: float = 1.0,
) -> BestParentResult:
    """Select best and second-best parents per offspring using EMM (expected mismatch).

    This mirrors the 'EMM' branch of bestmatch(), using mismatch_par_comb to
    build the offspring × parent expected mismatch matrix and choosing parents
    with the smallest EMM.
    """
    offspring_idx = np.asarray(offspring_idx, dtype=int)
    parent_idx = np.asarray(parent_idx, dtype=int)

    mm = mismatch_par_comb(
        depth=depth,
        genon=genon,
        p=p,
        depth2K=depth2K,
        offspring_idx=offspring_idx,
        parent_idx=parent_idx,
        snpsubset=snpsubset,
        mindepth_mm=mindepth_mm,
    )

    EMM = mm.mmrate - mm.exp_mmrate  # shape (n_off, n_par)

    # Best parents are those with minimal EMM per offspring.
    if not np.any(np.isfinite(EMM)):
        raise ValueError("best_parents_by_emm requires at least one finite EMM value.")
    best_pos = np.nanargmin(EMM, axis=1)
    # Mask best and re-compute second best.
    EMM_temp = EMM.copy()
    EMM_temp[np.arange(EMM_temp.shape[0]), best_pos] = np.nanmax(EMM_temp) + 1.0
    second_pos = np.nanargmin(EMM_temp, axis=1)

    # For EMM-based selection, relatedness metrics can be filled later using GRMOperator.
    n = offspring_idx.size
    nan_arr = np.full(n, np.nan)

    return BestParentResult(
        offspring_idx=offspring_idx,
        best_parent_idx=parent_idx[best_pos],
        second_parent_idx=parent_idx[second_pos],
        rel_best=nan_arr,
        rel_second=nan_arr,
        rel_parents=nan_arr,
    )


@dataclass
class ParentCheckResult:
    rel: Optional[float]
    emm: Optional[float]
    match: Optional[bool]
    inb_parent: Optional[float]


def _build_sample_index_maps(sample_ids: Sequence[str], keep_ind_mask: np.ndarray):
    """Helper mapping: sample_id -> RA index, RA index -> QC index."""
    name_to_ra = {sid: i for i, sid in enumerate(sample_ids)}
    ra_keep_idx = np.where(keep_ind_mask)[0]
    ra_to_qc = {int(ra_idx): int(pos) for pos, ra_idx in enumerate(ra_keep_idx)}
    return name_to_ra, ra_to_qc


def _lookup_qc_index(
    seq_id: str,
    indiv_to_seq: Dict[str, str],
    name_to_ra: Dict[str, int],
    ra_to_qc: Dict[int, int],
) -> Optional[int]:
    """Get QC index from an IndivID or seqID; return None if missing."""
    if seq_id is None or seq_id == "" or pd.isna(seq_id):
        return None
    # seq_id is already the sequencing ID here.
    if seq_id not in name_to_ra:
        return None
    ra_idx = name_to_ra[seq_id]
    return ra_to_qc.get(ra_idx)


def check_parents(
    ped_df: pd.DataFrame,
    sample_ids: Sequence[str],
    keep_ind_mask: np.ndarray,
    G5_diag: np.ndarray,
    grm_op: GRMOperator,
    depth_np: np.ndarray,
    genon_np: np.ndarray,
    p_np: np.ndarray,
    depth2K: Depth2KFn,
    rel_threshF: float = 0.4,
    rel_threshM: float = 0.4,
    emm_thresh: float = 0.01,
    mindepth_mm: float = 1.0,
) -> pd.DataFrame:
    """Depth-aware parentage verification for given pedigree.

    This mirrors the core of GBSPed:
    - For each row in ped_df, treat it as an offspring record.
    - If FatherID / MotherID exist and are genotyped, compute:
      - KGD relatedness G4 between offspring and parent.
      - Mismatch rate & expected mismatch rate (EMM).
      - Inbreeding for parent from G5 diagonal.
      - Boolean match flag: (rel > rel_thresh & EMM < emm_thresh).
    """
    ped = ped_df.copy()

    required_cols = {"IndivID", "seqID"}
    missing_cols = required_cols - set(ped.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"Pedigree file missing required columns: {missing_str}")

    # Build mapping IndivID -> seqID for parents.
    indiv_to_seq: Dict[str, str] = {}
    for _, row in ped.iterrows():
        indiv = str(row["IndivID"])
        seq = str(row["seqID"])
        indiv_to_seq[indiv] = seq

    name_to_ra, ra_to_qc = _build_sample_index_maps(sample_ids, keep_ind_mask)

    # Inbreeding from G5 diagonal for QC-kept individuals.
    F_hat = G5_diag - 1.0

    father_rel = []
    father_emm = []
    father_match = []
    father_inb = []

    mother_rel = []
    mother_emm = []
    mother_match = []
    mother_inb = []

    inb_offspring = []

    for _, row in ped.iterrows():
        seq_off = str(row["seqID"])
        qc_off = _lookup_qc_index(seq_off, indiv_to_seq, name_to_ra, ra_to_qc)

        if qc_off is None:
            inb_offspring.append(np.nan)
        else:
            inb_offspring.append(float(F_hat[qc_off]))

        # Father
        if "FatherID" in ped.columns:
            father_id = row["FatherID"]
            father_seq = None
            if isinstance(father_id, str) and father_id in indiv_to_seq:
                father_seq = indiv_to_seq[father_id]
            elif not isinstance(father_id, str) and not pd.isna(father_id):
                father_seq = indiv_to_seq.get(str(father_id))

            qc_father = (
                _lookup_qc_index(father_seq, indiv_to_seq, name_to_ra, ra_to_qc)
                if father_seq is not None
                else None
            )

            if qc_off is None or qc_father is None:
                father_rel.append(np.nan)
                father_emm.append(np.nan)
                father_match.append(np.nan)
                father_inb.append(np.nan)
            else:
                G4_block = grm_op.submatrix_G4([qc_off], [qc_father])
                rel = float(np.asarray(G4_block)[0, 0])
                mm = mismatch_par_pair(
                    depth_np,
                    genon_np,
                    p_np,
                    depth2K,
                    i=qc_off,
                    j=qc_father,
                    mindepth_mm=mindepth_mm,
                )
                emm = mm.mmrate - mm.exp_mmrate
                match = bool((rel > rel_threshF) and (emm < emm_thresh))
                father_rel.append(rel)
                father_emm.append(emm)
                father_match.append(match)
                father_inb.append(float(F_hat[qc_father]))
        else:
            father_rel.append(np.nan)
            father_emm.append(np.nan)
            father_match.append(np.nan)
            father_inb.append(np.nan)

        # Mother
        if "MotherID" in ped.columns:
            mother_id = row["MotherID"]
            mother_seq = None
            if isinstance(mother_id, str) and mother_id in indiv_to_seq:
                mother_seq = indiv_to_seq[mother_id]
            elif not isinstance(mother_id, str) and not pd.isna(mother_id):
                mother_seq = indiv_to_seq.get(str(mother_id))

            qc_mother = (
                _lookup_qc_index(mother_seq, indiv_to_seq, name_to_ra, ra_to_qc)
                if mother_seq is not None
                else None
            )

            if qc_off is None or qc_mother is None:
                mother_rel.append(np.nan)
                mother_emm.append(np.nan)
                mother_match.append(np.nan)
                mother_inb.append(np.nan)
            else:
                G4_block = grm_op.submatrix_G4([qc_off], [qc_mother])
                rel = float(np.asarray(G4_block)[0, 0])
                mm = mismatch_par_pair(
                    depth_np,
                    genon_np,
                    p_np,
                    depth2K,
                    i=qc_off,
                    j=qc_mother,
                    mindepth_mm=mindepth_mm,
                )
                emm = mm.mmrate - mm.exp_mmrate
                match = bool((rel > rel_threshM) and (emm < emm_thresh))
                mother_rel.append(rel)
                mother_emm.append(emm)
                mother_match.append(match)
                mother_inb.append(float(F_hat[qc_mother]))
        else:
            mother_rel.append(np.nan)
            mother_emm.append(np.nan)
            mother_match.append(np.nan)
            mother_inb.append(np.nan)

    ped["Inb"] = inb_offspring
    ped["FatherRel"] = father_rel
    ped["FatherEMM"] = father_emm
    ped["FatherMatch"] = father_match
    ped["FatherInb"] = father_inb

    ped["MotherRel"] = mother_rel
    ped["MotherEMM"] = mother_emm
    ped["MotherMatch"] = mother_match
    ped["MotherInb"] = mother_inb

    return ped
