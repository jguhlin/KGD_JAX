from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .grm import Depth2KFn, make_depth2K


@dataclass
class HeterozygosityResult:
    """Depth-aware heterozygosity summary (KGD GBS-PopGen heterozygosity)."""

    neff: float
    ohetstar: float
    ehetstar: float
    ohet: float
    ohet2: float
    ehet: float


def heterozygosity(
    depth: np.ndarray,
    genon: np.ndarray,
    p: np.ndarray,
    depth2K: Depth2KFn,
    maxiter: int = 100,
    convtol: float = 1e-3,
) -> HeterozygosityResult:
    """Port of heterozygosity() from GBS-PopGen.R (matrix-free in n_ind×n_snp).

    Args:
        depth: depth matrix (n_ind, n_snp), zeros allowed.
        genon: genotype matrix (n_ind, n_snp) with NaN for missing.
        p: allele frequencies per SNP (n_snp,) corresponding to depth/genon.
        depth2K: function mapping depth matrix to K matrix (same shape).
        maxiter: EM max iterations per SNP.
        convtol: EM convergence tolerance (L1 on genotype-class probs).
    """
    depth = np.asarray(depth, dtype=np.float32)
    genon = np.asarray(genon, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)

    n_ind, n_snp = genon.shape

    # depth.use with 0 -> NA
    depth_use = depth.astype(np.float32)
    depth_use[depth_use == 0.0] = np.nan

    # Observed genotype frequencies per SNP.
    mask_g = ~np.isnan(genon)
    denom = mask_g.sum(axis=0).astype(np.float32)  # per SNP
    denom_safe = np.where(denom == 0.0, np.nan, denom)

    counts0 = np.sum((genon == 0) & mask_g, axis=0).astype(np.float32)
    counts1 = np.sum((genon == 1) & mask_g, axis=0).astype(np.float32)
    counts2 = np.sum((genon == 2) & mask_g, axis=0).astype(np.float32)

    obsgfreq = np.stack(
        [counts0 / denom_safe, counts1 / denom_safe, counts2 / denom_safe], axis=1
    )  # shape (n_snp, 3)

    obshet = np.nanmean(obsgfreq[:, 1])

    # True expected het based on P (no depth errors).
    maf = np.minimum(p, 1.0 - p)
    ehettrue = np.nanmean(2.0 * maf * (1.0 - maf))

    # Depth to K on depth_use (NaNs propagate).
    K_depth_use = np.asarray(depth2K(depth_use))
    K_bar = np.nanmean(K_depth_use, axis=0)  # per SNP

    ehetstar = np.nanmean(2.0 * maf * (1.0 - maf) * (1.0 - 2.0 * K_bar))

    # ohet2: overall observed het scaled by mean(1-2K).
    denom_ohet2 = np.nanmean(1.0 - 2.0 * K_depth_use)
    ohet2 = float(obshet / denom_ohet2) if denom_ohet2 not in (0.0, np.nan) else np.nan

    # EM-refined heterozygosity per SNP.
    ohet = np.full(n_snp, np.nan, dtype=np.float32)

    for s in range(n_snp):
        g_col = genon[:, s]
        d_col = depth_use[:, s]

        if np.all(np.isnan(d_col)):
            continue

        # Initial genotype-class proportions from obsgfreq row.
        pnew = obsgfreq[s, :].copy()
        if not np.all(np.isfinite(pnew)) or pnew.sum() == 0.0:
            # Fall back to Hardy–Weinberg from p if needed.
            p_s = float(p[s])
            pnew = np.array([(1 - p_s) ** 2, 2 * p_s * (1 - p_s), p_s**2], dtype=np.float32)

        ng = np.sum(~np.isnan(d_col)).astype(float)
        if ng == 0.0:
            continue

        convtest = 1.0
        itcount = 0

        while convtest > convtol and itcount < maxiter:
            itcount += 1
            pcurrent = pnew.copy()

            # Depth vectors for AA (2) and BB (0); NA in depth or genon are dropped.
            mask2 = (g_col == 2.0) & ~np.isnan(d_col)
            mask0 = (g_col == 0.0) & ~np.isnan(d_col)

            d2 = d_col[mask2]
            d0 = d_col[mask0]

            # Avoid divisions by ~0 by checking class probabilities.
            pAA = max(pcurrent[2], 1e-8)
            pAB = max(pcurrent[1], 1e-8)
            pBB = max(pcurrent[0], 1e-8)

            if d2.size > 0:
                K2 = np.asarray(depth2K(d2), dtype=np.float32)
                paanew = np.sum(1.0 / (1.0 + K2 * pAB / pAA)) / ng
            else:
                paanew = pAA

            if d0.size > 0:
                K0 = np.asarray(depth2K(d0), dtype=np.float32)
                pbbnew = np.sum(1.0 / (1.0 + K0 * pAB / pBB)) / ng
            else:
                pbbnew = pBB

            pabnew = 1.0 - paanew - pbbnew
            pnew = np.array([pbbnew, pabnew, paanew], dtype=np.float32)

            convtest = float(np.sum(np.abs(pnew - pcurrent)))
            if not np.isfinite(convtest):
                convtest = 0.0

        ohet[s] = pnew[1]

    # Effective number of individuals: mean over SNPs of Σ_i (1-K_is).
    K_full = np.asarray(depth2K(depth), dtype=np.float32)
    effnumind = float(np.mean(np.sum(1.0 - K_full, axis=0)))

    return HeterozygosityResult(
        neff=effnumind,
        ohetstar=float(obshet),
        ehetstar=float(ehetstar),
        ohet=float(np.nanmean(ohet)),
        ohet2=float(ohet2),
        ehet=float(ehettrue),
    )


@dataclass
class FstResult:
    fst: np.ndarray  # per-SNP Fst, NaN where undefined


@dataclass
class HWResult:
    """Hardy–Weinberg diagnostics per population and SNP (KGD HWpops)."""

    popnames: List[str]
    HWdis: np.ndarray      # (npops, n_snp)
    l10LRT: np.ndarray     # (npops, n_snp)
    x2star: np.ndarray     # (npops, n_snp)
    l10pstar: np.ndarray   # (npops, n_snp)
    maf: np.ndarray        # (npops, n_snp)
    l10pstar_pop: Optional[np.ndarray] = None  # combined over populations


@dataclass
class DAPCResult:
    """Discriminant Analysis of Principal Components on genotype-based PCs."""

    sample_ids: np.ndarray          # shape (n_ind,)
    populations: np.ndarray         # shape (n_ind,)
    pc_scores: np.ndarray           # shape (n_ind, n_pc)
    pc_eigenvalues: np.ndarray      # shape (n_pc,)
    lda_scores: np.ndarray          # shape (n_ind, n_disc)
    lda_loadings: np.ndarray        # shape (n_pc, n_disc)
    lda_eigenvalues: np.ndarray     # shape (n_disc,)


def _group_indices(labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Map population labels to integer indices 0..(k-1)."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    mapping: Dict[str, int] = {lab: i for i, lab in enumerate(uniq)}
    idx = np.array([mapping[lab] for lab in labels], dtype=np.int32)
    return idx, list(uniq)


def fst_gbs(
    depth: np.ndarray,
    genon: np.ndarray,
    populations: Sequence[str],
    depth2K: Depth2KFn,
    varadj: int = 0,
) -> FstResult:
    """Port of Fst.GBS (no per-SNP p-values, matrix-free in GRM).

    Args:
        depth: depth matrix (n_ind, n_snp).
        genon: genotype matrix (n_ind, n_snp) with NaN for missing.
        populations: sequence of population labels (length n_ind).
        depth2K: depth-to-K function.
        varadj: 0 for usual Fst, 1 for Weir p.166 adjustment.
    """
    depth = np.asarray(depth, dtype=np.float32)
    genon = np.asarray(genon, dtype=np.float32)

    n_ind, n_snp = genon.shape
    pop_idx, pop_names = _group_indices(populations)
    npops = len(pop_names)

    # Depth per population and SNP.
    snppopdepth = np.zeros((npops, n_snp), dtype=np.float32)
    for i in range(n_ind):
        snppopdepth[pop_idx[i], :] += depth[i, :]

    # Global allele freq per SNP using genotypes.
    mask_g = ~np.isnan(genon)
    if not np.any(mask_g):
        raise ValueError("fst_gbs requires at least one non-NaN genotype.")
    pgsub = np.nansum(genon, axis=0) / (2.0 * mask_g.sum(axis=0))

    # SNPs usable for Fst.
    use = (snppopdepth.min(axis=0) > 0.0) & (pgsub > 0.0) & (pgsub < 1.0)
    fst_vals = np.full(n_snp, np.nan, dtype=np.float64)

    # Effective allele numbers.
    K_full = np.asarray(depth2K(depth), dtype=np.float32)
    effnuma1 = 2.0 * (1.0 - K_full)  # n_ind × n_snp

    snppopeffn = np.zeros((npops, n_snp), dtype=np.float32)
    for i in range(n_ind):
        snppopeffn[pop_idx[i], :] += effnuma1[i, :]

    effnuma = np.sum(2.0 * (1.0 - K_full), axis=0)  # length n_snp

    # Pre-construct population indices for allele rows.
    pop_idx2 = np.concatenate([pop_idx, pop_idx])

    for s in np.where(use)[0]:
        g_col = genon[:, s]

        # Pseudo-alleles 0/1 via round/ceil of g/2, ignoring NaN.
        a1 = np.round(g_col / 2.0)
        a2 = np.ceil(g_col / 2.0)
        alleles = np.concatenate([a1, a2])

        mask_valid = ~np.isnan(alleles)
        if not np.any(mask_valid):
            continue

        alleles_v = alleles[mask_valid].astype(int)
        pops_v = pop_idx2[mask_valid]

        # 2 x npops contingency table of allele (0/1) by population.
        table = np.zeros((2, npops), dtype=np.float64)
        np.add.at(table, (alleles_v, pops_v), 1.0)

        # Scale each population column to effective counts.
        col_tot = table.sum(axis=0)
        eff_col = snppopeffn[:, s]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = eff_col / col_tot
        scale[~np.isfinite(scale)] = 0.0
        table2 = table * scale

        grand = table2.sum()
        if grand <= 0.0:
            continue

        row_tot = table2.sum(axis=1)
        col_tot2 = table2.sum(axis=0)
        expected = np.outer(row_tot, col_tot2) / grand
        mask_exp = expected > 0.0
        chi2 = np.sum(((table2 - expected) ** 2 / expected)[mask_exp])

        if effnuma[s] > 0.0 and npops > varadj:
            fst_vals[s] = npops * chi2 / (effnuma[s] * (npops - varadj))

    return FstResult(fst=fst_vals)


def hw_pops(
    genon: np.ndarray,
    depth: np.ndarray,
    populations: Optional[Sequence[str]],
    depth2K: Depth2KFn,
) -> HWResult:
    """Port of HWpops (Hardy–Weinberg diagnostics) to numpy.

    Args:
        genon: genotype matrix (n_ind, n_snp) with NaN for missing, values 0/1/2.
        depth: depth matrix (n_ind, n_snp).
        populations: length-n_ind labels; if None, all in one group.
        depth2K: depth-to-K function.
    """
    genon = np.asarray(genon, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    n_ind, n_snp = genon.shape

    if populations is None:
        populations = ["A"] * n_ind
    populations = np.asarray(populations, dtype=object)

    # Preserve order of first appearance.
    seen = {}
    popnames: List[str] = []
    for lab in populations:
        if lab not in seen:
            seen[lab] = True
            popnames.append(lab)
    npops = len(popnames)

    HWdis = np.full((npops, n_snp), np.nan, dtype=np.float64)
    l10LRT = np.full_like(HWdis, np.nan)
    x2star = np.full_like(HWdis, np.nan)
    l10pstar = np.full_like(HWdis, np.nan)
    maf = np.full_like(HWdis, np.nan)

    # Helper: chi-square -log10(p) via survival function using gammaincc.
    def chisq_l10p(x: np.ndarray, df: int) -> np.ndarray:
        import jax.numpy as jnp
        from jax.scipy.special import gammaincc

        x_j = jnp.asarray(x, dtype=jnp.float32)
        df_j = jnp.asarray(df, dtype=jnp.float32)
        p = gammaincc(df_j / 2.0, x_j / 2.0)
        p_np = np.asarray(p, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = -np.log10(p_np)
        out[~np.isfinite(out)] = np.nan
        return out

    for ipop, pop in enumerate(popnames):
        indgroup = np.where(populations == pop)[0]
        if indgroup.size == 0:
            continue

        g_sub = genon[indgroup, :]

        naa = np.nansum(g_sub == 2.0, axis=0)
        nab = np.nansum(g_sub == 1.0, axis=0)
        nbb = np.nansum(g_sub == 0.0, axis=0)

        n1 = 2.0 * naa + nab
        n2 = nab + 2.0 * nbb
        n = n1 + n2  # allele count

        with np.errstate(divide="ignore", invalid="ignore"):
            p1 = n1 / n
        p2 = 1.0 - p1

        # Hardy–Weinberg disequilibrium: freq(AA) - p1^2.
        denom_geno = naa + nab + nbb
        with np.errstate(divide="ignore", invalid="ignore"):
            freq_AA = naa / denom_geno
        HWd = freq_AA - p1**2
        HWdis[ipop, :] = HWd

        # Pearson chi-square X^2 (not directly output but used to check).
        with np.errstate(divide="ignore", invalid="ignore"):
            x2 = denom_geno * HWd**2 / (p1**2 * p2**2)

        # Likelihood ratio test (LRT).
        # Using the same form as in R: df=1 chi-square.
        n_safe = np.where(n == 0.0, np.nan, n)
        # Avoid log(0) by using pmax(1, count) style via masking.
        naa_safe = np.where(naa <= 0, 1.0, naa)
        nab_safe = np.where(nab <= 0, 1.0, nab)
        nbb_safe = np.where(nbb <= 0, 1.0, nbb)
        n1_safe = np.where(n1 <= 0, 1.0, n1)
        n2_safe = np.where(n2 <= 0, 1.0, n2)

        # LRT formula from GBS-Chip-Gmatrix.R.
        term = (
            n_safe * np.log(n_safe)
            + naa_safe * np.log(naa_safe)
            + nab_safe * np.log(nab_safe)
            + nbb_safe * np.log(nbb_safe)
            - (n_safe / 2.0) * np.log(n_safe / 2.0)
            - n1_safe * np.log(n1_safe)
            - n2_safe * np.log(n2_safe)
            - nab_safe * np.log(2.0)
        )
        LRT = 2.0 * term
        l10LRT[ipop, :] = chisq_l10p(LRT, df=1)

        # Depth-adjusted x2* and -log10(p*).
        depth_sub = depth[indgroup, :]
        Kdepth = np.asarray(depth2K(depth_sub), dtype=np.float64)
        Kdepth[depth_sub == 0.0] = np.nan

        with np.errstate(invalid="ignore"):
            Kbar = np.nanmean(Kdepth, axis=0)
        esnphetstar = 2.0 * p1 * p2 * (1.0 - 2.0 * Kbar)
        with np.errstate(divide="ignore", invalid="ignore"):
            osnphetstar = nab / denom_geno

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = osnphetstar / esnphetstar
        ratio[~np.isfinite(ratio)] = np.nan
        one_minus_ratio = 1.0 - ratio

        with np.errstate(invalid="ignore"):
            sum_1_minus_2K = np.nansum(1.0 - 2.0 * Kdepth, axis=0)
        x2s = np.full_like(sum_1_minus_2K, np.nan, dtype=np.float64)
        valid = np.isfinite(sum_1_minus_2K) & np.isfinite(one_minus_ratio)
        if np.any(valid):
            x2s[valid] = sum_1_minus_2K[valid] * (one_minus_ratio[valid] ** 2)
        x2star[ipop, :] = x2s
        l10pstar[ipop, :] = chisq_l10p(x2s, df=1)

        maf[ipop, :] = np.where(p1 > 0.5, p2, p1)

    # Combined across populations (if >1) using sum of x2* and df equal to count.
    l10pstar_pop = None
    if npops > 1:
        with np.errstate(invalid="ignore"):
            x2star_sum = np.nansum(x2star, axis=0)
            df_pop = np.sum(~np.isnan(x2star), axis=0)
        valid = df_pop > 0
        l10pstar_pop = np.full(n_snp, np.nan, dtype=np.float64)
        if np.any(valid):
            l10pstar_pop[valid] = chisq_l10p(x2star_sum[valid], df=df_pop[valid])

    return HWResult(
        popnames=popnames,
        HWdis=HWdis,
        l10LRT=l10LRT,
        x2star=x2star,
        l10pstar=l10pstar,
        maf=maf,
        l10pstar_pop=l10pstar_pop,
    )


def dapc_from_genotypes(
    genon: np.ndarray,
    p: np.ndarray,
    populations: Sequence[str],
    sample_ids: Sequence[str],
    n_pca: Optional[int] = None,
    perc_pca: float = 90.0,
) -> DAPCResult:
    """DAPC on centered genotypes, mirroring DAPC.GBS behaviour.

    Steps:
      1. Center genotypes as in calcG (g' = g - 2p, missing set to 0).
      2. PCA via SVD on the centered matrix.
      3. Choose n_pca PCs based on perc_pca if not given.
      4. LDA on these PC scores with class labels 'populations'.
    """
    genon = np.asarray(genon, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    populations = np.asarray(populations)
    sample_ids = np.asarray(sample_ids)

    n_ind, n_snp = genon.shape
    if p.ndim != 1 or p.shape[0] != n_snp:
        raise ValueError("p must be a 1D array of length n_snp.")

    # Center genotypes like calcG: g' = g - 2p, missing → 0.
    p_mat = np.broadcast_to(p[None, :], genon.shape)
    usegeno = ~np.isnan(genon)
    X = np.where(usegeno, genon - 2.0 * p_mat, 0.0)

    # Column-center X (minor adjustment; expected mean already ~0).
    X -= X.mean(axis=0, keepdims=True)

    # PCA via SVD on X.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    evals = np.sign(S) * (S**2) / np.sum(S**2)
    varexpl = 100.0 * np.cumsum(evals) / np.sum(evals)

    if n_pca is None:
        k = int(min(np.sum(varexpl < perc_pca) + 1, len(evals)))
    else:
        k = int(min(n_pca, len(evals)))
    if k < 1:
        k = 1

    pc_scores = U[:, :k] * S[:k]
    pc_evals = evals[:k]

    # LDA on PC scores (multi-class).
    classes = populations.astype(object)
    uniq = np.unique(classes)
    n_classes = len(uniq)
    if n_classes < 2:
        raise ValueError("DAPC requires at least two populations.")

    mu = pc_scores.mean(axis=0)
    Sw = np.zeros((k, k), dtype=np.float64)
    Sb = np.zeros((k, k), dtype=np.float64)

    for lab in uniq:
        idx = np.where(classes == lab)[0]
        if idx.size == 0:
            continue
        Xc = pc_scores[idx, :]
        mu_c = Xc.mean(axis=0)
        Sw += (Xc - mu_c).T @ (Xc - mu_c)
        diff = (mu_c - mu).reshape(-1, 1)
        Sb += idx.size * (diff @ diff.T)

    # Generalized eigenproblem Sw^{-1} Sb.
    Sw_inv = np.linalg.pinv(Sw)
    A = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_disc = min(k, n_classes - 1)
    lda_loadings = eigvecs[:, :n_disc]
    lda_scores = pc_scores @ lda_loadings

    return DAPCResult(
        sample_ids=sample_ids,
        populations=classes,
        pc_scores=pc_scores,
        pc_eigenvalues=pc_evals,
        lda_scores=lda_scores,
        lda_loadings=lda_loadings,
        lda_eigenvalues=eigvals[:n_disc],
    )


def fst_pairwise(
    depth: np.ndarray,
    genon: np.ndarray,
    populations: Sequence[str],
    depth2K: Depth2KFn,
    varadj: int = 0,
) -> Dict[str, np.ndarray]:
    """Pairwise Fst between all population pairs (Fst.GBS.pairwise analogue).

    Returns a dict with keys:
      - 'Fst': 3D array [npops, npops, n_snp] with NaNs off upper triangle.
      - 'mean': 2D array [npops, npops] of per-pair mean Fst.
      - 'median': 2D array [npops, npops] of per-pair median Fst.
      - 'popnames': list of population names in order.
    """
    populations = np.asarray(populations)
    if not np.any(~np.isnan(genon)):
        raise ValueError("fst_pairwise requires at least one non-NaN genotype.")
    popnames = np.unique(populations)
    npops = len(popnames)

    Fst_cube = np.full((npops, npops, genon.shape[1]), np.nan, dtype=np.float64)
    Fst_mean = np.full((npops, npops), np.nan, dtype=np.float64)
    Fst_median = np.full_like(Fst_mean, np.nan)

    for i in range(npops - 1):
        for j in range(i + 1, npops):
            pop_i = popnames[i]
            pop_j = popnames[j]
            mask = (populations == pop_i) | (populations == pop_j)
            fst_res = fst_gbs(
                depth=depth[mask, :],
                genon=genon[mask, :],
                populations=populations[mask],
                depth2K=depth2K,
                varadj=varadj,
            )
            Fst_cube[i, j, :] = fst_res.fst
            finite = np.isfinite(fst_res.fst)
            if np.any(finite):
                Fst_mean[i, j] = np.nanmean(fst_res.fst[finite])
                Fst_median[i, j] = np.nanmedian(fst_res.fst[finite])

    return {
        "Fst": Fst_cube,
        "mean": Fst_mean,
        "median": Fst_median,
        "popnames": popnames,
    }


def popmaf(
    genon: np.ndarray,
    populations: Sequence[str],
    minsamps: int = 10,
    mafmin: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Compute MAF distributions per population (no plotting).

    Returns dict mapping population label -> MAF array (filtered by minsamps, mafmin).
    """
    genon = np.asarray(genon, dtype=np.float32)
    populations = np.asarray(populations)
    popnames = np.unique(populations)

    maf_by_pop: Dict[str, np.ndarray] = {}
    for pop in popnames:
        idx = np.where(populations == pop)[0]
        if idx.size < minsamps:
            continue
        g_sub = genon[idx, :]
        # Allele frequency per SNP from genotypes, ignoring NaNs.
        mask = ~np.isnan(g_sub)
        num = np.nansum(g_sub, axis=0)
        den = 2.0 * mask.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = num / den
        maf = np.minimum(p, 1.0 - p)
        maf = maf[np.isfinite(maf)]
        maf = maf[maf >= mafmin]
        maf_by_pop[str(pop)] = maf
    return maf_by_pop


def popG(
    G: np.ndarray,
    populations: Sequence[str],
    diag: bool = False,
) -> Dict[str, np.ndarray]:
    """Population-averaged GRM and within-pop inbreeding (popG analogue).

    Args:
        G: GRM (n_ind, n_ind).
        populations: population labels.
        diag: if False, diagonal of popG excludes self-rel (as in R).
    """
    G = np.asarray(G, dtype=np.float64)
    populations = np.asarray(populations)

    popnames = np.unique(populations)
    n_ind = G.shape[0]
    # Xpops: N×P indicator matrix.
    X = np.zeros((n_ind, popnames.size), dtype=np.float64)
    for i, pop in enumerate(popnames):
        X[populations == pop, i] = 1.0
    npops = X.sum(axis=0)

    # Average G by population.
    inv_npops = np.where(npops > 0, 1.0 / npops, 0.0)
    W = X * inv_npops  # each column scaled by 1/npops
    popG_mat = W.T @ G @ W

    # Mean self-rel per population from diagonal of G.
    diagG = np.diag(G)
    popSelf = W.T @ diagG

    if not diag:
        # Adjust diagonal of popG to exclude self-rel when population size > 1.
        popG_mat = popG_mat.copy()
        for i, n_pop in enumerate(npops):
            if n_pop <= 1:
                continue
            diag_val = popG_mat[i, i]
            new_diag = (diag_val * n_pop - popSelf[i]) / (n_pop - 1.0)
            popG_mat[i, i] = new_diag

    inb = popSelf[:, None] - 1.0

    return {
        "G": popG_mat,
        "Inb": inb,
        "popnames": popnames,
    }


def Nefromr2(
    r2auto: np.ndarray,
    nLD: np.ndarray,
    alpha: float = 1.0,
    weighted: bool = False,
    minN: int = 1,
) -> Dict[str, float]:
    """LD-based Ne estimator (Nefromr2 analogue).

    Args:
        r2auto: array of r^2 values across autosomes.
        nLD: effective sample sizes per r^2 value.
    """
    r2auto = np.asarray(r2auto, dtype=np.float64)
    nLD = np.asarray(nLD, dtype=np.float64)
    if nLD.size == 1:
        nLD = np.full_like(r2auto, nLD[0])

    uN = np.where(nLD >= minN)[0]
    r2auto = r2auto[uN]
    nLD = nLD[uN]

    wt = nLD if weighted else np.ones_like(r2auto)

    def wmean(x, w):
        return np.average(x, weights=w)

    meanN = float(np.mean(nLD))

    def _safe_ne_adj(r2_val: float, beta: float) -> float:
        denom = r2_val - 1.0 / (beta * meanN)
        if denom <= 0.0 or not np.isfinite(denom):
            return float("nan")
        return float((1.0 / denom - alpha) / 2.0)

    r2_mean = float(wmean(r2auto, wt))
    if r2_mean <= 0.0 or not np.isfinite(r2_mean):
        Neauto = float("nan")
    else:
        Neauto = float((1.0 / r2_mean - 1.0) / 2.0)
    Neauto_adj_b1 = _safe_ne_adj(r2_mean, beta=1.0)
    Neauto_adj_b2 = _safe_ne_adj(r2_mean, beta=2.0)

    med = float(np.median(r2auto))
    if med <= 0.0 or not np.isfinite(med):
        Neauto_med = float("nan")
    else:
        Neauto_med = float((1.0 / med - 1.0) / 2.0)
    Neauto_med_adj_b1 = _safe_ne_adj(med, beta=1.0)
    Neauto_med_adj_b2 = _safe_ne_adj(med, beta=2.0)

    return {
        "n": meanN,
        "Neauto": Neauto,
        "Neauto_adj_b1": Neauto_adj_b1,
        "Neauto_adj_b2": Neauto_adj_b2,
        "Neauto_med": Neauto_med,
        "Neauto_med_adj_b1": Neauto_med_adj_b1,
        "Neauto_med_adj_b2": Neauto_med_adj_b2,
    }


def snpselection(
    chromosome: np.ndarray,
    position: np.ndarray,
    nsnpperchrom: int = 100,
    seltype: str = "centre",
    snpsubset: Optional[Sequence[int]] = None,
    chromuse: Optional[Sequence] = None,
    randseed: Optional[int] = None,
) -> np.ndarray:
    """Select SNP indices for LD / Ne analysis (snpselection analogue).

    Returns:
        pairs: array of shape (n_pairs, 2) with 0-based SNP indices into
               the original chromosome/position arrays.
    """
    chromosome = np.asarray(chromosome)
    position = np.asarray(position)
    n_snp = chromosome.shape[0]

    if snpsubset is None:
        snpsubset = np.arange(n_snp, dtype=int)
    else:
        snpsubset = np.asarray(snpsubset, dtype=int)

    if chromuse is None:
        chromuse = np.unique(chromosome)
    chromuse = list(chromuse)

    seltype = seltype.lower()
    if seltype == "center":
        seltype = "centre"

    usnp = snpsubset[np.isin(chromosome[snpsubset], chromuse)]
    chromlist = np.unique(chromosome[usnp])

    # Helper to select up to nsnpperchrom SNPs for one chromosome.
    def choose_for_chr(chr_label):
        idx = np.where(chromosome[usnp] == chr_label)[0]
        if idx.size == 0:
            return np.array([], dtype=int)
        nsel = min(idx.size, nsnpperchrom)
        if seltype == "random":
            rng = np.random.default_rng(randseed)
            chosen_local = rng.choice(idx, size=nsel, replace=False)
            return chosen_local
        if seltype == "even":
            # Evenly spaced in physical position.
            pos_sub = position[usnp][idx]
            order = np.argsort(pos_sub)
            idx_sorted = idx[order]
            if nsel == 1:
                return np.array([idx_sorted[len(idx_sorted) // 2]], dtype=int)
            step = len(idx_sorted) / float(nsnpperchrom)
            centres = np.round(
                np.arange(step / 2.0, step * nsnpperchrom, step)
            ).astype(int)
            centres = np.clip(centres, 0, len(idx_sorted) - 1)
            return idx_sorted[centres[:nsel]]
        # "centre": pick SNPs closest to chromosome mean position.
        pos_sub = position[usnp][idx]
        meanpos = float(np.mean(pos_sub))
        order = np.argsort(np.abs(pos_sub - meanpos))
        chosen_local = idx[order[:nsel]]
        return np.sort(chosen_local)

    # Build list of selected index sets per chromosome.
    snp_lists = [choose_for_chr(chr_label) for chr_label in chromlist]
    n_chrom = len(snp_lists)

    # Construct all pairs between chromosomes (1 vs 2.., 2 vs 3.., etc.).
    pairs = []
    for i in range(n_chrom - 1):
        idx_i = snp_lists[i]
        if idx_i.size == 0:
            continue
        for j in range(i + 1, n_chrom):
            idx_j = snp_lists[j]
            if idx_j.size == 0:
                continue
            grid_i, grid_j = np.meshgrid(idx_i, idx_j, indexing="ij")
            pairs.append(
                np.stack([usnp[grid_i.ravel()], usnp[grid_j.ravel()]], axis=1)
            )

    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.vstack(pairs)
