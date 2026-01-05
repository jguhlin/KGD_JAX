from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


Depth2KFn = Callable[[jnp.ndarray], jnp.ndarray]


def depth2K_bb(depth: jnp.ndarray, alpha: float = jnp.inf) -> jnp.ndarray:
    """K(d) under the beta-binomial model (KGD_CORE §3.2)."""
    # Respect input dtype
    if jnp.isinf(alpha):
        return 0.5 ** depth
    from jax.scipy.special import beta as jbeta

    alpha_f = jnp.array(alpha, dtype=depth.dtype)
    return jbeta(alpha_f, depth + alpha_f) / jbeta(alpha_f, alpha_f)


def depth2K_modp(depth: jnp.ndarray, modp: float = 0.5) -> jnp.ndarray:
    """K(d) under the Markov persistence model (KGD_CORE §3.2)."""
    # Respect input dtype
    modp_f = jnp.array(modp, dtype=depth.dtype)
    K = 0.5 * modp_f ** (depth - 1.0)
    return jnp.where(depth == 0.0, 1.0, K)


def make_depth2K(dmodel: str = "bb", param: Optional[float] = None) -> Depth2KFn:
    """Factory mirroring depth2Kchoose in KGD."""
    dmodel = dmodel.lower()
    if dmodel not in {"bb", "modp"}:
        raise ValueError(f"Unsupported depth model '{dmodel}' (expected 'bb' or 'modp').")

    if dmodel == "bb":
        if param is None:
            param = float("inf")

        def fn(depth: jnp.ndarray) -> jnp.ndarray:
            return depth2K_bb(depth, alpha=param)  # type: ignore[arg-type]

        return fn

    # dmodel == "modp"
    if param is None:
        param = 0.5

    def fn(depth: jnp.ndarray) -> jnp.ndarray:
        return depth2K_modp(depth, modp=param)  # type: ignore[arg-type]

    return fn


@dataclass
class GRMOperator:
    """Matrix-free wrapper around the KGD G5 construction.

    This holds genotype, depth and allele-frequency matrices, and exposes
    diagonal and small-block access without ever forming the full n_ind × n_ind
    GRM explicitly.
    """

    depth: jnp.ndarray  # (n_ind, n_snp)
    genon: jnp.ndarray  # (n_ind, n_snp), float32 with nan for missing
    p: jnp.ndarray  # (n_snp,) or (n_ind, n_snp)
    depth2K: Depth2KFn

    def __post_init__(self) -> None:
        if self.depth.shape != self.genon.shape:
            raise ValueError("depth and genon must have the same shape")

    @property
    def n_ind(self) -> int:
        return int(self.depth.shape[0])

    @property
    def n_snp(self) -> int:
        return int(self.depth.shape[1])

    def diag_G5(self, depth_min: float = 0.0, depth_max: float = jnp.inf) -> jnp.ndarray:
        """Compute the KGD G5 diagonal via calcGdiag (KGD_CORE §4.4)."""
        num, div = diag_G5_partial(
            self.depth, self.genon, self.p, self.depth2K, depth_min, depth_max
        )
        return 1.0 + num / div


def diag_G5_partial(
    depth: jnp.ndarray,
    genon: jnp.ndarray,
    p: jnp.ndarray,
    depth2K: Depth2KFn,
    depth_min: float = 0.0,
    depth_max: float = jnp.inf,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute numerator and denominator terms for G5 diagonal for a chunk of SNPs."""
    dtype = depth.dtype
    if p.ndim == 1:
        p = jnp.broadcast_to(p[None, :], depth.shape)

    d4i = jnp.array(1.001, dtype=dtype)

    usegeno = ~jnp.isnan(genon)
    if depth_min > 1.0 or depth_max < float("inf"):
        depth_mask = (depth >= depth_min) & (depth <= depth_max)
        usegeno = usegeno & depth_mask

    two = jnp.array(2.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    
    genon0 = jnp.where(usegeno, genon - two * p, 0.0)
    genon01 = jnp.where(depth >= d4i, genon0, 0.0)

    P0 = jnp.where(usegeno & (depth >= d4i), p, 0.0)
    P1 = jnp.where(usegeno & (depth >= d4i), one - p, 0.0)

    div0 = two * jnp.sum(P0 * P1, axis=1)  # (n_ind,)

    depth_clamped = jnp.where(depth < d4i, d4i, depth)
    Kdepth = depth2K(depth_clamped)

    # Ensure constants in num are also matching dtype
    num = jnp.sum(
        (genon01**2 - two * P0 * P1 * (one + two * Kdepth)) / (one - two * Kdepth),
        axis=1,
    )
    return num, div0

    def submatrix_G4(
        self,
        ind_i: Sequence[int],
        ind_j: Optional[Sequence[int]] = None,
        depth_min: float = 0.0,
        depth_max: float = jnp.inf,
    ) -> jnp.ndarray:
        """Compute a small G4 block for given row/column index sets.

        This mirrors the off-diagonal construction of calcG (KGD_CORE §4.2),
        but only for a subset, to stay matrix-free at large n_ind.
        """
        if ind_j is None:
            ind_j = ind_i

        i_idx = jnp.asarray(ind_i, dtype=jnp.int32)
        j_idx = jnp.asarray(ind_j, dtype=jnp.int32)

        depth = self.depth
        genon = self.genon
        p = self.p
        if p.ndim == 1:
            p = jnp.broadcast_to(p[None, :], depth.shape)

        depth_sub = depth[i_idx][:, :]
        genon_sub = genon[i_idx][:, :]
        p_sub_i = p[i_idx][:, :]

        depth_sub_j = depth[j_idx][:, :]
        genon_sub_j = genon[j_idx][:, :]
        p_sub_j = p[j_idx][:, :]

        usegeno_i = ~jnp.isnan(genon_sub)
        usegeno_j = ~jnp.isnan(genon_sub_j)

        if depth_min > 1.0 or depth_max < float("inf"):
            mask_i = (depth_sub >= depth_min) & (depth_sub <= depth_max)
            mask_j = (depth_sub_j >= depth_min) & (depth_sub_j <= depth_max)
            usegeno_i = usegeno_i & mask_i
            usegeno_j = usegeno_j & mask_j

        # Centered genotypes, missing set to 0 for tcrossprod.
        gi = jnp.where(usegeno_i, genon_sub - 2.0 * p_sub_i, 0.0)
        gj = jnp.where(usegeno_j, genon_sub_j - 2.0 * p_sub_j, 0.0)

        # Numerator N_ij via tcrossprod.
        N = gi @ gj.T  # (len(i_idx), len(j_idx))

        # Denominator D_ij based on Q0 and cocall.
        Qi = p_sub_i * (1.0 - p_sub_i)
        Qj = p_sub_j * (1.0 - p_sub_j)

        cocall = (usegeno_i.astype(jnp.float32)) @ (
            usegeno_j.astype(jnp.float32).T
        )  # (i,j)

        # Following calcG: div0a_ij = 2 * sum_s Q_is * u_is * u_js
        div0a_i = 2.0 * (Qi @ usegeno_j.astype(jnp.float32).T)
        div0a_j = 2.0 * (Qj @ usegeno_i.astype(jnp.float32).T).T
        D = jnp.sqrt(div0a_i) * jnp.sqrt(div0a_j)

        # Avoid division by zero by masking entries with no cocalls.
        G4 = jnp.where(cocall > 0, N / D, 0.0)
        return G4


def build_grm_operator(
    depth: jnp.ndarray,
    genon: jnp.ndarray,
    p: jnp.ndarray,
    dmodel: str = "bb",
    dparam: Optional[float] = None,
) -> GRMOperator:
    """Helper to construct a GRMOperator from QC outputs."""
    depth2K_fn = make_depth2K(dmodel=dmodel, param=dparam)
    return GRMOperator(depth=depth, genon=genon, p=p, depth2K=depth2K_fn)

