from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from .grm import GRMOperator, make_depth2K


@dataclass
class DepthParamFitResult:
    dmodel: str
    param_opt: float
    ss_min: float
    n_ind: int
    n_snp: int


def _ssd_inb_for_param(
    param: float,
    dmodel: str,
    depth: jnp.ndarray,
    genon: jnp.ndarray,
    p: jnp.ndarray,
    inb_target: np.ndarray,
    ind_indices: np.ndarray,
    snp_indices: Optional[np.ndarray],
) -> float:
    """Sum of squared deviations in inbreeding for a given depth2K param."""
    # Restrict individuals / SNPs.
    depth_sub = depth[ind_indices, :]
    genon_sub = genon[ind_indices, :]
    if p.ndim == 1:
        p_full = jnp.broadcast_to(p[None, :], depth.shape)
    else:
        p_full = p
    p_sub_full = p_full[ind_indices, :]

    if snp_indices is not None:
        depth_sub = depth_sub[:, snp_indices]
        genon_sub = genon_sub[:, snp_indices]
        p_sub_full = p_sub_full[:, snp_indices]

    depth2K_fn = make_depth2K(dmodel=dmodel, param=float(param))
    op = GRMOperator(depth=depth_sub, genon=genon_sub, p=p_sub_full, depth2K=depth2K_fn)
    G5d = op.diag_G5()
    NInb = np.asarray(G5d - 1.0, dtype=np.float64)

    # Align target vector (already in same order).
    diff = NInb - inb_target
    return float(np.nansum(diff * diff))


def fit_depth_param_inb(
    depth: jnp.ndarray,
    genon: jnp.ndarray,
    p: jnp.ndarray,
    inb_target: np.ndarray,
    dmodel: str = "bb",
    ind_indices: Optional[Sequence[int]] = None,
    snp_indices: Optional[Sequence[int]] = None,
    bounds: Tuple[float, float] = (0.1, 200.0),
    tol: float = 0.05,
    max_iter: int = 60,
) -> DepthParamFitResult:
    """Fit depth-model parameter (e.g. beta-binomial alpha) to inbreeding targets.

    This is a Python/JAX analogue of ssdInb + optimise in GBS-Chip-Gmatrix.R.
    """
    dmodel = dmodel.lower()
    if dmodel not in {"bb", "modp"}:
        raise ValueError("dmodel must be 'bb' or 'modp'.")

    depth = jnp.asarray(depth, dtype=jnp.float32)
    genon = jnp.asarray(genon, dtype=jnp.float32)
    p = jnp.asarray(p, dtype=jnp.float32)

    n_ind, n_snp = depth.shape

    if ind_indices is None:
        ind_indices_np = np.arange(n_ind, dtype=np.int32)
    else:
        ind_indices_np = np.asarray(ind_indices, dtype=np.int32)

    if snp_indices is None:
        snp_indices_np = None
    else:
        snp_indices_np = np.asarray(snp_indices, dtype=np.int32)

    inb_target = np.asarray(inb_target, dtype=np.float64)
    if inb_target.shape[0] != ind_indices_np.shape[0]:
        raise ValueError("inb_target length must match number of individuals in ind_indices.")

    lo, hi = bounds
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError("bounds must be positive with hi > lo.")

    # Golden-section search on [lo, hi].
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    invphi = 1.0 / phi
    invphi2 = 1.0 / (phi**2)

    a, b = float(lo), float(hi)
    h = b - a
    if h <= tol:
        ss = _ssd_inb_for_param(
            (a + b) / 2.0, dmodel, depth, genon, p, inb_target, ind_indices_np, snp_indices_np
        )
        return DepthParamFitResult(
            dmodel=dmodel,
            param_opt=(a + b) / 2.0,
            ss_min=ss,
            n_ind=ind_indices_np.size,
            n_snp=(snp_indices_np.size if snp_indices_np is not None else n_snp),
        )

    # Required steps to get interval size <= tol.
    n_iter = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    fc = _ssd_inb_for_param(
        c, dmodel, depth, genon, p, inb_target, ind_indices_np, snp_indices_np
    )
    fd = _ssd_inb_for_param(
        d, dmodel, depth, genon, p, inb_target, ind_indices_np, snp_indices_np
    )

    for _ in range(min(n_iter, max_iter)):
        if fc < fd:
            b, d, fd = d, c, fc
            h = b - a
            c = a + invphi2 * h
            fc = _ssd_inb_for_param(
                c, dmodel, depth, genon, p, inb_target, ind_indices_np, snp_indices_np
            )
        else:
            a, c, fc = c, d, fd
            h = b - a
            d = a + invphi * h
            fd = _ssd_inb_for_param(
                d, dmodel, depth, genon, p, inb_target, ind_indices_np, snp_indices_np
            )
        if h <= tol:
            break

    if fc < fd:
        param_opt = c
        ss_opt = fc
    else:
        param_opt = d
        ss_opt = fd

    return DepthParamFitResult(
        dmodel=dmodel,
        param_opt=float(param_opt),
        ss_min=float(ss_opt),
        n_ind=ind_indices_np.size,
        n_snp=(snp_indices_np.size if snp_indices_np is not None else n_snp),
    )

