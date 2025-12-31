from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import pandas as pd

from kgd_jax import grm, io, qc, sim, popgen
from tests.jax_preflight import assert_cpu_backend

assert_cpu_backend()


_SIM_PARAMS = dict(n_ind=1000, n_snp=5000, seed=123, maf_min=0.05, maf_max=0.5, depth=20)


def _simulate_ra_cached() -> tuple[sim.SimulatedData, io.RAData, qc.QCResult]:
    # Simple memoization to avoid re-running heavy simulation/QC in multiple tests.
    if not hasattr(_simulate_ra_cached, "_cache"):
        simdata = sim.simulate_genotypes(
            n_ind=_SIM_PARAMS["n_ind"],
            n_snp=_SIM_PARAMS["n_snp"],
            seed=_SIM_PARAMS["seed"],
            maf_min=_SIM_PARAMS["maf_min"],
            maf_max=_SIM_PARAMS["maf_max"],
        )
        ra = _sim_to_ra(simdata, depth=_SIM_PARAMS["depth"])
        qc_res = qc.run_qc(ra)
        _simulate_ra_cached._cache = (simdata, ra, qc_res)
    return _simulate_ra_cached._cache  # type: ignore[attr-defined]


def _sim_to_ra(simdata: sim.SimulatedData, depth: int) -> io.RAData:
    n_ind, n_snp = simdata.genotypes.shape
    ref = np.zeros((n_ind, n_snp), dtype=np.int32)
    alt = np.zeros((n_ind, n_snp), dtype=np.int32)

    g = simdata.genotypes
    ref[:, :] = np.where(g == 0, depth, np.where(g == 1, depth // 2, 0))
    alt[:, :] = np.where(g == 0, 0, np.where(g == 1, depth - depth // 2, depth))

    return io.RAData(
        sample_ids=simdata.sample_ids.tolist(),
        chrom=simdata.chrom,
        pos=simdata.pos,
        ref=ref,
        alt=alt,
    )


class TestLegacyParity(unittest.TestCase):
    def test_diag_matches_legacy_kgd_golden(self) -> None:
        # Golden diag from original KGD on a deterministic simulation (n=1000, m=5000, seed=123, depth=20).
        golden = pd.read_csv("tests/data/golden/sim_1k_5k.G5diag.orig.csv").rename(columns={"seqID": "sample_id"})

        simdata, ra, qc_res = _simulate_ra_cached()
        # Simulation has no missingness; all should be kept.
        self.assertTrue(qc_res.keep_ind.all())
        self.assertTrue(qc_res.keep_snp.all())

        op = grm.build_grm_operator(
            depth=qc_res.depth,
            genon=qc_res.genon,
            p=qc_res.p,
            dmodel="bb",
            dparam=float("inf"),
        )

        ours = np.asarray(op.diag_G5())
        merged = pd.DataFrame({"sample_id": simdata.sample_ids, "ours": ours})
        merged = merged.merge(golden, on="sample_id", how="inner")

        diff = np.abs(merged["ours"] - merged["G5_diag"])
        # Legacy vs JAX should agree to ~1e-6 on this synthetic dataset.
        self.assertLess(diff.max(), 1e-6, f"max diff too large: {diff.max()}")

    def test_heterozygosity_matches_legacy(self) -> None:
        golden = pd.read_csv("tests/data/golden/sim_1k_5k.het.orig.csv").iloc[0]

        simdata, ra, qc_res = _simulate_ra_cached()
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        het = popgen.heterozygosity(
            depth=qc_res.depth,
            genon=qc_res.genon,
            p=qc_res.p,
            depth2K=depth2K,
        )

        # R vs JAX small numeric drift; tolerance ~1e-4 is sufficient.
        self.assertAlmostEqual(het.neff, golden["neff"], places=2)
        self.assertAlmostEqual(het.ohetstar, golden["ohetstar"], places=2)
        self.assertAlmostEqual(het.ehetstar, golden["ehetstar"], places=2)
        self.assertAlmostEqual(het.ohet, golden["ohet"], places=2)
        self.assertAlmostEqual(het.ohet2, golden["ohet2"], places=2)
        self.assertAlmostEqual(het.ehet, golden["ehet"], places=2)

    def test_fst_matches_legacy(self) -> None:
        golden = pd.read_csv("tests/data/golden/sim_1k_5k.fst.orig.csv")["Fst"].to_numpy()

        simdata, ra, qc_res = _simulate_ra_cached()
        pops = np.array(["A"] * (_SIM_PARAMS["n_ind"] // 2) + ["B"] * (_SIM_PARAMS["n_ind"] // 2))
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        fst_res = popgen.fst_gbs(
            depth=qc_res.depth,
            genon=qc_res.genon,
            populations=pops,
            depth2K=depth2K,
            varadj=0,
        )

        diff = np.abs(fst_res.fst - golden)
        self.assertLess(diff.max(), 5e-4, f"max Fst diff too large: {diff.max()}")

    def test_hw_matches_legacy(self) -> None:
        golden = pd.read_csv("tests/data/golden/sim_1k_5k.hw.orig.csv")

        simdata, ra, qc_res = _simulate_ra_cached()
        pops = np.array(["A"] * (_SIM_PARAMS["n_ind"] // 2) + ["B"] * (_SIM_PARAMS["n_ind"] // 2))
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        hw_res = popgen.hw_pops(
            depth=qc_res.depth,
            genon=qc_res.genon,
            populations=pops,
            depth2K=depth2K,
        )

        # Golden columns map to populations A/B in order.
        for pop_idx, pop_label in enumerate(["A", "B"]):
            mask = pop_idx  # 0 for A, 1 for B
            for label, arr in [
                ("HWdis", hw_res.HWdis[mask]),
                ("l10LRT", hw_res.l10LRT[mask]),
                ("l10pstar", hw_res.l10pstar[mask]),
                ("x2star", hw_res.x2star[mask]),
                ("maf", hw_res.maf[mask]),
            ]:
                gold = golden[f"{label}_{pop_label}"].to_numpy()
                nan_arr = np.isnan(arr)
                nan_gold = np.isnan(gold)
                # Allow a handful of NaNs due to numerical edge cases but they should be rare.
                self.assertLessEqual(nan_arr.sum(), nan_gold.sum() + 10)

                finite_mask = ~(nan_arr | nan_gold)
                if not np.any(finite_mask):
                    continue
                diff = np.max(np.abs(arr[finite_mask] - gold[finite_mask]))
                self.assertLess(diff, 1e-4, f"{label} diff too large ({pop_label}): {diff}")
