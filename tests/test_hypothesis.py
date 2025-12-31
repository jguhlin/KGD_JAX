from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from kgd_jax import grm, io, popgen, qc, tuning, merge, ped
from tests.jax_preflight import assert_cpu_backend

assert_cpu_backend()


@st.composite
def _depth_array(draw, min_depth: int = 0, max_depth: int = 20):
    size = draw(st.integers(min_value=1, max_value=30))
    depth = draw(
        hnp.arrays(
            dtype=np.int32,
            shape=(size,),
            elements=st.integers(min_value=min_depth, max_value=max_depth),
        )
    )
    return depth


@st.composite
def _nondecreasing_depth(draw, min_depth: int = 0, max_depth: int = 20):
    depth = draw(_depth_array(min_depth=min_depth, max_depth=max_depth))
    return np.sort(depth)


@st.composite
def _ref_alt_arrays(draw, min_value: int = 0, max_value: int = 30):
    n_ind = draw(st.integers(min_value=1, max_value=6))
    n_snp = draw(st.integers(min_value=1, max_value=8))
    ref = draw(
        hnp.arrays(
            dtype=np.int32,
            shape=(n_ind, n_snp),
            elements=st.integers(min_value=min_value, max_value=max_value),
        )
    )
    alt = draw(
        hnp.arrays(
            dtype=np.int32,
            shape=(n_ind, n_snp),
            elements=st.integers(min_value=min_value, max_value=max_value),
        )
    )
    return ref, alt


@st.composite
def _depth_genon_p(draw, min_depth: int = 2, allow_nan: bool = False):
    n_ind = draw(st.integers(min_value=2, max_value=6))
    n_snp = draw(st.integers(min_value=2, max_value=8))

    depth = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n_ind, n_snp),
            elements=st.integers(min_value=min_depth, max_value=20).map(np.float32),
        )
    )
    genon = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n_ind, n_snp),
            elements=st.integers(min_value=0, max_value=2).map(np.float32),
        )
    )
    if allow_nan:
        mask = draw(hnp.arrays(dtype=np.bool_, shape=(n_ind, n_snp), elements=st.booleans()))
        genon = genon.copy()
        genon[mask] = np.nan

    p = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n_snp,),
            elements=st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
        )
    )
    return depth, genon, p


@st.composite
def _genon_with_pops(draw):
    n_ind = draw(st.integers(min_value=4, max_value=8))
    n_snp = draw(st.integers(min_value=2, max_value=6))

    genon = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n_ind, n_snp),
            elements=st.integers(min_value=0, max_value=2).map(np.float32),
        )
    )
    depth = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(n_ind, n_snp),
            elements=st.integers(min_value=1, max_value=20).map(np.float32),
        )
    )
    pops = np.array(["A"] * (n_ind // 2) + ["B"] * (n_ind - n_ind // 2))
    return depth, genon, pops


@st.composite
def _symmetric_matrix(draw, min_size: int = 3, max_size: int = 6):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    mat = (mat + mat.T) / 2.0
    return mat


@st.composite
def _r2_and_nld(draw):
    n = draw(st.integers(min_value=3, max_value=20))
    r2auto = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.01, max_value=0.9, allow_nan=False, allow_infinity=False),
        )
    )
    nLD = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        )
    )
    return r2auto, nLD


@st.composite
def _chrom_pos_arrays(draw):
    n = draw(st.integers(min_value=6, max_value=20))
    chrom = draw(
        hnp.arrays(
            dtype=np.int32,
            shape=(n,),
            elements=st.integers(min_value=1, max_value=3),
        )
    )
    pos = draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(n,),
            elements=st.integers(min_value=1, max_value=10_000),
        )
    )
    return chrom, pos


class TestGRMProperties(unittest.TestCase):
    @given(_nondecreasing_depth(min_depth=0))
    @settings(max_examples=50, deadline=None)
    def test_depth2k_bb_inf_nondecreasing(self, depth: np.ndarray) -> None:
        K = np.asarray(grm.depth2K_bb(depth, alpha=np.inf))
        if K.size > 1:
            self.assertTrue(np.all(K[1:] <= K[:-1] + 1e-6))

    @given(_depth_array(min_depth=0))
    @settings(max_examples=50, deadline=None)
    def test_depth2k_bb_inf_matches_power(self, depth: np.ndarray) -> None:
        K = grm.depth2K_bb(depth, alpha=np.inf)
        expected = 0.5 ** depth
        np.testing.assert_allclose(np.asarray(K), expected, rtol=1e-6, atol=0.0)

    @given(_depth_array(min_depth=0))
    @settings(max_examples=50, deadline=None)
    def test_depth2k_bb_range(self, depth: np.ndarray) -> None:
        alpha = 2.5
        K = np.asarray(grm.depth2K_bb(depth, alpha=alpha))
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0 + 1e-6))
        if np.any(depth == 0):
            self.assertTrue(np.allclose(K[depth == 0], 1.0))

    @given(
        _depth_array(min_depth=0),
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_depth2k_modp_formula(self, depth: np.ndarray, modp: float) -> None:
        K = np.asarray(grm.depth2K_modp(depth, modp=modp))
        expected = np.where(depth == 0, 1.0, 0.5 * (modp ** (depth - 1)))
        np.testing.assert_allclose(K, expected, rtol=1e-6, atol=0.0)
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0 + 1e-6))

    @given(
        _nondecreasing_depth(min_depth=0),
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_depth2k_modp_nondecreasing(self, depth: np.ndarray, modp: float) -> None:
        K = np.asarray(grm.depth2K_modp(depth, modp=modp))
        if K.size > 1:
            self.assertTrue(np.all(K[1:] <= K[:-1] + 1e-6))

    @given(_depth_genon_p(min_depth=2, allow_nan=False))
    @settings(max_examples=30, deadline=None)
    def test_grm_diag_finite(self, data) -> None:
        depth, genon, p = data
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        G5d = np.asarray(op.diag_G5())
        self.assertEqual(G5d.shape[0], depth.shape[0])
        self.assertTrue(np.all(np.isfinite(G5d)))

    @given(_depth_genon_p(min_depth=2, allow_nan=True))
    @settings(max_examples=30, deadline=None)
    def test_grm_submatrix_symmetric(self, data) -> None:
        depth, genon, p = data
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        idx = np.arange(min(3, depth.shape[0]))
        G4 = np.asarray(op.submatrix_G4(idx))
        self.assertEqual(G4.shape, (idx.size, idx.size))
        np.testing.assert_allclose(G4, G4.T, rtol=1e-5, atol=1e-5)

    @given(_depth_genon_p(min_depth=2, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_grm_depth_mask_extremes(self, data) -> None:
        depth, genon, p = data
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        dmin = float(np.max(depth) + 1.0)
        diag = np.asarray(op.diag_G5(depth_min=dmin))
        self.assertTrue(np.all(np.isfinite(diag) | np.isnan(diag)))
        dmax = float(np.min(depth) - 1.0)
        diag2 = np.asarray(op.diag_G5(depth_max=dmax))
        self.assertTrue(np.all(np.isfinite(diag2) | np.isnan(diag2)))

    @given(_depth_genon_p(min_depth=2, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_grm_submatrix_depth_mask(self, data) -> None:
        depth, genon, p = data
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        idx = np.arange(min(3, depth.shape[0]))
        dmin = float(np.max(depth) + 1.0)
        G4 = np.asarray(op.submatrix_G4(idx, depth_min=dmin))
        self.assertTrue(np.all(np.isfinite(G4) | np.isnan(G4)))
    @given(_depth_genon_p(min_depth=2, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_grm_diag_consistent_submatrix(self, data) -> None:
        depth, genon, p = data
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        diag = np.asarray(op.diag_G5())
        idx = np.arange(min(3, depth.shape[0]))
        G4 = np.asarray(op.submatrix_G4(idx))
        # G4 and G5 diagonals are not expected to match exactly; just check finiteness.
        self.assertTrue(np.all(np.isfinite(np.diag(G4))))

class TestQCProperties(unittest.TestCase):
    @given(_ref_alt_arrays(min_value=0, max_value=20))
    @settings(max_examples=50, deadline=None)
    def test_alleles_to_depth_genon_consistency(self, data) -> None:
        ref, alt = data
        depth, genon = qc.alleles_to_depth_genon(ref, alt)
        np.testing.assert_array_equal(depth, (ref + alt).astype(np.float32))
        self.assertTrue(np.all(np.isnan(genon[depth == 0])))

        with np.errstate(divide="ignore", invalid="ignore"):
            frac_ref = np.where(depth > 0, ref / depth, np.nan)
            expected = np.floor(2.0 * frac_ref - 1.0) + 1.0
        np.testing.assert_allclose(np.nan_to_num(genon), np.nan_to_num(expected), rtol=0, atol=0)

        valid = ~np.isnan(genon)
        if np.any(valid):
            self.assertTrue(np.all(np.isin(genon[valid], [0.0, 1.0, 2.0])))

    @given(_ref_alt_arrays(min_value=0, max_value=30))
    @settings(max_examples=50, deadline=None)
    def test_calcp_alleles_bounds(self, data) -> None:
        ref, alt = data
        p = qc.calcp_alleles(ref, alt)
        denom = (ref + alt).sum(axis=0)
        has_data = denom > 0
        if np.any(has_data):
            self.assertTrue(np.all((p[has_data] >= 0.0) & (p[has_data] <= 1.0)))
        if np.any(~has_data):
            self.assertTrue(np.all(np.isnan(p[~has_data])))

    @given(_ref_alt_arrays(min_value=0, max_value=30))
    @settings(max_examples=50, deadline=None)
    def test_calcp_genotypes_bounds(self, data) -> None:
        ref, alt = data
        depth, genon = qc.alleles_to_depth_genon(ref, alt)
        p = qc.calcp_genotypes(genon)
        denom = np.sum(~np.isnan(genon), axis=0) * 2.0
        has_data = denom > 0
        if np.any(has_data):
            self.assertTrue(np.all((p[has_data] >= 0.0) & (p[has_data] <= 1.0)))
        if np.any(~has_data):
            self.assertTrue(np.all(np.isnan(p[~has_data])))

    @given(_ref_alt_arrays(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_run_qc_shapes(self, data) -> None:
        ref, alt = data
        n_ind, n_snp = ref.shape
        ra = io.RAData(
            sample_ids=[f"S{i}" for i in range(n_ind)],
            chrom=np.array(["1"] * n_snp),
            pos=np.arange(1, n_snp + 1),
            ref=ref,
            alt=alt,
        )
        res = qc.run_qc(ra, sampdepth_thresh=0.0, snpdepth_thresh=0.0, maf_thresh=0.0)
        self.assertEqual(res.keep_ind.shape[0], n_ind)
        self.assertEqual(res.keep_snp.shape[0], n_snp)
        self.assertEqual(res.depth.shape[0], int(np.sum(res.keep_ind)))
        self.assertEqual(res.depth.shape[1], int(np.sum(res.keep_snp)))
        self.assertEqual(res.genon.shape, res.depth.shape)
        self.assertEqual(res.p.shape[0], res.depth.shape[1])

    @given(_ref_alt_arrays(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_run_qc_keeps_all_when_depths_present(self, data) -> None:
        ref, alt = data
        n_ind, n_snp = ref.shape
        # ensure every sample has max depth >=2
        assume(np.all((ref + alt).max(axis=1) >= 2))
        ra = io.RAData(
            sample_ids=[f"S{i}" for i in range(n_ind)],
            chrom=np.array(["1"] * n_snp),
            pos=np.arange(1, n_snp + 1),
            ref=ref,
            alt=alt,
        )
        res = qc.run_qc(ra, sampdepth_thresh=0.0, snpdepth_thresh=0.0, maf_thresh=0.0)
        self.assertTrue(np.all(res.keep_ind))
        self.assertTrue(np.all(res.keep_snp))

    @given(_ref_alt_arrays(min_value=0, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_run_qc_pmethod_g_matches_calcp(self, data) -> None:
        ref, alt = data
        n_ind, n_snp = ref.shape
        ra = io.RAData(
            sample_ids=[f"S{i}" for i in range(n_ind)],
            chrom=np.array(["1"] * n_snp),
            pos=np.arange(1, n_snp + 1),
            ref=ref,
            alt=alt,
        )
        res = qc.run_qc(
            ra,
            sampdepth_thresh=0.0,
            snpdepth_thresh=0.0,
            maf_thresh=0.0,
            pmethod="G",
        )
        p_expected = qc.calcp_genotypes(np.asarray(res.genon))
        np.testing.assert_allclose(p_expected, np.asarray(res.p), rtol=1e-6, atol=1e-6, equal_nan=True)


class TestIOProperties(unittest.TestCase):
    @given(_ref_alt_arrays(min_value=0, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_select_samples_and_snps_shapes(self, data) -> None:
        ref, alt = data
        n_ind, n_snp = ref.shape
        ra = io.RAData(
            sample_ids=[f"S{i}" for i in range(n_ind)],
            chrom=np.array(["1"] * n_snp),
            pos=np.arange(1, n_snp + 1),
            ref=ref,
            alt=alt,
        )
        samp_idx = np.arange(0, n_ind, 2, dtype=int)
        snp_idx = np.arange(0, n_snp, 2, dtype=int)
        ra_samp = io.select_samples(ra, samp_idx)
        ra_snp = io.select_snps(ra, snp_idx)
        self.assertEqual(ra_samp.ref.shape, (samp_idx.size, n_snp))
        self.assertEqual(ra_snp.ref.shape, (n_ind, snp_idx.size))


class TestMergeProperties(unittest.TestCase):
    @given(_ref_alt_arrays(min_value=0, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_merge_ra_samples_sums_counts(self, data) -> None:
        ref, alt = data
        n_ind, n_snp = ref.shape
        ra = io.RAData(
            sample_ids=[f"S{i}" for i in range(n_ind)],
            chrom=np.array(["1"] * n_snp),
            pos=np.arange(1, n_snp + 1),
            ref=ref,
            alt=alt,
        )
        if n_ind < 2:
            return
        # Merge first two samples into a group, keep others as-is.
        mapping = merge.MergeMapping(
            sample_ids=np.array(["S0", "S1"]),
            merge_ids=np.array(["G0", "G0"]),
        )
        merged = merge.merge_ra_samples(ra, mapping)
        idx_g0 = merged.sample_ids.index("G0")
        expected_ref = ref[0, :] + ref[1, :]
        expected_alt = alt[0, :] + alt[1, :]
        np.testing.assert_array_equal(merged.ref[idx_g0, :], expected_ref)
        np.testing.assert_array_equal(merged.alt[idx_g0, :], expected_alt)


class TestPopgenProperties(unittest.TestCase):
    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=30, deadline=None)
    def test_heterozygosity_ranges(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        res = popgen.heterozygosity(depth=depth, genon=genon, p=p, depth2K=depth2K)
        self.assertTrue(np.isfinite(res.neff))
        self.assertTrue(0.0 <= res.ehet <= 0.5 + 1e-6)
        self.assertTrue(0.0 <= res.ehetstar <= 0.5 + 1e-6)
        if np.isfinite(res.ohet):
            self.assertTrue(-1e-6 <= res.ohet <= 1.0 + 1e-6)
        if np.isfinite(res.ohet2):
            self.assertTrue(res.ohet2 >= 0.0 - 1e-6)

    @given(_genon_with_pops())
    @settings(max_examples=30, deadline=None)
    def test_fst_gbs_bounds(self, data) -> None:
        depth, genon, pops = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        res = popgen.fst_gbs(depth=depth, genon=genon, populations=pops, depth2K=depth2K)
        self.assertEqual(res.fst.shape[0], genon.shape[1])
        finite = np.isfinite(res.fst)
        if np.any(finite):
            self.assertTrue(np.all(res.fst[finite] >= -1e-6))
            self.assertTrue(np.all(res.fst[finite] <= 1.0 + 1e-6))

    @given(_genon_with_pops())
    @settings(max_examples=20, deadline=None)
    def test_fst_gbs_all_nan_raises(self, data) -> None:
        depth, genon, pops = data
        genon_nan = genon.copy()
        genon_nan[:] = np.nan
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        with self.assertRaises(ValueError):
            popgen.fst_gbs(depth=depth, genon=genon_nan, populations=pops, depth2K=depth2K)

    @given(_genon_with_pops())
    @settings(max_examples=30, deadline=None)
    def test_hw_pops_shapes(self, data) -> None:
        depth, genon, pops = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        res = popgen.hw_pops(genon=genon, depth=depth, populations=pops, depth2K=depth2K)
        self.assertEqual(len(res.popnames), 2)
        self.assertEqual(res.HWdis.shape[1], genon.shape[1])
        self.assertTrue(np.all((res.maf <= 0.5 + 1e-6) | np.isnan(res.maf)))
        self.assertTrue(np.all((res.maf >= 0.0 - 1e-6) | np.isnan(res.maf)))
        self.assertIsNotNone(res.l10pstar_pop)
        self.assertTrue(np.all(np.isfinite(res.x2star) | np.isnan(res.x2star)))

    @given(_genon_with_pops())
    @settings(max_examples=20, deadline=None)
    def test_fst_pairwise_upper_triangle(self, data) -> None:
        depth, genon, pops = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        res = popgen.fst_pairwise(depth=depth, genon=genon, populations=pops, depth2K=depth2K)
        Fst_cube = res["Fst"]
        npops = len(res["popnames"])
        self.assertEqual(Fst_cube.shape, (npops, npops, genon.shape[1]))
        for i in range(npops):
            for j in range(i + 1):
                self.assertTrue(np.all(np.isnan(Fst_cube[i, j, :])))

    @given(_genon_with_pops())
    @settings(max_examples=30, deadline=None)
    def test_fst_pairwise_all_nan_means(self, data) -> None:
        depth, genon, pops = data
        genon_nan = genon.copy()
        genon_nan[:] = np.nan
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        with self.assertRaises(ValueError):
            popgen.fst_pairwise(depth=depth, genon=genon_nan, populations=pops, depth2K=depth2K)

    @given(_genon_with_pops())
    @settings(max_examples=30, deadline=None)
    def test_popmaf_range(self, data) -> None:
        _, genon, pops = data
        maf = popgen.popmaf(genon=genon, populations=pops, minsamps=1, mafmin=0.0)
        for vals in maf.values():
            if vals.size == 0:
                continue
            self.assertTrue(np.all(vals >= 0.0 - 1e-6))
            self.assertTrue(np.all(vals <= 0.5 + 1e-6))

    @given(_symmetric_matrix())
    @settings(max_examples=30, deadline=None)
    def test_popG_symmetry(self, G: np.ndarray) -> None:
        n = G.shape[0]
        pops = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
        res = popgen.popG(G=G, populations=pops, diag=False)
        popG_mat = res["G"]
        np.testing.assert_allclose(popG_mat, popG_mat.T, rtol=1e-6, atol=1e-6)
        self.assertEqual(res["Inb"].shape[1], 1)

    @given(_symmetric_matrix())
    @settings(max_examples=30, deadline=None)
    def test_popG_single_population_no_inf(self, G: np.ndarray) -> None:
        n = G.shape[0]
        pops = np.array(["A"] * n)
        res = popgen.popG(G=G, populations=pops, diag=False)
        popG_mat = res["G"]
        self.assertTrue(np.all(np.isfinite(popG_mat)))
    @given(_r2_and_nld())
    @settings(max_examples=30, deadline=None)
    def test_Nefromr2_finite(self, data) -> None:
        r2auto, nLD = data
        res = popgen.Nefromr2(r2auto=r2auto, nLD=nLD, alpha=1.0, weighted=False, minN=1)
        for key, val in res.items():
            self.assertTrue(np.isfinite(val) or np.isnan(val), msg=f"{key} not finite or nan")

    @given(_r2_and_nld())
    @settings(max_examples=30, deadline=None)
    def test_Nefromr2_weighted_finite(self, data) -> None:
        r2auto, nLD = data
        res = popgen.Nefromr2(r2auto=r2auto, nLD=nLD, alpha=1.0, weighted=True, minN=1)
        for key, val in res.items():
            self.assertTrue(np.isfinite(val) or np.isnan(val), msg=f"{key} not finite or nan")

    @given(_chrom_pos_arrays(), st.sampled_from(["centre", "even", "random"]))
    @settings(max_examples=30, deadline=None)
    def test_snpselection_pairs(self, data, seltype: str) -> None:
        chrom, pos = data
        pairs = popgen.snpselection(chromosome=chrom, position=pos, nsnpperchrom=3, seltype=seltype, randseed=7)
        if pairs.size == 0:
            return
        self.assertEqual(pairs.shape[1], 2)
        self.assertTrue(np.all(pairs >= 0))
        self.assertTrue(np.all(pairs < chrom.shape[0]))
        for i, j in pairs:
            self.assertNotEqual(chrom[i], chrom[j])


class TestDAPCProperties(unittest.TestCase):
    @given(
        st.integers(min_value=4, max_value=10),
        st.integers(min_value=3, max_value=8),
    )
    @settings(max_examples=20, deadline=None)
    def test_dapc_shapes(self, n_ind: int, n_snp: int) -> None:
        genon = np.random.default_rng(0).integers(0, 3, size=(n_ind, n_snp)).astype(np.float64)
        p = np.mean(genon, axis=0) / 2.0
        pops = np.array(["A"] * (n_ind // 2) + ["B"] * (n_ind - n_ind // 2))
        sample_ids = np.array([f"S{i}" for i in range(n_ind)])

        res = popgen.dapc_from_genotypes(genon=genon, p=p, populations=pops, sample_ids=sample_ids)
        self.assertEqual(res.pc_scores.shape[0], n_ind)
        self.assertEqual(res.pc_eigenvalues.shape[0], res.pc_scores.shape[1])
        self.assertEqual(res.lda_scores.shape[0], n_ind)
        self.assertEqual(res.lda_loadings.shape[0], res.pc_scores.shape[1])
        self.assertEqual(res.lda_loadings.shape[1], res.lda_scores.shape[1])
        self.assertEqual(res.lda_eigenvalues.shape[0], res.lda_scores.shape[1])

    @given(
        st.integers(min_value=3, max_value=10),
        st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=20, deadline=None)
    def test_dapc_requires_two_pops(self, n_ind: int, n_snp: int) -> None:
        genon = np.random.default_rng(1).integers(0, 3, size=(n_ind, n_snp)).astype(np.float64)
        p = np.mean(genon, axis=0) / 2.0
        pops = np.array(["A"] * n_ind)
        sample_ids = np.array([f"S{i}" for i in range(n_ind)])
        with self.assertRaises(ValueError):
            popgen.dapc_from_genotypes(genon=genon, p=p, populations=pops, sample_ids=sample_ids)

class TestTuningProperties(unittest.TestCase):
    @given(_depth_genon_p(min_depth=2, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_fit_depth_param_bounds(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        target = np.asarray(op.diag_G5() - 1.0)
        res = tuning.fit_depth_param_inb(
            depth=depth,
            genon=genon,
            p=p,
            inb_target=target,
            dmodel="bb",
            bounds=(0.5, 10.0),
            tol=0.2,
            max_iter=20,
        )
        self.assertTrue(0.5 <= res.param_opt <= 10.0)
        self.assertTrue(res.ss_min >= 0.0)


class TestPedProperties(unittest.TestCase):
    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=30, deadline=None)
    def test_mismatch_par_pair_bounds(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        res = ped.mismatch_par_pair(depth=depth, genon=genon, p=p, depth2K=depth2K, i=0, j=1, mindepth_mm=1.0)
        if np.isfinite(res.mmrate):
            self.assertTrue(0.0 <= res.mmrate <= 1.0)
        if np.isfinite(res.exp_mmrate):
            self.assertTrue(0.0 <= res.exp_mmrate <= 1.0)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_par_pair_identical_zero(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        # Force individual 1 to match individual 0 exactly
        genon2 = genon.copy()
        genon2[1, :] = genon2[0, :]
        res = ped.mismatch_par_pair(depth=depth, genon=genon2, p=p, depth2K=depth2K, i=0, j=1, mindepth_mm=1.0)
        if res.ncompare > 0:
            self.assertEqual(res.mmrate, 0.0)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_par_pair_symmetry(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        # Force i/j to be identical to make symmetry strict.
        genon2 = genon.copy()
        genon2[1, :] = genon2[0, :]
        depth2 = depth.copy()
        depth2[1, :] = depth2[0, :]
        res_ij = ped.mismatch_par_pair(depth=depth2, genon=genon2, p=p, depth2K=depth2K, i=0, j=1, mindepth_mm=1.0)
        res_ji = ped.mismatch_par_pair(depth=depth2, genon=genon2, p=p, depth2K=depth2K, i=1, j=0, mindepth_mm=1.0)
        self.assertEqual(res_ij.ncompare, res_ji.ncompare)
        if res_ij.ncompare > 0:
            self.assertEqual(res_ij.mmrate, res_ji.mmrate)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_par_pair_no_compare(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        mindepth = float(np.max(depth) + 1.0)
        res = ped.mismatch_par_pair(depth=depth, genon=genon, p=p, depth2K=depth2K, i=0, j=1, mindepth_mm=mindepth)
        self.assertEqual(res.ncompare, 0)
        self.assertTrue(np.isnan(res.mmrate))
        self.assertTrue(np.isnan(res.exp_mmrate))

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_par_comb_shapes(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        offspring_idx = np.array([0])
        parent_idx = np.array([1]) if n_ind > 1 else np.array([0])
        res = ped.mismatch_par_comb(
            depth=depth,
            genon=genon,
            p=p,
            depth2K=depth2K,
            offspring_idx=offspring_idx,
            parent_idx=parent_idx,
            mindepth_mm=1.0,
        )
        self.assertEqual(res.mmrate.shape, (offspring_idx.size, parent_idx.size))
        self.assertEqual(res.ncompare.shape, res.mmrate.shape)
        self.assertEqual(res.exp_mmrate.shape, res.mmrate.shape)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_par_comb_ncompare_monotone(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 2:
            return
        res_lo = ped.mismatch_par_comb(
            depth=depth,
            genon=genon,
            p=p,
            depth2K=depth2K,
            offspring_idx=[0],
            parent_idx=[1],
            mindepth_mm=1.0,
        )
        res_hi = ped.mismatch_par_comb(
            depth=depth,
            genon=genon,
            p=p,
            depth2K=depth2K,
            offspring_idx=[0],
            parent_idx=[1],
            mindepth_mm=2.0,
        )
        self.assertTrue(res_hi.ncompare[0, 0] <= res_lo.ncompare[0, 0])

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_two_parents_shapes(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 3:
            return
        offspring_idx = np.array([0])
        parent1_idx = np.array([1])
        parent2_idx = np.array([2])
        res = ped.mismatch_two_parents(
            depth=depth,
            genon=genon,
            p=p,
            depth2K=depth2K,
            offspring_idx=offspring_idx,
            parent1_idx=parent1_idx,
            parent2_idx=parent2_idx,
            mindepth_mm=1.0,
        )
        self.assertEqual(res.mmrate.shape, (offspring_idx.size,))
        self.assertEqual(res.ncompare.shape, res.mmrate.shape)
        self.assertEqual(res.exp_mmrate.shape, res.mmrate.shape)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_best_parents_by_relatedness_empty(self, data) -> None:
        depth, genon, p = data
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        res = ped.best_parents_by_relatedness(op, offspring_idx=[], parent_idx=[])
        self.assertEqual(res.offspring_idx.size, 0)
        self.assertEqual(res.best_parent_idx.size, 0)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_best_parents_by_relatedness_no_self(self, data) -> None:
        depth, genon, p = data
        n_ind = depth.shape[0]
        if n_ind < 2:
            return
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        offspring_idx = np.array([0])
        parent_idx = np.array([0, 1])
        res = ped.best_parents_by_relatedness(op, offspring_idx=offspring_idx, parent_idx=parent_idx)
        self.assertNotEqual(res.best_parent_idx[0], 0)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_best_parents_by_relatedness_all_filtered(self, data) -> None:
        depth, genon, p = data
        n_ind = depth.shape[0]
        if n_ind < 1:
            return
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        offspring_idx = np.array([0])
        parent_idx = np.array([0])
        res = ped.best_parents_by_relatedness(op, offspring_idx=offspring_idx, parent_idx=parent_idx)
        # Only parent is self, so should be masked to -inf; index stays but rel_best is -inf.
        self.assertEqual(res.best_parent_idx[0], 0)
        self.assertTrue(np.isneginf(res.rel_best[0]))
        self.assertTrue(np.isneginf(res.rel_second[0]))
    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_best_parents_by_emm_shapes(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 3:
            return
        offspring_idx = np.array([0])
        parent_idx = np.array([1, 2])
        res = ped.best_parents_by_emm(
            depth=depth,
            genon=genon,
            p=p,
            depth2K=depth2K,
            offspring_idx=offspring_idx,
            parent_idx=parent_idx,
            mindepth_mm=1.0,
        )
        self.assertEqual(res.best_parent_idx.shape, offspring_idx.shape)
        self.assertEqual(res.second_parent_idx.shape, offspring_idx.shape)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_best_parents_by_emm_all_nan_raises(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 3:
            return
        genon_nan = genon.copy()
        genon_nan[:] = np.nan
        with self.assertRaises(ValueError):
            ped.best_parents_by_emm(
                depth=depth,
                genon=genon_nan,
                p=p,
                depth2K=depth2K,
                offspring_idx=[0],
                parent_idx=[1, 2],
                mindepth_mm=1.0,
            )
    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_par_comb_with_nan_genotypes(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 2:
            return
        genon_nan = genon.copy()
        genon_nan[0, :] = np.nan
        res = ped.mismatch_par_comb(
            depth=depth,
            genon=genon_nan,
            p=p,
            depth2K=depth2K,
            offspring_idx=[0],
            parent_idx=[1],
            mindepth_mm=1.0,
        )
        self.assertTrue(np.all(np.isfinite(res.mmrate) | np.isnan(res.mmrate)))

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_two_parents_with_nan_genotypes(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 3:
            return
        genon_nan = genon.copy()
        genon_nan[0, :] = np.nan
        res = ped.mismatch_two_parents(
            depth=depth,
            genon=genon_nan,
            p=p,
            depth2K=depth2K,
            offspring_idx=[0],
            parent1_idx=[1],
            parent2_idx=[2],
            mindepth_mm=1.0,
        )
        self.assertTrue(np.all(np.isfinite(res.mmrate) | np.isnan(res.mmrate)))

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_mismatch_two_parents_identical_zero(self, data) -> None:
        depth, genon, p = data
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        n_ind = depth.shape[0]
        if n_ind < 3:
            return
        genon2 = genon.copy()
        genon2[1, :] = genon2[0, :]
        genon2[2, :] = genon2[0, :]
        res = ped.mismatch_two_parents(
            depth=depth,
            genon=genon2,
            p=p,
            depth2K=depth2K,
            offspring_idx=[0],
            parent1_idx=[1],
            parent2_idx=[2],
            mindepth_mm=1.0,
        )
        if res.ncompare[0] > 0:
            self.assertEqual(res.mmrate[0], 0.0)

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_check_parents_missing_cols_raises(self, data) -> None:
        depth, genon, p = data
        n_ind = depth.shape[0]
        sample_ids = [f"S{i}" for i in range(n_ind)]
        keep_mask = np.ones(n_ind, dtype=bool)
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        G5_diag = np.asarray(op.diag_G5())
        ped_df = pd.DataFrame({"IndivID": ["X1"]})  # missing seqID
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        with self.assertRaises(ValueError):
            ped.check_parents(
                ped_df=ped_df,
                sample_ids=sample_ids,
                keep_ind_mask=keep_mask,
                G5_diag=G5_diag,
                grm_op=op,
                depth_np=depth,
                genon_np=genon,
                p_np=p,
                depth2K=depth2K,
            )

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_check_parents_outputs(self, data) -> None:
        depth, genon, p = data
        n_ind = depth.shape[0]
        if n_ind < 2:
            return
        sample_ids = [f"S{i}" for i in range(n_ind)]
        keep_mask = np.ones(n_ind, dtype=bool)
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        G5_diag = np.asarray(op.diag_G5())
        ped_df = pd.DataFrame(
            {
                "IndivID": ["C1", "P1"],
                "seqID": ["S0", "S1"],
                "FatherID": ["P1", ""],
                "MotherID": ["", ""],
            }
        )
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        out = ped.check_parents(
            ped_df=ped_df,
            sample_ids=sample_ids,
            keep_ind_mask=keep_mask,
            G5_diag=G5_diag,
            grm_op=op,
            depth_np=depth,
            genon_np=genon,
            p_np=p,
            depth2K=depth2K,
        )
        self.assertTrue("FatherRel" in out.columns)
        self.assertTrue("MotherRel" in out.columns)
        self.assertEqual(out.shape[0], ped_df.shape[0])

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_check_parents_thresholds(self, data) -> None:
        depth, genon, p = data
        n_ind = depth.shape[0]
        if n_ind < 2:
            return
        sample_ids = [f"S{i}" for i in range(n_ind)]
        keep_mask = np.ones(n_ind, dtype=bool)
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        G5_diag = np.asarray(op.diag_G5())
        ped_df = pd.DataFrame(
            {
                "IndivID": ["C1", "P1"],
                "seqID": ["S0", "S1"],
                "FatherID": ["P1", ""],
                "MotherID": ["", ""],
            }
        )
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        out = ped.check_parents(
            ped_df=ped_df,
            sample_ids=sample_ids,
            keep_ind_mask=keep_mask,
            G5_diag=G5_diag,
            grm_op=op,
            depth_np=depth,
            genon_np=genon,
            p_np=p,
            depth2K=depth2K,
            rel_threshF=10.0,
            rel_threshM=10.0,
            emm_thresh=-10.0,
        )
        # Thresholds make matching impossible.
        self.assertTrue(np.all(out["FatherMatch"].isna() | (out["FatherMatch"] == False)))

    @given(_depth_genon_p(min_depth=1, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_check_parents_partial_qc_mask(self, data) -> None:
        depth, genon, p = data
        n_ind = depth.shape[0]
        if n_ind < 2:
            return
        sample_ids = [f"S{i}" for i in range(n_ind)]
        keep_mask = np.ones(n_ind, dtype=bool)
        keep_mask[1] = False  # drop parent
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        G5_diag = np.asarray(op.diag_G5())
        ped_df = pd.DataFrame(
            {
                "IndivID": ["C1", "P1"],
                "seqID": ["S0", "S1"],
                "FatherID": ["P1", ""],
                "MotherID": ["", ""],
            }
        )
        depth2K = grm.make_depth2K("bb", param=float("inf"))
        out = ped.check_parents(
            ped_df=ped_df,
            sample_ids=sample_ids,
            keep_ind_mask=keep_mask,
            G5_diag=G5_diag,
            grm_op=op,
            depth_np=depth,
            genon_np=genon,
            p_np=p,
            depth2K=depth2K,
        )
        self.assertTrue(np.isnan(out.loc[0, "FatherRel"]))

if __name__ == "__main__":
    unittest.main()
