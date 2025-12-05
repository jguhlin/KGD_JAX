from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from kgd_jax import grm, io, popgen, qc, tuning


class TestQCAndGRM(unittest.TestCase):
    def setUp(self) -> None:
        # Tiny synthetic RA dataset: 3 samples × 2 SNPs.
        # S1: homozygous ref at SNP1, alt at SNP2
        # S2: homozygous alt at SNP1, ref at SNP2
        # S3: balanced reads (approx het) at both SNPs.
        chrom = np.array(["1", "1"])
        pos = np.array([100, 200])
        ref = np.array(
            [
                [10, 0],  # S1
                [0, 10],  # S2
                [5, 5],  # S3
            ],
            dtype=np.int32,
        )
        alt = np.array(
            [
                [0, 10],  # S1
                [10, 0],  # S2
                [5, 5],  # S3
            ],
            dtype=np.int32,
        )
        self.ra = io.RAData(
            sample_ids=["S1", "S2", "S3"],
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
        )

    def test_qc_and_grm_diag(self) -> None:
        qc_res = qc.run_qc(self.ra)
        # All individuals/SNPs should pass in this tiny example.
        self.assertTrue(qc_res.keep_ind.all())
        self.assertTrue(qc_res.keep_snp.all())
        self.assertEqual(qc_res.depth.shape, (3, 2))

        op = grm.build_grm_operator(
            depth=qc_res.depth,
            genon=qc_res.genon,
            p=qc_res.p,
            dmodel="bb",
            dparam=float("inf"),
        )
        G5d = op.diag_G5()
        self.assertEqual(G5d.shape, (3,))
        self.assertTrue(np.all(np.isfinite(np.asarray(G5d))))

    def test_depth_tuning_trivial(self) -> None:
        qc_res = qc.run_qc(self.ra)
        depth = qc_res.depth
        genon = qc_res.genon
        p = qc_res.p

        # Use current inbreeding as "target" to ensure ss_min ~ 0
        op = grm.build_grm_operator(depth=depth, genon=genon, p=p, dmodel="bb", dparam=float("inf"))
        G5d = op.diag_G5()
        F_hat = np.asarray(G5d - 1.0)

        fit_res = tuning.fit_depth_param_inb(
            depth=depth,
            genon=genon,
            p=p,
            inb_target=F_hat,
            dmodel="bb",
            bounds=(0.1, 200.0),
            tol=0.1,
        )
        self.assertLess(fit_res.ss_min, 1e-6)


class TestPopgen(unittest.TestCase):
    def test_heterozygosity_basic(self) -> None:
        # Simple balanced genotypes across 4 individuals / 3 SNPs.
        depth = np.full((4, 3), 10.0, dtype=np.float32)
        genon = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        depth2K_fn = grm.make_depth2K("bb", param=float("inf"))

        het_res = popgen.heterozygosity(
            depth=depth,
            genon=genon,
            p=p,
            depth2K=depth2K_fn,
        )
        # Expect heterozygosity around 0.5 given symmetric genotypes.
        self.assertTrue(np.isfinite(het_res.ohet))
        self.assertGreater(het_res.ohet, 0.1)
        self.assertLess(het_res.ohet, 0.9)

    def test_hw_pops_and_fst(self) -> None:
        # Two populations, each with 2 individuals and 2 SNPs.
        depth = np.full((4, 2), 10.0, dtype=np.float32)
        genon = np.array(
            [
                [0.0, 2.0],  # pop A
                [0.0, 2.0],  # pop A
                [2.0, 0.0],  # pop B
                [2.0, 0.0],  # pop B
            ],
            dtype=np.float32,
        )
        pops = np.array(["A", "A", "B", "B"])
        depth2K_fn = grm.make_depth2K("bb", param=float("inf"))

        hw_res = popgen.hw_pops(
            genon=genon,
            depth=depth,
            populations=pops,
            depth2K=depth2K_fn,
        )
        self.assertEqual(len(hw_res.popnames), 2)
        self.assertEqual(hw_res.HWdis.shape, (2, 2))

        fst_res = popgen.fst_gbs(
            depth=depth,
            genon=genon,
            populations=pops,
            depth2K=depth2K_fn,
        )
        self.assertEqual(fst_res.fst.shape, (2,))


class TestRAStore(unittest.TestCase):
    def test_write_and_read_store_roundtrip(self) -> None:
        # Tiny RAData object.
        chrom = np.array(["1", "1"])
        pos = np.array([100, 200])
        ref = np.array([[10, 0], [0, 10]], dtype=np.int32)
        alt = np.array([[0, 10], [10, 0]], dtype=np.int32)
        ra = io.RAData(
            sample_ids=["S1", "S2"],
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
        )

        store_path = "test_ra_store.kgd.zarr"
        io.write_ra_store(ra, store_path)

        ra2 = io.read_ra_store(store_path)

        self.assertEqual(ra2.sample_ids, ra.sample_ids)
        np.testing.assert_array_equal(ra2.chrom, ra.chrom)
        np.testing.assert_array_equal(ra2.pos, ra.pos)
        np.testing.assert_array_equal(ra2.ref, ra.ref)
        np.testing.assert_array_equal(ra2.alt, ra.alt)


if __name__ == "__main__":
    unittest.main()
