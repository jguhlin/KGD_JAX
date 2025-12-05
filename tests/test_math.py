from __future__ import annotations

import unittest
import numpy as np

from kgd_jax import grm


class TestDepth2K(unittest.TestCase):
    def test_depth2k_bb_inf_basic(self) -> None:
        depth = np.array([0, 1, 2, 3], dtype=np.float32)
        K = grm.depth2K_bb(depth, alpha=np.inf)
        np.testing.assert_allclose(K, np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32))

    def test_depth2k_modp_basic(self) -> None:
        depth = np.array([0, 1, 2, 3], dtype=np.float32)
        K = grm.depth2K_modp(depth, modp=0.8)
        expected = np.array([1.0, 0.5, 0.5 * 0.8, 0.5 * 0.8 ** 2], dtype=np.float32)
        np.testing.assert_allclose(K, expected)

    def test_make_depth2k_invalid_model(self) -> None:
        with self.assertRaises(ValueError):
            grm.make_depth2K("bad-model")
