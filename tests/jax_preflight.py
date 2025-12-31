from __future__ import annotations

import jax


def assert_cpu_backend() -> None:
    backend = jax.default_backend()
    assert backend == "cpu", f"JAX backend is '{backend}', expected 'cpu' for tests."
