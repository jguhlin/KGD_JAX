from __future__ import annotations

import os

# Force CPU backend for tests (GPU stays available for production runs).
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
