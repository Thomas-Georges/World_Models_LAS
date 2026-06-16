"""Pytest session setup shared by the whole suite.

Pin BLAS/OpenMP thread counts to 1 *before* any heavy numeric library is
imported. The torch-backed tests otherwise oversubscribe a many-core CPU
(each test process spins up as many BLAS threads as there are cores, and they
contend), which makes the suite appear to hang or run many times slower. With
the limits in place the full suite finishes in a few seconds regardless of core
count, and no reviewer needs to know hidden environment-variable settings.

``setdefault`` is deliberate: an explicit value in the environment (e.g. a CI
job that wants more threads) still wins.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

try:  # torch is optional locally (the torch-backed tests skip without it)
    import torch
except Exception:  # pragma: no cover - torch import failures are environmental
    torch = None

if torch is not None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # pragma: no cover - already set once the pool is warm
        pass
