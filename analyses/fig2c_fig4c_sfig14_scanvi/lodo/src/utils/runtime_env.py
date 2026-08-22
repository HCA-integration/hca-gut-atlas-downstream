"""
Process-wide limits for scimilarity / PyTorch on macOS and multi-core BLAS.

Matches patterns from hca-gut-atlas-downstream vignettes (MPS fallback, batch size
in notebooks) and caps BLAS/OpenMP threads so one job does not spawn many threads
and spike memory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional


def apply_scimilarity_runtime_limits(
    cfg: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Set environment variables before importing NumPy (ideal) or as early as possible.

    Parameters
    ----------
    cfg
        Optional ``runtime`` block from YAML, e.g.::

            runtime:
              omp_threads: 1
              pytorch_mps_fallback: "1"
    """
    cfg = dict(cfg or {})

    def _set(name: str, key: str, default: Optional[str]) -> None:
        val = cfg.get(key, default)
        if val is not None:
            os.environ[name] = str(val)

    # BLAS / OpenMP (cap threads to reduce memory spikes on large matrices)
    threads = cfg.get("omp_threads", cfg.get("blas_threads", 1))
    if threads is not None:
        t = str(int(threads))
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(name, t)

    _set("TOKENIZERS_PARALLELISM", "tokenizers_parallelism", "false")

    # Vignettes: scimilarity_in_context uses fallback 1; scimilarity_models uses 0 + watermark 0
    _set("PYTORCH_ENABLE_MPS_FALLBACK", "pytorch_mps_fallback", "1")
    hw = cfg.get("pytorch_mps_high_watermark_ratio")
    if hw is not None:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(hw)

    seed = cfg.get("pythonhashseed")
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
