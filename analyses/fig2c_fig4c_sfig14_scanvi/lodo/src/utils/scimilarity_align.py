"""
Scimilarity gene-space alignment for AnnData.

Kept separate from ``data.preparation`` so notebooks can import a small module
(helps when Jupyter caches an older ``preparation`` without reload).
"""

from __future__ import annotations

import logging
from typing import Sequence

import anndata

logger = logging.getLogger(__name__)


def align_anndata_to_scimilarity(
    adata: anndata.AnnData,
    gene_order: Sequence[str],
    *,
    gene_overlap_threshold: int = 2500,
    subset_to_intersection: bool = True,
    copy: bool = True,
) -> anndata.AnnData:
    """
    Map ``adata`` into scimilarity's fixed gene space.

    HGCA may use updated symbols vs the model's ``gene_order`` (v1.1), so overlap can be
    below scimilarity's default 5000-gene check. Optionally **subset** to genes present in
    both (saves memory); ``align_dataset`` then pads to full length with zeros.

    Parameters
    ----------
    copy
        If True (default), copy ``adata`` before in-place var/index fixes. If False, avoid
        an extra full matrix copy when the caller already owns a disposable object (vignette
        style: load → set ``layers['counts']`` → align). Duplicate removal / gene subsetting
        still uses ``.copy()`` when required.
    """
    from scimilarity.utils import align_dataset

    if copy:
        adata = adata.copy()
    if "gene_symbol" in adata.var.columns:
        adata.var.set_index("gene_symbol", inplace=True, drop=False)
    if adata.var.index.has_duplicates:
        dup = adata.var.index.duplicated(keep="first")
        adata = adata[:, ~dup].copy()

    n_in_model = int(adata.var.index.isin(gene_order).sum())
    logger.info(
        "Genes in adata also in scimilarity model list: %s (of %s adata genes)",
        n_in_model,
        adata.n_vars,
    )

    if subset_to_intersection:
        mask = adata.var.index.isin(gene_order)
        if int(mask.sum()) == 0:
            raise ValueError(
                "No overlapping gene symbols between adata.var and scimilarity gene_order. "
                "Check gene_symbol / var.index."
            )
        adata = adata[:, mask].copy()

    logger.info(
        "align_dataset(gene_overlap_threshold=%s)",
        gene_overlap_threshold,
    )
    out = align_dataset(
        adata,
        list(gene_order),
        gene_overlap_threshold=gene_overlap_threshold,
    )
    logger.info("Aligned to scimilarity gene space: %s", out.shape)
    return out
