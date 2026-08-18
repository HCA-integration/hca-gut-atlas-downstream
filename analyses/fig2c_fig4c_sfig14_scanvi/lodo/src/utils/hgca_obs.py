"""
HGCA AnnData helpers for obs / var naming (separate from ``data.preparation`` so Jupyter
imports stay reliable when ``preparation`` is cached).
"""

from __future__ import annotations

import logging

import anndata

logger = logging.getLogger(__name__)


def normalize_hgca_obs_columns(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Map alternate HGCA spellings in obs (e.g. ``hgca_celltype_level1`` → ``hgca_celltype_level_1``)
    to names used in configs. Only renames when the canonical column is missing.
    """
    pairs = [
        ("hgca_celltype_level_1", ("hgca_celltype_level1",)),
        ("hgca_celltype_level2", ("hgca_celltype_level_2",)),
        ("hgca_celltype_level3", ("hgca_celltype_level_3",)),
        ("hgca_celltype_level_4", ("hgca_celltype_level4",)),
        ("hgca_celltype_level_5", ("hgca_celltype_level5",)),
        ("hgca_celltype_v1", ()),
    ]
    rename_map = {}
    for canon, alts in pairs:
        if canon in adata.obs.columns:
            continue
        for alt in alts:
            if alt in adata.obs.columns:
                rename_map[alt] = canon
                break
    if not rename_map:
        return adata
    adata = adata.copy()
    adata.obs.rename(columns=rename_map, inplace=True)
    logger.info("Renamed HGCA label columns: %s", rename_map)
    return adata


def ensure_var_index_gene_symbols(adata: anndata.AnnData) -> anndata.AnnData:
    """
    scimilarity's ``align_dataset`` intersects on ``adata.var.index`` with the model gene list.
    Many HGCA h5ad objects keep symbols in ``adata.var['gene_symbol']`` while ``var_names`` are
    Ensembl IDs or other keys — resulting in zero overlap unless the index is updated.
    """
    if "gene_symbol" not in adata.var.columns:
        return adata
    adata = adata.copy()
    adata.var.set_index("gene_symbol", inplace=True, drop=False)
    if adata.var.index.has_duplicates:
        dup = adata.var.index.duplicated(keep="first")
        n_dup = int(dup.sum())
        if n_dup:
            logger.info("Removing %s duplicated gene_symbol rows", n_dup)
        adata = adata[:, ~dup].copy()
    return adata
