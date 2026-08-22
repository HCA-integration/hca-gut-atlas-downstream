"""
Train/test splits for benchmarking — including leave-one-dataset-out (LODO).

Random stratified splits mix cells from the same study; LODO is closer to
“annotate a completely unseen dataset” when ``dataset_id`` marks study/batch.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

import anndata


def train_test_mask_leave_one_dataset(
    adata: anndata.AnnData,
    dataset_col: str,
    holdout_dataset_id: Union[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return boolean masks (train, test) where test is exactly cells with
    ``adata.obs[dataset_col] == holdout_dataset_id``.
    """
    if dataset_col not in adata.obs.columns:
        raise KeyError(f"{dataset_col!r} not in adata.obs")
    ds = adata.obs[dataset_col]
    # compare as string to avoid dtype surprises
    hold = ds.astype(str) == str(holdout_dataset_id)
    train = ~hold.to_numpy()
    test = hold.to_numpy()
    if int(test.sum()) == 0:
        raise ValueError(f"No cells for holdout_dataset_id={holdout_dataset_id!r}")
    if int(train.sum()) == 0:
        raise ValueError("Train mask is empty — check holdout id")
    return train, test


def iter_leave_one_dataset_folds(
    adata: anndata.AnnData,
    dataset_col: str = "dataset_id",
    *,
    max_folds: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[Tuple[Any, np.ndarray, np.ndarray]]:
    """
    Yield ``(holdout_id, train_mask, test_mask)`` for each unique ``dataset_col`` value.

    If ``max_folds`` is set, subsample that many holdout datasets (for pilot LODO).
    """
    if dataset_col not in adata.obs.columns:
        raise KeyError(f"{dataset_col!r} not in adata.obs")
    ids = pd.unique(adata.obs[dataset_col].astype(str))
    ids = np.array(sorted(ids), dtype=object)
    if max_folds is not None and max_folds < len(ids):
        if rng is not None:
            pick = rng.choice(len(ids), size=max_folds, replace=False)
            ids = ids[np.sort(pick)]
        else:
            ids = ids[:max_folds]

    for hid in ids:
        train, test = train_test_mask_leave_one_dataset(adata, dataset_col, hid)
        yield hid, train, test


def subset_adata_masks(
    adata: anndata.AnnData,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    copy: bool = True,
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Materialize train/test ``AnnData`` from boolean row masks."""
    idx_train = np.flatnonzero(train_mask)
    idx_test = np.flatnonzero(test_mask)
    if copy:
        return adata[idx_train].copy(), adata[idx_test].copy()
    return adata[idx_train], adata[idx_test]
