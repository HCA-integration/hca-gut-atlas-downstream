"""
Annotation resolution (annotated depth) for HGCA hierarchy columns.

``annotated_depth`` counts how many of ``hgca_celltype_level_1`` … ``hgca_celltype_level_5``
are present per cell (non-missing across the five levels). Used after transfer predictions
are written into ``adata.obs``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve column names: canonical ``hgca_celltype_level_1`` or ``hgca_celltype_level1``, etc.
LEVEL_INDEXES = (1, 2, 3, 4, 5)


def level_column_names_for_obs(obs: pd.DataFrame) -> List[Optional[str]]:
    """Return the obs column name used for each level 1..5, or None if absent."""
    # PanGI / CELLxGENE-style three-level hierarchy (when HGCA columns are absent)
    pangi = ("level_1_annot", "level_2_annot", "level_3_annot")
    if all(c in obs.columns for c in pangi) and "hgca_celltype_level_1" not in obs.columns:
        return [pangi[0], pangi[1], pangi[2], None, None]

    # HGCA: try underscore form first, then legacy spellings (e.g. ``hgca_celltype_level2``).
    candidates_by_level = {
        1: ("hgca_celltype_level_1", "hgca_celltype_level1"),
        2: ("hgca_celltype_level_2", "hgca_celltype_level2"),
        3: ("hgca_celltype_level_3", "hgca_celltype_level3"),
        4: ("hgca_celltype_level_4", "hgca_celltype_level4"),
        5: ("hgca_celltype_level_5", "hgca_celltype_level5"),
    }
    names: List[Optional[str]] = []
    for i in LEVEL_INDEXES:
        found = None
        for c in candidates_by_level[i]:
            if c in obs.columns:
                found = c
                break
        names.append(found)
    return names


def is_missing_annotation_value(val: Any) -> bool:
    """True for NA, None, NaN, empty string, and string 'nan' / 'none' (case-insensitive)."""
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except TypeError:
        pass
    s = str(val).strip()
    if s == "":
        return True
    low = s.lower()
    if low in ("nan", "none"):
        return True
    return False


def compute_annotated_depth_series(obs: pd.DataFrame) -> pd.Series:
    """
    Per-row count of non-missing values across the five HGCA level columns present in ``obs``.
    If a level column is absent, counts as missing for that level.
    """
    cols = level_column_names_for_obs(obs)
    n = len(obs)
    counts = np.zeros(n, dtype=np.int32)
    for col in cols:
        if col is None:
            continue
        s = obs[col]
        for i in range(n):
            if not is_missing_annotation_value(s.iloc[i]):
                counts[i] += 1
    return pd.Series(counts, index=obs.index, name="annotated_depth")


def attach_annotated_depth(adata: anndata.AnnData) -> anndata.AnnData:
    """Write ``adata.obs['annotated_depth']`` (int 0–5 for HGCA, 0–3 for PanGI three-level)."""
    adata.obs["annotated_depth"] = compute_annotated_depth_series(adata.obs)
    return adata


def merge_predictions_into_obs(
    adata_test: anndata.AnnData,
    predicted_label: Union[np.ndarray, pd.Series, List],
    label_key: str,
) -> anndata.AnnData:
    """
    Copy test AnnData, attach predicted labels and ``annotated_depth``.
    Predictions column: ``predicted_{label_key}``.
    """
    out = adata_test.copy()
    col = f"predicted_{label_key}"
    out.obs[col] = np.asarray(predicted_label)
    attach_annotated_depth(out)
    return out


def summarize_annotated_depth(
    depth: pd.Series,
) -> Dict[str, float]:
    """Mean, median, and fraction of cells at each depth 1..5 (and optionally 0)."""
    d = depth.astype(float)
    n = int(np.sum(d >= 0))
    if n == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "frac_depth_0": float("nan"),
            **{f"frac_depth_{k}": float("nan") for k in range(1, 6)},
        }
    out: Dict[str, float] = {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "frac_depth_0": float((d == 0).sum() / n),
    }
    for k in range(1, 6):
        out[f"frac_depth_{k}"] = float((d == k).sum() / n)
    return out


def per_celltype_v1_summary(
    obs: pd.DataFrame,
    depth: pd.Series,
    celltype_col: str = "hgca_celltype_v1",
) -> pd.DataFrame:
    """Mean / median / count of ``annotated_depth`` per ``hgca_celltype_v1``."""
    if celltype_col not in obs.columns:
        logger.warning("Column %r not in obs — skipping per-celltype summary", celltype_col)
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            celltype_col: obs[celltype_col].astype(str),
            "annotated_depth": depth.astype(int).values,
        }
    )
    g = df.groupby(celltype_col, dropna=False)["annotated_depth"].agg(
        ["count", "mean", "median", "std"]
    )
    g = g.reset_index().sort_values(celltype_col)
    return g


def save_annotation_depth_benchmark_outputs(
    adata_test_annotated: anndata.AnnData,
    output_dir: Union[str, Path],
    method_name: str,
    *,
    label_key: str,
    split_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path, Path, Path]:
    """
    Save CSVs (per-cell, summary by method, per-celltype) and one comparison figure.

    Returns paths: (per_cell_csv, summary_csv, by_celltype_csv, figure_png).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs = adata_test_annotated.obs
    if "annotated_depth" not in obs.columns:
        attach_annotated_depth(adata_test_annotated)
        obs = adata_test_annotated.obs

    depth = obs["annotated_depth"].astype(int)

    fold_tag = ""
    if split_meta is not None:
        hold = split_meta.get("holdout_dataset_id")
        if hold is not None:
            fold_tag = f"_holdout_{str(hold)}"
        safe = split_meta.get("run_tag", "")
        if safe:
            fold_tag = f"{fold_tag}_{safe}" if fold_tag else f"_{safe}"

    safe_method = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in method_name)[:80]

    # Per-cell
    cells = pd.DataFrame({"cell_id": obs.index.astype(str), "annotated_depth": depth.values})
    pred_col = f"predicted_{label_key}"
    if pred_col in obs.columns:
        cells[pred_col] = obs[pred_col].values
    if "hgca_celltype_v1" in obs.columns:
        cells["hgca_celltype_v1"] = obs["hgca_celltype_v1"].astype(str).values
    for col in ("dataset_id", "study"):
        if col in obs.columns:
            cells[col] = obs[col].astype(str).values
    if "level_3_annot" in obs.columns:
        cells["level_3_annot"] = obs["level_3_annot"].astype(str).values

    per_cell_path = out / f"annotation_depth_per_cell_{safe_method}{fold_tag}.csv"
    cells.to_csv(per_cell_path, index=False)

    # Summary by method (one row)
    summ = summarize_annotated_depth(depth)
    row = {"method": method_name, "n_cells": int(len(depth)), **summ}
    if split_meta:
        row.update(
            {
                "holdout_dataset_id": split_meta.get("holdout_dataset_id"),
                "run_tag": split_meta.get("run_tag"),
            }
        )
    summary_path = out / f"annotation_depth_summary_by_method_{safe_method}{fold_tag}.csv"
    pd.DataFrame([row]).to_csv(summary_path, index=False)

    # Per cell type (HGCA v1 or PanGI level 3)
    if "hgca_celltype_v1" in obs.columns:
        ct_path = out / f"annotation_depth_by_hgca_celltype_v1_{safe_method}{fold_tag}.csv"
        ct_df = per_celltype_v1_summary(obs, depth)
    elif "level_3_annot" in obs.columns:
        ct_path = out / f"annotation_depth_by_level_3_annot_{safe_method}{fold_tag}.csv"
        ct_df = per_celltype_v1_summary(obs, depth, celltype_col="level_3_annot")
    else:
        ct_path = out / f"annotation_depth_by_celltype_{safe_method}{fold_tag}.csv"
        ct_df = pd.DataFrame()
    if len(ct_df) > 0:
        ct_df.to_csv(ct_path, index=False)
    else:
        ct_empty = ct_path
        pd.DataFrame({"note": ["no cell type column for per-type summary"]}).to_csv(ct_empty, index=False)

    # Figure: stacked fraction at depth 1..5 (+ 0 for context)
    fig_path = out / f"annotation_depth_comparison_{safe_method}{fold_tag}.png"
    _plot_depth_distribution_stack(summ, method_name, fig_path)

    return per_cell_path, summary_path, ct_path, fig_path


def append_summary_row_for_cross_run(
    summary_csv: Path,
    row: Dict[str, Any],
) -> None:
    """Append one row to a cumulative summary CSV (for LODO CV aggregation)."""
    summary_csv = Path(summary_csv)
    df_new = pd.DataFrame([row])
    if summary_csv.exists():
        df_old = pd.read_csv(summary_csv)
        pd.concat([df_old, df_new], ignore_index=True).to_csv(summary_csv, index=False)
    else:
        df_new.to_csv(summary_csv, index=False)


def _plot_depth_distribution_stack(
    summ: Dict[str, float],
    method_name: str,
    out_path: Path,
) -> None:
    """Stacked horizontal bar: fraction of cells at depth 0..5."""
    fracs = [summ.get(f"frac_depth_{k}", 0.0) for k in range(0, 6)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, 6))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    left = 0.0
    for k, frac in enumerate(fracs):
        ax.barh(
            0,
            frac,
            left=left,
            height=0.55,
            color=colors[k],
            label=f"depth {k} ({frac:.2f})",
        )
        left += frac
    ax.set_yticks([0])
    ax.set_yticklabels([method_name[:50]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of test cells")
    ax.set_title("Annotation depth distribution (levels 1–5 filled)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_annotation_depth_multi_method_figure(
    summary_csv_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    *,
    title: str = "Annotation depth distribution by method",
) -> None:
    """
    Load one or more ``annotation_depth_summary_by_method_*.csv`` files (single-row or multi-row)
    and save a stacked-bar comparison figure.
    """
    rows: List[Dict[str, Any]] = []
    for p in summary_csv_paths:
        df = pd.read_csv(Path(p))
        rows.extend(df.to_dict("records"))
    plot_annotation_depth_comparison_multi_method(rows, out_path, title=title)


def plot_annotation_depth_comparison_multi_method(
    summary_rows: List[Dict[str, Any]],
    out_path: Union[str, Path],
    *,
    title: str = "Annotation depth distribution by method",
) -> None:
    """
    Stacked horizontal bars per method (same test cells → identical stacks).
    ``summary_rows`` are dicts with ``method`` and ``frac_depth_0`` … ``frac_depth_5``.
    """
    out_path = Path(out_path)
    if not summary_rows:
        return
    methods = [str(r.get("method", "?"))[:40] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(methods))))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, 6))
    y_pos = np.arange(len(methods))
    for i, meth in enumerate(methods):
        left = 0.0
        r = summary_rows[i]
        for k in range(0, 6):
            frac = float(r.get(f"frac_depth_{k}", 0.0))
            ax.barh(
                i,
                frac,
                left=left,
                height=0.65,
                color=colors[k],
                label=f"depth {k}" if i == 0 else None,
            )
            left += frac
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
