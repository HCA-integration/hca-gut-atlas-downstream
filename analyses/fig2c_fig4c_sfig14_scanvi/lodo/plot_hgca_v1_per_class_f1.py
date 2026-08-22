#!/usr/bin/env python3
"""
Per-lineage bar plots: HGCA v1 (final) cell-type F1 from LODO folds (SCANVI / HGCA reference).

Aggregates ``fold_*/per_class_metrics.csv`` under an HGCA v1 benchmark run with
support-weighted mean F1 across holdouts. Cell types below a threshold are highlighted
as candidates for label refinement.

Example::

    python workflows/slurm/plot_hgca_v1_per_class_f1.py \\
        --hgca-run results/myeloid/slurm_benchmarks/e3_full_lodo_20260416T183819Z_hgca_myeloid \\
        --output-dir results/comparisons/e3_full_lodo_20260416T183819Z/myeloid_hgca_v0_v1_pangi \\
        --lineage myeloid

Batch (all lineages under a comparison root)::

    python workflows/slurm/plot_hgca_v1_per_class_f1.py \\
        --comparison-root results/comparisons/e3_full_lodo_20260416T183819Z \\
        --run-base e3_full_lodo_20260416T183819Z
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LINEAGES: Tuple[str, ...] = ("myeloid", "lymphoid", "epithelial", "stroma")
COLOR_OK = "#3a6ea5"
COLOR_REVIEW = "#c44e52"
COLOR_THRESH_LINE = "#888888"

_V1_PATTERNS = (
    re.compile(r"^hgca[_\s]*celltype[_\s]*v1$", re.I),
    re.compile(r"^hgca_celltype_v1$", re.I),
)


def _is_hgca_v1_label_type(label_type: str) -> bool:
    norm = str(label_type).strip().lower().replace(" ", "_")
    if "v0" in norm:
        return False
    return any(p.match(norm.replace("__", "_")) for p in _V1_PATTERNS) or norm == "hgca_celltype_v1"


def _load_fold_metrics(hgca_run: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for fold_dir in sorted(hgca_run.glob("fold_*")):
        mf = fold_dir / "per_class_metrics.csv"
        if not mf.is_file():
            continue
        df = pd.read_csv(mf)
        if df.empty:
            continue
        holdout = fold_dir.name[len("fold_") :]
        df = df.copy()
        df["holdout_dataset_id"] = holdout
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No fold_*/per_class_metrics.csv under {hgca_run}"
        )
    return pd.concat(rows, ignore_index=True)


def aggregate_v1_f1(raw: pd.DataFrame) -> pd.DataFrame:
    sub = raw[raw["label_type"].map(_is_hgca_v1_label_type)].copy()
    if sub.empty:
        raise ValueError(
            "No rows for HGCA v1 label_type in per_class_metrics "
            f"(got label_type values: {sorted(raw['label_type'].dropna().unique())})"
        )
    sub["f1"] = pd.to_numeric(sub["f1"], errors="coerce")
    sub["support"] = pd.to_numeric(sub["support"], errors="coerce").fillna(0)

    out_rows: List[dict] = []
    for cell_type, g in sub.groupby("cell_type", sort=False):
        w = g["support"].to_numpy(dtype=float)
        f1 = g["f1"].to_numpy(dtype=float)
        mask = w > 0
        if not mask.any():
            continue
        w_ok, f1_ok = w[mask], f1[mask]
        weighted = float(np.average(f1_ok, weights=w_ok))
        out_rows.append(
            {
                "cell_type": cell_type,
                "mean_f1_weighted": weighted,
                "std_f1_across_folds": float(np.nanstd(f1_ok, ddof=0)),
                "min_f1_fold": float(np.nanmin(f1_ok)),
                "max_f1_fold": float(np.nanmax(f1_ok)),
                "total_support": int(w.sum()),
                "n_folds_present": int(g["holdout_dataset_id"].nunique()),
            }
        )
    agg = pd.DataFrame(out_rows)
    if agg.empty:
        raise ValueError("No cell types with support > 0 for HGCA v1")
    return agg.sort_values("mean_f1_weighted", ascending=False).reset_index(drop=True)


def plot_per_class_f1(
    agg: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    lineage: str,
    low_f1_threshold: float,
) -> None:
    plot_df = agg.sort_values("mean_f1_weighted", ascending=True)
    n = len(plot_df)
    y = np.arange(n)
    f1 = plot_df["mean_f1_weighted"].to_numpy()
    needs = f1 < low_f1_threshold
    colors = [COLOR_REVIEW if r else COLOR_OK for r in needs]

    fig_h = max(5.5, 0.38 * n + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(
        y,
        f1,
        color=colors,
        height=0.72,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        low_f1_threshold,
        color=COLOR_THRESH_LINE,
        linestyle="--",
        linewidth=1.2,
        label=f"review threshold ({low_f1_threshold:.2f})",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["cell_type"], fontsize=8)
    ax.set_xlabel("Support-weighted mean F1 (LODO folds)", fontsize=10)
    ax.set_ylabel("HGCA v1 cell type", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=COLOR_OK, label=f"F1 ≥ {low_f1_threshold:.2f}"),
        Patch(facecolor=COLOR_REVIEW, label=f"F1 < {low_f1_threshold:.2f} (review)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.95)

    n_review = int(needs.sum())
    ax.text(
        0.99,
        0.01,
        f"{lineage}: {n} types, {n_review} below {low_f1_threshold:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one(
    hgca_run: Path,
    output_dir: Path,
    lineage: str,
    *,
    low_f1_threshold: float,
    title: Optional[str] = None,
) -> Tuple[Path, Path]:
    raw = _load_fold_metrics(hgca_run.resolve())
    agg = aggregate_v1_f1(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "hgca_v1_per_class_f1_summary.csv"
    agg.to_csv(out_csv, index=False)
    n_folds = len(list(hgca_run.resolve().glob("fold_*")))
    t = title or (
        f"HGCA v1 per cell-type F1 — {lineage} "
        f"(SCANVI LODO, {n_folds} holdouts)"
    )
    out_png = output_dir / "hgca_v1_per_class_f1_by_celltype.png"
    plot_per_class_f1(
        agg,
        out_png,
        title=t,
        lineage=lineage,
        low_f1_threshold=low_f1_threshold,
    )
    return out_csv, out_png


def _comparison_lineage_dirs(comp_root: Path) -> List[Tuple[str, Path]]:
    found: List[Tuple[str, Path]] = []
    for lin in LINEAGES:
        for suffix in ("_hgca_v0_v1_pangi", "_hgca_vs_pangi"):
            d = comp_root / f"{lin}{suffix}"
            if d.is_dir():
                found.append((lin, d))
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hgca-run", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--lineage", type=str, default=None)
    ap.add_argument("--comparison-root", type=Path, default=None)
    ap.add_argument(
        "--run-base",
        type=str,
        default="e3_full_lodo_20260416T183819Z",
        help="HGCA slurm_benchmarks token when using --comparison-root.",
    )
    ap.add_argument(
        "--low-f1-threshold",
        type=float,
        default=0.75,
        help="Cell types with mean F1 below this are highlighted for review.",
    )
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    if args.comparison_root is not None:
        comp = args.comparison_root.resolve()
        if not comp.is_dir():
            print(f"Not a directory: {comp}", file=sys.stderr)
            return 1
        pairs = _comparison_lineage_dirs(comp)
        if not pairs:
            print(f"No lineage comparison dirs under {comp}", file=sys.stderr)
            return 1
        for lin, out_dir in pairs:
            hgca = (
                PROJECT_ROOT
                / "results"
                / lin
                / "slurm_benchmarks"
                / f"{args.run_base}_hgca_{lin}"
            )
            if not hgca.is_dir():
                print(f"SKIP {lin}: missing {hgca}", file=sys.stderr)
                continue
            csv_p, png_p = run_one(
                hgca,
                out_dir,
                lin,
                low_f1_threshold=args.low_f1_threshold,
                title=args.title,
            )
            print(f"{lin}: {csv_p}, {png_p}")
        return 0

    if not args.hgca_run or not args.output_dir:
        print(
            "Provide --hgca-run and --output-dir, or --comparison-root.",
            file=sys.stderr,
        )
        return 1
    lin = args.lineage or args.hgca_run.name.split("_hgca_")[-1]
    csv_p, png_p = run_one(
        args.hgca_run.resolve(),
        args.output_dir.resolve(),
        lin,
        low_f1_threshold=args.low_f1_threshold,
        title=args.title,
    )
    print(f"Wrote {csv_p}, {png_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
