#!/usr/bin/env python3
"""Supplement: gene-level HEOCA vs HGCA comparison by mapped cell type.

For each focus identity, compare mean log1p(CP10k) expression of shared genes in:
  - healthy HGCA cells of that identity
  - PSC intestine HEOCA cells confidently mapped to that identity at ≤14 d
  - same at ≥56 d

Writes one scatter figure per cell type plus machine-readable tables.
"""
from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats

import common as C

MM = 1 / 25.4

FOCUS_IDENTITIES = [
    "EEC Progenitors",
    "Intestinal Stem Cells (ISC)",
    "Transiently Amplifying Cells (TA)",
    "Enterocyte Progenitors",
    "Mid Villus Enterocytes",
    "Villus Tip Enterocytes",
    "Crypt Top Colonocytes",
    "Lower Crypt Colonocytes",
    "BEST4 Colonocytes",
    "Colonocyte Progenitors",
]

SI_TISSUES = {"duodenum", "jejunum", "ileum", "small intestine"}
COLON_TISSUES = {
    "colon",
    "ascending colon",
    "transverse colon",
    "descending colon",
    "sigmoid colon",
    "rectum",
    "cecum",
    "caecum",
}

ALWAYS_LABEL = {
    "EEC Progenitors": [
        "NEUROG3",
        "NEUROD1",
        "PAX4",
        "NKX2-2",
        "CHGA",
        "CHGB",
        "TPH1",
        "FEV",
        "SOX4",
        "INSM1",
    ],
    "Intestinal Stem Cells (ISC)": [
        "LGR5",
        "OLFM4",
        "ASCL2",
        "SMOC2",
        "RGMB",
        "PROM1",
    ],
    "Transiently Amplifying Cells (TA)": [
        "MKI67",
        "TOP2A",
        "PCNA",
        "HMGB2",
        "UBE2C",
    ],
    "Enterocyte Progenitors": [
        "FABP1",
        "FABP2",
        "APOA1",
        "APOA4",
        "DMBT1",
        "SI",
    ],
    "Mid Villus Enterocytes": [
        "FABP2",
        "APOA1",
        "APOA4",
        "SI",
        "ALPI",
        "ANPEP",
    ],
    "Villus Tip Enterocytes": [
        "APOA4",
        "APOB",
        "RBP2",
        "ADA",
        "CREB3L3",
    ],
    "Crypt Top Colonocytes": [
        "CA2",
        "AQP8",
        "GUCA2A",
        "GUCA2B",
        "MS4A12",
        "SATB2",
    ],
    "Lower Crypt Colonocytes": [
        "CA2",
        "BEST4",
        "OTOP2",
        "SLC26A3",
        "SATB2",
    ],
    "BEST4 Colonocytes": ["BEST4", "OTOP2", "CFTR", "GUCA2A"],
    "Colonocyte Progenitors": ["MKI67", "PCNA", "SATB2", "CDX2"],
}


def configure_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )


def parse_day(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(
        r"(?i)^\s*(\d+(?:\.\d+)?)\s*day\s*$"
    )[0]
    return pd.to_numeric(extracted, errors="coerce")


def time_bin(day: float) -> str:
    if not np.isfinite(day):
        return "unknown"
    if day <= 14:
        return "≤14 d"
    if day < 56:
        return "15–55 d"
    return "≥56 d"


def identity_reference_tissues(identity: str) -> set[str] | None:
    text = identity.lower()
    if "colonocyte" in text:
        return COLON_TISSUES
    if any(
        token in text
        for token in (
            "enterocyte",
            "villus",
            "isc",
            "ta",
            "eec",
            "goblet",
            "tuft",
            "paneth",
            "secretory",
        )
    ):
        return SI_TISSUES
    return None


def to_log1p_cp10k(matrix, target_sum: float = 1e4) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float64)
        lib = np.asarray(matrix.sum(axis=1)).ravel()
        lib[lib == 0] = 1.0
        scaled = matrix.multiply((target_sum / lib)[:, None])
        return np.log1p(np.asarray(scaled.mean(axis=0))).ravel()
    arr = np.asarray(matrix, dtype=np.float64)
    lib = arr.sum(axis=1)
    lib[lib == 0] = 1.0
    scaled = arr * (target_sum / lib)[:, None]
    return np.log1p(scaled.mean(axis=0))


def mean_log1p_cp10k(adata: ad.AnnData, use_counts_layer: bool) -> np.ndarray:
    if use_counts_layer:
        matrix = adata.layers["counts"]
    else:
        matrix = adata.X
    return to_log1p_cp10k(matrix)


def gene_symbols(adata: ad.AnnData, genes: list[str]) -> dict[str, str]:
    if "gene_symbol" in adata.var.columns:
        symbols = adata.var.loc[genes, "gene_symbol"].astype(str)
        return dict(zip(genes, symbols))
    if "feature_name" in adata.var.columns:
        symbols = adata.var.loc[genes, "feature_name"].astype(str)
        return dict(zip(genes, symbols))
    return {gene: gene for gene in genes}


def subset_indices(mask: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_cells:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_cells, replace=False))


def load_group_means(
    path: Path,
    cell_mask: np.ndarray,
    genes: list[str],
    use_counts_layer: bool,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    idx = subset_indices(cell_mask, max_cells=max_cells, seed=seed)
    if len(idx) == 0:
        return np.full(len(genes), np.nan), 0
    view = ad.read_h5ad(path, backed="r")
    try:
        sub = view[idx, genes].to_memory()
    finally:
        view.file.close()
    return mean_log1p_cp10k(sub, use_counts_layer=use_counts_layer), int(len(idx))


def draw_celltype_gene_scatter(
    frame: pd.DataFrame,
    identity: str,
    n_hgca: int,
    n_early: int,
    n_late: int,
    out_stem: Path,
) -> None:
    plot = frame.dropna(
        subset=["hgca_mean", "heoca_early_mean", "heoca_late_mean"]
    ).copy()
    if plot.empty:
        return
    label_genes = set(ALWAYS_LABEL.get(identity, []))
    # Genes that close the absolute gap to HGCA the most.
    plot["gap_early"] = (plot["heoca_early_mean"] - plot["hgca_mean"]).abs()
    plot["gap_late"] = (plot["heoca_late_mean"] - plot["hgca_mean"]).abs()
    plot["gap_closed"] = plot["gap_early"] - plot["gap_late"]
    closers = (
        plot.sort_values("gap_closed", ascending=False).head(8)["symbol"].tolist()
    )
    worseners = (
        plot.sort_values("gap_closed", ascending=True).head(4)["symbol"].tolist()
    )
    label_genes.update(closers)
    label_genes.update(worseners)

    r_early = stats.pearsonr(plot["hgca_mean"], plot["heoca_early_mean"])[0]
    r_late = stats.pearsonr(plot["hgca_mean"], plot["heoca_late_mean"])[0]

    figure, axes = plt.subplots(1, 2, figsize=(140 * MM, 70 * MM), sharex=True, sharey=True)
    limits = [
        0.0,
        float(
            np.nanmax(
                [
                    plot["hgca_mean"].max(),
                    plot["heoca_early_mean"].max(),
                    plot["heoca_late_mean"].max(),
                ]
            )
            * 1.05
        ),
    ]
    panels = [
        (axes[0], "heoca_early_mean", f"HEOCA ≤14 d (n={n_early})", r_early, "#56B4E9"),
        (axes[1], "heoca_late_mean", f"HEOCA ≥56 d (n={n_late})", r_late, "#D55E00"),
    ]
    for axis, ycol, title, r_value, color in panels:
        axis.scatter(
            plot["hgca_mean"],
            plot[ycol],
            s=6,
            c=color,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.plot(limits, limits, color="#666666", lw=0.7, zorder=0)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("HGCA healthy mean log1p(CP10k)")
        axis.set_ylabel("HEOCA mean log1p(CP10k)")
        axis.set_title(
            f"{identity}\n{title}\nPearson r={r_value:.3f} vs HGCA (n_ref={n_hgca})",
            fontsize=7,
            loc="left",
        )
        labeled = plot[plot["symbol"].isin(label_genes)]
        for _, row in labeled.iterrows():
            axis.scatter(
                row["hgca_mean"],
                row[ycol],
                s=14,
                c="black",
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )
            axis.text(
                row["hgca_mean"],
                row[ycol],
                str(row["symbol"]),
                fontsize=4.5,
                ha="left",
                va="bottom",
            )
        sns.despine(ax=axis)

    figure.suptitle(
        "Shared-gene expression: HEOCA PSC intestine vs matched HGCA identity",
        fontsize=8,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    stem = re.sub(r"[^A-Za-z0-9]+", "_", identity).strip("_").lower()
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            out_stem.parent / f"{out_stem.name}_{stem}.{extension}",
            dpi=400,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def main() -> None:
    configure_style()
    logger = C.setup_logging("07_supp_heoca_hgca_gene_by_celltype")
    config = C.load_config()
    C.require_files(config)
    C.set_seed(int(config["project"]["seed"]))

    hgca_path = Path(config["inputs"]["hgca_epithelial"])
    heoca_path = Path(config["inputs"]["heoca_query"])
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv")
    metadata["day"] = parse_day(metadata["time"])
    metadata["time_bin"] = metadata["day"].map(time_bin)
    meta_lookup = metadata.set_index("sample_id")

    hgca = ad.read_h5ad(hgca_path, backed="r")
    heoca = ad.read_h5ad(heoca_path, backed="r")
    shared_genes = sorted(set(hgca.var_names).intersection(heoca.var_names))
    symbols = gene_symbols(heoca, shared_genes)
    # Prefer HGCA symbols when available.
    symbols.update(
        {
            gene: symbol
            for gene, symbol in gene_symbols(hgca, shared_genes).items()
            if symbol and symbol != "nan"
        }
    )
    logger.info(
        "Shared genes=%s | HGCA cells=%s | HEOCA cells=%s",
        len(shared_genes),
        hgca.n_obs,
        heoca.n_obs,
    )

    heoca_obs = heoca.obs.copy()
    heoca_obs["day"] = heoca_obs["sample_id"].map(meta_lookup["day"])
    heoca_obs["time_bin"] = heoca_obs["sample_id"].map(meta_lookup["time_bin"])
    heoca_obs["region_broad"] = heoca_obs["sample_id"].map(
        meta_lookup["region_broad"]
    )
    heoca_obs["source_standardized"] = heoca_obs["sample_id"].map(
        meta_lookup["source_standardized"]
    )
    label_col = config["columns"]["query_label_confident"]
    conf_col = config["columns"]["confidence"]
    heoca_confident = (
        heoca_obs[conf_col].astype(float)
        >= float(config["filters"]["confidence_threshold"])
    ) & heoca_obs[label_col].astype(str).ne("Unknown")

    out_dir = C.OUT / "supp_heoca_hgca_gene_by_celltype"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_rows = []

    for identity in FOCUS_IDENTITIES:
        tissues = identity_reference_tissues(identity)
        hgca_mask = (
            hgca.obs["hgca_celltype_v1"].astype(str).eq(identity).to_numpy()
            & hgca.obs["sampled_site_condition"]
            .astype(str)
            .eq(config["filters"]["reference_healthy_value"])
            .to_numpy()
        )
        if tissues is not None:
            tissue = hgca.obs["tissue"].astype(str).str.lower()
            hgca_mask &= tissue.isin(tissues).to_numpy()

        early_mask = (
            heoca_confident.to_numpy()
            & heoca_obs["source_standardized"].astype(str).eq("PSC").to_numpy()
            & heoca_obs["region_broad"].astype(str).eq("Intestine").to_numpy()
            & heoca_obs["time_bin"].astype(str).eq("≤14 d").to_numpy()
            & heoca_obs[label_col].astype(str).eq(identity).to_numpy()
        )
        late_mask = (
            heoca_confident.to_numpy()
            & heoca_obs["source_standardized"].astype(str).eq("PSC").to_numpy()
            & heoca_obs["region_broad"].astype(str).eq("Intestine").to_numpy()
            & heoca_obs["time_bin"].astype(str).eq("≥56 d").to_numpy()
            & heoca_obs[label_col].astype(str).eq(identity).to_numpy()
        )

        n_hgca = int(hgca_mask.sum())
        n_early = int(early_mask.sum())
        n_late = int(late_mask.sum())
        logger.info(
            "%s | HGCA=%s early=%s late=%s", identity, n_hgca, n_early, n_late
        )
        if min(n_hgca, n_early, n_late) < 20:
            logger.warning("Skipping %s (insufficient cells)", identity)
            continue

        hgca_mean, n_hgca_used = load_group_means(
            hgca_path,
            hgca_mask,
            shared_genes,
            use_counts_layer=False,
            max_cells=3000,
            seed=int(config["project"]["seed"]),
        )
        early_mean, n_early_used = load_group_means(
            heoca_path,
            early_mask,
            shared_genes,
            use_counts_layer=True,
            max_cells=3000,
            seed=int(config["project"]["seed"]) + 1,
        )
        late_mean, n_late_used = load_group_means(
            heoca_path,
            late_mask,
            shared_genes,
            use_counts_layer=True,
            max_cells=3000,
            seed=int(config["project"]["seed"]) + 2,
        )

        frame = pd.DataFrame(
            {
                "gene_id": shared_genes,
                "symbol": [symbols.get(gene, gene) for gene in shared_genes],
                "identity": identity,
                "hgca_mean": hgca_mean,
                "heoca_early_mean": early_mean,
                "heoca_late_mean": late_mean,
            }
        )
        frame["gap_early"] = (frame["heoca_early_mean"] - frame["hgca_mean"]).abs()
        frame["gap_late"] = (frame["heoca_late_mean"] - frame["hgca_mean"]).abs()
        frame["gap_closed"] = frame["gap_early"] - frame["gap_late"]
        frame["delta_heoca_late_minus_early"] = (
            frame["heoca_late_mean"] - frame["heoca_early_mean"]
        )
        frame["n_hgca_cells"] = n_hgca_used
        frame["n_heoca_early_cells"] = n_early_used
        frame["n_heoca_late_cells"] = n_late_used
        rows.append(frame)

        r_early = stats.pearsonr(frame["hgca_mean"], frame["heoca_early_mean"])[0]
        r_late = stats.pearsonr(frame["hgca_mean"], frame["heoca_late_mean"])[0]
        summary_rows.append(
            {
                "identity": identity,
                "n_hgca_cells": n_hgca_used,
                "n_heoca_early_cells": n_early_used,
                "n_heoca_late_cells": n_late_used,
                "pearson_r_early_vs_hgca": r_early,
                "pearson_r_late_vs_hgca": r_late,
                "delta_r_late_minus_early": r_late - r_early,
                "mean_abs_gap_early": float(frame["gap_early"].mean()),
                "mean_abs_gap_late": float(frame["gap_late"].mean()),
                "mean_gap_closed": float(frame["gap_closed"].mean()),
                "top_gap_closing_genes": ";".join(
                    frame.sort_values("gap_closed", ascending=False)
                    .head(10)["symbol"]
                    .tolist()
                ),
            }
        )
        draw_celltype_gene_scatter(
            frame,
            identity=identity,
            n_hgca=n_hgca_used,
            n_early=n_early_used,
            n_late=n_late_used,
            out_stem=out_dir / "supp_heoca_hgca_genes",
        )

    hgca.file.close()
    heoca.file.close()

    if not rows:
        raise RuntimeError("No cell types had sufficient cells for gene comparison")

    gene_table = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    gene_table.to_csv(
        C.DATA / "supp_heoca_hgca_gene_means_by_celltype.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(
        C.DATA / "supp_heoca_hgca_gene_similarity_summary.csv", index=False
    )
    logger.info("Wrote %s cell-type gene plots to %s", len(summary), out_dir)
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
