"""Theil's U confounding heatmap for HGCA all-cells metadata."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "figures"
OUT_TAB = ROOT / "tables"

REPO = Path(__file__).resolve().parents[4]
H5AD = Path(os.environ.get("HGCA_H5AD", REPO / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"))

# column -> broad-audience label (axis ticks)
# Note: dissociation_protocol omitted — filled for only ~34% cells / ~20% samples.
# manner_of_death omitted — sparse + unstable pairwise U.
COVARIATES = [
    ("age_range", "Age range"),
    ("sex_ontology_term", "Sex"),
    ("sampled_site_condition", "Disease-adjacent vs healthy"),
    ("disease", "Disease"),
    ("tissue_level_1", "Intestinal segment"),
    ("radial_tissue_term", "Radial layer"),
    ("sample_collection_method", "Biopsy vs resection"),
    ("sample_preservation_method", "Fresh vs Frozen"),
    ("cell_enrichment", "Cell enrichment method"),
    ("assay", "Library preparation method"),
    ("sequenced_fragment", "Sequenced fragment"),
    ("alignment_software", "Alignment software"),
    ("reference_genome", "Reference genome"),
    ("gene_annotation_version", "Gene annotation version"),
    ("author_cell_type", "Author cell type"),
    ("dataset_id", "Dataset"),
    ("donor_id", "Donor"),
    ("sample_id", "Sample ID"),
]


def read_obs_codes(path: Path, cols: list[str]) -> tuple[pd.DataFrame, dict[str, list]]:
    """Read categorical obs columns as integer codes + category lists."""
    codes = {}
    cats = {}
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        for c in cols:
            if c not in obs:
                continue
            node = obs[c]
            if isinstance(node, h5py.Group) and "categories" in node:
                cat = [
                    x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                    for x in node["categories"][:]
                ]
                code = np.asarray(node["codes"][:], dtype=np.int32)
                codes[c] = code
                cats[c] = cat
            else:
                arr = node[:]
                arr = [
                    x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr
                ]
                s = pd.Series(arr, dtype="object")
                cat = sorted(s.dropna().unique().tolist())
                mapping = {v: i for i, v in enumerate(cat)}
                code = s.map(mapping).fillna(-1).astype(np.int32).to_numpy()
                codes[c] = code
                cats[c] = cat
    return pd.DataFrame(codes), cats


def is_unknown_label(lab: str) -> bool:
    t = str(lab).strip().lower()
    return t in {"", "unknown", "nan", "none", "n/a", "na", "not applicable"}


def theils_u_from_codes(x: np.ndarray, y: np.ndarray) -> float:
    """U(x|y): fraction of X entropy explained by Y. Codes are int; -1 = missing."""
    ok = (x >= 0) & (y >= 0)
    x = x[ok]
    y = y[ok]
    if x.size < 10:
        return np.nan
    x_u, x_inv = np.unique(x, return_inverse=True)
    y_u, y_inv = np.unique(y, return_inverse=True)
    # No residual uncertainty to explain, or predictor is constant in the overlap
    if x_u.size < 2 or y_u.size < 2:
        return 0.0
    tab = np.zeros((x_u.size, y_u.size), dtype=np.float64)
    np.add.at(tab, (x_inv, y_inv), 1.0)
    n = tab.sum()
    if n <= 0:
        return np.nan
    px = tab.sum(axis=1) / n
    py = tab.sum(axis=0) / n
    pxy = tab / n
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    if hx <= 0:
        return 0.0
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    hxy = -(pxy[pxy > 0] * np.log(pxy[pxy > 0])).sum()
    hx_given_y = hxy - hy
    return float((hx - hx_given_y) / hx)


def hclust_order(mat: np.ndarray) -> np.ndarray:
    """Leaf order from average-linkage on 1 - symmetrized U. No dendrogram drawn."""
    sim = np.nan_to_num(0.5 * (mat + mat.T), nan=0.0)
    np.fill_diagonal(sim, 1.0)
    dist = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    # numerical safety for squareform
    dist = 0.5 * (dist + dist.T)
    Z = linkage(squareform(dist, checks=False), method="average")
    return leaves_list(Z)


def tile_text_color(cmap, v: float) -> str:
    """White on dark tiles, black on light tiles (by magma luminance)."""
    r, g, b, _ = cmap(Normalize(vmin=0, vmax=1)(v))
    # relative luminance (sRGB approx)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else "black"


def main():
    global H5AD, OUT_FIG, OUT_TAB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--outdir", type=Path, default=ROOT / "tables")
    args = parser.parse_args()
    H5AD = args.h5ad
    OUT_TAB = args.outdir
    OUT_FIG = args.outdir
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    if "demo" in H5AD.name:
        print("DEMO MODE: results are for software checking, not manuscript figures.")
    cols = [c for c, _ in COVARIATES]
    labels = [lab for _, lab in COVARIATES]
    print(f"Reading {H5AD} …", flush=True)
    df, cats = read_obs_codes(H5AD, cols)
    present = [c for c in cols if c in df.columns]
    labels = [lab for c, lab in zip(cols, labels) if c in present]
    cols = present
    if not cols:
        raise SystemExit("No requested covariate columns found in the h5ad obs.")
    n = len(df)
    print(f"n cells = {n:,}", flush=True)

    for c in cols:
        unk_idx = {i for i, lab in enumerate(cats[c]) if is_unknown_label(lab)}
        if unk_idx:
            code = df[c].to_numpy().copy()
            code[np.isin(code, list(unk_idx))] = -1
            df[c] = code

    keep = []
    for c, lab in zip(cols, labels):
        known = df[c] >= 0
        nlev = pd.Series(df.loc[known, c]).nunique()
        print(f"  {lab}: known={int(known.sum()):,} levels={nlev}", flush=True)
        if nlev >= 2:
            keep.append((c, lab))
        else:
            print("    SKIP (constant / <2 levels)", flush=True)

    cols = [c for c, _ in keep]
    labels = [lab for _, lab in keep]
    k = len(cols)

    # Matrix: rows = Y (guess), columns = X (knowing) → U(Y|X)
    mat = np.full((k, k), np.nan, dtype=float)
    for i, cy in enumerate(cols):
        for j, cx in enumerate(cols):
            if i == j:
                mat[i, j] = 1.0
                continue
            mat[i, j] = theils_u_from_codes(df[cy].to_numpy(), df[cx].to_numpy())
        print(f"  done row {i+1}/{k}: {labels[i]}", flush=True)

    order = hclust_order(mat)
    mat = mat[np.ix_(order, order)]
    labels = [labels[i] for i in order]
    print("hclust order:", labels, flush=True)

    mat_df = pd.DataFrame(mat, index=labels, columns=labels)
    mat_df.index.name = "guess_Y"
    mat_df.columns.name = "knowing_X"
    csv_path = OUT_TAB / "theils_u_confounding_matrix.csv"
    mat_df.to_csv(csv_path)
    print(f"Wrote {csv_path}", flush=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    tick_pt = 11
    title_pt = 13
    cell = 0.55
    fig_w = max(10.0, k * cell + 3.2)
    fig_h = max(9.0, k * cell + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=tick_pt)
    ax.set_yticklabels(labels, fontsize=tick_pt)
    ax.tick_params(axis="both", which="both", length=0, pad=4)

    for i in range(k):
        for j in range(k):
            v = mat[i, j]
            if not np.isfinite(v):
                txt = "—"
                color = "0.35"
            else:
                txt = f"{v:.2f}"
                color = tile_text_color(cmap, float(v))
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=7.5, color=color, fontweight="medium"
            )

    ax.set_xlabel("Knowing X", fontsize=tick_pt, labelpad=10)
    ax.set_ylabel("Guess Y", fontsize=tick_pt, labelpad=10)

    # Title + subtitle as separate fig texts so they never overlap
    fig.subplots_adjust(left=0.28, right=0.86, bottom=0.28, top=0.84)
    fig.text(
        0.57,
        0.98,
        "Confounding with Theil's U",
        ha="center",
        va="top",
        fontsize=title_pt,
        fontweight="semibold",
        transform=fig.transFigure,
    )
    fig.text(
        0.57,
        0.935,
        "How well does knowing X allow you to guess Y?",
        ha="center",
        va="top",
        fontsize=tick_pt,
        color="0.15",
        transform=fig.transFigure,
    )

    cax = fig.add_axes([0.90, 0.28, 0.02, 0.50])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Theil's U", fontsize=tick_pt)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=tick_pt - 1)

    stem = "theils_u_confounding_heatmap"
    for ext, kwargs in [
        (".pdf", dict(bbox_inches="tight")),
        (".svg", dict(bbox_inches="tight")),
        (".png", dict(dpi=300, bbox_inches="tight")),
    ]:
        out = OUT_FIG / f"{stem}{ext}"
        fig.savefig(out, facecolor="white", **kwargs)
        print(f"Wrote {out}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
