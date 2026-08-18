#!/usr/bin/env python3
"""Supplementary Figure 5 marker dot plots.

Source: archive vignettes/RareCellTypes.ipynb. Gene lists follow the
submitted panels, not every exploratory cell in that notebook.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[2]

INFLARE_24_4 = {
    "Brunner": ["MUC6", "TFF2", "PGC", "HDAC9"],
    "TA": ["TFF3", "AQP5", "MKI67", "TOP2A", "MCM5", "HMGB2"],
    "Goblet": ["MUC2", "SPDEF", "AGR2"],
    "Enteroc.": ["CLCA1", "ALPI", "APOA1", "FABP1", "SI"],
}
INFLARE_11_3 = {
    **INFLARE_24_4,
    "Foveo.": ["REG4", "TFF1", "MUC5AC"],
}
SINUS_MARKERS = {
    "LEC core": ["PROX1", "PDPN", "LYVE1"],
    "Medullary sinus": ["STAB2", "CLEC4M"],
    "Ceiling sinus": ["ACKR4", "CAV1", "NT5E"],
}
SINUS_CLUSTERS = ["13_0", "13_1", "13_2", "13_3", "13_4", "13_5"]
MAC_MARKERS = {
    "Macrophage Core": ["MS4A7", "GPNMB", "CXCL2"],
    "Residence": ["SELENOP", "CNDP1"],
    "Cycling": ["STMN1", "MKI67"],
    "Follicle Associated": ["PLA2G2D", "PTPRB", "MMP9"],
    "Complement": ["C1QA", "C1QB", "C1QC"],
    "Myeloid Resident": ["FOLR2"],
    "Homeostatic": ["CD163L1"],
    "Perivascular": ["LYVE1", "COLEC12", "MRC1"],
}
MAC_TYPES = [
    "Cycling Resident Macrophages",
    "Cycling Macrophages",
    "Follicle Associated Resident Macrophages",
    "Homeostatic Macrophages",
    "M0 Macrophages",
    "Perivascular Resident Macrophages",
]
GRAN_MARKERS = {
    "Eosinophils": ["CLC", "RNASE2", "IL5RA", "CCR3", "IL4"],
    "Basophils": ["ENPP3", "FCER1A", "FCER1G", "MS4A2", "HDC", "GATA2", "IL3RA"],
    "Neutrophils": ["PROK2", "IFIT2", "CCL3L1", "CXCR4", "HCAR3", "G0S2"],
    "Mast cells": ["TPSAB1", "TPSB2", "KIT", "MS4A2", "CPA3", "GATA2"],
    "Monocytes/Macrophages": [
        "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "CTSS", "LST1", "LGALS3", "CST3"
    ],
    "DCs": ["FCER1A", "CLEC10A", "CD1C", "CLEC9A", "BATF3", "IRF7", "LAMP3"],
}


def setup_nature_style(font_size_pt: int = 6) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": font_size_pt,
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def objects_dir(cli_objects: str | None) -> Path:
    raw = cli_objects or os.environ.get("HGCA_OBJECTS")
    if not raw:
        raise SystemExit("Set HGCA_OBJECTS or pass --objects to the lineage h5ad directory.")
    path = Path(raw)
    if not path.is_dir():
        raise SystemExit(f"HGCA_OBJECTS is not a directory: {path}")
    return path


def read_lineage(objects: Path, name: str):
    path = objects / f"{name}.h5ad"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    print(f"reading {path}", flush=True)
    return sc.read_h5ad(path)


def use_gene_symbols(adata):
    ad = adata.copy()
    if "gene_symbol" in ad.var.columns:
        ad.var_names = ad.var["gene_symbol"].astype(str).values
        ad.var_names_make_unique()
    return ad


def present_panel(adata, panel: dict[str, list[str]]) -> dict[str, list[str]]:
    names = set(adata.var_names.astype(str))
    out = {k: [g for g in genes if g in names] for k, genes in panel.items()}
    return {k: v for k, v in out.items() if v}


def save_dotplot(dp, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dp.make_figure()
    fig = dp.fig
    fig.savefig(dest.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(dest.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {dest.with_suffix('.pdf')}", flush=True)
    plt.close(fig)


def term_condition(ad, tissue_col: str, cond_col: str = "sampled_site_condition"):
    grp = ad.obs[tissue_col].astype(str) + " | " + ad.obs[cond_col].astype(str)
    order = sorted(grp.unique(), key=lambda s: (s.split(" | ")[0], s.split(" | ")[1]))
    ad.obs["term_cond"] = pd.Categorical(grp, categories=order, ordered=True)
    return ad


def panel_inflares(objects: Path, out: Path) -> None:
    raw = read_lineage(objects, "epithelial")
    if "leiden_lineage_l2" not in raw.obs.columns:
        raise SystemExit("epithelial.h5ad is missing leiden_lineage_l2")
    ad = use_gene_symbols(raw)
    del raw
    setup_nature_style()
    specs = (
        ("24_4", INFLARE_24_4, "tissue_level_1", "sfig5_a_inflare_24_4"),
        ("11_3", INFLARE_11_3, "tissue_ontology_term", "sfig5_a_inflare_11_3"),
    )
    for cluster, panel, tissue_col, stem in specs:
        sub = ad[ad.obs["leiden_lineage_l2"].astype(str) == cluster].copy()
        if sub.n_obs == 0:
            print(f"skip {stem}: no cells in {cluster}", flush=True)
            continue
        if tissue_col not in sub.obs.columns:
            tissue_col = "tissue_level_1"
        sub = term_condition(sub, tissue_col)
        markers = present_panel(sub, panel)
        dp = sc.pl.dotplot(
            sub,
            markers,
            groupby="term_cond",
            use_raw=False,
            swap_axes=True,
            standard_scale="var",
            dendrogram=False,
            show=False,
            return_fig=True,
            title=f"Subcluster {cluster}",
        )
        save_dotplot(dp, out / stem)


def resolve_sinus_clusters(adata, chosen: list[str]) -> list[str]:
    have = set(adata.obs["leiden_lineage_l2"].astype(str))
    found = [c for c in chosen if c in have]
    if not found:
        lec = sorted(
            have
            & set(
                adata.obs.loc[
                    adata.obs["hgca_celltype_v1"].astype(str).str.contains(
                        "sinus|lymphatic", case=False, na=False
                    ),
                    "leiden_lineage_l2",
                ].astype(str)
            )
        )
        raise SystemExit(
            f"No sinus clusters {chosen} in stroma.h5ad. "
            f"Nearby LEC Leiden IDs: {lec[:20]}"
        )
    if found != chosen:
        print(f"sinus clusters present: {found} (requested {chosen})", flush=True)
    return found


def panel_sinus(objects: Path, out: Path, sinus_clusters: list[str]) -> None:
    raw = read_lineage(objects, "stroma")
    ad = use_gene_symbols(raw)
    del raw
    clusters = resolve_sinus_clusters(ad, sinus_clusters)
    sub = ad[ad.obs["leiden_lineage_l2"].astype(str).isin(clusters)].copy()
    setup_nature_style()
    markers = present_panel(sub, SINUS_MARKERS)
    dp = sc.pl.dotplot(
        sub,
        var_names=markers,
        groupby="leiden_lineage_l2",
        standard_scale="var",
        dot_max=0.7,
        show=False,
        return_fig=True,
        title="Sinus endothelium",
    )
    save_dotplot(dp, out / "sfig5_b_sinus_endothelium")


def panel_macrophages(objects: Path, out: Path) -> None:
    raw = read_lineage(objects, "myeloid")
    ad = use_gene_symbols(raw)
    del raw
    have = set(ad.obs["hgca_celltype_v1"].astype(str))
    keep = [t for t in MAC_TYPES if t in have]
    if not keep:
        keep = sorted(
            x for x in have if "macrophage" in x.lower() or x.startswith("M0")
        )
    sub = ad[ad.obs["hgca_celltype_v1"].astype(str).isin(keep)].copy()
    setup_nature_style()
    markers = present_panel(sub, MAC_MARKERS)
    dp = sc.pl.dotplot(
        sub,
        var_names=markers,
        groupby="hgca_celltype_v1",
        standard_scale="var",
        dot_max=0.7,
        show=False,
        return_fig=True,
        title="Resident macrophages",
    )
    save_dotplot(dp, out / "sfig5_c_resident_macrophages")


def panel_granulocytes(objects: Path, out: Path) -> None:
    raw = read_lineage(objects, "myeloid")
    ad = use_gene_symbols(raw)
    del raw
    setup_nature_style()
    markers = present_panel(ad, GRAN_MARKERS)
    dp = sc.pl.dotplot(
        ad,
        var_names=markers,
        groupby="hgca_celltype_v1",
        standard_scale="var",
        dot_max=0.7,
        show=False,
        return_fig=True,
        title="Granulocytes vs other myeloid",
    )
    save_dotplot(dp, out / "sfig5_d_granulocytes")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects", default=None, help="Directory with lineage h5ads")
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default analyses/sfig5_rare_celltypes/out)",
    )
    p.add_argument(
        "--panel",
        default="a,b,c,d",
        help="Comma-separated subset of a,b,c,d",
    )
    p.add_argument(
        "--sinus-clusters",
        default=",".join(SINUS_CLUSTERS),
        help="Comma-separated leiden_lineage_l2 IDs for panel b (default 13_0–13_5)",
    )
    args = p.parse_args(argv)
    objects = objects_dir(args.objects)
    out = Path(args.outdir) if args.outdir else ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    wanted = {x.strip().lower() for x in args.panel.split(",") if x.strip()}
    if "a" in wanted:
        panel_inflares(objects, out)
    if "b" in wanted:
        panel_sinus(
            objects,
            out,
            [x.strip() for x in args.sinus_clusters.split(",") if x.strip()],
        )
    if "c" in wanted:
        panel_macrophages(objects, out)
    if "d" in wanted:
        panel_granulocytes(objects, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
