#!/usr/bin/env python3
"""Build a Nature-scale demo slice of hgca_all_lineages_v1.h5ad.

The slice keeps every hgca_celltype_v1 label and the hard-coded covariate
columns used across figure scripts, with enough samples to exercise
Mann-Whitney (>=5 per ileum/colon x biopsy/resection) and the follicle
k=3 rule. It is not intended to reproduce paper numbers.

Size target: well under GitHub's 50 MB warning (prefer <25 MB) so the
file can live in the review repo without LFS. Code Ocean treats local
/data as a "small or example dataset" inside a 5 GB capsule workspace;
Zenodo's 50 GB record limit is not the constraint.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(os.environ["HGCA_H5AD"]) if os.environ.get("HGCA_H5AD") else None
DEFAULT_OUT = ROOT / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
TAXONOMY = Path(
    os.environ.get("HGCA_TAXONOMY", ROOT / "data" / "demo" / "GCA_taxonomy_2026_CAP.csv")
)
GC_GENES = ROOT / "data" / "demo" / "follicle_gsva_gc_b_gene_list.csv"

KEEP_OBS = [
    "sample_id",
    "donor_id",
    "dataset_id",
    "author_cell_type",
    "closest_GCA_celltype",
    "hgca_celltype_v0",
    "hgca_celltype_v1",
    "hgca_celltype_level1",
    "hgca_celltype_level2",
    "hgca_celltype_level3",
    "hgca_celltype_level4",
    "hgca_celltype_level5",
    "tissue_level_1",
    "tissue_level_2",
    "radial_tissue_term",
    "sample_collection_method",
    "sample_preservation_method",
    "sampled_site_condition",
    "disease",
    "sex_ontology_term",
    "age_range",
    "assay",
    "sequenced_fragment",
    "gene_annotation_version",
    "n_counts",
    "n_genes",
]

FEATURED = [
    "GC B Light Zone (GC B LZ)",
    "GC B Dark Zone (GC B DZ)",
    "BEST4 Enterocytes",
    "BEST4 Colonocytes",
    "Tuft Progenitors",
    "Tuft Cells",
    "Paneth Cells",
    "Brunners Gland Cells",
    "CD4 Tfr",
    "Follicle Associated Resident Macrophages",
    "Post Arteriole Capillary Endothelial (PAC)",
    "Submucosal Fibroblasts (S3)",
    "Goblet Cells",
    "Mature Goblet Cells",
]
GC_TYPES = {"GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"}
BEST4_MARKERS = [
    "CFTR", "FOLH1", "ONECUT2", "CACNA2D1", "SMIM24", "CPA2", "ADGRG4", "LYZ",
    "BEST4", "OTOP2", "SPIB",
]
STAPLE_GENES = [
    "EPCAM", "PTPRC", "COL1A1", "PECAM1", "CD3D", "CD68", "CD79A", "MS4A1",
    "LGR5", "MUC2", "DEFA5", "AICDA", "BCL6", "RGS13", "CXCR4", "CD83",
    "MME", "LMO2", "SUGCT", "ACTB", "GAPDH",
]
CONTRASTS = [
    ("ileum", "biopsy"),
    ("ileum", "surgical resection"),
    ("colon", "biopsy"),
    ("colon", "surgical resection"),
]
UNKNOWN = {"", "unknown", "nan", "none", "n/a", "na", "not applicable"}


def _s(series: pd.Series) -> pd.Series:
    return series.astype(str)


def _is_unknown(series: pd.Series) -> pd.Series:
    return _s(series).str.strip().str.lower().isin(UNKNOWN)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def pick_n(ids: pd.Index | list, n: int, rng: np.random.Generator) -> list:
    ids = list(pd.Index(ids).unique())
    if len(ids) <= n:
        return ids
    return [ids[i] for i in rng.choice(len(ids), size=n, replace=False)]


def sample_meta(obs: pd.DataFrame) -> pd.DataFrame:
    gcb = _s(obs["hgca_celltype_v1"]).isin(GC_TYPES)
    counts = (
        obs.assign(_gcb=gcb)
        .groupby("sample_id", observed=True)
        .agg(
            n_cells=("sample_id", "size"),
            n_gcb=("_gcb", "sum"),
            n_lymph=("hgca_celltype_level1", lambda s: (_s(s) == "Lymphoid").sum()),
            tissue=("tissue_level_1", "first"),
            collection=("sample_collection_method", "first"),
            radial=("radial_tissue_term", "first"),
            disease=("disease", "first"),
            condition=("sampled_site_condition", "first"),
            preservation=("sample_preservation_method", "first"),
            fragment=("sequenced_fragment", "first"),
            assay=("assay", "first"),
            age=("age_range", "first"),
            sex=("sex_ontology_term", "first"),
            dataset=("dataset_id", "first"),
            donor=("donor_id", "first"),
        )
    )
    counts["tissue"] = _s(counts["tissue"])
    counts["collection"] = _s(counts["collection"])
    counts.index = _s(counts.index)
    return counts


def choose_samples(meta: pd.DataFrame, rng: np.random.Generator, n_contrast: int) -> list[str]:
    chosen: list[str] = []

    def add(ids, n):
        for sid in pick_n(ids, n, rng):
            if sid not in chosen:
                chosen.append(sid)

    for tissue, collection in CONTRASTS:
        block = meta[(meta["tissue"] == tissue) & (meta["collection"] == collection)]
        # Prefer healthy/normal, then smaller libraries so the slice stays tiny.
        healthy = block[_s(block["condition"]).str.lower().eq("healthy")]
        pool = healthy if len(healthy) >= n_contrast else block
        pool = pool.sort_values(["n_cells", "n_gcb"])
        add(pool.index[: max(n_contrast * 3, n_contrast)], n_contrast)

    # Extra segments and technical levels that scripts hard-code.
    add(meta[meta["tissue"] == "duodenum"].index, 2)
    add(meta[meta["tissue"] == "jejunum"].index, 2)
    add(meta[_s(meta["radial"]).eq("EPI")].index, 2)
    add(meta[_s(meta["radial"]).eq("LP")].index, 2)
    add(meta[_s(meta["radial"]).eq("EPI_LP_MUSC")].index, 2)
    add(meta[_s(meta["preservation"]).str.contains("frozen", case=False, na=False)].index, 2)
    add(meta[_s(meta["fragment"]).str.contains("5", na=False)].index, 2)
    add(meta[_s(meta["condition"]).str.lower().eq("adjacent")].index, 2)

    # Follicle k=3 needs samples with and without GC B.
    add(meta[meta["n_gcb"] >= 3].sort_values("n_cells").index, 3)
    add(meta[meta["n_gcb"] == 0].sort_values("n_cells").index, 3)

    # Age decades and both sexes, cheap extras.
    for age in ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]:
        block = meta[_s(meta["age"]).eq(age)]
        if len(block):
            add(block.sort_values("n_cells").index, 1)
    for sex in ["male", "female"]:
        add(meta[_s(meta["sex"]).str.lower().eq(sex)].index, 1)

    return chosen


def choose_cells(
    obs: pd.DataFrame,
    sample_ids: list[str],
    follicle_neg: set[str],
    rng: np.random.Generator,
    max_per_sample: int,
    featured_cap: int,
    missing_cap: int,
) -> pd.Index:
    keep: set[str] = set()
    obs = obs.copy()
    obs["sample_id"] = _s(obs["sample_id"])
    ct = _s(obs["hgca_celltype_v1"])
    lin = _s(obs["hgca_celltype_level1"])

    for sid in sample_ids:
        sub = obs.loc[obs["sample_id"] == sid]
        gcb = sub.index[ct.loc[sub.index].isin(GC_TYPES)]
        lymph = sub.index[lin.loc[sub.index].eq("Lymphoid")]
        if sid not in follicle_neg:
            keep.update(pick_n(gcb, min(len(gcb), 12), rng))
        keep.update(pick_n([i for i in lymph if i not in gcb], min(len(lymph), 25), rng))
        remaining = [i for i in sub.index if i not in keep and i not in gcb]
        keep.update(pick_n(remaining, max(0, max_per_sample - len(keep.intersection(sub.index))), rng))

    # Guarantee every terminal identity appears. Do not pull GC B cells
    # from follicle-negative samples, or those samples stop being negatives.
    for typ, sub in obs.groupby(obs["hgca_celltype_v1"].astype(str), observed=True):
        cap = featured_cap if typ in FEATURED else missing_cap
        already = sum(1 for i in sub.index if i in keep)
        want = 8 if typ in FEATURED else 1
        if already >= want:
            continue
        pool = [i for i in sub.index if i not in keep]
        if typ in GC_TYPES:
            pool = [i for i in pool if sub.loc[i, "sample_id"] not in follicle_neg]
        keep.update(pick_n(pool, cap - already, rng))

    # Extra BEST4 / goblet cells in ileum and colon so a DESeq2-style filter
    # (>10 cells/sample) can still run for one or two types.
    for typ in ["Goblet Cells", "BEST4 Enterocytes", "BEST4 Colonocytes"]:
        sub = obs.loc[ct.eq(typ)]
        for tissue in ["ileum", "colon"]:
            tsub = sub.loc[_s(sub["tissue_level_1"]).eq(tissue)]
            for sid in pick_n(_s(tsub["sample_id"]), 4, rng):
                cells = tsub.index[_s(tsub["sample_id"]) == sid]
                keep.update(pick_n(cells, 12, rng))

    return pd.Index(sorted(keep))


def choose_genes(var: pd.DataFrame, gc_csv: Path, n_extra: int) -> list[str]:
    """Return Ensembl IDs. Named paper genes are matched on gene_symbol."""
    symbols: list[str] = []
    if gc_csv.exists():
        symbols.extend(pd.read_csv(gc_csv)["gene"].astype(str).tolist())
    symbols.extend(BEST4_MARKERS)
    symbols.extend(STAPLE_GENES)
    symbols = list(dict.fromkeys(symbols))
    sym = var["gene_symbol"].astype(str)
    named_ids = var.index[sym.isin(symbols)].astype(str).tolist()
    extra_pool = [i for i in var.index.astype(str) if i not in set(named_ids)]
    if extra_pool and n_extra:
        step = max(1, len(extra_pool) // n_extra)
        extra = extra_pool[::step][:n_extra]
    else:
        extra = []
    return named_ids + extra


def coverage_report(obs: pd.DataFrame) -> dict:
    meta = sample_meta(obs)
    contrasts = {}
    for tissue, collection in CONTRASTS:
        n = int(((meta["tissue"] == tissue) & (meta["collection"] == collection)).sum())
        contrasts[f"{tissue}|{collection}"] = n
    return {
        "n_cells": int(obs.shape[0]),
        "n_celltypes": int(_s(obs["hgca_celltype_v1"]).nunique()),
        "n_samples": int(obs["sample_id"].nunique()),
        "n_donors": int(obs["donor_id"].nunique()),
        "n_datasets": int(obs["dataset_id"].nunique()),
        "contrasts_n_samples": contrasts,
        "n_samples_gcb_ge3": int((meta["n_gcb"] >= 3).sum()),
        "n_samples_gcb_zero": int((meta["n_gcb"] == 0).sum()),
        "tissues": _s(obs["tissue_level_1"]).value_counts().to_dict(),
        "collections": _s(obs["sample_collection_method"]).value_counts().to_dict(),
        "radials": _s(obs["radial_tissue_term"]).value_counts().to_dict(),
        "conditions": _s(obs["sampled_site_condition"]).value_counts().to_dict(),
        "ages": _s(obs["age_range"]).value_counts().to_dict(),
        "sexes": _s(obs["sex_ontology_term"]).value_counts().to_dict(),
        "featured_counts": {
            t: int((_s(obs["hgca_celltype_v1"]) == t).sum()) for t in FEATURED
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--n-contrast", type=int, default=5)
    p.add_argument("--max-per-sample", type=int, default=60)
    p.add_argument("--featured-cap", type=int, default=12)
    p.add_argument("--missing-cap", type=int, default=6)
    p.add_argument("--n-extra-genes", type=int, default=1600)
    args = p.parse_args()
    if args.source is None:
        raise SystemExit(
            "Pass --source /path/to/hgca_all_lineages_v1.h5ad "
            "or set HGCA_H5AD. The bundled demo slice is already in data/demo/."
        )

    print(f"Reading backed obs from {args.source}", flush=True)
    src = ad.read_h5ad(args.source, backed="r")
    obs = src.obs.copy()
    rng = _rng(args.seed)
    meta = sample_meta(obs)
    sample_ids = choose_samples(meta, rng, args.n_contrast)
    follicle_neg = set(meta.index[meta["n_gcb"] == 0].intersection(sample_ids))
    if len(follicle_neg) < 3:
        extra_neg = [s for s in meta.index[meta["n_gcb"] == 0] if s not in sample_ids]
        sample_ids.extend(extra_neg[: 3 - len(follicle_neg)])
        follicle_neg = set(meta.index[meta["n_gcb"] == 0].intersection(sample_ids))
    print(f"Follicle-negative samples pinned: {len(follicle_neg)}", flush=True)
    cell_ids = choose_cells(
        obs,
        sample_ids,
        follicle_neg,
        rng,
        args.max_per_sample,
        args.featured_cap,
        args.missing_cap,
    )
    genes = choose_genes(src.var, GC_GENES, args.n_extra_genes)
    print(
        f"Selected {len(cell_ids)} cells from {pd.Index(sample_ids).nunique()} "
        f"planned samples; {len(genes)} genes",
        flush=True,
    )

    pos_map = pd.Series(np.arange(obs.shape[0]), index=obs.index)
    idx = np.sort(pos_map.loc[cell_ids].to_numpy())
    gene_mask = src.var_names.isin(genes)
    print("Materializing subset…", flush=True)
    demo = src[idx, gene_mask].to_memory()
    keep_cols = [c for c in KEEP_OBS if c in demo.obs.columns]
    demo.obs = demo.obs[keep_cols].copy()
    for col in demo.obs.columns:
        if isinstance(demo.obs[col].dtype, pd.CategoricalDtype):
            demo.obs[col] = demo.obs[col].cat.remove_unused_categories()
    # Figure scripts look up AICDA / BEST4 / etc. by symbol.
    if "gene_symbol" in demo.var.columns:
        demo.var["gene_id"] = demo.var_names.astype(str)
        seen: dict[str, int] = {}
        symbols = []
        for s in demo.var["gene_symbol"].astype(str):
            n = seen.get(s, 0)
            symbols.append(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
        demo.var_names = pd.Index(symbols)
        demo.var.index = demo.var_names
    demo.obsm = None
    demo.obsp = None
    demo.uns = {
        "title": "HGCA v1 Nature software demo slice",
        "demo_mode": True,
        "banner": "DEMO MODE: results are for software checking, not manuscript figures.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    demo.write_h5ad(args.out, compression="gzip")
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({size_mb:.1f} MB)", flush=True)

    report = coverage_report(demo.obs)
    report.update(
        {
            "source": str(args.source),
            "out": str(args.out),
            "n_genes": int(demo.n_vars),
            "size_mb": round(size_mb, 2),
            "seed": args.seed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "github_warn_mb": 50,
            "github_block_mb": 100,
            "codeocean_workspace_gb": 5,
            "zenodo_record_gb": 50,
        }
    )
    prov = args.out.with_suffix(".provenance.json")
    prov.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in [
        "n_cells", "n_celltypes", "n_samples", "n_datasets",
        "contrasts_n_samples", "n_samples_gcb_ge3", "n_samples_gcb_zero",
        "size_mb",
    ]}, indent=2), flush=True)

    dest_demo = args.out.parent
    if TAXONOMY.exists():
        shutil.copy2(TAXONOMY, dest_demo / TAXONOMY.name)
    if GC_GENES.exists():
        shutil.copy2(GC_GENES, dest_demo / GC_GENES.name)

    missing_types = sorted(set(_s(obs["hgca_celltype_v1"]).unique()) - set(_s(demo.obs["hgca_celltype_v1"]).unique()))
    if missing_types:
        raise SystemExit(f"Slice dropped cell types: {missing_types}")
    weak = [k for k, n in report["contrasts_n_samples"].items() if n < 5]
    if weak:
        print(f"WARNING: contrasts with <5 samples: {weak}", flush=True)
    if size_mb >= 50:
        print("WARNING: file is at or above GitHub's 50 MB warning threshold", flush=True)


if __name__ == "__main__":
    main()
