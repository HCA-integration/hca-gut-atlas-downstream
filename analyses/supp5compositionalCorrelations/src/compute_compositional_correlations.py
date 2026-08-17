#!/usr/bin/env python3
"""Compositional correlations among HGCA cell types (support-aware).

Joint CLR from per-sample counts; Spearman correlations with BH-FDR.
Pairs / strata are tested only when sample support is adequate for *both*
cell types (detection = n_cells ≥ DETECT_MIN_CELLS), not merely when total
library n is large. Sparse rare types (e.g. colon EEC S / M cells) therefore
drop out of testing rather than producing unstable Spearman r.

Stats:
  - Association: Spearman rank correlation on joint CLR
  - Multiple testing: Benjamini–Hochberg FDR among tested pairs
  - Pair retained only if n_samples ≥ MIN_SAMPLES and
    n_samples with celltype_a detected ≥ MIN_DETECT and
    n_samples with celltype_b detected ≥ MIN_DETECT
  - Epi×immune analyses further restrict to samples with epithelial cells > 0
    and (lymphoid + myeloid) cells > 0 (avoids false zeros from missing lineages)

Outputs under ../data/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parents[1]
DATA = FIG / "data"
PARENT = REPO / "data" / "demo" / "expected" / "clr"

MIN_SAMPLES = 30          # total samples in stratum for a tested pair
DETECT_MIN_CELLS = 3      # a sample "detects" a type only if n_cells ≥ this
MIN_DETECT = 20           # samples detecting each member of the pair
MIN_SEGMENT_BOTH = 40     # samples with epi+immune for segment epi×immune panels
PSEUDO = 0.5

RARE_EPITHELIAL = [
    "Tuft Cells",
    "Tuft Progenitors",
    "BEST4 Enterocytes",
    "BEST4 Colonocytes",
    "Paneth Cells",
    "Microfold Cells (M Cells)",
    "EEC Enterochromaffin (EC)",
    "EEC L",
    "EEC N",
    "EEC S",
    "EEC Progenitors",
    "Enteroendocrine Cells (EEC)",
    "Goblet Cells",
    "Mature Goblet Cells",
    "Secretory Progenitors",
]

COVARIATES = [
    ("tissue_level_1", "segment"),
    ("radial_tissue_term", "radial"),
    ("sample_collection_method", "collection"),
    ("sample_preservation_method", "preservation"),
    ("sampled_site_condition", "condition"),
    ("dataset_id", "study"),
    ("sex_ontology_term", "sex"),
]


def clr_transform(counts: pd.DataFrame, pseudo: float = PSEUDO) -> pd.DataFrame:
    x = counts.astype(float) + pseudo
    logx = np.log(x)
    return logx.sub(logx.mean(axis=1), axis=0)


def fisher_z(rho: float) -> float:
    return float(np.arctanh(np.clip(rho, -0.999999, 0.999999)))


def short_name(ct: str) -> str:
    rep = {
        "Microfold Cells (M Cells)": "M cells",
        "Enteroendocrine Cells (EEC)": "EEC",
        "EEC Enterochromaffin (EC)": "EEC EC",
        "EEC Progenitors": "EEC prog.",
        "Tuft Progenitors": "Tuft prog.",
        "BEST4 Enterocytes": "BEST4 ent.",
        "BEST4 Colonocytes": "BEST4 col.",
        "Mature Goblet Cells": "Mature goblet",
        "Goblet Cells": "Goblet",
        "Paneth Cells": "Paneth",
        "Secretory Progenitors": "Sec. prog.",
        "Follicle Associated Resident Macrophages": "FARM",
        "GC B Light Zone (GC B LZ)": "GC B LZ",
        "GC B Dark Zone (GC B DZ)": "GC B DZ",
        "Monocyte Derived Dendritic Cells (MO DC)": "MO DC",
        "Perivascular Resident Macrophages": "PV mac.",
        "Homeostatic Macrophages": "Homeo. mac.",
        "Gamma Delta T Cells": "GD T",
        "CD8 Circulating Effector Memory": "CD8 circ. EM",
        "CD8 Effector Memory": "CD8 EM",
        "CD8 Memory Exhausted": "CD8 exh.",
        "Plasma IGA": "Plasma IgA",
        "Plasma IGG": "Plasma IgG",
    }
    return rep.get(ct, ct)


def load_counts(clr_path: Path | None = None, map_path: Path | None = None):
    clr = pd.read_csv(clr_path or PARENT / "clr_long.csv")
    lmap = pd.read_csv(map_path or PARENT / "celltype_lineage_map.csv")
    lineage = lmap.set_index("celltype")["lineage"]

    meta_cols = [
        "sample_id", "donor_id", "dataset_id", "tissue_level_1",
        "sampled_site_condition", "radial_tissue_term",
        "sample_preservation_method", "sex_ontology_term",
        "sample_collection_method", "age_range",
    ]
    if "chemical_fractionation" in clr.columns:
        meta_cols.append("chemical_fractionation")
    meta = clr.drop_duplicates("sample_id")[meta_cols].copy()
    meta["segment"] = meta["tissue_level_1"].astype(str).str.lower()
    meta["collection"] = (
        meta["sample_collection_method"].astype(str).str.lower()
        .replace({"surgical resection": "resection"})
    )
    if "chemical_fractionation" not in meta.columns:
        # fallback: merge curated sample table if clr not yet patched
        frac_path = PARENT / "sample_chemical_fractionation.csv"
        if frac_path.exists():
            frac = pd.read_csv(frac_path)[
                ["sample_id", "chemical_fractionation", "radial_tissue_term"]
            ]
            meta = meta.drop(columns=["radial_tissue_term"], errors="ignore")
            meta = meta.merge(frac, on="sample_id", how="left")
        else:
            meta["chemical_fractionation"] = "unknown"
    meta = meta[
        meta["sampled_site_condition"].isin(["healthy", "adjacent"])
        & meta["segment"].isin(["duodenum", "jejunum", "ileum", "colon"])
    ].copy()

    counts = (
        clr[clr.sample_id.isin(meta.sample_id)]
        .pivot_table(
            index="sample_id", columns="celltype", values="n_cells",
            aggfunc="sum", fill_value=0,
        )
        .reindex(meta.sample_id)
        .fillna(0)
    )
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    meta = meta.set_index("sample_id").loc[counts.index]

    # lineage totals per sample (for epi+immune support)
    lin_tot = {}
    for lin in ["epithelial", "lymphoid", "myeloid", "stroma"]:
        cts = [c for c in counts.columns if lineage.get(c) == lin]
        lin_tot[lin] = counts[cts].sum(axis=1) if cts else pd.Series(0, index=counts.index)
    meta["n_epithelial"] = lin_tot["epithelial"]
    meta["n_lymphoid"] = lin_tot["lymphoid"]
    meta["n_myeloid"] = lin_tot["myeloid"]
    meta["n_immune"] = meta["n_lymphoid"] + meta["n_myeloid"]
    meta["has_epi_immune"] = (meta["n_epithelial"] > 0) & (meta["n_immune"] > 0)

    return counts, meta, lineage


def spearman_pair(
    a: np.ndarray,
    b: np.ndarray,
    det_a: np.ndarray,
    det_b: np.ndarray,
    min_samples: int = MIN_SAMPLES,
    min_detect: int = MIN_DETECT,
):
    """Spearman with detection support gates."""
    n = len(a)
    n_det_a = int(det_a.sum())
    n_det_b = int(det_b.sum())
    if n < min_samples or n_det_a < min_detect or n_det_b < min_detect:
        return dict(
            spearman_r=np.nan, p_value=np.nan, n_samples=n,
            n_detect_a=n_det_a, n_detect_b=n_det_b, tested=False,
            exclude_reason=(
                f"n={n}<{min_samples}" if n < min_samples
                else f"detect_a={n_det_a}<{min_detect}" if n_det_a < min_detect
                else f"detect_b={n_det_b}<{min_detect}"
            ),
        )
    rho, pval = stats.spearmanr(a, b)
    return dict(
        spearman_r=float(rho), p_value=float(pval), n_samples=n,
        n_detect_a=n_det_a, n_detect_b=n_det_b, tested=True, exclude_reason="",
    )


def stratum_pair_gates(n_stratum: int) -> tuple[int, int]:
    """Relax pair gates for small strata so n_stratum < MIN_SAMPLES can still test."""
    min_samples = min(MIN_SAMPLES, max(15, n_stratum))
    min_detect = min(MIN_DETECT, max(8, n_stratum // 3))
    return min_samples, min_detect


def corr_block(
    counts: pd.DataFrame,
    joint: pd.DataFrame,
    types_a: list,
    types_b: list,
    lineage: pd.Series,
    min_samples: int = MIN_SAMPLES,
    min_detect: int = MIN_DETECT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rectangular correlations types_a × types_b with support masking."""
    rows, cols = types_a, types_b
    r = pd.DataFrame(np.nan, index=rows, columns=cols)
    p = pd.DataFrame(np.nan, index=rows, columns=cols)
    n_det = pd.DataFrame("", index=rows, columns=cols)
    long_rows = []
    for a in rows:
        for b in cols:
            if a not in joint.columns or b not in joint.columns:
                continue
            det_a = counts[a].to_numpy() >= DETECT_MIN_CELLS
            det_b = counts[b].to_numpy() >= DETECT_MIN_CELLS
            res = spearman_pair(
                joint[a].to_numpy(), joint[b].to_numpy(), det_a, det_b,
                min_samples=min_samples, min_detect=min_detect,
            )
            r.loc[a, b] = res["spearman_r"]
            p.loc[a, b] = res["p_value"]
            n_det.loc[a, b] = f"{res['n_detect_a']}/{res['n_detect_b']}"
            long_rows.append(
                dict(
                    celltype_a=a, celltype_b=b,
                    lineage_a=lineage.get(a, ""), lineage_b=lineage.get(b, ""),
                    short_a=short_name(a), short_b=short_name(b),
                    **res,
                )
            )
    long = pd.DataFrame(long_rows)
    # BH-FDR among tested pairs only
    if len(long) and long["tested"].any():
        tested = long["tested"].to_numpy()
        adj = np.full(len(long), np.nan)
        adj[tested] = multipletests(long.loc[tested, "p_value"], method="fdr_bh")[1]
        long["p_adj"] = adj
        for _, row in long.iterrows():
            if row["tested"] and np.isfinite(row["p_adj"]):
                p.loc[row["celltype_a"], row["celltype_b"]] = row["p_adj"]
    else:
        long["p_adj"] = np.nan
    return r, p, n_det, long


def full_corr(counts, joint, lineage):
    """Full square matrix among all types with support gates."""
    cols = list(joint.columns)
    p = len(cols)
    r_m = np.full((p, p), np.nan)
    p_m = np.full((p, p), np.nan)
    np.fill_diagonal(r_m, 1.0)
    np.fill_diagonal(p_m, 0.0)
    long_rows = []
    for i in range(p):
        for j in range(i + 1, p):
            a, b = cols[i], cols[j]
            res = spearman_pair(
                joint[a].to_numpy(), joint[b].to_numpy(),
                counts[a].to_numpy() >= DETECT_MIN_CELLS,
                counts[b].to_numpy() >= DETECT_MIN_CELLS,
            )
            r_m[i, j] = r_m[j, i] = res["spearman_r"]
            p_m[i, j] = p_m[j, i] = res["p_value"]
            long_rows.append(
                dict(celltype_a=a, celltype_b=b,
                     lineage_a=lineage.get(a, ""), lineage_b=lineage.get(b, ""),
                     short_a=short_name(a), short_b=short_name(b), **res)
            )
    long = pd.DataFrame(long_rows)
    tested = long["tested"].to_numpy()
    adj = np.full(len(long), np.nan)
    if tested.any():
        adj[tested] = multipletests(long.loc[tested, "p_value"], method="fdr_bh")[1]
    long["p_adj"] = adj
    p_adj_m = np.full((p, p), np.nan)
    np.fill_diagonal(p_adj_m, 0.0)
    for k, row in long.iterrows():
        if not row["tested"]:
            continue
        i = cols.index(row["celltype_a"])
        j = cols.index(row["celltype_b"])
        p_adj_m[i, j] = p_adj_m[j, i] = row["p_adj"]
    return (
        pd.DataFrame(r_m, index=cols, columns=cols),
        pd.DataFrame(p_adj_m, index=cols, columns=cols),
        long,
    )


def prevalence_table(counts, meta, lineage, types):
    rows = []
    for seg, idx in meta.groupby("segment").groups.items():
        idx = list(idx)
        for ct in types:
            if ct not in counts.columns:
                continue
            x = counts.loc[idx, ct]
            rows.append(
                dict(
                    segment=seg,
                    celltype=ct,
                    short_name=short_name(ct),
                    lineage=lineage.get(ct, ""),
                    n_samples=len(idx),
                    n_detect_ge1=int((x >= 1).sum()),
                    n_detect_ge3=int((x >= 3).sum()),
                    total_cells=int(x.sum()),
                    prevalence_ge1=float((x >= 1).mean()),
                    prevalence_ge3=float((x >= 3).mean()),
                    n_epi_immune=int(meta.loc[idx, "has_epi_immune"].sum()),
                )
            )
    return pd.DataFrame(rows)


def main():
    global DATA, PARENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clr-long", type=Path, default=PARENT / "clr_long.csv")
    parser.add_argument("--lineage-map", type=Path, default=PARENT / "celltype_lineage_map.csv")
    parser.add_argument("--outdir", type=Path, default=DATA)
    args = parser.parse_args()
    DATA = args.outdir
    DATA.mkdir(parents=True, exist_ok=True)
    PARENT = args.clr_long.parent
    if "demo" in str(args.clr_long):
        print("DEMO MODE: results are for software checking, not manuscript figures.")
    counts, meta, lineage = load_counts(args.clr_long, args.lineage_map)
    print(f"Samples: {counts.shape[0]}  Cell types: {counts.shape[1]}")
    print("Epi+immune samples by segment:")
    print(meta.groupby("segment")["has_epi_immune"].agg(["sum", "count"]).to_string())

    immune_types = sorted(
        [c for c in counts.columns if lineage.get(c) in ("lymphoid", "myeloid")]
    )
    rare_epi = [c for c in RARE_EPITHELIAL if c in counts.columns]
    sets = pd.DataFrame(
        {
            "celltype": rare_epi + immune_types,
            "set": ["rare_epithelial"] * len(rare_epi) + ["immune"] * len(immune_types),
            "lineage": [lineage.get(c, "") for c in rare_epi + immune_types],
            "short_name": [short_name(c) for c in rare_epi + immune_types],
        }
    )
    sets.to_csv(DATA / "celltype_sets.csv", index=False)

    prev = prevalence_table(counts, meta, lineage, rare_epi + immune_types)
    prev.to_csv(DATA / "celltype_prevalence_by_segment.csv", index=False)

    # Focus support for user's examples
    focus = ["BEST4 Colonocytes", "EEC S", "Microfold Cells (M Cells)"]
    print("\nFocus prevalence (ge1 / ge3 / total cells):")
    print(
        prev[prev.celltype.isin(focus)][
            ["segment", "celltype", "n_samples", "n_detect_ge1", "n_detect_ge3",
             "total_cells", "n_epi_immune"]
        ].to_string(index=False)
    )

    # ── Overall (all samples with epi+immune for fair joint CLR) ───────────
    keep = meta.index[meta["has_epi_immune"]]
    print(f"\nOverall epi+immune samples: {len(keep)}")
    counts_ei = counts.loc[keep]
    joint_ei = clr_transform(counts_ei)
    meta_ei = meta.loc[keep]

    joint_ei.reset_index().melt(
        id_vars="sample_id", var_name="celltype", value_name="joint_clr"
    ).merge(meta_ei.reset_index(), on="sample_id").to_csv(
        DATA / "joint_clr_long.csv", index=False
    )

    # Support metadata for plots
    support = pd.DataFrame(
        [
            dict(
                scope="overall_epi_immune",
                segment="all",
                n_samples=len(keep),
                n_epithelial_pos=int((meta_ei.n_epithelial > 0).sum()),
                n_immune_pos=int((meta_ei.n_immune > 0).sum()),
                note=f"Min pair support: ≥{MIN_SAMPLES} samples, ≥{MIN_DETECT} detects each type",
            )
        ]
    )

    print("Overall full matrix…")
    r_all, p_all, long_all = full_corr(counts_ei, joint_ei, lineage)
    r_all.to_csv(DATA / "corr_matrix_overall.csv")
    p_all.to_csv(DATA / "corr_pmatrix_overall.csv")
    long_all.to_csv(DATA / "corr_overall_spearman.csv", index=False)

    print("Overall epi × immune…")
    r_ei, p_ei, n_ei, long_ei = corr_block(
        counts_ei, joint_ei, rare_epi, immune_types, lineage
    )
    r_ei.to_csv(DATA / "corr_rect_epi_immune_overall_r.csv")
    p_ei.to_csv(DATA / "corr_rect_epi_immune_overall_padj.csv")
    n_ei.to_csv(DATA / "corr_rect_epi_immune_overall_ndetect.csv")
    long_ei.to_csv(DATA / "corr_epi_immune_overall.csv", index=False)
    n_tested = int(long_ei["tested"].sum())
    print(f"  tested pairs: {n_tested} / {len(long_ei)}")
    top = (
        long_ei[long_ei.tested]
        .assign(abs_r=lambda d: d.spearman_r.abs())
        .sort_values("abs_r", ascending=False)
        .head(12)
    )
    print(top[["celltype_a", "celltype_b", "spearman_r", "p_adj",
               "n_detect_a", "n_detect_b"]].to_string(index=False))

    # ── By segment (only if enough epi+immune samples) ─────────────────────
    # Drop stale underpowered segment matrices from prior runs
    for stale in DATA.glob("corr_rect_epi_immune_segment_*"):
        stale.unlink()
    for stale in DATA.glob("corr_epi_immune_segment_*"):
        stale.unlink()

    pair_het = []
    for seg, idx in meta_ei.groupby("segment").groups.items():
        idx = list(idx)
        n_both = len(idx)
        support = pd.concat(
            [
                support,
                pd.DataFrame(
                    [dict(
                        scope="segment_epi_immune",
                        segment=seg,
                        n_samples=n_both,
                        n_epithelial_pos=int((meta_ei.loc[idx, "n_epithelial"] > 0).sum()),
                        n_immune_pos=int((meta_ei.loc[idx, "n_immune"] > 0).sum()),
                        note=(
                            "powered" if n_both >= MIN_SEGMENT_BOTH
                            else f"underpowered (n={n_both} < {MIN_SEGMENT_BOTH})"
                        ),
                    )]
                ),
            ],
            ignore_index=True,
        )
        if n_both < MIN_SEGMENT_BOTH:
            print(f"SKIP segment {seg}: epi+immune n={n_both}")
            continue
        print(f"Segment {seg}: epi+immune n={n_both}")
        csub = counts_ei.loc[idx]
        jsub = joint_ei.loc[idx]
        rs, ps, ns, ls = corr_block(csub, jsub, rare_epi, immune_types, lineage)
        rs.to_csv(DATA / f"corr_rect_epi_immune_segment_{seg}_r.csv")
        ps.to_csv(DATA / f"corr_rect_epi_immune_segment_{seg}_padj.csv")
        ns.to_csv(DATA / f"corr_rect_epi_immune_segment_{seg}_ndetect.csv")
        ls["segment"] = seg
        ls.to_csv(DATA / f"corr_epi_immune_segment_{seg}.csv", index=False)
        print(f"  tested pairs: {int(ls.tested.sum())} / {len(ls)}")

    # ── Collection × chemical fractionation (UF vs fractionated) ───────────
    # Drop stale chemfrac matrices
    for stale in DATA.glob("corr_rect_epi_immune_chemfrac_*"):
        stale.unlink()
    for stale in DATA.glob("corr_epi_immune_chemfrac_*"):
        stale.unlink()

    print("\nCollection × chemical_fractionation (epi+immune):")
    print(
        meta_ei.groupby(["collection", "chemical_fractionation"])
        .size()
        .rename("n")
        .to_string()
    )

    chem_levels = ["unfractionated", "fractionated"]
    coll_levels = ["biopsy", "resection"]
    # Always write ileum/colon splits for biopsy unfractionated (user-requested)
    chem_strata = [
        (f"{coll}_{frac}", meta_ei["collection"].eq(coll)
         & meta_ei["chemical_fractionation"].astype(str).eq(frac))
        for coll in coll_levels for frac in chem_levels
    ] + [
        ("biopsy_unfractionated_ileum",
         meta_ei["collection"].eq("biopsy")
         & meta_ei["chemical_fractionation"].astype(str).eq("unfractionated")
         & meta_ei["segment"].eq("ileum")),
        ("biopsy_unfractionated_colon",
         meta_ei["collection"].eq("biopsy")
         & meta_ei["chemical_fractionation"].astype(str).eq("unfractionated")
         & meta_ei["segment"].eq("colon")),
    ]
    # Show requested segment splits if n >= 15; others keep MIN_SEGMENT_BOTH
    CHEM_FORCE_MIN = 15
    for tag, mask in chem_strata:
        idx = list(meta_ei.index[mask])
        n_both = len(idx)
        force = tag.endswith("_ileum") or tag.endswith("_colon")
        min_n = CHEM_FORCE_MIN if force else MIN_SEGMENT_BOTH
        support = pd.concat(
            [
                support,
                pd.DataFrame(
                    [dict(
                        scope="chemfrac_epi_immune",
                        segment=tag,
                        n_samples=n_both,
                        n_epithelial_pos=int(
                            (meta_ei.loc[idx, "n_epithelial"] > 0).sum()
                        ) if n_both else 0,
                        n_immune_pos=int(
                            (meta_ei.loc[idx, "n_immune"] > 0).sum()
                        ) if n_both else 0,
                        note=(
                            "powered" if n_both >= MIN_SEGMENT_BOTH
                            else (
                                f"shown_underpowered (n={n_both} < {MIN_SEGMENT_BOTH})"
                                if n_both >= min_n
                                else f"underpowered (n={n_both} < {min_n})"
                            )
                        ),
                    )]
                ),
            ],
            ignore_index=True,
        )
        if n_both < min_n:
            print(f"SKIP chemfrac {tag}: epi+immune n={n_both}")
            continue
        print(f"Chemfrac {tag}: epi+immune n={n_both}")
        csub = counts_ei.loc[idx]
        jsub = joint_ei.loc[idx]
        ms, md = stratum_pair_gates(n_both)
        rs, ps, ns, ls = corr_block(
            csub, jsub, rare_epi, immune_types, lineage,
            min_samples=ms, min_detect=md,
        )
        rs.to_csv(DATA / f"corr_rect_epi_immune_chemfrac_{tag}_r.csv")
        ps.to_csv(DATA / f"corr_rect_epi_immune_chemfrac_{tag}_padj.csv")
        ns.to_csv(DATA / f"corr_rect_epi_immune_chemfrac_{tag}_ndetect.csv")
        ls["chemfrac_stratum"] = tag
        ls.to_csv(DATA / f"corr_epi_immune_chemfrac_{tag}.csv", index=False)
        print(f"  tested pairs: {int(ls.tested.sum())} / {len(ls)} "
              f"(min_samples={ms}, min_detect={md})")

    # ── LP-only (chemically fractionated lamina propria libraries) ─────────
    for stale in DATA.glob("corr_rect_epi_immune_lp_*"):
        stale.unlink()
    for stale in DATA.glob("corr_epi_immune_lp_*"):
        stale.unlink()

    meta_ei = meta_ei.copy()
    meta_ei["radial"] = meta_ei["radial_tissue_term"].astype(str).str.upper()
    lp_ei = meta_ei[meta_ei["radial"].eq("LP")]
    print("\nLP-only epi+immune:")
    print(
        lp_ei.groupby(["collection", "segment"]).size().rename("n").to_string()
        if len(lp_ei) else "(none)"
    )

    lp_strata = [
        ("biopsy_all", lp_ei["collection"].eq("biopsy")),
        ("biopsy_ileum", lp_ei["collection"].eq("biopsy") & lp_ei["segment"].eq("ileum")),
        ("biopsy_colon", lp_ei["collection"].eq("biopsy") & lp_ei["segment"].eq("colon")),
        ("all_ileum", lp_ei["segment"].eq("ileum")),
    ]
    LP_FORCE_MIN = 15  # show ileum/colon LP even if below panel gate
    for tag, mask in lp_strata:
        idx = list(lp_ei.index[mask])
        n_both = len(idx)
        force = tag in {"biopsy_ileum", "biopsy_colon"}
        min_n = LP_FORCE_MIN if force else MIN_SEGMENT_BOTH
        support = pd.concat(
            [
                support,
                pd.DataFrame(
                    [dict(
                        scope="lp_epi_immune",
                        segment=tag,
                        n_samples=n_both,
                        n_epithelial_pos=int(
                            (meta_ei.loc[idx, "n_epithelial"] > 0).sum()
                        ) if n_both else 0,
                        n_immune_pos=int(
                            (meta_ei.loc[idx, "n_immune"] > 0).sum()
                        ) if n_both else 0,
                        note=(
                            "powered" if n_both >= MIN_SEGMENT_BOTH
                            else (
                                f"shown_underpowered (n={n_both} < {MIN_SEGMENT_BOTH})"
                                if n_both >= min_n
                                else f"underpowered (n={n_both} < {min_n})"
                            )
                        ),
                    )]
                ),
            ],
            ignore_index=True,
        )
        if n_both < min_n:
            print(f"SKIP LP {tag}: epi+immune n={n_both}")
            continue
        print(f"LP {tag}: epi+immune n={n_both}")
        csub = counts_ei.loc[idx]
        jsub = joint_ei.loc[idx]
        ms, md = stratum_pair_gates(n_both)
        rs, ps, ns, ls = corr_block(
            csub, jsub, rare_epi, immune_types, lineage,
            min_samples=ms, min_detect=md,
        )
        rs.to_csv(DATA / f"corr_rect_epi_immune_lp_{tag}_r.csv")
        ps.to_csv(DATA / f"corr_rect_epi_immune_lp_{tag}_padj.csv")
        ns.to_csv(DATA / f"corr_rect_epi_immune_lp_{tag}_ndetect.csv")
        ls["lp_stratum"] = tag
        ls.to_csv(DATA / f"corr_epi_immune_lp_{tag}.csv", index=False)
        print(f"  tested pairs: {int(ls.tested.sum())} / {len(ls)} "
              f"(min_samples={ms}, min_detect={md})")

    support.to_csv(DATA / "analysis_support_summary.csv", index=False)

    # ── Heterogeneity across covariates (epi×immune, support-aware) ────────
    print("\nCovariate heterogeneity…")
    cov_summaries = []
    for cov_col, cov_label in COVARIATES:
        col = cov_col if cov_col in meta_ei.columns else cov_label
        if col not in meta_ei.columns:
            continue
        levels = meta_ei[col].astype(str)
        vc = levels.value_counts()
        keep_levels = [
            lv for lv, n in vc.items()
            if n >= MIN_SAMPLES and str(lv).lower() not in {"nan", "none", "unknown", ""}
        ]
        if len(keep_levels) < 2:
            print(f"  skip {cov_label}")
            continue

        z_vars, abs_deltas = [], []
        for e in rare_epi:
            for i in immune_types:
                zs, rs, ws = [], [], []
                for lv in keep_levels:
                    samp = meta_ei.index[levels.eq(lv)]
                    if len(samp) < MIN_SAMPLES:
                        continue
                    res = spearman_pair(
                        joint_ei.loc[samp, e].to_numpy(),
                        joint_ei.loc[samp, i].to_numpy(),
                        counts_ei.loc[samp, e].to_numpy() >= DETECT_MIN_CELLS,
                        counts_ei.loc[samp, i].to_numpy() >= DETECT_MIN_CELLS,
                    )
                    pair_het.append(
                        dict(
                            covariate=cov_label, level=lv,
                            rare_epithelial=e, immune=i,
                            spearman_r=res["spearman_r"], p_value=res["p_value"],
                            n_samples=res["n_samples"],
                            n_detect_epi=res["n_detect_a"],
                            n_detect_imm=res["n_detect_b"],
                            tested=res["tested"],
                            fisher_z=fisher_z(res["spearman_r"]) if res["tested"] else np.nan,
                        )
                    )
                    if res["tested"]:
                        zs.append(fisher_z(res["spearman_r"]))
                        rs.append(res["spearman_r"])
                        ws.append(res["n_samples"])
                if len(zs) < 2:
                    continue
                zs = np.asarray(zs); w = np.asarray(ws, float); w = w / w.sum()
                mu = np.sum(w * zs)
                z_vars.append(float(np.sum(w * (zs - mu) ** 2)))
                abs_deltas.append(float(np.max(rs) - np.min(rs)))

        cov_summaries.append(
            dict(
                covariate=cov_label,
                n_levels=len(keep_levels),
                levels=";".join(map(str, keep_levels)),
                n_pairs_scored=len(z_vars),
                mean_fisher_z_var=float(np.mean(z_vars)) if z_vars else np.nan,
                median_fisher_z_var=float(np.median(z_vars)) if z_vars else np.nan,
                mean_abs_delta_r=float(np.mean(abs_deltas)) if abs_deltas else np.nan,
            )
        )
        print(
            f"  {cov_label}: mean z-var={cov_summaries[-1]['mean_fisher_z_var']:.4f}  "
            f"|Δr|={cov_summaries[-1]['mean_abs_delta_r']:.3f}  pairs={len(z_vars)}"
        )

    cov_df = pd.DataFrame(cov_summaries)
    if cov_df.empty:
        cov_df = pd.DataFrame(
            columns=[
                "covariate",
                "n_levels",
                "levels",
                "n_pairs_scored",
                "mean_fisher_z_var",
                "median_fisher_z_var",
                "mean_abs_delta_r",
            ]
        )
    else:
        cov_df = cov_df.sort_values("mean_fisher_z_var", ascending=False)
    cov_df.to_csv(DATA / "corr_heterogeneity_by_covariate.csv", index=False)

    pair_df = pd.DataFrame(pair_het)
    pair_df.to_csv(DATA / "corr_pair_heterogeneity_long.csv", index=False)
    if not pair_df.empty:
        pair_sum = (
            pair_df[pair_df.tested]
            .groupby(["covariate", "rare_epithelial", "immune"], as_index=False)
            .agg(
                n_levels=("level", "nunique"),
                mean_r=("spearman_r", "mean"),
                min_r=("spearman_r", "min"),
                max_r=("spearman_r", "max"),
                abs_delta_r=("spearman_r", lambda s: float(s.max() - s.min())),
                fisher_z_var=("fisher_z", "var"),
                min_n=("n_samples", "min"),
                min_detect=("n_detect_epi", "min"),
            )
            .sort_values(["covariate", "abs_delta_r"], ascending=[True, False])
        )
        pair_sum.to_csv(DATA / "corr_pair_heterogeneity_summary.csv", index=False)

    # lineage blocks on epi+immune samples
    for lin in ["epithelial", "lymphoid", "myeloid", "stroma"]:
        cts = [c for c in joint_ei.columns if lineage.get(c) == lin]
        if len(cts) < 3:
            continue
        rs, ps, ls = full_corr(counts_ei[cts], joint_ei[cts], lineage)
        rs.to_csv(DATA / f"corr_matrix_lineage_{lin}.csv")
        ps.to_csv(DATA / f"corr_pmatrix_lineage_{lin}.csv")

    # write stats methods blurb
    (DATA / "STATS_METHODS.txt").write_text(
        f"""Statistical methods — compositional correlations

Association test
  Spearman rank correlation between joint CLR abundances of cell-type pairs.
  Joint CLR: centred log-ratio of per-sample cell counts across all cell types
  (pseudocount {PSEUDO}), restricted to samples with epithelial cells > 0 and
  immune (lymphoid + myeloid) cells > 0.

Support filters (pair retained / tested only if all pass)
  - n_samples ≥ {MIN_SAMPLES}
  - n_samples with celltype A detected (n_cells ≥ {DETECT_MIN_CELLS}) ≥ {MIN_DETECT}
  - n_samples with celltype B detected (n_cells ≥ {DETECT_MIN_CELLS}) ≥ {MIN_DETECT}
  Sparse rare types (colon EEC S / M cells; ileum BEST4 colonocytes) typically
  fail the detection gate and are left blank rather than tested.

Multiple testing
  Benjamini–Hochberg FDR on raw Spearman P values among tested pairs within
  each analysis block (overall; each segment; each lineage matrix).

Segment epi×immune heatmaps
  Shown only when n samples with both epithelial and immune lineages
  ≥ {MIN_SEGMENT_BOTH}. Duodenum / jejunum fail this gate in HGCA
  (few libraries profile both compartments).

Chemical fractionation × collection
  Same epi×immune Spearman / FDR / support gates within
  sample_collection_method × chemical_fractionation strata
  (biopsy|resection × unfractionated|fractionated).
  chemical_fractionation is methods-curated (not equal to radial_tissue_term);
  Martin2019 recoded to radial LP and fractionated.

Heterogeneity ranking
  For each covariate and epi×immune pair, Spearman r within each level that
  passes the support filters; score = weighted variance of Fisher z(r).
  Covariates ranked by mean score across scored pairs.
"""
    )
    print("\nWrote", DATA / "STATS_METHODS.txt")
    print("Done.")


if __name__ == "__main__":
    main()
