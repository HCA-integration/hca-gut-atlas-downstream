#!/usr/bin/env python3
"""Sample-level GSVA of niche marker programs + bimodality diagnostics.

Hypothesis: when a follicle / LN niche is captured in a biopsy, marker-gene
GSVA scores are high; when absent, scores hug zero → bimodal across samples.

Uses CAP taxonomy markers (alias-resolved), expanded with data-driven top
Wilcoxon genes when a set has < MIN_SET genes. Scores via decoupler.run_gsva
on lineage sample pseudobulks (log2 CPM+1).

Outputs under ../data/:
  niche_gsva_gene_sets.csv
  niche_gsva_scores_long.csv
  niche_gsva_bimodality_global.csv
  niche_gsva_bimodality_strata.csv
  niche_gsva_vs_detection.csv
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
_OBJECTS = os.environ.get("HGCA_OBJECTS")
OBJ = Path(_OBJECTS) if _OBJECTS else Path()
CAP_MARKERS = Path(os.environ["HGCA_CAP_BRIDGE"]) if os.environ.get("HGCA_CAP_BRIDGE") else Path()
DATA.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PB = 20
MIN_SET = 5
TOP_DE = 20
MIN_STRATUM_N = 15
RANDOM_STATE = 0

# Alias cleanup for CAP strings that are protein / CD names
ALIAS = {
    "CD10": "MME",
    "CD206": "MRC1",
    "CD278": "ICOS",
    "CD279": "PDCD1",
    "RANKL": "TNFSF11",
    "CD38": "CD38",
    "CD4": "CD4",
    "CD7": "CD7",
    "CD69": "CD69",
    "CD160": "CD160",
    "CD244": "CD244",
    "CD40LG": "CD40LG",
    "KRT": "KRT8",  # generic family label → drop-ish; keep one
}

# Programs: (label, celltype(s) for DE expansion, lineage file, CAP row name(s), curated extra)
PROGRAMS = [
    {
        "program": "GC_module",
        "lineage": "lymphoid",
        "cap_names": [
            "GC B Dark Zone (GC B DZ)",
            "GC B Light Zone (GC B LZ)",
            "Germinal Center B Cells (GC B)",
        ],
        "de_celltypes": [
            "GC B Dark Zone (GC B DZ)",
            "GC B Light Zone (GC B LZ)",
        ],
        "curated": ["AICDA", "RGS13", "BCL6", "CXCR4", "CD83", "MME", "SUGCT", "LMO2"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "GC_DZ",
        "lineage": "lymphoid",
        "cap_names": ["GC B Dark Zone (GC B DZ)"],
        "de_celltypes": ["GC B Dark Zone (GC B DZ)"],
        "curated": ["AICDA", "CXCR4", "MKI67", "RGS13", "NEIL1"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "GC_LZ",
        "lineage": "lymphoid",
        "cap_names": ["GC B Light Zone (GC B LZ)"],
        "de_celltypes": ["GC B Light Zone (GC B LZ)"],
        "curated": ["CD83", "BANK1", "TNFRSF13C", "CXCR5", "BCL6"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "Tfh",
        "lineage": "lymphoid",
        "cap_names": ["CD4 Tfh"],
        "de_celltypes": ["CD4 Tfh"],
        "curated": ["CXCR5", "BCL6", "PDCD1", "CD40LG", "ICOS", "CXCL13", "TOX2", "IL21"],
        "drop": ["FOXP3"],  # Tfr contaminant in CAP list
        "role": "featured",
    },
    {
        "program": "Tfr",
        "lineage": "lymphoid",
        "cap_names": ["CD4 Tfr"],
        "de_celltypes": ["CD4 Tfr"],
        "curated": ["FOXP3", "IL2RA", "CXCR5", "BCL6", "IL10"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "FARM",
        "lineage": "myeloid",
        "cap_names": ["Follicle Associated Resident Macrophages"],
        "de_celltypes": ["Follicle Associated Resident Macrophages"],
        "curated": ["CD209", "CD163L1", "FOLR2", "MRC1", "PLA2G2D", "PTGDS", "MMP9"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "fDC",
        "lineage": "stroma",
        "cap_names": ["Follicular Dendritic Cells (fDC)"],
        "de_celltypes": ["Follicular Dendritic Cells (fDC)"],
        "curated": ["CR2", "CR1", "CXCL13", "TNFSF13B", "FCER2"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "Med_sinus",
        "lineage": "stroma",
        "cap_names": ["Medullary Sinus Endothelial"],
        "de_celltypes": ["Medullary Sinus Endothelial"],
        "curated": ["STAB2", "MARCO", "MRC1", "LYVE1", "CLEC4G"],
        "drop": [],
        "role": "featured",
    },
    {
        "program": "Goblet",
        "lineage": "epithelial",
        "cap_names": ["Goblet Cells", "Mature Goblet Cells"],
        "de_celltypes": ["Goblet Cells"],
        "curated": ["MUC2", "TFF3", "CLCA1", "SPINK4", "REG4", "ITLN1"],
        "drop": [],
        "role": "negative_control",
    },
    {
        "program": "CD8_IEL",
        "lineage": "lymphoid",
        "cap_names": ["CD8 IEL"],
        "de_celltypes": ["CD8 IEL"],
        "curated": ["ITGAE", "CD160", "CD7", "CD69", "NKG7", "GZMA"],
        "drop": [],
        "role": "negative_control",
    },
]

LINEAGE_FILES = {
    "lymphoid": OBJ / "lymphoid.h5ad",
    "myeloid": OBJ / "myeloid.h5ad",
    "stroma": OBJ / "stroma.h5ad",
    "epithelial": OBJ / "epithelial.h5ad",
}

# Composition detection labels (exact atlas names) for concordance
DETECTION_MAP = {
    "GC_module": ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"],
    "GC_DZ": ["GC B Dark Zone (GC B DZ)"],
    "GC_LZ": ["GC B Light Zone (GC B LZ)"],
    "Tfh": ["CD4 Tfh"],
    "Tfr": ["CD4 Tfr"],
    "FARM": ["Follicle Associated Resident Macrophages"],
    "fDC": ["Follicular Dendritic Cells (fDC)"],
    "Med_sinus": ["Medullary Sinus Endothelial"],
    "Goblet": ["Goblet Cells"],
    "CD8_IEL": ["CD8 IEL"],
}


def _resolve_symbol(g: str) -> str:
    g = str(g).strip()
    return ALIAS.get(g, g)


def load_cap_genes(names: list[str]) -> list[str]:
    cap = pd.read_csv(CAP_MARKERS)
    genes: list[str] = []
    for name in names:
        hit = cap[cap.hgca_celltype_v1.astype(str) == name]
        if hit.empty:
            continue
        raw = str(hit.iloc[0]["markers_final"])
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if tok and tok.lower() not in {"nan", "none"}:
                genes.append(_resolve_symbol(tok))
    return genes


def filter_obs(adata: ad.AnnData) -> ad.AnnData:
    obs = adata.obs
    m = obs["sampled_site_condition"].astype(str).isin(["healthy", "adjacent"])
    m &= obs["tissue_level_1"].astype(str).str.lower().isin(
        ["duodenum", "jejunum", "ileum", "colon", "mesentery", "accessory"]
    )
    out = adata[m].copy()
    # gene symbols
    if "gene_symbol" in out.var.columns:
        out.var_names = out.var["gene_symbol"].astype(str).values
        out.var_names_make_unique()
    out.obs["sample_id"] = out.obs["sample_id"].astype(str)
    out.obs["dataset_id"] = out.obs["dataset_id"].astype(str)
    out.obs["segment"] = out.obs["tissue_level_1"].astype(str).str.lower()
    out.obs["radial"] = out.obs["radial_tissue_term"].astype(str).str.upper()
    return out


def pseudobulk_samples(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum raw counts per sample_id; return (counts, meta)."""
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()
    samples = adata.obs["sample_id"].astype(str).values
    uniq, inv = np.unique(samples, return_inverse=True)
    ind = sparse.csr_matrix(
        (np.ones(len(inv), dtype=np.float64), (inv, np.arange(len(inv)))),
        shape=(len(uniq), X.shape[0]),
    )
    pb = ind @ X
    n_cells = np.bincount(inv, minlength=len(uniq))
    keep = n_cells >= MIN_CELLS_PB
    pb = pb[keep]
    uniq = uniq[keep]
    n_cells = n_cells[keep]
    counts = pd.DataFrame(
        np.asarray(pb.todense() if sparse.issparse(pb) else pb),
        index=uniq,
        columns=adata.var_names.astype(str),
    )
    meta_cols = [
        c
        for c in [
            "dataset_id",
            "segment",
            "radial",
            "sample_collection_method",
            "sampled_site_condition",
            "donor_id",
        ]
        if c in adata.obs.columns
    ]
    meta = (
        adata.obs.groupby("sample_id", observed=True)[meta_cols]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    )
    meta = meta.reindex(uniq)
    meta.index.name = "sample_id"
    meta["n_cells_lineage"] = n_cells
    meta["total_counts"] = counts.sum(axis=1).values
    return counts, meta


def to_log2cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=1).values.astype(float)
    lib[lib <= 0] = 1.0
    cpm = counts.values / lib[:, None] * 1e6
    return pd.DataFrame(np.log2(cpm + 1.0), index=counts.index, columns=counts.columns)


def expand_with_de(adata: ad.AnnData, celltypes: list[str], present: set[str]) -> list[str]:
    """Top Wilcoxon markers for celltype(s) vs rest; subsample for speed."""
    genes: list[str] = []
    ct = adata.obs["hgca_celltype_v1"].astype(str)
    for celltype in celltypes:
        mask = ct == celltype
        n_pos = int(mask.sum())
        if n_pos < 30:
            print(f"  DE skip {celltype}: only {n_pos} cells")
            continue
        # balanced subsample
        rng = np.random.default_rng(RANDOM_STATE)
        pos_idx = np.flatnonzero(mask.values)
        neg_idx = np.flatnonzero(~mask.values)
        n_take_pos = min(n_pos, 1500)
        n_take_neg = min(len(neg_idx), 4000)
        take = np.concatenate(
            [
                rng.choice(pos_idx, n_take_pos, replace=False),
                rng.choice(neg_idx, n_take_neg, replace=False),
            ]
        )
        sub = adata[take].copy()
        sub.obs["__grp"] = np.where(
            sub.obs["hgca_celltype_v1"].astype(str) == celltype, "pos", "neg"
        )
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        sc.tl.rank_genes_groups(
            sub, groupby="__grp", groups=["pos"], reference="neg",
            method="wilcoxon", n_genes=TOP_DE, use_raw=False,
        )
        names = list(sub.uns["rank_genes_groups"]["names"]["pos"])
        genes.extend([g for g in names if g in present])
        print(f"  DE {celltype}: top={names[:8]}")
    return genes


def build_gene_sets_for_lineage(adata: ad.AnnData, lineage: str) -> pd.DataFrame:
    rows = []
    present = set(adata.var_names.astype(str))
    for prog in PROGRAMS:
        if prog["lineage"] != lineage:
            continue
        genes = load_cap_genes(prog["cap_names"])
        genes += [_resolve_symbol(g) for g in prog["curated"]]
        genes = [g for g in genes if g not in prog["drop"]]
        genes = [g for g in genes if g in present]
        # Always add top DE markers so GSVA sets are atlas-grounded (≥ MIN_SET)
        print(f"Expanding {prog['program']} with DE markers (seed n={len(set(genes))})")
        genes += expand_with_de(adata, prog["de_celltypes"], present)
        genes = sorted(set(g for g in genes if g in present))
        if len(genes) < MIN_SET:
            print(f"  WARNING: {prog['program']} still has only {len(genes)} genes")
        print(f"  set {prog['program']}: n={len(genes)} → {genes[:12]}...")
        for g in genes:
            rows.append(
                dict(
                    program=prog["program"],
                    lineage=prog["lineage"],
                    role=prog["role"],
                    gene=g,
                )
            )
    return pd.DataFrame(rows)


def run_gsva_for_lineage(
    log2cpm: pd.DataFrame, gene_sets: pd.DataFrame, lineage: str
) -> pd.DataFrame:
    net = gene_sets[gene_sets.lineage == lineage][["program", "gene"]].rename(
        columns={"program": "source", "gene": "target"}
    )
    if net.empty:
        return pd.DataFrame()
    # keep genes present
    net = net[net.target.isin(log2cpm.columns)]
    # drop programs below min_n
    keep_src = net.groupby("source").size()
    keep_src = keep_src[keep_src >= MIN_SET].index
    net = net[net.source.isin(keep_src)]
    if net.empty:
        return pd.DataFrame()
    # filter mat to expressed genes (reduce noise / memory)
    mat = log2cpm
    print(f"  GSVA {lineage}: {mat.shape[0]} samples × {mat.shape[1]} genes; "
          f"{net.source.nunique()} programs")
    estimates, = dc.run_gsva(mat, net, min_n=MIN_SET, verbose=False)
    return estimates


def ashman_d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    if len(x) < 10:
        return np.nan
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE, n_init=5)
    gmm.fit(x)
    mu = gmm.means_.ravel()
    var = gmm.covariances_.ravel()
    order = np.argsort(mu)
    mu, var = mu[order], var[order]
    return float(abs(mu[1] - mu[0]) / np.sqrt((var[0] + var[1]) / 2.0))


def gmm_delta_bic(x: np.ndarray) -> tuple[float, float, int]:
    """Return (bic1 - bic2, weight_of_low_component, preferred_k). Positive ΔBIC favors 2."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    if len(x) < 10:
        return np.nan, np.nan, 1
    m1 = GaussianMixture(1, random_state=RANDOM_STATE, n_init=3).fit(x)
    m2 = GaussianMixture(2, random_state=RANDOM_STATE, n_init=5).fit(x)
    d = float(m1.bic(x) - m2.bic(x))
    # low component = smaller mean
    order = np.argsort(m2.means_.ravel())
    w_low = float(m2.weights_.ravel()[order[0]])
    pref = 2 if d > 0 else 1
    return d, w_low, pref


def bimodality_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return np.nan
    n = len(x)
    g = stats.skew(x, bias=False)
    # excess kurtosis → convert to kurtosis used by Sarle: m4/m2^2
    k = stats.kurtosis(x, fisher=False, bias=False)
    if k == 0:
        return np.nan
    return float((g ** 2 + 1.0) / k)


def summarize_bimodality(scores: pd.DataFrame, group_cols: list[str] | None) -> pd.DataFrame:
    rows = []
    if group_cols:
        grouped = scores.groupby(group_cols + ["program"], dropna=False)
    else:
        grouped = scores.groupby(["program"], dropna=False)
    for key, g in grouped:
        x = g["gsva"].dropna().values
        n = len(x)
        if n < 8:
            continue
        d_bic, w_low, pref = gmm_delta_bic(x)
        ad = ashman_d(x)
        bc = bimodality_coefficient(x)
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            rec = {c: v for c, v in zip(group_cols + ["program"], key)}
        else:
            rec = {"program": key if not isinstance(key, tuple) else key[0]}
        rec.update(
            dict(
                n_samples=n,
                mean=float(np.mean(x)),
                median=float(np.median(x)),
                q05=float(np.quantile(x, 0.05)),
                q95=float(np.quantile(x, 0.95)),
                frac_neg=float(np.mean(x < 0)),
                frac_pos=float(np.mean(x > 0)),
                delta_bic=d_bic,
                ashman_d=ad,
                bimodality_coef=bc,
                gmm_weight_low=w_low,
                prefer_k=pref,
                likely_bimodal=bool(
                    (d_bic is not None)
                    and np.isfinite(d_bic)
                    and d_bic > 10
                    and np.isfinite(ad)
                    and ad > 2
                ),
            )
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def load_detection_table() -> pd.DataFrame:
    """Per-sample cell counts for concordance with GSVA."""
    clr = pd.read_csv(HERE.parent.parent / "data" / "clr_long.csv")
    want = sorted({ct for cts in DETECTION_MAP.values() for ct in cts})
    sub = clr[clr.celltype.isin(want)].copy()
    piv = sub.pivot_table(
        index="sample_id", columns="celltype", values="n_cells",
        aggfunc="sum", fill_value=0,
    )
    for ct in want:
        if ct not in piv.columns:
            piv[ct] = 0
    piv = piv.reset_index()
    piv["gc_module"] = (
        (piv["GC B Light Zone (GC B LZ)"] >= 3)
        | (piv["GC B Dark Zone (GC B DZ)"] >= 3)
    )
    piv["tfh_program"] = piv["CD4 Tfh"] >= 3
    return piv


def main() -> None:
    det = load_detection_table()

    needed_lineages = sorted({p["lineage"] for p in PROGRAMS})
    score_frames = []
    gene_set_frames = []

    for lin in needed_lineages:
        path = LINEAGE_FILES[lin]
        print(f"\n=== Load {lin}: {path.name} ===")
        raw = sc.read_h5ad(path)
        adata = filter_obs(raw)
        del raw
        print(
            f"  filtered: {adata.n_obs:,} cells, {adata.n_vars:,} genes, "
            f"{adata.obs.sample_id.nunique()} samples"
        )

        print(f"=== Gene sets: {lin} ===")
        gs = build_gene_sets_for_lineage(adata, lin)
        gene_set_frames.append(gs)

        print(f"=== Pseudobulk + GSVA: {lin} ===")
        counts, meta = pseudobulk_samples(adata)
        del adata
        log2cpm = to_log2cpm(counts)
        del counts
        est = run_gsva_for_lineage(log2cpm, gs, lin)
        del log2cpm
        if est.empty:
            continue
        long = (
            est.rename_axis("sample_id")
            .reset_index()
            .melt(id_vars="sample_id", var_name="program", value_name="gsva")
        )
        long["lineage"] = lin
        long = long.merge(meta.reset_index(), on="sample_id", how="left")
        score_frames.append(long)
        del est, meta

    gene_sets = pd.concat(gene_set_frames, ignore_index=True)
    gene_sets.to_csv(DATA / "niche_gsva_gene_sets.csv", index=False)

    scores = pd.concat(score_frames, ignore_index=True)
    role_map = gene_sets.drop_duplicates("program").set_index("program")["role"]
    scores["role"] = scores["program"].map(role_map)

    # attach composition detection
    det_cols = ["sample_id", "gc_module", "tfh_program"]
    extra_cts = sorted({ct for cts in DETECTION_MAP.values() for ct in cts})
    for ct in extra_cts:
        if ct in det.columns and ct not in det_cols:
            det_cols.append(ct)
    scores = scores.merge(det[det_cols], on="sample_id", how="left")

    def detected_for_row(r) -> bool:
        cts = DETECTION_MAP.get(r["program"], [])
        if not cts:
            return np.nan
        vals = [r[c] for c in cts if c in r.index and pd.notna(r[c])]
        if not vals:
            return np.nan
        return bool(np.any(np.asarray(vals) >= 3))

    scores["detected_ge3"] = scores.apply(detected_for_row, axis=1)
    # scopes
    scores["scope_gut"] = scores["segment"].isin(
        ["duodenum", "jejunum", "ileum", "colon"]
    )
    scores["scope_ln"] = scores["segment"].isin(
        ["duodenum", "jejunum", "ileum", "colon", "mesentery", "accessory"]
    )
    scores.to_csv(DATA / "niche_gsva_scores_long.csv", index=False)
    print(f"Wrote scores: {len(scores):,} rows")

    # Bimodality: global (gut wall featured samples), by radial, by segment×dataset
    gut = scores[scores.scope_gut].copy()
    global_bi = summarize_bimodality(gut, None)
    global_bi["strata"] = "global_gut"
    radial_bi = summarize_bimodality(gut, ["radial"])
    radial_bi["strata"] = "radial"
    seg_bi = summarize_bimodality(gut, ["segment"])
    seg_bi["strata"] = "segment"
    # tissue × dataset (require enough n)
    td = gut.copy()
    td["tissue_dataset"] = td["segment"] + " | " + td["dataset_id"]
    # filter strata size before summarize — do inside via n check
    td_bi = summarize_bimodality(td, ["segment", "dataset_id"])
    td_bi = td_bi[td_bi.n_samples >= MIN_STRATUM_N].copy()
    td_bi["strata"] = "segment×dataset"

    # LN scope global (includes mesentery)
    ln = scores[scores.scope_ln].copy()
    ln_bi = summarize_bimodality(ln, None)
    ln_bi["strata"] = "global_ln"
    mes = scores[scores.segment.eq("mesentery")].copy()
    mes_bi = summarize_bimodality(mes, None)
    mes_bi["strata"] = "mesentery_only"

    bimod = pd.concat(
        [global_bi, radial_bi, seg_bi, td_bi, ln_bi, mes_bi], ignore_index=True, sort=False
    )
    bimod.to_csv(DATA / "niche_gsva_bimodality_strata.csv", index=False)
    bimod[bimod.strata.isin(["global_gut", "global_ln", "mesentery_only"])].to_csv(
        DATA / "niche_gsva_bimodality_global.csv", index=False
    )

    # Concordance: GSVA vs cell-count detection
    conc_rows = []
    for prog, g in scores.groupby("program"):
        for scope_name, mask in [("gut", g.scope_gut), ("ln", g.scope_ln)]:
            gg = g.loc[mask]
            if gg["detected_ge3"].isna().all():
                continue
            for det_flag, sub in gg.groupby("detected_ge3", dropna=True):
                x = sub["gsva"].dropna()
                if len(x) == 0:
                    continue
                conc_rows.append(
                    dict(
                        program=prog,
                        scope=scope_name,
                        detected_ge3=bool(det_flag),
                        n=len(x),
                        mean_gsva=float(x.mean()),
                        median_gsva=float(x.median()),
                        q25=float(x.quantile(0.25)),
                        q75=float(x.quantile(0.75)),
                    )
                )
    conc = pd.DataFrame(conc_rows)
    conc.to_csv(DATA / "niche_gsva_vs_detection.csv", index=False)

    print("\n=== Global gut bimodality (featured) ===")
    show = bimod[
        (bimod.strata == "global_gut")
        & (bimod.program.isin([p["program"] for p in PROGRAMS if p["role"] == "featured"]))
    ][
        ["program", "n_samples", "delta_bic", "ashman_d", "bimodality_coef",
         "prefer_k", "likely_bimodal", "frac_neg", "frac_pos"]
    ]
    print(show.to_string(index=False))
    print("\nLikely bimodal segment×dataset strata:")
    hit = td_bi[td_bi.likely_bimodal].sort_values("delta_bic", ascending=False)
    print(
        hit[["program", "segment", "dataset_id", "n_samples", "delta_bic", "ashman_d"]]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
