"""Gene-expression analog of the composition sampling-depth analysis.

Two questions, per cell type, over the depth categories
  radial_tissue_term        (5 layers)
  sample_collection_method  (biopsy vs surgical resection)
  full_thickness            (EPI_LP_MUSC vs all other radial layers)

1) VARIANCE EXPLAINED: how much of a cell type's pseudobulk gene-expression
   variance does each depth category explain? Same variance-weighted omega^2
   Principal Component Regression used for composition
   (scripts/composition_vs_expression_pcr.py), computed within each cell type
   on per-sample pseudobulk log-CPM over the top 2000 HVGs. Written for every
   eligible cell type so we can rank which cell types' expression covaries most
   with sampling depth. -> data/depth_gex_variance_explained.csv

2) DIFFERENTIAL EXPRESSION: per-gene pseudobulk two-group tests for the two
   binary depth contrasts (biopsy vs resection; full-thickness vs rest), so the
   top depth-covarying cell types (+ Glia) can be shown as gene volcanoes.
   Vectorised Welch t-test on log2(CPM+1); effect = mean_B - mean_A (a log2
   fold-change), BH-FDR across tested genes. Computed for ALL eligible cell
   types (cheap); only the selected set is rendered as volcanoes downstream.
   -> data/depth_de/<contrast>/<celltype>_de.csv

Selection of cell types to volcano (rendered by render_depth_expression.R):
   top by max depth omega^2 across the three categories, plus Glia if absent.
   -> data/depth_de/selected_celltypes.csv

Run in the scvi-tools env (needs scanpy + the integrated h5ads).
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

_OBJECTS = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else None
if _OBJECTS is None:
    LINEAGE_PATHS = {}
else:
    LINEAGE_PATHS = {
        "epithelial": str(_OBJECTS / "epithelial.h5ad"),
        "lymphoid": str(_OBJECTS / "lymphoid.h5ad"),
        "myeloid": str(_OBJECTS / "myeloid.h5ad"),
        "stroma": str(_OBJECTS / "stroma.h5ad"),
    }
DATA = Path(__file__).resolve().parent.parent / "data"
DE_DIR = DATA / "depth_de"
DE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_KEY = "sample_id"
CT_COL = "hgca_celltype_v1"

DEPTH_COVS = ["radial_tissue_term", "sample_collection_method", "full_thickness"]

MIN_CELLS_PER_PSEUDOBULK = 10
MIN_SAMPLES_PER_CELLTYPE = 25
MIN_SAMPLES_PER_GROUP = 3
N_HVG = 2000
N_PCS = 50
# DE gene filter: keep genes with mean log2CPM above this in at least one group
MIN_EXPR_LOG2CPM = 0.5

N_TOP_VOLCANO = 8  # per depth category, before union

GLIA_MARKER_GROUPS = {
    "Intra-ganglionic": ["APOE", "FRZB", "RLBP1", "CRYM"],
    "Myenteric plexus / type I": ["TF", "BCAN"],
    # CACLB was supplied as a candidate marker but is not a gene symbol in the
    # atlas feature space; it is deliberately not substituted with a guessed
    # CACN-family gene.
    "Submucosal plexus / type I": ["ENTPD2", "SFRP5"],
    "Extra-ganglionic": ["GFRA3"],
    "Mucosal / type III": ["RELN", "MBP", "PLLP"],
    "Muscularis propria / type IV": ["SLC5A7", "APOD", "KCNS3"],
}

BEST4_SEGMENT_MARKERS = [
    "CFTR", "FOLH1", "ONECUT2", "CACNA2D1", "SMIM24", "CPA2", "ADGRG4", "LYZ",
    "ANXA13", "CLDN15", "CCL25", "ALDOB", "ALDH1A1", "SI", "SULT1E1", "FAM3B",
    "DPEP1", "MALRD1", "CKB", "CEACAM5", "VSIG2",
    "C10orf99", "CA2", "HMGCS2", "FABP5", "MUC12",
]


def _clean(s):
    return " ".join(str(s).split())


def _mode_or_nan(x):
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan


# ---------------- omega^2 PCR (verbatim logic from composition_vs_expression_pcr) --
def _anova_r2_all_pcs(scores, groups):
    n, K = scores.shape
    grand = scores.mean(axis=0)
    ss_total = ((scores - grand) ** 2).sum(axis=0)
    codes, inv = np.unique(groups, return_inverse=True)
    G = len(codes)
    if G < 2 or n - G < 1:
        return np.zeros(K)
    ss_between = np.zeros(K)
    for gi in range(G):
        m = inv == gi
        n_g = m.sum()
        if n_g == 0:
            continue
        mean_g = scores[m].mean(axis=0)
        ss_between += n_g * (mean_g - grand) ** 2
    ss_within = np.clip(ss_total - ss_between, 0.0, None)
    ms_within = ss_within / (n - G)
    num = ss_between - (G - 1) * ms_within
    den = ss_total + ms_within
    with np.errstate(divide="ignore", invalid="ignore"):
        omega2 = np.where(den > 0, num / den, 0.0)
    return np.clip(np.nan_to_num(omega2, nan=0.0), 0.0, 1.0)


def _isunknown(s):
    t = s.astype(str).str.strip().str.lower()
    return s.isna() | t.isin(["", "unknown", "nan", "none", "n/a", "na"])


def pcr_per_covariate(embedding, var_weights, meta, covariates):
    out = {}
    w = np.asarray(var_weights, dtype=float)
    w_sum = w.sum()
    meta = meta.reset_index(drop=True)
    for cov in covariates:
        if cov not in meta.columns:
            out[cov] = np.nan
            continue
        cvals = meta[cov].astype("object")
        mask = ~_isunknown(cvals)
        if mask.sum() < 10 or cvals[mask].nunique() < 2:
            out[cov] = np.nan
            continue
        r2 = _anova_r2_all_pcs(embedding[mask.values], cvals[mask].astype(str).values)
        out[cov] = float((w * r2).sum() / w_sum) if w_sum > 0 else np.nan
    return out


def embed_matrix(X, n_pcs):
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    X = X[:, sd > 1e-12]
    if X.shape[1] == 0:
        return None, None
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    n_comp = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    if n_comp < 1:
        return None, None
    p = PCA(n_components=n_comp, svd_solver="full")
    return p.fit_transform(X), p.explained_variance_


# ---------------- pseudobulk ----------------
def build_meta(adata):
    cols = ["radial_tissue_term", "sample_collection_method", "tissue_level_1",
            "dataset_id"]
    cols = [c for c in cols if c in adata.obs.columns]
    sm = adata.obs.groupby(SAMPLE_KEY).agg({c: _mode_or_nan for c in cols})
    sm.index = sm.index.astype(str)
    r = sm["radial_tissue_term"].astype(str).str.strip().str.lower()
    sm["full_thickness"] = r.map(lambda x: "full_thickness" if x == "epi_lp_musc"
                                 else ("rest" if x in {"epi", "epi_lp", "lp", "wm"} else np.nan))
    coll = sm["sample_collection_method"].astype(str).str.strip().str.lower()
    sm["sample_collection_method"] = coll.map(
        {"biopsy": "biopsy", "surgical resection": "surgical resection"})
    return sm


def pseudobulk_counts(adata, celltype):
    """Per-sample summed raw counts for one cell type. Returns (df[samples x genes], samples)."""
    mask = (adata.obs[CT_COL] == celltype).values
    if mask.sum() == 0:
        return None
    X = adata.X[mask]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    samples = adata.obs.loc[mask, SAMPLE_KEY].astype(str).values
    uniq, inv = np.unique(samples, return_inverse=True)
    ind = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                            shape=(len(uniq), X.shape[0]))
    pb = np.asarray((ind @ X).todense(), dtype=float)
    cps = np.bincount(inv, minlength=len(uniq))
    keep = cps >= MIN_CELLS_PER_PSEUDOBULK
    pb, uniq = pb[keep], uniq[keep]
    if pb.shape[0] < MIN_SAMPLES_PER_CELLTYPE:
        return None
    return pd.DataFrame(pb, index=uniq, columns=adata.var_names.astype(str))


def to_log2cpm(counts_df):
    lib = counts_df.values.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    cpm = counts_df.values / lib * 1e6
    return pd.DataFrame(np.log2(cpm + 1.0), index=counts_df.index, columns=counts_df.columns)


def export_glia_marker_pseudobulk(log2cpm, meta, symbol_map):
    """Write sample-level Glia marker expression for the figure dot plot.

    If more than one Ensembl feature maps to a symbol, retain the feature with
    the highest mean log2(CPM+1) in Glia. Dot size downstream is the fraction
    of sample pseudobulks above log2(CPM+1) > 1 (CPM > 1), not a cell-level
    detection rate, so samples remain the independent observational units.
    """
    marker_to_subtype = {
        marker: subtype
        for subtype, markers in GLIA_MARKER_GROUPS.items()
        for marker in markers
    }
    symbol_to_genes = {}
    for gene_id in log2cpm.columns:
        symbol = symbol_map.get(gene_id, gene_id)
        if symbol in marker_to_subtype:
            symbol_to_genes.setdefault(symbol, []).append(gene_id)

    rows = []
    aligned_meta = meta.reindex(log2cpm.index)
    for marker, subtype in marker_to_subtype.items():
        candidates = symbol_to_genes.get(marker, [])
        if not candidates:
            print(f"  Glia marker absent after pseudobulk: {marker}", flush=True)
            continue
        gene_id = max(candidates, key=lambda g: float(log2cpm[g].mean()))
        vals = log2cpm[gene_id]
        for sample_id, value in vals.items():
            m = aligned_meta.loc[sample_id]
            rows.append({
                "sample_id": sample_id,
                "marker": marker,
                "marker_group": subtype,
                "gene_id": gene_id,
                "log2cpm": float(value),
                "detected_cpm_gt_1": bool(value > 1.0),
                "sample_collection_method": m.get("sample_collection_method", np.nan),
                "radial_tissue_term": m.get("radial_tissue_term", np.nan),
                "full_thickness": m.get("full_thickness", np.nan),
                "tissue_level_1": m.get("tissue_level_1", np.nan),
                "dataset_id": m.get("dataset_id", np.nan),
            })
    pd.DataFrame(rows).to_csv(DATA / "glia_depth_marker_pseudobulk.csv", index=False)


def export_best4_segment_marker_dotplot():
    """Export the established ileum-enriched BEST4 program across gut segments."""
    adata = sc.read_h5ad(LINEAGE_PATHS["epithelial"])
    adata.obs[CT_COL] = adata.obs[CT_COL].astype(str).map(_clean)
    segments = ["duodenum", "jejunum", "ileum", "colon"]
    keep = (
        adata.obs[CT_COL].isin(["BEST4 Enterocytes", "BEST4 Colonocytes"])
        & adata.obs["tissue_level_1"].astype(str).str.lower().isin(segments)
    ).values
    sub = adata[keep]

    symbols = adata.var["gene_symbol"].astype(str)
    selected_ids = []
    selected_symbols = []
    for marker in BEST4_SEGMENT_MARKERS:
        ids = adata.var_names[symbols == marker].astype(str).tolist()
        if not ids:
            print(f"BEST4 marker absent: {marker}", flush=True)
            continue
        selected_ids.append(ids[0])
        selected_symbols.append(marker)

    X = sub.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1
    gene_idx = adata.var_names.get_indexer(selected_ids)
    raw = np.asarray(X[:, gene_idx].todense(), dtype=float)
    lognorm = np.log1p(raw / lib[:, None] * 1e4)

    frame = sub.obs[[SAMPLE_KEY, CT_COL, "tissue_level_1"]].copy()
    frame["tissue_level_1"] = frame["tissue_level_1"].astype(str).str.lower()
    rows = []
    for (segment, celltype), idx in frame.groupby(
        ["tissue_level_1", CT_COL], observed=True
    ).indices.items():
        idx = np.asarray(idx)
        for j, marker in enumerate(selected_symbols):
            rows.append({
                "tissue_level_1": segment,
                "celltype": celltype,
                "marker": marker,
                "mean_log1p_10k": float(lognorm[idx, j].mean()),
                "pct_cells_detected": float((raw[idx, j] > 0).mean() * 100),
                "n_cells": int(len(idx)),
                "n_samples": int(frame.iloc[idx][SAMPLE_KEY].nunique()),
            })
    for segment, idx in frame.groupby(
        "tissue_level_1", observed=True
    ).indices.items():
        idx = np.asarray(idx)
        for j, marker in enumerate(selected_symbols):
            rows.append({
                "tissue_level_1": segment,
                "celltype": "All BEST4 cells",
                "marker": marker,
                "mean_log1p_10k": float(lognorm[idx, j].mean()),
                "pct_cells_detected": float((raw[idx, j] > 0).mean() * 100),
                "n_cells": int(len(idx)),
                "n_samples": int(frame.iloc[idx][SAMPLE_KEY].nunique()),
            })
    pd.DataFrame(rows).to_csv(
        DATA / "best4_segment_marker_dotplot.csv", index=False
    )


def adjusted_de(log2cpm, meta, cov, label_a, label_b, symbol_map):
    """dataset_id-adjusted pseudobulk DE for a binary contrast.

    Per gene, OLS  log2CPM ~ 1 + group(B) + C(dataset_id), solved jointly over
    all genes; the reported effect / p-value are for the group coefficient, i.e.
    the biopsy/resection (or full-thickness) difference *after* absorbing
    study-of-origin as fixed effects. Because collection method is almost fully
    nested in dataset_id (only Elmentaite2020 spans both arms), the estimate is
    driven by within-study variation and the contrast is skipped entirely when no
    single dataset contains both arms (otherwise unidentifiable).
    """
    m = meta.reindex(log2cpm.index)
    grp = m[cov].astype(str)
    keep_s = grp.isin([label_a, label_b]).values
    if keep_s.sum() < 2 * MIN_SAMPLES_PER_GROUP:
        return None
    y_all = log2cpm.values[keep_s]
    g = (grp[keep_s] == label_b).astype(float).values
    ds = m["dataset_id"].astype(str).values[keep_s] if "dataset_id" in m.columns \
        else np.array(["_"] * keep_s.sum())
    na, nb = int((g == 0).sum()), int((g == 1).sum())
    if na < MIN_SAMPLES_PER_GROUP or nb < MIN_SAMPLES_PER_GROUP:
        return None

    # need >=1 dataset spanning both arms, else group is collinear with batch
    ds_both = [d for d in np.unique(ds)
               if len(np.unique(g[ds == d])) == 2]
    if not ds_both:
        return None

    # design: intercept + group + dataset dummies (drop-first)
    ds_levels = pd.Index(np.unique(ds))
    D = pd.get_dummies(pd.Categorical(ds, categories=ds_levels),
                       drop_first=True).values.astype(float)
    X = np.column_stack([np.ones(len(g)), g, D])
    n, p = X.shape
    # drop collinear columns for a stable inverse
    q, r = np.linalg.qr(X)
    rank_keep = np.abs(np.diag(r)) > 1e-8
    if not rank_keep[1]:  # group column dropped => unidentifiable
        return None
    X = X[:, rank_keep]
    p = X.shape[1]
    g_idx = 1  # group is 2nd column, retained
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y_all)           # [p x n_genes]
    resid = y_all - X @ beta
    dof = n - p
    if dof < 1:
        return None
    sigma2 = (resid ** 2).sum(axis=0) / dof  # [n_genes]
    se_g = np.sqrt(XtX_inv[g_idx, g_idx] * sigma2)

    ma = y_all[g == 0].mean(0)
    mb = y_all[g == 1].mean(0)
    keep = ((ma > MIN_EXPR_LOG2CPM) | (mb > MIN_EXPR_LOG2CPM)) & (se_g > 0)
    if keep.sum() < 10:
        return None
    eff = beta[g_idx][keep]                  # adjusted log2 fold change (B - A)
    se_k = se_g[keep]
    t = eff / se_k
    pval = 2 * stats.t.sf(np.abs(t), dof)
    pval = np.where(np.isfinite(pval), pval, 1.0)
    padj = multipletests(pval, method="fdr_bh")[1]
    genes = log2cpm.columns.values[keep]
    out = pd.DataFrame({
        "gene_id": genes,
        "gene_symbol": [symbol_map.get(gg, gg) for gg in genes],
        "log2fc_B_minus_A": eff,
        "mean_A": ma[keep], "mean_B": mb[keep],
        "p_value": pval, "p_adj": padj,
        "neglog10_p_adj": -np.log10(np.clip(padj, 1e-300, None)),
        "n_A": na, "n_B": nb, "n_datasets_shared": len(ds_both),
        "adjusted": True,
    })
    return out.sort_values("p_value").reset_index(drop=True)


def unadjusted_de(log2cpm, meta, cov, label_a, label_b, symbol_map):
    """Plain Welch t-test on log2CPM, NO study adjustment. Used only as a
    labelled fallback for cell types whose depth contrast is fully nested in
    dataset_id (no dataset spans both arms), so the study-adjusted model is
    unidentifiable. The effect here is confounded with study-of-origin and must
    be presented as such.
    """
    m = meta.reindex(log2cpm.index)
    grp = m[cov].astype(str)
    ia = (grp == label_a).values
    ib = (grp == label_b).values
    na, nb = int(ia.sum()), int(ib.sum())
    if na < MIN_SAMPLES_PER_GROUP or nb < MIN_SAMPLES_PER_GROUP:
        return None
    A = log2cpm.values[ia]; B = log2cpm.values[ib]
    ma, mb = A.mean(0), B.mean(0)
    va, vb = A.var(0, ddof=1), B.var(0, ddof=1)
    keep = ((ma > MIN_EXPR_LOG2CPM) | (mb > MIN_EXPR_LOG2CPM)) & ((va + vb) > 0)
    if keep.sum() < 10:
        return None
    ma, mb, va, vb = ma[keep], mb[keep], va[keep], vb[keep]
    genes = log2cpm.columns.values[keep]
    se = np.sqrt(va / na + vb / nb); se[se == 0] = np.nan
    t = (mb - ma) / se
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), dof)
    p = np.where(np.isfinite(p), p, 1.0)
    padj = multipletests(p, method="fdr_bh")[1]
    out = pd.DataFrame({
        "gene_id": genes,
        "gene_symbol": [symbol_map.get(gg, gg) for gg in genes],
        "log2fc_B_minus_A": mb - ma,
        "mean_A": ma, "mean_B": mb,
        "p_value": p, "p_adj": padj,
        "neglog10_p_adj": -np.log10(np.clip(padj, 1e-300, None)),
        "n_A": na, "n_B": nb, "n_datasets_shared": 0,
        "adjusted": False,
    })
    return out.sort_values("p_value").reset_index(drop=True)


CONTRASTS = {
    "collection": ("sample_collection_method", "biopsy", "surgical resection"),
    "full_thickness": ("full_thickness", "rest", "full_thickness"),
}


def main():
    var_rows = []
    for contrast in CONTRASTS:
        cdir = DE_DIR / contrast
        cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*_de.csv"):  # clear stale unadjusted outputs
            old.unlink()

    for lineage, path in LINEAGE_PATHS.items():
        print(f"\n=== {lineage} ===", flush=True)
        adata = sc.read_h5ad(path)
        adata.obs[SAMPLE_KEY] = adata.obs[SAMPLE_KEY].astype(str)
        adata.obs[CT_COL] = adata.obs[CT_COL].astype(str).map(_clean)
        meta = build_meta(adata)
        symbol_map = dict(zip(adata.var_names.astype(str),
                              adata.var["gene_symbol"].astype(str)))

        cts = [c for c in adata.obs[CT_COL].unique() if c and c != "nan"]
        for ct in cts:
            counts = pseudobulk_counts(adata, ct)
            if counts is None:
                continue
            log2cpm = to_log2cpm(counts)
            ct_meta = meta.reindex(counts.index)
            if lineage == "stroma" and ct == "Glia":
                export_glia_marker_pseudobulk(log2cpm, meta, symbol_map)

            # variance explained (HVG PCA + omega^2 PCR)
            v = log2cpm.var(0).sort_values(ascending=False)
            hvg = log2cpm[v.head(N_HVG).index]
            scores, evr = embed_matrix(hvg.values, N_PCS)
            if scores is not None:
                pcr = pcr_per_covariate(scores, evr, ct_meta, DEPTH_COVS)
                for cov in DEPTH_COVS:
                    var_rows.append(dict(lineage=lineage, celltype=ct, covariate=cov,
                                         omega2=pcr.get(cov, np.nan),
                                         n_samples=counts.shape[0]))

            # DE for the two binary contrasts: study-adjusted where identifiable,
            # else an explicitly-flagged unadjusted (study-confounded) fallback.
            slug = re.sub(r"[^0-9a-zA-Z]+", "_", ct).strip("_")
            de_status = []
            for contrast, (cov, la, lb) in CONTRASTS.items():
                de = adjusted_de(log2cpm, meta, cov, la, lb, symbol_map)
                mode = "adj"
                if de is None:
                    de = unadjusted_de(log2cpm, meta, cov, la, lb, symbol_map)
                    mode = "unadj"
                if de is not None:
                    de.insert(0, "celltype", ct)
                    de.insert(0, "lineage", lineage)
                    de.to_csv(DE_DIR / contrast / f"{slug}_de.csv", index=False)
                    de_status.append(f"{contrast}:{(de['p_adj']<0.05).sum()}sig[{mode}]")
                else:
                    de_status.append(f"{contrast}:skip")
            print(f"  {ct}: {counts.shape[0]} samples  [{', '.join(de_status)}]", flush=True)

        del adata

    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(DATA / "depth_gex_variance_explained.csv", index=False)
    print(f"\ndepth_gex_variance_explained.csv: {var_df.shape[0]} rows")

    # ---- select cell types to volcano ----
    wide = var_df.pivot_table(index=["lineage", "celltype"], columns="covariate",
                              values="omega2").reset_index()
    for cov in DEPTH_COVS:
        if cov not in wide.columns:
            wide[cov] = np.nan
    wide["max_depth_omega2"] = wide[DEPTH_COVS].max(axis=1)

    selected = set()
    reasons = {}
    for cov in ["sample_collection_method", "full_thickness", "radial_tissue_term"]:
        top = wide.sort_values(cov, ascending=False).head(N_TOP_VOLCANO)
        for _, r in top.iterrows():
            selected.add(r["celltype"])
            reasons.setdefault(r["celltype"], []).append(f"top {cov} ({r[cov]:.2f})")
    if "Glia" in set(wide["celltype"]) and "Glia" not in selected:
        selected.add("Glia")
        grow = wide[wide["celltype"] == "Glia"].iloc[0]
        reasons["Glia"] = [f"requested (max depth omega2 {grow['max_depth_omega2']:.2f})"]

    sel_df = wide[wide["celltype"].isin(selected)].copy()
    sel_df["reason"] = sel_df["celltype"].map(lambda c: "; ".join(reasons.get(c, [])))
    sel_df = sel_df.sort_values("max_depth_omega2", ascending=False)
    sel_df.to_csv(DE_DIR / "selected_celltypes.csv", index=False)
    print(f"selected {len(sel_df)} cell types for volcanoes:")
    print(sel_df[["lineage", "celltype", "sample_collection_method",
                  "full_thickness", "radial_tissue_term", "max_depth_omega2"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
