#!/usr/bin/env python
"""
Concept expression + believability validation on the HGCA all-lineages v1 atlas.

For the 4 CCC concepts (+ venular endothelial chemokine sink / HEV), this:
  1. Computes per-(segment, hgca_celltype_v1) mean log-norm expression + fraction
     expressing, for all genes involved (normal cells, 4 major segments).
  2. Believability / metadata-breadth validation: for each concept's key
     (gene, cell type) it reports in how many of the 27 datasets and 265 donors
     the gene is robustly detected in that cell type, and the cross-dataset
     spread of the mean -- i.e. is the signal broad or one-dataset-driven.
  3. HEV check: HEV / venular markers within endothelial subsets.

Outputs -> <OUT>/concept_validation/
  expr_by_segment_celltype.csv         full table (all target genes)
  believability_by_dataset.csv         per (concept,gene,celltype) dataset/donor support
  hev_endothelial.csv                  HEV marker expression in endothelial subsets
  fig_concept{1..4}_expression.png     per-concept dotplots
  fig_pillar_chemokine_sink.png        venular endothelial sink
  fig_hev_check.png                    HEV marker dotplot
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import matplotlib.pyplot as plt
import gca_plot_style as gps
gps.set_style()

AD_PATH = "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/hgca_all_lineages_v1.h5ad"
OUT = "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA/concept_validation"
os.makedirs(OUT, exist_ok=True)
SEG = ["duodenum", "jejunum", "ileum", "colon"]
CT_KEY = "hgca_celltype_v1"
MIN_CELLS = 20          # min cells per (segment,celltype) to report
DET_FRAC = 0.10         # gene "detected" in a group if frac expressing >= this

GENES = ["TPH1","HTR4","HTR7","PYY","NPY1R","GCG","VIPR1","SCT","NTS","SORT1","DPP4",
         "GUCA2A","GUCA2B","GUCY2C","PCSK1N","GPR171","TPSAB1","TPSB2","F2RL1",
         "NRXN1","NLGN1","NLGN2","PLP1","S100B","SOX10","GFAP","GDNF","NRG1",
         "ACKR1","ACKR2","ACKR3","ACKR4","CCL14","CCL21","CCL19","CCL5","CCL2",
         "CXCL1","CXCL2","CXCL3","CXCL8","CCR7","CXCR3","CXCR2","MADCAM1","CHST4",
         "CHST2","B3GNT3","FUT7","GLCE","SELE","SELP","PECAM1",
         # C6 perivascular / fibrosis-neovascular wiring
         "PDGFA","PDGFRA","PDGFRB","MFGE8","JAG1","NOTCH3","DLL4","VEGFC","LYVE1",
         "WNT5A","MCAM"]

CMAP = gps.SEQ

def load_matrix():
    print("Loading backed AnnData ...", flush=True)
    A = ad.read_h5ad(AD_PATH, backed="r")
    sym = A.var["gene_symbol"].astype(str).values
    present = [g for g in GENES if g in set(sym)]
    gi = [int(np.where(sym == g)[0][0]) for g in present]
    obs = A.obs
    keep = (obs["disease"].astype(str) == "normal") & \
           (obs["tissue_level_1"].astype(str).isin(SEG))
    keep = keep.values
    idx = np.where(keep)[0]
    print(f"  normal + 4-seg cells: {len(idx):,} / {A.n_obs:,}", flush=True)
    n = A.n_obs
    chunk = 50000
    mats = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = A.X[s:e]                      # backed CSR -> in-memory sparse
        if not sparse.issparse(x):
            x = sparse.csr_matrix(x)
        mats.append(x[:, gi])
    X = sparse.vstack(mats).tocsr()[idx]
    md = obs.iloc[idx][["tissue_level_1", CT_KEY, "dataset_id", "donor_id",
                        "n_counts"]].copy()
    md.columns = ["segment", "celltype", "dataset", "donor", "n_counts"]
    md["segment"] = pd.Categorical(md["segment"].astype(str), categories=SEG,
                                   ordered=True)
    # CP10k + log1p using per-cell total counts
    nc = md["n_counts"].values.astype(float)
    nc[nc <= 0] = 1.0
    Xn = X.multiply(1e4 / nc[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    print("  matrix:", Xn.shape, flush=True)
    return Xn, md, present

def summarise(Xn, md, genes):
    """mean log-norm + frac expressing per (segment, celltype, gene)."""
    rows = []
    binX = (Xn > 0).astype(np.int8)
    g_index = {g: i for i, g in enumerate(genes)}
    grp = md.groupby(["segment", "celltype"], observed=True).indices
    for (seg, ct), ii in grp.items():
        if len(ii) < MIN_CELLS:
            continue
        sub = Xn[ii]
        subb = binX[ii]
        mean = np.asarray(sub.mean(axis=0)).ravel()
        frac = np.asarray(subb.mean(axis=0)).ravel()
        for g, gi in g_index.items():
            rows.append((seg, ct, g, len(ii), mean[gi], frac[gi]))
    df = pd.DataFrame(rows, columns=["segment","celltype","gene","n_cells",
                                     "mean_lognorm","frac_expr"])
    return df

def believability(Xn, md, checks):
    """For each (gene, celltype) key: dataset/donor breadth of detection."""
    binX = (Xn > 0).astype(np.int8)
    g_index = {g: i for i, g in enumerate(md_genes)}
    out = []
    for concept, gene, ct_pat in checks:
        cell_mask = md["celltype"].astype(str).str.contains(ct_pat, regex=True)
        sub_idx = np.where(cell_mask.values)[0]
        if len(sub_idx) == 0 or gene not in g_index:
            out.append((concept, gene, ct_pat, 0, 0, 0, 0, 0, np.nan, np.nan))
            continue
        gi = g_index[gene]
        d = md.iloc[sub_idx].copy()
        expr = np.asarray(Xn[sub_idx][:, gi].todense()).ravel()
        binv = (expr > 0).astype(int)
        d["expr"] = expr
        d["bin"] = binv
        # per-dataset detection
        dd = d.groupby("dataset", observed=True).agg(
            n=("bin", "size"), frac=("bin", "mean"), mean=("expr", "mean"))
        dd = dd[dd["n"] >= MIN_CELLS]
        n_ds = len(dd)
        n_ds_det = int((dd["frac"] >= DET_FRAC).sum())
        # per-donor detection
        do = d.groupby("donor", observed=True).agg(
            n=("bin", "size"), frac=("bin", "mean"))
        do = do[do["n"] >= 10]
        n_dn = len(do)
        n_dn_det = int((do["frac"] >= DET_FRAC).sum())
        out.append((concept, gene, ct_pat, len(sub_idx),
                    n_ds, n_ds_det, n_dn, n_dn_det,
                    float(dd["frac"].median()) if n_ds else np.nan,
                    float(dd["frac"].std()) if n_ds else np.nan))
    return pd.DataFrame(out, columns=["concept","gene","celltype_pat","n_cells",
        "n_datasets","n_datasets_detected","n_donors","n_donors_detected",
        "median_dataset_frac","sd_dataset_frac"])

def dotplot(df, pairs, title, stem, note=""):
    """pairs: list of (row_label, celltype_exact, gene). x = segment.

    Right-hand column is split into a colourbar (top) and a dedicated size-legend
    axis (bottom) so the '% cells expressing' key never overlaps the data.
    """
    labels = [p[0] for p in pairs]
    n = len(pairs)
    width_mm = 120
    height_mm = min(170, 24 + 5.0 * n)
    fig = plt.figure(figsize=(width_mm * gps.MM, height_mm * gps.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.24], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    gsr = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.9)
    cax = fig.add_subplot(gsr[0, 0])
    lax = fig.add_subplot(gsr[1, 0]); lax.axis("off")

    xmap = {s: i for i, s in enumerate(SEG)}
    pts, vmax = [], 0.0
    for r, (lab, ct, g) in enumerate(pairs):
        sub = df[(df["celltype"] == ct) & (df["gene"] == g)]
        for _, row in sub.iterrows():
            xi = xmap.get(str(row["segment"]))
            if xi is None:
                continue
            pts.append((xi, n - 1 - r, row["mean_lognorm"], row["frac_expr"]))
            vmax = max(vmax, row["mean_lognorm"])
    vmax = vmax or 1.0
    SMAX = 130
    for xi, yi, m, f in pts:
        ax.scatter(xi, yi, s=6 + SMAX * f, c=[m], cmap=CMAP, vmin=0, vmax=vmax,
                   edgecolors="none")
    ax.set_xticks(range(len(SEG)))
    ax.set_xticklabels(SEG, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1])
    ax.set_xlim(-0.5, len(SEG) - 0.5); ax.set_ylim(-0.6, n - 0.4)
    ax.set_title(title)
    gps.open_axes(ax)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax); cb.set_label("Mean log-norm expr")
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(width=0.5)

    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    lax.text(0.0, 1.06, "% cells expressing", fontweight="bold", fontsize=6)
    for i, f in enumerate([0.1, 0.5, 0.9]):
        yy = 0.78 - i * 0.30
        lax.scatter(0.16, yy, s=6 + SMAX * f, c="#555555", edgecolors="none")
        lax.text(0.45, yy, f"{int(f*100)}%", va="center", fontsize=6)
    if note:
        fig.text(0.01, 0.005, note, fontsize=5, color="#666666")
    gps.save(fig, stem)
    print("wrote", stem)

# ------------------------------------------------------------------ main
Xn, md, md_genes = load_matrix()
df = summarise(Xn, md, md_genes)
df.to_csv(f"{OUT}/expr_by_segment_celltype.csv", index=False)
print("celltypes present:", df['celltype'].nunique())

# ---- concept dotplots (role, exact celltype, gene) ----
C1 = [
  ("EC : TPH1 (5-HT synth)", "EEC Enterochromaffin (EC)", "TPH1"),
  ("EEC L : PYY", "EEC L", "PYY"),
  ("EEC L : GCG", "EEC L", "GCG"),
  ("EEC N : NTS", "EEC N", "NTS"),
  ("EEC S : SCT", "EEC S", "SCT"),
  ("ISC : HTR4 (5-HT4 rec)", "Intestinal Stem Cells (ISC)", "HTR4"),
  ("TA : HTR4", "Transiently Amplifying Cells (TA)", "HTR4"),
  ("Homeostatic Mac : HTR7", "Homeostatic Macrophages", "HTR7"),
  ("BEST4 Ent : GUCY2C", "BEST4 Enterocytes", "GUCY2C"),
  ("EC : GUCA2B", "EEC Enterochromaffin (EC)", "GUCA2B"),
  ("Contractile Peri : NPY1R", "Contractile Pericytes", "NPY1R"),
  ("Tuft : VIPR1", "Tuft Cells", "VIPR1"),
]
C2 = [
  ("EEC EC : PCSK1N (proSAAS)", "EEC Enterochromaffin (EC)", "PCSK1N"),
  ("EEC L : PCSK1N", "EEC L", "PCSK1N"),
  ("EEC N : PCSK1N", "EEC N", "PCSK1N"),
  ("CD8 IEL : GPR171", "CD8 IEL", "GPR171"),
  ("CD8 TRM : GPR171", "CD8 TRM", "GPR171"),
  ("CD8 Eff Mem : GPR171", "CD8 Effector Memory", "GPR171"),
]
C3 = [
  ("Mast : TPSAB1", "Mast Cells", "TPSAB1"),
  ("Mast : TPSB2", "Mast Cells", "TPSB2"),
  ("Mature Goblet : F2RL1 (PAR2)", "Mature Goblet Cells", "F2RL1"),
  ("Goblet : F2RL1", "Goblet Cells", "F2RL1"),
  ("BEST4 Ent : F2RL1", "BEST4 Enterocytes", "F2RL1"),
  ("Paneth : F2RL1", "Paneth Cells", "F2RL1"),
]
C4 = [
  ("Glia : NRXN1", "Glia", "NRXN1"),
  ("Glia : GDNF", "Glia", "GDNF"),
  ("Glia : NRG1", "Glia", "NRG1"),
  ("Glia : PLP1 (identity)", "Glia", "PLP1"),
  ("SMC : NLGN1", "Smooth Muscle Cells (SMC)", "NLGN1"),
  ("Myofibroblast : NLGN2", "Myofibroblasts", "NLGN2"),
]
PILLAR = [
  ("Venular Endo : ACKR1", "Venular Endothelial", "ACKR1"),
  ("Med. Sinus Endo : ACKR1", "Medullary Sinus Endothelial", "ACKR1"),
  ("PVC : ACKR1", "Pre Venule Capillary Endothelial (PVC)", "ACKR1"),
  ("Med. Sinus Endo : CCL21", "Medullary Sinus Endothelial", "CCL21"),
  ("Lymphatic Endo : CCL21", "Lymphatic Endothelial", "CCL21"),
  ("CD8 TRM : CCL5", "CD8 TRM", "CCL5"),
  ("Homeostatic Mac : CCL2", "Homeostatic Macrophages", "CCL2"),
  ("Neutrophils : CXCL8", "Neutrophils", "CXCL8"),
]
dotplot(df, C1, "Concept 1  EEC peptidergic command layer", f"{OUT}/fig_concept1_expression")
dotplot(df, C2, "Concept 2  EEC proSAAS/BigLEN -> CD8 GPR171 checkpoint", f"{OUT}/fig_concept2_expression")
dotplot(df, C3, "Concept 3  Mast tryptase -> epithelial PAR2 (F2RL1)", f"{OUT}/fig_concept3_expression")
dotplot(df, C4, "Concept 4  Enteric glia hub (NRXN1/GDNF/NRG1)", f"{OUT}/fig_concept4_expression")
dotplot(df, PILLAR, "Pillar  Venular endothelial chemokine sink (ACKR1)", f"{OUT}/fig_pillar_chemokine_sink")

C6 = [
  ("Glia : PDGFA", "Glia", "PDGFA"),
  ("Crypt Bottom Fib (S2A) : PDGFRA", "Crypt Bottom Fibroblasts (S2A)", "PDGFRA"),
  ("Submucosal Fib (S3) : MFGE8", "Submucosal Fibroblasts (S3)", "MFGE8"),
  ("Angiogenic Peri : PDGFRB", "Angiogenic Pericytes", "PDGFRB"),
  ("Arteriolar Endo : JAG1", "Arteriolar Endothelial", "JAG1"),
  ("Arteriolar Endo : DLL4", "Arteriolar Endothelial", "DLL4"),
  ("Contractile Peri : NOTCH3", "Contractile Pericytes", "NOTCH3"),
  ("Crypt Top Fib (S2B) : WNT5A", "Crypt Top Fibroblasts (S2B)", "WNT5A"),
  ("Contractile Peri : MCAM", "Contractile Pericytes", "MCAM"),
  ("Venular Endo : VEGFC", "Venular Endothelial", "VEGFC"),
  ("Lymphatic Endo : LYVE1", "Lymphatic Endothelial", "LYVE1"),
]
C7 = [
  ("Lamina propria Fib (S1) : CXCL1", "Lamina propria Fibroblasts (S1)", "CXCL1"),
  ("FRC : CXCL1", "Fibroblastic Reticular Cells (FRC)", "CXCL1"),
  ("Lamina propria Fib (S1) : CXCL2", "Lamina propria Fibroblasts (S1)", "CXCL2"),
  ("Homeostatic Mac : CXCL8", "Homeostatic Macrophages", "CXCL8"),
  ("Neutrophils : CXCR2", "Neutrophils", "CXCR2"),
  ("Neutrophils : CXCR1", "Neutrophils", "CXCR1"),
  ("Venular Endo : ACKR1 (sink)", "Venular Endothelial", "ACKR1"),
  ("PVC : ACKR1 (sink)", "Pre Venule Capillary Endothelial (PVC)", "ACKR1"),
]
dotplot(df, C6, "Concept 6  Perivascular gut-wall wiring (fibrosis / neovascular)", f"{OUT}/fig_concept6_expression")
dotplot(df, C7, "Concept 7  Chemokine recruitment amplifier vs ACKR1 sink", f"{OUT}/fig_concept7_expression")

# ---- HEV check within endothelial subsets ----
endo = ["Venular Endothelial","Medullary Sinus Endothelial",
        "Pre Venule Capillary Endothelial (PVC)",
        "Post Arteriole Capillary Endothelial (PAC)",
        "Capillary Endothelial","Arteriolar Endothelial","Lymphatic Endothelial"]
hev_genes = ["ACKR1","MADCAM1","CCL21","CCL19","CHST4","CHST2","B3GNT3","FUT7",
             "GLCE","SELE","SELP","PECAM1"]
hev = df[df["celltype"].isin(endo) & df["gene"].isin(hev_genes)].copy()
hev.to_csv(f"{OUT}/hev_endothelial.csv", index=False)
# dotplot: rows = endo subset, cols = gene, colour=mean(ileum+colon avg), here use per-gene across segments avg
hh = hev.groupby(["celltype","gene"], observed=True).agg(
    mean_lognorm=("mean_lognorm","mean"), frac_expr=("frac_expr","mean")).reset_index()
fig = plt.figure(figsize=(150 * gps.MM, 66 * gps.MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.2], wspace=0.05)
ax = fig.add_subplot(gs[0, 0])
gsr = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=1.0)
cax = fig.add_subplot(gsr[0, 0]); lax = fig.add_subplot(gsr[1, 0]); lax.axis("off")
gx = {g:i for i,g in enumerate(hev_genes)}
cy = {c:i for i,c in enumerate(endo)}
vmax = hh["mean_lognorm"].max() or 1
SMAX = 130
for _, r in hh.iterrows():
    ax.scatter(gx[r["gene"]], len(endo)-1-cy[r["celltype"]],
               s=6+SMAX*r["frac_expr"], c=[r["mean_lognorm"]], cmap=CMAP,
               vmin=0, vmax=vmax, edgecolors="none")
ax.set_xticks(range(len(hev_genes))); ax.set_xticklabels(hev_genes, rotation=45, ha="right")
ax.set_yticks(range(len(endo))); ax.set_yticklabels(endo[::-1])
ax.set_xlim(-0.5, len(hev_genes)-0.5); ax.set_ylim(-0.6, len(endo)-0.4)
ax.set_title("HEV / venular markers across endothelial subsets (mean over segments)")
gps.open_axes(ax)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
cb = fig.colorbar(sm, cax=cax); cb.set_label("Mean log-norm expr")
cb.outline.set_linewidth(0.5); cb.ax.tick_params(width=0.5)
lax.set_xlim(0,1); lax.set_ylim(0,1)
lax.text(0.0, 1.06, "% cells expressing", fontweight="bold", fontsize=6)
for i, f in enumerate([0.1,0.5,0.9]):
    yy = 0.78 - i*0.30
    lax.scatter(0.16, yy, s=6+SMAX*f, c="#555555", edgecolors="none")
    lax.text(0.5, yy, f"{int(f*100)}%", va="center", fontsize=6)
gps.save(fig, f"{OUT}/fig_hev_check"); print("wrote HEV check")

# ---- believability / metadata-breadth ----
checks = [
  ("C1_serotonin","TPH1","EEC Enterochromaffin"),
  ("C1_serotonin","HTR4","Intestinal Stem Cells|Transiently Amplifying"),
  ("C1_guanylin","GUCA2B","EEC|Enterocyte|BEST4"),
  ("C1_guanylin","GUCY2C","BEST4 Enterocytes"),
  ("C2_checkpoint","PCSK1N","EEC"),
  ("C2_checkpoint","GPR171","CD8"),
  ("C3_tryptase","TPSAB1","Mast Cells"),
  ("C3_tryptase","F2RL1","Goblet|Enterocyte|BEST4"),
  ("C4_glia","NRXN1","Glia"),
  ("C4_glia","NLGN1","Smooth Muscle|Myofibroblast"),
  ("pillar_sink","ACKR1","Venular Endothelial|Medullary Sinus"),
  ("pillar_sink","CCL21","Medullary Sinus|Lymphatic"),
  ("hev","MADCAM1","Venular Endothelial|Medullary Sinus"),
  ("C6_perivascular","PDGFA","Glia|Fibroblast"),
  ("C6_perivascular","PDGFRA","Fibroblast"),
  ("C6_perivascular","NOTCH3","Pericyte|Smooth Muscle"),
  ("C6_perivascular","VEGFC","Venular Endothelial"),
  ("C7_chemokine","CXCL1","Fibroblast|Reticular"),
  ("C7_chemokine","CXCR2","Neutrophil"),
]
bel = believability(Xn, md, checks)
bel.to_csv(f"{OUT}/believability_by_dataset.csv", index=False)
print("\n=== BELIEVABILITY (dataset/donor breadth) ===")
print(bel.to_string(index=False))
print("\nDONE ->", OUT)
