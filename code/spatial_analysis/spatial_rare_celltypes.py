"""
Spatial validation of rare cell types in the Teichmann adult gut Visium dataset.

Goal: take rare-cell-type marker modules used in the integrated HCA gut atlas
notebooks (see vignettes/RareCellTypes.ipynb, BEST4_heterogeneity.ipynb,
LIANA_analysis.ipynb) and check whether they are present and spatially organized
in 4 healthy adult gut biopsies profiled with 10x Visium:

  GUTsp9518706 -> A38-REC  (rectum)
  GUTsp9518707 -> A50_DUO  (duodenum)         ** focus for neuropod/EEC-glia-neuron
  GUTsp9518708 -> A50-SCL  (sigmoid colon)
  GUTsp9518709 -> A39-MLN  (mesenteric lymph node)

The script does NOT generate a notebook. It runs end-to-end from the command
line, caches per-sample AnnData, and writes PDF/PNG figures to:

  /Users/kylekimler/Projects/GCA/spatial/figs_rare_celltypes/

Run:
  /Users/kylekimler/miniforge3/envs/scanpy/bin/python code/spatial_analysis/spatial_rare_celltypes.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths and sample metadata
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/Users/kylekimler/Projects/GCA/spatial/extracted")
FIG_DIR = Path("/Users/kylekimler/Projects/GCA/spatial/figs_rare_celltypes")
CACHE_DIR = FIG_DIR / "_cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# capture area -> (biopsy_id, anatomical segment, donor)
SAMPLES: Dict[str, Dict[str, str]] = {
    "GUTsp9518706": dict(label="A38-REC", segment="rectum",        donor="A38"),
    "GUTsp9518707": dict(label="A50_DUO", segment="duodenum",      donor="A50"),
    "GUTsp9518708": dict(label="A50-SCL", segment="sigmoid_colon", donor="A50"),
    "GUTsp9518709": dict(label="A39-MLN", segment="mesenteric_LN", donor="A39"),
}

# scanpy figure defaults
sc.settings.set_figure_params(dpi=110, dpi_save=200, facecolor="white", frameon=False)
sc.settings.verbosity = 1
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif"})

# ---------------------------------------------------------------------------
# Marker dictionaries (compiled from RareCellTypes / BEST4 / LIANA vignettes)
# ---------------------------------------------------------------------------
TISSUE_LANDMARKS: Dict[str, List[str]] = {
    "Epithelium":  ["EPCAM", "VIL1", "KRT8", "KRT18"],
    "Stroma":      ["COL1A1", "COL3A1", "DCN", "LUM"],
    "Immune":      ["PTPRC", "CD3D", "CD79A", "MS4A1"],
    "SmoothMuscle":["ACTA2", "MYH11", "DES"],
}

# rare epithelial sensory / endocrine / BEST4 modules
EPI_RARE: Dict[str, List[str]] = {
    "EEC_general":     ["CHGA", "CHGB", "SCG2", "NEUROD1", "NEUROG3", "PCSK1", "PCSK1N"],
    "EEC_I_CCK_prox":  ["CCK"],                                      # I-cells, proximal/duo
    "EEC_L_GLP1":      ["GCG", "PYY", "GLP1R"],                      # L-cells, distal
    "EEC_EC_TPH1":     ["TPH1", "DDC", "SLC18A1", "CHGA"],           # enterochromaffin
    "EEC_N_NTS":       ["NTS"],
    "EEC_S_SCT":       ["SCT"],
    "Tuft":            ["POU2F3", "TRPM5", "GFI1B", "LRMP", "AVIL", "RGS13", "DCLK1"],
    "BEST4_core":      ["BEST4", "OTOP2", "CA7", "CFTR", "GUCA2A", "GUCA2B", "GUCY2C"],
    "BEST4_ileum_pgm": ["PDLIM4", "BRINP3", "PRKG1"],                # ileum-specific BEST4
    # segment-specific BEST4 marker panels (literature / atlas-derived)
    "BEST4_duo":       ["CFTR", "FOLH1", "ONECUT2", "CACNA2D1", "SMIM24",
                        "CPA2", "ADGRG4", "LYZ"],
    "BEST4_jej":       ["ANXA13", "CLDN12", "CCL25"],
    "BEST4_ile":       ["ALDOB", "ALDH1A1", "SI", "SULT1E1", "FAM3B",
                        "DPEP1", "MALRD1"],
    "BEST4_col":       ["CKB", "CEACAM5", "VSIG2", "C10orf99", "CA2",
                        "HMGCS2", "FABP5", "MUC12"],
}

# enteric glia subtypes (from RareCellTypes endo+glia panels)
GLIA_PANELS: Dict[str, List[str]] = {
    "Glia_pan":           ["SOX10", "PLP1", "S100B", "GFAP"],
    "Glia_progenitor":    ["SOX2", "NES", "VIM", "S100A1", "FABP7"],
    "Glia_mucosal_T3":    ["RELN", "MBP", "PLLP"],
    "Glia_muscularis_T4": ["SLC5A7", "APOD", "KCNS3"],
}

# enteric neuron module
NEURON_PANELS: Dict[str, List[str]] = {
    "Neuron_pan":     ["ELAVL3", "ELAVL4", "RBFOX3", "UCHL1", "TUBB3", "STMN2", "GAP43"],
    "Neurofilament":  ["NEFL", "NEFM", "NEFH", "PRPH", "INA"],
    "Neuron_IPAN":    ["CALCB", "NMU"],
    "Neuron_cholin":  ["CHAT", "SLC18A3"],
    "Neuron_nitrerg": ["NOS1", "VIP"],
}

# Neuropod adhesion / GDNF-RET / synaptic module (from LIANA notebook 2.2)
NEUROPOD_PANELS: Dict[str, List[str]] = {
    "Neuropod_adhesion": ["NCAM1", "NCAM2", "L1CAM", "CHL1", "CADM1", "CADM2", "CADM3",
                          "DSCAM", "OPCML", "PTPRF", "NRXN1", "NRXN3", "NLGN1", "CNTN1"],
    "Neuropod_GDNF_RET": ["GDNF", "NRTN", "ARTN", "PSPN", "RET",
                          "GFRA1", "GFRA2", "GFRA3", "GFRA4"],
    "Neuropod_synaptic": ["SYN1", "PCLO", "BSN", "UNC13B", "RIMS2", "SYT3",
                          "HOMER3", "DLG4"],
    # BEST4 neuropod-specific: cholinergic + nuclear TF + neurofilament + axon guidance.
    # MEIS1 and ROBO1 are flagged in the literature as BEST4-neuropod-enriched;
    # CHAT (cholinergic) is present in a BEST4 subset; neurofilaments mark
    # the ECM-like cytoskeletal program found in epithelial neuropods.
    "Neuropod_BEST4":   ["CHAT", "SLC18A3", "MEIS1", "MEIS2", "ROBO1", "ROBO2",
                         "NEFL", "NEFM", "NEFH", "PRPH", "INA",
                         "NCAM1", "GFRA1", "RET"],
}

# Macrophage subtypes (from gca_celltype_taxonomy.csv)
MACROPHAGE_PANELS: Dict[str, List[str]] = {
    "Mac_pan":         ["CD68", "AIF1", "CSF1R", "C1QA", "C1QB", "C1QC"],
    "Mac_general":     ["NRG1", "KLF4", "GPR183", "C1QB", "CCL3L1", "MS4A6A",
                        "MS4A7", "CXCL2"],
    "Mac_tissue_res":  ["C1QA", "C1QB", "C1QC", "SELENOP", "CD209", "FOLR2",
                        "CD163L1", "LYVE1", "MAF", "MAFB"],
    "Mac_M2":          ["CD209", "CD163L1", "FOLR2", "MRC1", "CD163"],
    "Mac_M1":          ["TNF", "CCL2", "CCL3", "IL1B", "IL6", "PTGS2"],
    "Mac_cycling":     ["MKI67", "TOP2A", "STMN1"],
    "Monocyte":        ["CD14", "FCN1", "LYST", "AREG", "PLAUR", "S100A8", "S100A9"],
}

# Congenital diarrhea-associated genes
# SLC26A3 = DRA = Congenital chloride diarrhea
# SLC9A3  = NHE3 = Congenital sodium diarrhea
# MYO5B / STX3 = Microvillus inclusion disease (MVID)
# DGAT1 = Familial diarrhea (lipid)
# NEUROG3 = Anendocrine enteric dysgenesis (EEC loss)
# GUCY2C = familial diarrhea gain-of-function
# EPCAM = Tufting enteropathy
# TTC7A = MIA-combined immunodeficiency / atresia
# HNF4A = MODY + congenital diarrhea
# SI    = Congenital sucrase-isomaltase deficiency
# LCT   = Congenital lactase deficiency
# KCNQ1 / KCNE3 / CFTR = secretory diarrhea machinery
CONGENITAL_DIARRHEA: Dict[str, List[str]] = {
    "CCD_DRA":         ["SLC26A3"],                              # chloride diarrhea
    "CSD_NHE3":        ["SLC9A3"],                               # sodium diarrhea
    "MVID":            ["MYO5B", "STX3", "RAB11A", "SLC15A1"],   # microvillus inclusion
    "Tufting_EPCAM":   ["EPCAM", "SPINT2"],
    "FamDiarrhea_GCC": ["GUCY2C", "GUCA2A", "GUCA2B"],
    "AnEEC_NEUROG3":   ["NEUROG3", "NEUROD1", "PAX4"],
    "Lipid_DGAT":      ["DGAT1", "DGAT2"],
    "Disacc_SI":       ["SI"],
    "Disacc_LCT":      ["LCT"],
    "MIA_TTC7A":       ["TTC7A"],
    "Diabetes_HNF":    ["HNF4A", "HNF1A"],
    "Secretory_cAMP":  ["CFTR", "ADCY6", "KCNQ1", "KCNE3", "GNAS", "PRKACA"],
}

# DALY / infectious diarrhea pathway signatures (from Diarrhea_Mechanisms notebook)
DIARRHEA_PATHWAYS: Dict[str, List[str]] = {
    "Sprue":           ["IL15", "IL15RA", "CXCL10", "CXCL11"],          # celiac-like
    "Cholera":         ["CFTR", "ADCY6", "KCNQ1", "KCNE3", "GNAS", "PRKACA"],
    "Dysentery":       ["CXCL8", "TLR4", "NOD1", "NOD2", "MUC2"],
    "Rotavirus":       ["TLR3", "DDX58", "IFIH1", "MAVS", "STAT1", "ISG15",
                        "MX1", "OAS1", "IFIT1", "IFNL1", "IFNL2", "IFNL3"],
    "Norovirus":       ["FUT2", "FUT3", "ABO", "TLR3", "STAT1", "OAS1",
                        "IFIT1", "MX1"],
    "ETEC_secretory":  ["GUCY2C", "GUCA2A", "GUCA2B", "SLC26A3", "SLC9A3"],
    "Crypto_IFN":      ["STAT1", "ISG15", "MX1", "OAS1", "IFIT1"],
    "Cdiff_chloride":  ["SLC26A3"],
}

# endothelial vasculature subtypes (sinus-relevant where noted)
ENDO_PANELS: Dict[str, List[str]] = {
    "Endo_pan":      ["PECAM1", "CDH5", "CLDN5", "VWF"],
    "Endo_arteriolar":["SEMA3G", "MGP", "IGFBP4", "GJA5", "HEY1"],
    "Endo_capillary":["CD36", "RAMP2", "IGFBP7"],
    "Endo_PAC":      ["RGCC", "FCN3"],
    "Endo_PVC":      ["ICAM1", "LITAF"],
    "Endo_venular":  ["ACKR1", "MADCAM1", "SELP"],     # MADCAM1 = gut HEV-like
    "Endo_lymphatic":["PROX1", "PDPN", "LYVE1"],
    "Endo_lacteals": ["FABP1", "PLVAP", "LYVE1"],
    "LEC_sinus":     ["STAB2", "LYVE1", "MARCO"],      # LN sinus-specific endothelium
    "LEC_med_sinus": ["MRC1", "STAB2"],
    "LEC_cort_sinus":["ACKR4", "STAB1", "CCRL2"],
    "HEV":           ["CD34", "ICAM1", "MADCAM1"],
}

# Group of all module dicts that we score with score_genes
ALL_MODULES = {}
for d in (EPI_RARE, GLIA_PANELS, NEURON_PANELS, NEUROPOD_PANELS, ENDO_PANELS,
          MACROPHAGE_PANELS, CONGENITAL_DIARRHEA, DIARRHEA_PATHWAYS):
    ALL_MODULES.update(d)

# Cell-type "presence" scores -- one composite per major rare type used for
# spatial co-occurrence work.
PRESENCE_GENES: Dict[str, List[str]] = {
    "EEC":         EPI_RARE["EEC_general"] + ["CCK", "GCG", "PYY", "TPH1", "NTS", "SCT"],
    "Tuft":        EPI_RARE["Tuft"],
    "BEST4":       ["BEST4", "OTOP2", "CA7"],
    "Glia":        GLIA_PANELS["Glia_pan"],
    "Neuron":      NEURON_PANELS["Neuron_pan"] + NEURON_PANELS["Neurofilament"],
    "Neuropod":    NEUROPOD_PANELS["Neuropod_adhesion"] + NEUROPOD_PANELS["Neuropod_GDNF_RET"],
    "Neuropod_B4": NEUROPOD_PANELS["Neuropod_BEST4"],
    "LEC_sinus":   ENDO_PANELS["LEC_sinus"] + ENDO_PANELS["LEC_med_sinus"],
    "Mac_TR":      MACROPHAGE_PANELS["Mac_tissue_res"],
    "Mac_M2":      MACROPHAGE_PANELS["Mac_M2"],
    "Mac_M1":      MACROPHAGE_PANELS["Mac_M1"],
    "Mac_pan":     MACROPHAGE_PANELS["Mac_pan"],
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_visium_mtx(sample_dir: Path) -> ad.AnnData:
    """Build a Visium-style AnnData from a 'filtered_feature_bc_matrix' folder
    and a 'spatial' folder (Spaceranger 3.x layout, no .h5)."""
    import json as _json
    from matplotlib.image import imread as _imread

    adata = sc.read_10x_mtx(sample_dir / "filtered_feature_bc_matrix",
                            var_names="gene_symbols", make_unique=True)
    sp_dir = sample_dir / "spatial"
    # tissue_positions.csv has a header in spaceranger >=2.0:
    # barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
    tp = pd.read_csv(sp_dir / "tissue_positions.csv")
    tp = tp.set_index("barcode")
    keep = tp.index.intersection(adata.obs_names)
    adata = adata[keep].copy()
    tp = tp.loc[keep]
    # only keep in-tissue spots
    in_tis = tp["in_tissue"].astype(int) == 1
    adata = adata[in_tis.values].copy()
    tp = tp.loc[in_tis]

    adata.obs["array_row"] = tp["array_row"].astype(int).values
    adata.obs["array_col"] = tp["array_col"].astype(int).values
    # obsm['spatial'] uses (x, y) = (pxl_col, pxl_row) at fullres
    adata.obsm["spatial"] = np.column_stack([
        tp["pxl_col_in_fullres"].astype(float).values,
        tp["pxl_row_in_fullres"].astype(float).values,
    ])

    # scalefactors + image
    with open(sp_dir / "scalefactors_json.json") as fh:
        sf = _json.load(fh)
    images = {}
    hires_p = sp_dir / "tissue_hires_image.png"
    lores_p = sp_dir / "tissue_lowres_image.png"
    if hires_p.exists():
        images["hires"] = _imread(str(hires_p))
    if lores_p.exists():
        images["lowres"] = _imread(str(lores_p))

    library_id = sample_dir.name
    adata.uns["spatial"] = {
        library_id: {
            "images": images,
            "scalefactors": sf,
            "metadata": {"chemistry_description": "Visium", "software_version": "spaceranger"},
        }
    }
    return adata


def load_sample(sid: str) -> ad.AnnData:
    """Read Visium directory, normalize, score modules. Cached as .h5ad."""
    cache = CACHE_DIR / f"{sid}.h5ad"
    if cache.exists():
        adata = sc.read_h5ad(cache)
        return adata

    sample_dir = DATA_ROOT / sid
    print(f"[load] {sid} <- {sample_dir}", flush=True)

    adata = _read_visium_mtx(sample_dir)
    adata.var_names_make_unique()

    # metadata
    md = SAMPLES[sid]
    adata.obs["sample"] = sid
    adata.obs["label"] = md["label"]
    adata.obs["segment"] = md["segment"]
    adata.obs["donor"] = md["donor"]

    # mitochondrial QC
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)

    # filter very low-quality spots
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=200)
    print(f"  spots: {n_before} -> {adata.n_obs} after min_counts=200", flush=True)

    # save raw counts in a layer for safekeeping
    adata.layers["counts"] = adata.X.copy()

    # normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # score every module that has at least 1 gene present in the dataset
    var_set = set(map(str.upper, adata.var_names))
    for name, genes in ALL_MODULES.items():
        present = [g for g in genes if g.upper() in var_set]
        if not present:
            adata.obs[f"score_{name}"] = 0.0
            continue
        try:
            sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}",
                              random_state=0, use_raw=False)
        except Exception as e:
            print(f"  [warn] {name}: {e}", flush=True)
            adata.obs[f"score_{name}"] = 0.0

    # composite "presence" scores for the major rare cell types
    for name, genes in PRESENCE_GENES.items():
        present = [g for g in genes if g.upper() in var_set]
        if not present:
            adata.obs[f"presence_{name}"] = 0.0
            continue
        sc.tl.score_genes(adata, gene_list=present, score_name=f"presence_{name}",
                          random_state=0, use_raw=False)

    # marker-positive call: require AT LEAST ONE canonical gene to have raw
    # count >= 1 in the spot AND the module score to be above the 75th
    # percentile (within sample). This avoids the trap where every spot has
    # score ~= 0 and the 90th percentile is itself 0.
    raw = adata.layers["counts"]
    if issparse(raw):
        raw = raw.tocsc()

    def _any_gene_pos(gene_list):
        present = [g for g in gene_list if g in adata.var_names]
        if not present:
            return np.zeros(adata.n_obs, dtype=bool)
        idx = [adata.var_names.get_loc(g) for g in present]
        if issparse(raw):
            sub = raw[:, idx]
            ge = (sub > 0).sum(axis=1)
            return np.asarray(ge).flatten() >= 1
        return (raw[:, idx] > 0).sum(axis=1) >= 1

    # gene sets used for the strict raw-count positive call
    POS_GENES_STRICT = {
        "EEC":         ["CHGA", "CHGB", "CCK", "GCG", "TPH1", "NTS", "SCT", "PYY",
                        "GHRL", "MLN", "NEUROD1", "NEUROG3"],
        "Tuft":        ["POU2F3", "TRPM5", "AVIL", "DCLK1"],
        "BEST4":       ["BEST4"],
        "Glia":        ["SOX10", "PLP1", "S100B"],
        "Neuron":      ["ELAVL3", "ELAVL4", "UCHL1", "STMN2", "GAP43",
                        "NEFL", "NEFM", "NEFH"],
        "Neuropod":    ["NCAM1", "L1CAM", "GDNF", "RET", "GFRA1"],
        "Neuropod_B4": ["CHAT", "MEIS1", "ROBO1", "NEFL", "NEFM"],
        "LEC_sinus":   ["STAB2", "LYVE1"],
        "Mac_TR":      ["C1QA", "C1QB", "FOLR2", "CD163L1", "LYVE1"],
        "Mac_M2":      ["CD163", "FOLR2", "MRC1"],
        "Mac_M1":      ["IL1B", "TNF", "CCL3", "CCL2"],
        "Mac_pan":     ["CD68", "AIF1", "CSF1R"],
    }
    for name, glist in POS_GENES_STRICT.items():
        raw_hit = _any_gene_pos(glist)
        s = adata.obs[f"presence_{name}"].astype(float).values
        thr = np.quantile(s, 0.75)
        pos = (raw_hit & (s > thr)).astype(int)
        # fall back to raw_hit alone if quantile filter wipes everything
        if pos.sum() < 5 and raw_hit.sum() >= 5:
            pos = raw_hit.astype(int)
        adata.obs[f"pos_{name}"] = pos

    # tissue landmark scores
    for name, genes in TISSUE_LANDMARKS.items():
        present = [g for g in genes if g.upper() in var_set]
        if present:
            sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}",
                              random_state=0, use_raw=False)
        else:
            adata.obs[f"score_{name}"] = 0.0

    adata.write_h5ad(cache)
    return adata


def load_all() -> Dict[str, ad.AnnData]:
    return {sid: load_sample(sid) for sid in SAMPLES}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def spatial_panel(adata: ad.AnnData, keys: List[str], outpath: Path,
                  ncols: int = 4, size_factor: float = 1.3, cmap: str = "magma",
                  title_prefix: str = "", vmax: str | float | None = "p99",
                  img_key: str = "lowres"):
    """Plot spatial maps; uses the lowres tissue image to keep PDF small.

    Pass img_key='hires' for higher-resolution H&E backgrounds (larger PDFs).
    """
    keys = [k for k in keys if k in adata.var_names or k in adata.obs.columns]
    if not keys:
        return
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.6))
    axes = np.atleast_2d(axes).flatten()
    for i, k in enumerate(keys):
        ax = axes[i]
        sc.pl.spatial(adata, color=k, show=False, ax=ax, size=size_factor,
                      cmap=cmap, vmax=vmax, colorbar_loc=None, frameon=False,
                      img_key=img_key, title=f"{title_prefix}{k}")
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}", flush=True)


def stacked_segment_panel(samples: Dict[str, ad.AnnData], keys: List[str],
                          outpath: Path, size_factor: float = 1.3,
                          cmap: str = "magma", vmax: str | float | None = "p99",
                          img_key: str = "lowres"):
    """Rows = samples, cols = keys."""
    n_rows = len(samples)
    n_cols = len(keys)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.6, n_rows * 3.6),
                             squeeze=False)
    for r, (sid, adata) in enumerate(samples.items()):
        for c, k in enumerate(keys):
            ax = axes[r, c]
            if k not in adata.var_names and k not in adata.obs.columns:
                ax.set_visible(False)
                continue
            sc.pl.spatial(adata, color=k, show=False, ax=ax, size=size_factor,
                          cmap=cmap, vmax=vmax, colorbar_loc=None, frameon=False,
                          img_key=img_key,
                          title=f"{SAMPLES[sid]['label']} / {k}")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath.name}", flush=True)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def fig_tissue_landmarks(samples):
    print("== tissue landmark QC ==", flush=True)
    keys = ["total_counts", "n_genes_by_counts",
            "score_Epithelium", "score_Stroma", "score_Immune", "score_SmoothMuscle"]
    stacked_segment_panel(samples, keys,
                          FIG_DIR / "01_tissue_landmarks_all_samples.png",
                          vmax="p99")


def fig_rare_celltype_scores(samples):
    print("== rare cell-type presence scores: all samples ==", flush=True)
    keys = ["presence_EEC", "presence_Tuft", "presence_BEST4",
            "presence_Glia", "presence_Neuron", "presence_Neuropod",
            "presence_LEC_sinus"]
    stacked_segment_panel(samples, keys,
                          FIG_DIR / "02_presence_scores_all_samples.png",
                          vmax="p99")


def fig_duo_neuropod(samples):
    print("== Duodenum: neuropod-focused panels ==", flush=True)
    duo = samples["GUTsp9518707"]
    epi_keys = ["score_EEC_general", "score_EEC_I_CCK_prox", "score_EEC_L_GLP1",
                "score_EEC_EC_TPH1", "score_EEC_N_NTS", "score_EEC_S_SCT",
                "score_Tuft", "score_BEST4_core"]
    spatial_panel(duo, epi_keys, FIG_DIR / "03_duo_epi_subtype_scores.png",
                  ncols=4, title_prefix="DUO ")

    npod_keys = ["score_Neuron_pan", "score_Neurofilament", "score_Glia_pan",
                 "score_Glia_progenitor", "score_Glia_mucosal_T3",
                 "score_Neuropod_adhesion", "score_Neuropod_GDNF_RET",
                 "score_Neuropod_synaptic"]
    spatial_panel(duo, npod_keys, FIG_DIR / "04_duo_glia_neuron_neuropod.png",
                  ncols=4, title_prefix="DUO ")

    # individual neuropod ligands/receptors
    gene_keys = [g for g in ["NCAM1", "L1CAM", "CADM1", "GDNF", "NRTN",
                             "RET", "GFRA1", "GFRA2", "NEFL", "NEFM", "NEFH",
                             "PLP1", "S100B", "SOX10", "ELAVL3", "UCHL1",
                             "CHGA", "CCK", "GCG", "TPH1", "POU2F3"]
                 if g in duo.var_names]
    spatial_panel(duo, gene_keys, FIG_DIR / "05_duo_neuropod_genes.png",
                  ncols=4, title_prefix="DUO ")


def fig_best4_across_segments(samples):
    print("== BEST4 heterogeneity across segments ==", flush=True)
    # exclude MLN -- not a gut epithelial segment
    gut = {sid: samples[sid] for sid in
           ["GUTsp9518707", "GUTsp9518708", "GUTsp9518706"]}  # DUO, SCL, REC

    # spatial maps of BEST4 module genes
    keys = ["BEST4", "OTOP2", "CA7", "CFTR", "GUCA2A", "GUCY2C"]
    stacked_segment_panel(gut, keys,
                          FIG_DIR / "06_best4_module_across_segments.png",
                          vmax="p99")

    # ileum-program markers across segments (expect low / null since no ileum)
    keys2 = ["PDLIM4", "BRINP3", "PRKG1", "score_BEST4_ileum_pgm"]
    stacked_segment_panel(gut, keys2,
                          FIG_DIR / "07_best4_ileum_pgm_across_segments.png",
                          vmax="p99")

    # per-segment BEST4+ summary: fraction CFTR-hi within BEST4+ spots
    rows = []
    for sid, adata in gut.items():
        s = adata.obs
        b4_pos = s["pos_BEST4"].values.astype(bool)
        if b4_pos.sum() == 0:
            continue
        # CFTR expression (gene symbol expected from features.tsv)
        if "CFTR" in adata.var_names:
            cftr = np.asarray(adata[:, "CFTR"].X.toarray() if issparse(adata.X)
                              else adata[:, "CFTR"].X).flatten()
        else:
            cftr = np.zeros(adata.n_obs)
        median_cftr_in_b4 = np.median(cftr[b4_pos])
        rows.append(dict(
            sample=sid,
            label=SAMPLES[sid]["label"],
            segment=SAMPLES[sid]["segment"],
            n_spots=int(adata.n_obs),
            n_BEST4_pos=int(b4_pos.sum()),
            frac_BEST4_pos=float(b4_pos.mean()),
            mean_BEST4_log1p=float(np.asarray(adata[:, "BEST4"].X.toarray()
                                              if "BEST4" in adata.var_names
                                              else np.zeros((adata.n_obs, 1))).mean()),
            median_CFTR_in_BEST4pos=float(median_cftr_in_b4),
            frac_CFTRhi_in_BEST4pos=float(np.mean(cftr[b4_pos] > median_cftr_in_b4)),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "08_best4_per_segment_summary.csv", index=False)
    print(df.to_string(index=False), flush=True)

    # bar plot
    if len(df):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        axes[0].bar(df["label"], df["frac_BEST4_pos"] * 100, color="#4477AA")
        axes[0].set_ylabel("% spots BEST4+")
        axes[0].set_title("BEST4+ spot fraction")
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[1].bar(df["label"], df["mean_BEST4_log1p"], color="#EE6677")
        axes[1].set_ylabel("mean BEST4 log1p")
        axes[1].set_title("BEST4 expression")
        axes[2].bar(df["label"], df["median_CFTR_in_BEST4pos"], color="#228833")
        axes[2].set_ylabel("median CFTR log1p in BEST4+ spots")
        axes[2].set_title("CFTR in BEST4+")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "08_best4_per_segment_bars.png", dpi=200)
        fig.savefig(FIG_DIR / "08_best4_per_segment_bars.pdf")
        plt.close(fig)


def fig_endothelial_subtypes(samples):
    print("== Endothelial subtype maps (including sinus-specific) ==", flush=True)
    # pan-endo
    keys_pan = ["PECAM1", "CDH5", "VWF", "score_Endo_pan"]
    stacked_segment_panel(samples, keys_pan,
                          FIG_DIR / "09_endo_pan.png", vmax="p99")

    # subtype scores
    keys_sub = ["score_Endo_arteriolar", "score_Endo_capillary",
                "score_Endo_venular", "score_Endo_lymphatic",
                "score_Endo_lacteals", "score_LEC_sinus",
                "score_LEC_med_sinus", "score_LEC_cort_sinus", "score_HEV"]
    stacked_segment_panel(samples, keys_sub,
                          FIG_DIR / "10_endo_subtype_scores.png", vmax="p99")

    # individual sinus markers
    sinus_genes = [g for g in ["STAB2", "LYVE1", "MARCO", "MRC1", "STAB1",
                               "ACKR4", "MADCAM1", "PLVAP", "PROX1", "PDPN",
                               "CCL21", "FOXC2"]
                   if any(g in s.var_names for s in samples.values())]
    if sinus_genes:
        stacked_segment_panel(samples, sinus_genes,
                              FIG_DIR / "11_endo_sinus_genes.png", vmax="p99")


def fig_neuropod_coexpression(samples):
    """For each sample compute spatial overlap statistics between EEC / Tuft /
    BEST4 / Glia / Neuron / Neuropod.

    1. Pearson r of per-spot scores  (linear association across the tissue)
    2. Fraction of EEC+ / Tuft+ / BEST4+ spots that are also Glia+ or Neuron+
    3. Distance from each EEC+ spot to the nearest Glia+ / Neuron+ spot vs a
       random null derived by shuffling Glia+ / Neuron+ assignments.
    """
    print("== Co-localization stats across samples ==", flush=True)
    score_keys = ["presence_EEC", "presence_Tuft", "presence_BEST4",
                  "presence_Glia", "presence_Neuron", "presence_Neuropod"]

    corr_rows = []
    coloc_rows = []
    dist_rows = []
    rng = np.random.default_rng(0)

    for sid, adata in samples.items():
        df = adata.obs[score_keys].astype(float)
        # pairwise pearson correlation
        for a in score_keys:
            for b in score_keys:
                if a >= b:
                    continue
                r, p = pearsonr(df[a].values, df[b].values)
                corr_rows.append(dict(sample=sid, label=SAMPLES[sid]["label"],
                                      a=a, b=b, pearson_r=r, pvalue=p,
                                      n_spots=adata.n_obs))

        # binary co-localization
        for primary in ["EEC", "Tuft", "BEST4"]:
            for partner in ["Glia", "Neuron", "Neuropod"]:
                p_mask = adata.obs[f"pos_{primary}"].values.astype(bool)
                q_mask = adata.obs[f"pos_{partner}"].values.astype(bool)
                if p_mask.sum() == 0:
                    continue
                obs_frac = float((p_mask & q_mask).sum() / max(1, p_mask.sum()))
                expected = float(q_mask.mean())
                fold = obs_frac / expected if expected > 0 else np.nan
                coloc_rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"],
                    primary=primary, partner=partner,
                    n_primary_pos=int(p_mask.sum()),
                    obs_double_pos_frac=obs_frac,
                    expected_frac=expected,
                    fold_over_expected=fold,
                ))

        # nearest-neighbor distance between EEC+ spots and Glia+/Neuron+/Neuropod+
        # use pixel coords from spatial slot
        sp_key = list(adata.uns["spatial"].keys())[0]
        coords = adata.obsm["spatial"]  # (n, 2) in pixels of fullres
        sf = adata.uns["spatial"][sp_key]["scalefactors"]
        # scaled to micrometers if spot diameter known (~ 55um)
        spot_um = 55.0
        # diameter in pixels at fullres
        spot_diam_px = sf.get("spot_diameter_fullres", 1.0)
        coords_um = coords * (spot_um / max(spot_diam_px, 1.0))

        for primary in ["EEC", "Tuft", "BEST4"]:
            p_mask = adata.obs[f"pos_{primary}"].values.astype(bool)
            if p_mask.sum() == 0:
                continue
            for partner in ["Glia", "Neuron", "Neuropod"]:
                q_mask = adata.obs[f"pos_{partner}"].values.astype(bool)
                if q_mask.sum() == 0:
                    continue
                tree = BallTree(coords_um[q_mask])
                d, _ = tree.query(coords_um[p_mask], k=1)
                d = d[:, 0]

                # permutation null: re-sample q-mask randomly with same n
                null_means = []
                k = q_mask.sum()
                for _ in range(50):
                    perm = np.zeros_like(q_mask)
                    idx = rng.choice(len(q_mask), size=k, replace=False)
                    perm[idx] = True
                    null_tree = BallTree(coords_um[perm])
                    nd, _ = null_tree.query(coords_um[p_mask], k=1)
                    null_means.append(nd[:, 0].mean())
                null_mean = float(np.mean(null_means))
                null_sd = float(np.std(null_means) + 1e-9)
                z = (null_mean - d.mean()) / null_sd  # positive z = closer than chance

                dist_rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"],
                    primary=primary, partner=partner,
                    n_primary_pos=int(p_mask.sum()),
                    n_partner_pos=int(q_mask.sum()),
                    mean_d_um=float(d.mean()),
                    median_d_um=float(np.median(d)),
                    null_mean_d_um=null_mean,
                    z_closer_than_random=float(z),
                ))

    pd.DataFrame(corr_rows).to_csv(FIG_DIR / "12_pairwise_pearson_corr.csv", index=False)
    pd.DataFrame(coloc_rows).to_csv(FIG_DIR / "13_binary_coloc_fold_over_expected.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(FIG_DIR / "14_nearest_neighbor_distance_vs_null.csv", index=False)

    # heatmap of correlations per sample
    samples_order = list(samples.keys())
    score_label = {s: s.replace("presence_", "") for s in score_keys}
    n_keys = len(score_keys)
    fig, axes = plt.subplots(1, len(samples_order),
                             figsize=(3.6 * len(samples_order), 3.4),
                             squeeze=False)
    axes = axes.flatten()
    corr_df = pd.DataFrame(corr_rows)
    for ax, sid in zip(axes, samples_order):
        sub = corr_df[corr_df["sample"] == sid]
        mat = np.zeros((n_keys, n_keys))
        for _, row in sub.iterrows():
            i = score_keys.index(row["a"]); j = score_keys.index(row["b"])
            mat[i, j] = row["pearson_r"]; mat[j, i] = row["pearson_r"]
        np.fill_diagonal(mat, 1.0)
        im = ax.imshow(mat, vmin=-0.6, vmax=0.6, cmap="RdBu_r")
        ax.set_xticks(range(n_keys));
        ax.set_xticklabels([score_label[s] for s in score_keys], rotation=45, ha="right")
        ax.set_yticks(range(n_keys));
        ax.set_yticklabels([score_label[s] for s in score_keys])
        ax.set_title(SAMPLES[sid]["label"])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Pairwise Pearson r of per-spot module scores", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_pairwise_pearson_corr_heatmaps.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "12_pairwise_pearson_corr_heatmaps.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # bar plot of nearest-neighbor z scores (primary x partner) per sample
    dist_df = pd.DataFrame(dist_rows)
    if len(dist_df):
        fig, ax = plt.subplots(figsize=(11, 3.6))
        pivot = (dist_df.assign(pair=lambda d: d["primary"] + "_x_" + d["partner"])
                 .pivot(index="pair", columns="label",
                        values="z_closer_than_random"))
        pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("z (closer to partner than random)")
        ax.set_title("Spatial proximity of rare epi-spots to glia/neuron/neuropod (z = null - obs)")
        ax.tick_params(axis="x", rotation=25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "14_nearest_neighbor_z_bar.png", dpi=200)
        fig.savefig(FIG_DIR / "14_nearest_neighbor_z_bar.pdf")
        plt.close(fig)


def fig_neighborhood_enrichment(samples, k: int = 18, n_perm: int = 300):
    """For each (primary, partner) pair compute the mean PARTNER score within
    k nearest neighbors of each PRIMARY+ spot, then test whether that mean
    exceeds what we'd see if PRIMARY+ labels were randomly placed on the SAME
    set of spatial coordinates.

    This null preserves the underlying tissue geometry (so the result is not
    driven by EEC+ and Glia+ living in different compartments) and tests
    whether being a PRIMARY+ spot specifically predicts a high partner
    neighborhood score.
    """
    print("== Neighborhood enrichment (compartment-aware null) ==", flush=True)
    primaries = ["EEC", "Tuft", "BEST4"]
    partners = ["Glia", "Neuron", "Neuropod"]

    rows = []
    rng = np.random.default_rng(0)
    for sid, adata in samples.items():
        sp_key = list(adata.uns["spatial"].keys())[0]
        coords = adata.obsm["spatial"]
        sf = adata.uns["spatial"][sp_key]["scalefactors"]
        spot_diam_px = sf.get("spot_diameter_fullres", 1.0)
        coords_um = coords * (55.0 / max(spot_diam_px, 1.0))

        # for each spot, find k nearest neighbors (excluding self)
        tree = BallTree(coords_um)
        _, knn = tree.query(coords_um, k=k + 1)
        knn = knn[:, 1:]  # drop self

        for partner in partners:
            partner_score = adata.obs[f"presence_{partner}"].astype(float).values
            # mean partner score in the k-neighborhood of every spot
            neigh_mean = partner_score[knn].mean(axis=1)

            for primary in primaries:
                p_mask = adata.obs[f"pos_{primary}"].values.astype(bool)
                if p_mask.sum() < 5:
                    continue
                obs_mean = float(neigh_mean[p_mask].mean())

                # permutation null: shuffle which spots are PRIMARY+ but
                # keep the same count and the same neigh_mean vector
                n_pos = int(p_mask.sum())
                null_means = np.empty(n_perm)
                n_spots = adata.n_obs
                for i in range(n_perm):
                    perm_idx = rng.choice(n_spots, size=n_pos, replace=False)
                    null_means[i] = neigh_mean[perm_idx].mean()
                null_mean = float(null_means.mean())
                null_sd = float(null_means.std() + 1e-9)
                z = (obs_mean - null_mean) / null_sd
                p_emp = float((null_means >= obs_mean).mean())

                rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"],
                    primary=primary, partner=partner,
                    n_primary_pos=n_pos,
                    obs_mean_partner_in_kNN=obs_mean,
                    null_mean=null_mean, null_sd=null_sd,
                    z_score=z, p_empirical=p_emp,
                ))

    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "16_kNN_enrichment_vs_null.csv", index=False)
    print(df.to_string(index=False), flush=True)

    if len(df):
        df["pair"] = df["primary"] + "->" + df["partner"]
        # heatmap of z-scores: rows = pair, cols = sample
        pivot = df.pivot(index="pair", columns="label", values="z_score")
        fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 3, 0.55 * pivot.shape[0] + 1.5))
        vmax = max(3.0, float(np.nanmax(np.abs(pivot.values))))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax * 0.55 else "black"
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            color=col, fontsize=8)
        plt.colorbar(im, ax=ax, label="z (neighborhood partner score, obs vs random PRIMARY label)")
        ax.set_title(f"k-NN partner-score enrichment around PRIMARY+ spots (k={k})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "16_kNN_enrichment_heatmap.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / "16_kNN_enrichment_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)
    print("  wrote 16_kNN_enrichment_heatmap.[png,pdf] and 16_kNN_enrichment_vs_null.csv",
          flush=True)


def fig_duo_double_positive_overlay(samples):
    """Spatial overlay of binary positive-call masks for the duodenum."""
    print("== Duodenum overlay: EEC+ / Tuft+ / BEST4+ / Glia+ / Neuron+ ==",
          flush=True)
    duo = samples["GUTsp9518707"]
    keys = ["pos_EEC", "pos_Tuft", "pos_BEST4", "pos_Glia",
            "pos_Neuron", "pos_Neuropod"]
    duo_for_plot = duo.copy()
    for k in keys:
        duo_for_plot.obs[k] = duo_for_plot.obs[k].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, k in zip(axes.flatten(), keys):
        sc.pl.spatial(duo_for_plot, color=k, show=False, ax=ax,
                      palette={"0": "#dddddd", "1": "#cc0033"},
                      size=1.4, title=f"DUO {k}", frameon=False,
                      legend_loc=None, img_key="lowres")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_duo_positive_spots_overlay.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / "15_duo_positive_spots_overlay.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote 15_duo_positive_spots_overlay.[png,pdf]", flush=True)


def fig_duo_deep_dive(samples):
    """Per-EEC+ spot inspection: NCAM/GDNF/RET ligand-receptor expression in
    the immediate (k=6) neighborhood.

    Also writes a CSV of every EEC+ spot in the duodenum with: distance to
    nearest Glia+ spot, distance to nearest Neuron+ spot, mean neuropod-module
    score in 6-NN neighborhood, and the dominant EEC subtype (max of EEC_I /
    EEC_L / EEC_EC scores).
    """
    print("== Duodenum deep dive: EEC+ spot table + neuropod neighborhood ==",
          flush=True)
    duo = samples["GUTsp9518707"]
    sp_key = list(duo.uns["spatial"].keys())[0]
    sf = duo.uns["spatial"][sp_key]["scalefactors"]
    spot_diam_px = sf.get("spot_diameter_fullres", 1.0)
    coords_um = duo.obsm["spatial"] * (55.0 / max(spot_diam_px, 1.0))
    tree = BallTree(coords_um)
    _, knn = tree.query(coords_um, k=7)
    knn = knn[:, 1:]

    eec_pos = duo.obs["pos_EEC"].values.astype(bool)
    if eec_pos.sum() == 0:
        print("  no EEC+ spots in duodenum")
        return

    score_cols = {
        "EEC_I_CCK":  "score_EEC_I_CCK_prox",
        "EEC_L_GLP1": "score_EEC_L_GLP1",
        "EEC_EC_TPH1":"score_EEC_EC_TPH1",
        "EEC_N_NTS":  "score_EEC_N_NTS",
        "EEC_S_SCT":  "score_EEC_S_SCT",
        "Tuft":       "score_Tuft",
        "BEST4":      "score_BEST4_core",
    }
    neighborhood_cols = {
        "Glia":     "presence_Glia",
        "Neuron":   "presence_Neuron",
        "Neuropod": "presence_Neuropod",
    }

    glia_pos_idx = np.where(duo.obs["pos_Glia"].values == 1)[0]
    neuron_pos_idx = np.where(duo.obs["pos_Neuron"].values == 1)[0]
    glia_tree = BallTree(coords_um[glia_pos_idx]) if len(glia_pos_idx) else None
    neuron_tree = BallTree(coords_um[neuron_pos_idx]) if len(neuron_pos_idx) else None

    # vectorized neighborhood means for every spot, then subset to EEC+
    score_arrays = {label: duo.obs[col].astype(float).values
                    for label, col in score_cols.items()}
    neigh_means = {label: duo.obs[col].astype(float).values[knn].mean(axis=1)
                   for label, col in neighborhood_cols.items()}

    eec_idx = np.where(eec_pos)[0]
    df = pd.DataFrame(index=range(len(eec_idx)))
    df["spot_index"] = eec_idx
    df["barcode"] = [str(duo.obs_names[i]) for i in eec_idx]
    for label, arr in score_arrays.items():
        df[label] = arr[eec_idx]
    sub_arr = np.column_stack([score_arrays[k] for k in
                               ("EEC_I_CCK", "EEC_L_GLP1", "EEC_EC_TPH1",
                                "EEC_N_NTS", "EEC_S_SCT")])
    sub_names = ["EEC_I_CCK", "EEC_L_GLP1", "EEC_EC_TPH1",
                 "EEC_N_NTS", "EEC_S_SCT"]
    df["dominant_EEC"] = [sub_names[j] for j in sub_arr[eec_idx].argmax(axis=1)]
    for label, arr in neigh_means.items():
        df[f"kNN6_{label}_score"] = arr[eec_idx]
    if glia_tree is not None:
        d, _ = glia_tree.query(coords_um[eec_idx], k=1)
        df["dist_to_nearest_Glia_um"] = d[:, 0]
    else:
        df["dist_to_nearest_Glia_um"] = np.nan
    if neuron_tree is not None:
        d, _ = neuron_tree.query(coords_um[eec_idx], k=1)
        df["dist_to_nearest_Neuron_um"] = d[:, 0]
    else:
        df["dist_to_nearest_Neuron_um"] = np.nan
    df.to_csv(FIG_DIR / "17_duo_EECpos_spot_table.csv", index=False)
    print(f"  wrote 17_duo_EECpos_spot_table.csv ({len(df)} EEC+ spots)", flush=True)

    # dominant-subtype counts
    sub_counts = df["dominant_EEC"].value_counts()
    print("  Duodenum EEC+ spot dominant subtype:")
    print("  " + sub_counts.to_string().replace("\n", "\n  "), flush=True)

    rng = np.random.default_rng(0)
    n = int(eec_pos.sum())
    neuropod_mean_all = duo.obs["presence_Neuropod"].astype(float).values[knn].mean(axis=1)
    obs = neuropod_mean_all[eec_pos]
    null = np.empty(1000)
    n_spots = duo.n_obs
    for ii in range(null.size):
        null[ii] = neuropod_mean_all[rng.choice(n_spots, size=n, replace=False)].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.hist(null, bins=40, color="#cccccc", label="random spot labels (2000 perms)", alpha=0.9)
    ax.axvline(obs.mean(), color="#cc0033", lw=2.0, label=f"EEC+ spots (n={n})")
    ax.set_xlabel("mean neuropod score in 6-NN neighborhood")
    ax.set_ylabel("# permutations")
    ax.set_title(f"DUO: neuropod score around EEC+ vs null  (z={(obs.mean()-null.mean())/null.std():.2f})")
    ax.legend(fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_duo_EEC_neuropod_neighborhood_null.png", dpi=200)
    fig.savefig(FIG_DIR / "18_duo_EEC_neuropod_neighborhood_null.pdf")
    plt.close(fig)
    print("  wrote 18_duo_EEC_neuropod_neighborhood_null.[png,pdf]", flush=True)

    # 2) Joint spatial overlay: EEC+ spots colored by NCAM1 expression in
    # their immediate 6-NN
    ncam_neigh = None
    if "NCAM1" in duo.var_names:
        ncam_vec = np.asarray(
            duo[:, "NCAM1"].X.toarray() if issparse(duo.X) else duo[:, "NCAM1"].X
        ).flatten()
        ncam_neigh = ncam_vec[knn].mean(axis=1)
        duo.obs["NCAM1_kNN6"] = ncam_neigh
    if "GFRA1" in duo.var_names:
        gfra_vec = np.asarray(
            duo[:, "GFRA1"].X.toarray() if issparse(duo.X) else duo[:, "GFRA1"].X
        ).flatten()
        duo.obs["GFRA1_kNN6"] = gfra_vec[knn].mean(axis=1)
    if "GDNF" in duo.var_names:
        gdnf_vec = np.asarray(
            duo[:, "GDNF"].X.toarray() if issparse(duo.X) else duo[:, "GDNF"].X
        ).flatten()
        duo.obs["GDNF_kNN6"] = gdnf_vec[knn].mean(axis=1)

    keys = [k for k in ["NCAM1_kNN6", "GFRA1_kNN6", "GDNF_kNN6"] if k in duo.obs.columns]
    if keys:
        spatial_panel(duo, keys, FIG_DIR / "19_duo_kNN_neuropod_genes.png",
                      ncols=3, title_prefix="DUO ", cmap="magma")


def fig_best4_segment_markers(samples):
    """Segment-specific BEST4 marker panels (duodenum/jejunum/ileum/colon).

    For each segment-specific gene list we:
      (a) plot each gene as a spatial map across the 3 gut segments
          (jejunum + ileum panels are still scored / plotted on DUO+SCL+REC
          because we have no jej/ile sample here; the comparison still tells us
          how 'jejunum-like' or 'ileum-like' our DUO/SCL/REC BEST4+ cells look),
      (b) plot the aggregate module score per panel,
      (c) compute a per-sample table of mean expression in BEST4+ vs
          other-epi vs all-spots and log2 fold,
      (d) assign each sample the most-enriched panel in BEST4+ vs other-epi as
          a 'segment identity' of its BEST4 cells (sanity check that DUO
          looks duodenum-like, SCL/REC look colon-like).
    """
    print("== BEST4 segment-specific marker panels (duo/jej/ile/col) ==",
          flush=True)
    gut = {sid: samples[sid] for sid in
           ["GUTsp9518707", "GUTsp9518708", "GUTsp9518706"]}  # DUO, SCL, REC

    panel_genes = {
        "Duodenum": EPI_RARE["BEST4_duo"],
        "Jejunum":  EPI_RARE["BEST4_jej"],
        "Ileum":    EPI_RARE["BEST4_ile"],
        "Colon":    EPI_RARE["BEST4_col"],
    }

    # spatial maps of all panel genes (one big stacked figure per segment-panel)
    for panel_name, genes in panel_genes.items():
        present = [g for g in genes
                   if any(g in a.var_names for a in gut.values())]
        if not present:
            continue
        outname = f"27_BEST4_panel_{panel_name.lower()}_genes"
        stacked_segment_panel(gut, present,
                              FIG_DIR / f"{outname}.png", vmax="p99")

    # spatial map of the four aggregate scores
    score_keys = [f"score_BEST4_{abbr}" for abbr in ("duo", "jej", "ile", "col")]
    stacked_segment_panel(gut, score_keys,
                          FIG_DIR / "28_BEST4_segment_scores.png", vmax="p99")

    # per-gene table: mean log1p expression in BEST4+ vs other-epi
    rows = []
    for sid, adata in gut.items():
        b4_pos = adata.obs["pos_BEST4"].values.astype(bool)
        epi = adata.obs["score_Epithelium"].astype(float).values
        epi_thr = np.quantile(epi, 0.75)
        other_epi = (epi > epi_thr) & ~b4_pos
        for panel_name, genes in panel_genes.items():
            for gene in genes:
                if gene not in adata.var_names:
                    continue
                v = np.asarray(adata[:, gene].X.toarray() if issparse(adata.X)
                               else adata[:, gene].X).flatten()
                rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"],
                    panel=panel_name, gene=gene,
                    n_BEST4_pos=int(b4_pos.sum()),
                    n_other_epi=int(other_epi.sum()),
                    mean_in_BEST4_pos=float(v[b4_pos].mean()) if b4_pos.any() else np.nan,
                    mean_in_other_epi=float(v[other_epi].mean()) if other_epi.any() else np.nan,
                    mean_in_all_spots=float(v.mean()),
                    frac_BEST4pos_with_gene=float((v[b4_pos] > 0).mean()) if b4_pos.any() else np.nan,
                    frac_other_epi_with_gene=float((v[other_epi] > 0).mean()) if other_epi.any() else np.nan,
                ))
    df = pd.DataFrame(rows)
    df["log2_fold_vs_otherEpi"] = np.log2(
        (df["mean_in_BEST4_pos"] + 1e-3) / (df["mean_in_other_epi"] + 1e-3))
    df.to_csv(FIG_DIR / "29_BEST4_segment_marker_table.csv", index=False)

    # heatmap of log2 fold (BEST4+ / other-epi), gene rows grouped by panel
    if len(df):
        # order rows by panel then by gene as supplied
        df["panel"] = pd.Categorical(df["panel"],
                                     categories=["Duodenum", "Jejunum",
                                                 "Ileum", "Colon"],
                                     ordered=True)
        df = df.sort_values(["panel", "gene"])
        pivot = df.pivot_table(index=["panel", "gene"], columns="label",
                               values="log2_fold_vs_otherEpi", sort=False)
        # column ordering: DUO, SCL, REC
        wanted_cols = ["A50_DUO", "A50-SCL", "A38-REC"]
        pivot = pivot.reindex(columns=[c for c in wanted_cols if c in pivot.columns])
        fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 4.5,
                                        0.35 * pivot.shape[0] + 1.5))
        vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 2.0
        vmax = max(vmax, 1.0)
        im = ax.imshow(pivot.values, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        # tick labels: include panel prefix
        ylabels = [f"{p[:3]} | {g}" for (p, g) in pivot.index]
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(ylabels)
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        # add horizontal lines between panels
        panel_starts = []
        prev = None
        for i, (p, _) in enumerate(pivot.index):
            if p != prev:
                if prev is not None:
                    ax.axhline(i - 0.5, color="black", linewidth=1)
                panel_starts.append(i)
                prev = p
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color=col, fontsize=8)
        plt.colorbar(im, ax=ax, label="log2 (BEST4+ / other epi mean log1p)")
        ax.set_title("Segment-specific BEST4 markers: enrichment in BEST4+ spots")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "29_BEST4_segment_marker_log2fold.png",
                    dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / "29_BEST4_segment_marker_log2fold.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    # per-panel module-score enrichment in BEST4+ vs other-epi
    score_rows = []
    for sid, adata in gut.items():
        b4_pos = adata.obs["pos_BEST4"].values.astype(bool)
        epi = adata.obs["score_Epithelium"].astype(float).values
        epi_thr = np.quantile(epi, 0.75)
        other_epi = (epi > epi_thr) & ~b4_pos
        for abbr, panel_name in zip(("duo", "jej", "ile", "col"),
                                    ("Duodenum", "Jejunum", "Ileum", "Colon")):
            col = f"score_BEST4_{abbr}"
            if col not in adata.obs.columns:
                continue
            s = adata.obs[col].astype(float).values
            score_rows.append(dict(
                sample=sid, label=SAMPLES[sid]["label"], panel=panel_name,
                n_BEST4_pos=int(b4_pos.sum()),
                mean_score_BEST4_pos=float(s[b4_pos].mean()) if b4_pos.any() else np.nan,
                mean_score_other_epi=float(s[other_epi].mean()) if other_epi.any() else np.nan,
                mean_score_all=float(s.mean()),
                delta_BEST4_vs_otherEpi=float(s[b4_pos].mean() - s[other_epi].mean())
                                       if (b4_pos.any() and other_epi.any())
                                       else np.nan,
            ))
    sdf = pd.DataFrame(score_rows)
    sdf.to_csv(FIG_DIR / "29_BEST4_segment_score_table.csv", index=False)

    # heatmap of delta (BEST4+ - other-epi) per panel per sample
    if len(sdf):
        pivot = sdf.pivot(index="panel", columns="label",
                          values="delta_BEST4_vs_otherEpi")
        pivot = pivot.reindex(index=["Duodenum", "Jejunum", "Ileum", "Colon"])
        pivot = pivot.reindex(columns=["A50_DUO", "A50-SCL", "A38-REC"])
        fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 3,
                                        0.5 * pivot.shape[0] + 1.5))
        vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 0.1
        im = ax.imshow(pivot.values, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                            color=col, fontsize=9)
        plt.colorbar(im, ax=ax, label="mean score (BEST4+) - (other epi)")
        ax.set_title("BEST4 segment-identity scores (delta vs other epithelium)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "29_BEST4_segment_score_heatmap.png",
                    dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / "29_BEST4_segment_score_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        # BEST4 identity assignment: aggregate-score delta is dominated by
        # genes with high absolute expression (e.g. ALDOB/SI in ileum panel are
        # broadly expressed in any small intestine BEST4 cell, not
        # BEST4-specific). The better metric is the *median per-gene log2 fold
        # of BEST4+/other-epi*, which captures BEST4-specific enrichment.
        median_lfc = (df.groupby(["label", "panel"], observed=True)
                        ["log2_fold_vs_otherEpi"].median().reset_index())
        med_pivot = median_lfc.pivot(index="panel", columns="label",
                                     values="log2_fold_vs_otherEpi")
        med_pivot = med_pivot.reindex(index=["Duodenum", "Jejunum",
                                              "Ileum", "Colon"])
        med_pivot = med_pivot.reindex(columns=[c for c in
                                                ["A50_DUO", "A50-SCL", "A38-REC"]
                                                if c in med_pivot.columns])
        med_pivot.to_csv(FIG_DIR / "29_BEST4_segment_median_log2fold.csv")

        # heatmap of median log2 fold per panel per sample
        fig, ax = plt.subplots(figsize=(0.9 * med_pivot.shape[1] + 3,
                                        0.5 * med_pivot.shape[0] + 1.5))
        vmax = max(1.0, float(np.nanmax(np.abs(med_pivot.values))))
        im = ax.imshow(med_pivot.values, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(med_pivot.shape[1])); ax.set_xticklabels(med_pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(med_pivot.shape[0])); ax.set_yticklabels(med_pivot.index)
        for i in range(med_pivot.shape[0]):
            for j in range(med_pivot.shape[1]):
                v = med_pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color=col, fontsize=9)
        plt.colorbar(im, ax=ax, label="median per-gene log2 (BEST4+ / other epi)")
        ax.set_title("BEST4 segment identity via median per-gene log2 fold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "29_BEST4_segment_median_log2fold_heatmap.png",
                    dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / "29_BEST4_segment_median_log2fold_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        ident_score = (pivot.idxmax(axis=0)
                       .rename("by_aggregate_score_delta"))
        ident_lfc = (med_pivot.idxmax(axis=0)
                     .rename("by_median_log2fold"))
        ident = pd.concat([ident_score, ident_lfc], axis=1).reset_index()
        ident.to_csv(FIG_DIR / "29_BEST4_segment_identity.csv", index=False)
        print("  BEST4 segment-identity per sample:", flush=True)
        print("  " + ident.to_string(index=False).replace("\n", "\n  "),
              flush=True)
        print("  (use 'by_median_log2fold' -- aggregate score gets dominated by"
              " broadly-expressed SI genes like ALDOB/SI in the Ileum panel)",
              flush=True)

    print("  wrote 27-29 BEST4 segment marker figures", flush=True)


def fig_best4_neuropod(samples):
    """BEST4-neuropod-specific markers (ChAT, MEIS1, ROBO1, neurofilaments).

    For each segment we plot the BEST4-neuropod module score + each member
    gene, then ask within each sample whether BEST4+ spots are enriched for
    these markers vs (a) all spots and (b) other epithelium (epithelial score
    above 75th percentile). Output: spatial maps + a per-sample table.
    """
    print("== BEST4 neuropod-specific markers (ChAT/MEIS1/ROBO1/NEFL...) ==",
          flush=True)
    gut = {sid: samples[sid] for sid in
           ["GUTsp9518707", "GUTsp9518708", "GUTsp9518706"]}  # DUO, SCL, REC

    # spatial maps -- score + key individual genes
    keys = ["score_Neuropod_BEST4", "presence_Neuropod_B4",
            "CHAT", "MEIS1", "MEIS2", "ROBO1", "ROBO2", "NEFL", "NEFM"]
    stacked_segment_panel(gut, keys,
                          FIG_DIR / "20_best4_neuropod_markers.png", vmax="p99")

    # quantitative enrichment in BEST4+ spots vs other epithelium
    rows = []
    for sid, adata in gut.items():
        b4_pos = adata.obs["pos_BEST4"].values.astype(bool)
        # other epithelial spots: epithelium score above 75th pct, NOT BEST4+
        epi = adata.obs["score_Epithelium"].astype(float).values
        epi_thr = np.quantile(epi, 0.75)
        other_epi = (epi > epi_thr) & ~b4_pos
        all_spots = np.ones(adata.n_obs, dtype=bool)

        for gene in ["CHAT", "MEIS1", "MEIS2", "ROBO1", "NEFL", "NEFM", "NEFH",
                     "NCAM1", "GFRA1", "RET"]:
            if gene not in adata.var_names:
                continue
            v = np.asarray(adata[:, gene].X.toarray() if issparse(adata.X)
                           else adata[:, gene].X).flatten()
            rows.append(dict(
                sample=sid, label=SAMPLES[sid]["label"], gene=gene,
                mean_in_BEST4_pos=float(v[b4_pos].mean()) if b4_pos.any() else np.nan,
                mean_in_other_epi=float(v[other_epi].mean()) if other_epi.any() else np.nan,
                mean_in_all_spots=float(v[all_spots].mean()),
                frac_BEST4pos_with_gene=float((v[b4_pos] > 0).mean()) if b4_pos.any() else np.nan,
                frac_other_epi_with_gene=float((v[other_epi] > 0).mean()) if other_epi.any() else np.nan,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "20_best4_neuropod_gene_table.csv", index=False)
    print(df.to_string(index=False), flush=True)

    # bar plot: log2 fold (BEST4+ / other_epi mean log1p) per gene per segment
    if len(df):
        df["log2_fold_vs_otherEpi"] = np.log2(
            (df["mean_in_BEST4_pos"] + 1e-3) / (df["mean_in_other_epi"] + 1e-3))
        pivot = df.pivot(index="gene", columns="label",
                         values="log2_fold_vs_otherEpi")
        fig, ax = plt.subplots(figsize=(7, 0.35 * pivot.shape[0] + 1.5))
        vmax = float(np.nanmax(np.abs(pivot.values)))
        im = ax.imshow(pivot.values, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax*0.6 else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color=col, fontsize=8)
        plt.colorbar(im, ax=ax, label="log2 (mean BEST4+ / other epi)")
        ax.set_title("BEST4-neuropod gene enrichment in BEST4+ vs other epithelium")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "20_best4_neuropod_log2fold.png", dpi=200, bbox_inches="tight")
        fig.savefig(FIG_DIR / "20_best4_neuropod_log2fold.pdf", bbox_inches="tight")
        plt.close(fig)
    print("  wrote 20_best4_neuropod_markers + log2fold", flush=True)


def fig_macrophage_subtypes(samples):
    """Spatial maps of macrophage subtypes (tissue resident, M2, M1) and their
    co-localization with rare epithelial cell types.

    The atlas calls 4 macrophage subgroups: pan, tissue-resident, M2, M1.
    Tissue-resident gut macs are typically deep in lamina propria; LYVE1+
    tissue macs are perivascular/submucosal. M2 markers overlap with TR.
    """
    print("== Macrophage subtypes spatial maps + epi colocalization ==",
          flush=True)
    keys = ["score_Mac_pan", "score_Mac_tissue_res", "score_Mac_M2",
            "score_Mac_M1", "score_Monocyte"]
    stacked_segment_panel(samples, keys,
                          FIG_DIR / "21_macrophage_subtype_scores.png", vmax="p99")

    # individual genes (a few canonical per subtype)
    gene_keys = [g for g in ["C1QA", "C1QB", "FOLR2", "CD163", "CD163L1",
                              "LYVE1", "MRC1", "MAFB", "CD14", "FCN1",
                              "IL1B", "TNF"]
                 if any(g in s.var_names for s in samples.values())]
    if gene_keys:
        stacked_segment_panel(samples, gene_keys,
                              FIG_DIR / "22_macrophage_genes.png", vmax="p99")

    # kNN enrichment: are rare epi spots enriched for macrophage subtypes
    print("  kNN partner enrichment vs random label null", flush=True)
    primaries = ["EEC", "Tuft", "BEST4"]
    partners = ["Mac_TR", "Mac_M2", "Mac_M1", "Mac_pan"]
    rows = []
    rng = np.random.default_rng(0)
    n_perm = 300
    k = 18

    for sid, adata in samples.items():
        sp_key = list(adata.uns["spatial"].keys())[0]
        sf = adata.uns["spatial"][sp_key]["scalefactors"]
        spot_diam_px = sf.get("spot_diameter_fullres", 1.0)
        coords_um = adata.obsm["spatial"] * (55.0 / max(spot_diam_px, 1.0))
        tree = BallTree(coords_um)
        _, knn = tree.query(coords_um, k=k + 1)
        knn = knn[:, 1:]

        for partner in partners:
            partner_score = adata.obs[f"presence_{partner}"].astype(float).values
            neigh_mean = partner_score[knn].mean(axis=1)
            for primary in primaries:
                p_mask = adata.obs[f"pos_{primary}"].values.astype(bool)
                if p_mask.sum() < 5:
                    continue
                obs_mean = float(neigh_mean[p_mask].mean())
                n_pos = int(p_mask.sum())
                null = np.empty(n_perm)
                for ii in range(n_perm):
                    perm_idx = rng.choice(adata.n_obs, size=n_pos, replace=False)
                    null[ii] = neigh_mean[perm_idx].mean()
                null_mean = float(null.mean())
                null_sd = float(null.std() + 1e-9)
                z = (obs_mean - null_mean) / null_sd
                p_emp = float((null >= obs_mean).mean())
                rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"],
                    primary=primary, partner=partner,
                    n_primary_pos=n_pos,
                    obs=obs_mean, null=null_mean, z=z, p_empirical=p_emp,
                ))

    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "23_macrophage_kNN_enrichment.csv", index=False)
    if len(df):
        df["pair"] = df["primary"] + "->" + df["partner"]
        pivot = df.pivot(index="pair", columns="label", values="z")
        fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 3, 0.45 * pivot.shape[0] + 1.5))
        vmax = max(3.0, float(np.nanmax(np.abs(pivot.values))))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", color=col, fontsize=8)
        plt.colorbar(im, ax=ax, label="z (vs random label null)")
        ax.set_title("kNN partner-score enrichment: rare epi vs macrophage subtypes")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "23_macrophage_kNN_enrichment_heatmap.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / "23_macrophage_kNN_enrichment_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)
    print("  wrote 21-23 macrophage figures", flush=True)


def fig_congenital_diarrhea(samples):
    """For each congenital diarrhea gene/module:

    1. Per-segment spatial map (lowres).
    2. Per-segment table: mean expression in BEST4+ / EEC+ / Tuft+ / Macrophage+
       spots vs other epithelial spots vs all spots.
    3. For SLC26A3 specifically: identify the dominant cell-type compartment
       by ranking each compartment by mean expression and write a focused
       table.
    """
    print("== Congenital diarrhea genes (DRA/SLC26A3 focus) ==", flush=True)
    gut = {sid: samples[sid] for sid in
           ["GUTsp9518707", "GUTsp9518708", "GUTsp9518706"]}  # DUO, SCL, REC

    # individual genes of interest
    primary_genes = ["SLC26A3", "SLC9A3", "GUCY2C", "GUCA2A", "GUCA2B",
                     "CFTR", "MYO5B", "STX3", "EPCAM", "NEUROG3",
                     "HNF4A", "SI", "LCT", "DGAT1", "TTC7A"]
    primary_genes_present = [g for g in primary_genes
                             if any(g in s.var_names for s in gut.values())]
    stacked_segment_panel(gut, primary_genes_present,
                          FIG_DIR / "24_congenital_diarrhea_genes_spatial.png",
                          vmax="p99")

    # quantitative stratification: mean expression of each gene in each
    # cell-type-positive spot set, per segment
    compartment_cols = {
        "BEST4+":   "pos_BEST4",
        "EEC+":     "pos_EEC",
        "Tuft+":    "pos_Tuft",
        "MacTR+":   "pos_Mac_TR",
        "MacM2+":   "pos_Mac_M2",
        "MacM1+":   "pos_Mac_M1",
        "Neuron+":  "pos_Neuron",
        "Glia+":    "pos_Glia",
    }
    rows = []
    for sid, adata in gut.items():
        epi_score = adata.obs["score_Epithelium"].astype(float).values
        epi_thr = np.quantile(epi_score, 0.75)
        epi_mask = epi_score > epi_thr
        for gene in primary_genes_present:
            if gene not in adata.var_names:
                continue
            v = np.asarray(adata[:, gene].X.toarray() if issparse(adata.X)
                           else adata[:, gene].X).flatten()
            for cname, col in compartment_cols.items():
                if col not in adata.obs.columns:
                    continue
                mask = adata.obs[col].values.astype(bool)
                rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"], gene=gene,
                    compartment=cname,
                    n_in_compartment=int(mask.sum()),
                    mean_log1p_in_compartment=float(v[mask].mean()) if mask.any() else np.nan,
                    frac_pos_in_compartment=float((v[mask] > 0).mean()) if mask.any() else np.nan,
                ))
            rows.append(dict(
                sample=sid, label=SAMPLES[sid]["label"], gene=gene,
                compartment="OtherEpi",
                n_in_compartment=int(epi_mask.sum()),
                mean_log1p_in_compartment=float(v[epi_mask].mean()) if epi_mask.any() else np.nan,
                frac_pos_in_compartment=float((v[epi_mask] > 0).mean()) if epi_mask.any() else np.nan,
            ))
            rows.append(dict(
                sample=sid, label=SAMPLES[sid]["label"], gene=gene,
                compartment="AllSpots",
                n_in_compartment=int(adata.n_obs),
                mean_log1p_in_compartment=float(v.mean()),
                frac_pos_in_compartment=float((v > 0).mean()),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "24_congenital_diarrhea_compartment_table.csv", index=False)

    # heatmap of mean log1p across (gene x compartment), one panel per segment
    if len(df):
        order = ["BEST4+", "EEC+", "Tuft+", "Glia+", "Neuron+",
                 "MacTR+", "MacM2+", "MacM1+", "OtherEpi", "AllSpots"]
        fig, axes = plt.subplots(1, len(gut),
                                 figsize=(3.4 * len(gut) + 0.5,
                                          0.32 * len(primary_genes_present) + 1.6),
                                 sharey=True, squeeze=False)
        axes = axes.flatten()
        for ax, sid in zip(axes, gut):
            sub = df[df["sample"] == sid]
            pivot = sub.pivot(index="gene", columns="compartment",
                              values="mean_log1p_in_compartment")
            pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
            pivot = pivot.reindex(index=primary_genes_present)
            vmax = float(np.nanmax(pivot.values)) if pivot.size else 1.0
            im = ax.imshow(pivot.values, cmap="magma", vmin=0,
                           vmax=max(vmax, 0.05), aspect="auto")
            ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
            ax.set_title(SAMPLES[sid]["label"])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean log1p")
        fig.suptitle("Congenital-diarrhea genes: mean expression per compartment per segment", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "24_congenital_diarrhea_heatmap.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / "24_congenital_diarrhea_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)

    # SLC26A3 focused: rank compartments by mean expression per sample
    if "SLC26A3" in primary_genes_present:
        sub = df[df["gene"] == "SLC26A3"].copy()
        sub = sub.sort_values(["sample", "mean_log1p_in_compartment"],
                              ascending=[True, False])
        sub.to_csv(FIG_DIR / "25_SLC26A3_compartment_rank.csv", index=False)
        print("  SLC26A3 compartment ranking (top entries):")
        for sid in gut:
            top = sub[sub["sample"] == sid].head(5)
            print(f"  -- {SAMPLES[sid]['label']} --", flush=True)
            print("  " + top[["compartment", "n_in_compartment",
                              "mean_log1p_in_compartment",
                              "frac_pos_in_compartment"]
                             ].to_string(index=False).replace("\n", "\n  "),
                  flush=True)

        # spatial map: SLC26A3 + BEST4+, EEC+ overlay per sample (mucosa-only)
        keys = ["SLC26A3", "BEST4", "score_Epithelium", "pos_BEST4", "pos_EEC"]
        # the pos_X cols are int 0/1 -- promote to obs for cmap rendering
        gut_copy = {sid: a.copy() for sid, a in gut.items()}
        stacked_segment_panel(gut_copy, keys,
                              FIG_DIR / "25_SLC26A3_with_BEST4_EEC.png", vmax="p99")

        # discover what is co-elevated with SLC26A3: for each sample, take
        # top 10% SLC26A3 spots and report their mean module scores.
        focus_rows = []
        modules = ["EEC", "Tuft", "BEST4", "Glia", "Neuron", "Neuropod",
                   "Neuropod_B4", "Mac_TR", "Mac_M2", "Mac_M1"]
        for sid, adata in gut.items():
            if "SLC26A3" not in adata.var_names:
                continue
            v = np.asarray(adata[:, "SLC26A3"].X.toarray() if issparse(adata.X)
                           else adata[:, "SLC26A3"].X).flatten()
            thr = np.quantile(v, 0.90)
            hi = v > thr
            if hi.sum() < 5:
                continue
            for m in modules:
                col = f"presence_{m}"
                if col not in adata.obs.columns:
                    continue
                s = adata.obs[col].astype(float).values
                focus_rows.append(dict(
                    sample=sid, label=SAMPLES[sid]["label"],
                    module=m,
                    n_DRA_hi=int(hi.sum()),
                    mean_module_score_DRA_hi=float(s[hi].mean()),
                    mean_module_score_all=float(s.mean()),
                    delta=float(s[hi].mean() - s.mean()),
                ))
        fdf = pd.DataFrame(focus_rows)
        fdf.to_csv(FIG_DIR / "25_SLC26A3_neighborhood_modules.csv", index=False)
        if len(fdf):
            pivot = fdf.pivot(index="module", columns="label", values="delta")
            fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 3, 0.45 * pivot.shape[0] + 1.5))
            vmax = float(np.nanmax(np.abs(pivot.values)))
            im = ax.imshow(pivot.values, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
            ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if np.isfinite(v):
                        col = "white" if abs(v) > vmax * 0.6 else "black"
                        ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                                color=col, fontsize=8)
            plt.colorbar(im, ax=ax, label="Delta mean module score (DRA-hi - all)")
            ax.set_title("SLC26A3-hi (top 10%) spots: which modules are elevated?")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "25_SLC26A3_neighborhood_heatmap.png", dpi=200,
                        bbox_inches="tight")
            fig.savefig(FIG_DIR / "25_SLC26A3_neighborhood_heatmap.pdf",
                        bbox_inches="tight")
            plt.close(fig)
    print("  wrote 24-25 congenital diarrhea figures", flush=True)


def fig_diarrhea_pathways(samples):
    """Spatial maps of infectious-diarrhea pathway scores (Sprue / Cholera /
    Dysentery / Rotavirus / Norovirus / ETEC / Crypto / Cdiff) across the 3
    gut segments. These come from the Diarrhea_Mechanisms notebook.
    """
    print("== Infectious diarrhea pathway signatures across segments ==",
          flush=True)
    gut = {sid: samples[sid] for sid in
           ["GUTsp9518707", "GUTsp9518708", "GUTsp9518706"]}  # DUO, SCL, REC
    keys = [f"score_{k}" for k in
            ["Sprue", "Cholera", "Dysentery", "Rotavirus", "Norovirus",
             "ETEC_secretory", "Crypto_IFN"]]
    stacked_segment_panel(gut, keys,
                          FIG_DIR / "26_diarrhea_pathway_scores.png", vmax="p99")

    # quantify mean score per pathway per segment
    rows = []
    for sid, adata in gut.items():
        for k in keys:
            if k not in adata.obs.columns:
                continue
            s = adata.obs[k].astype(float).values
            rows.append(dict(sample=sid, label=SAMPLES[sid]["label"],
                             pathway=k.replace("score_", ""),
                             mean=float(s.mean()), median=float(np.median(s)),
                             p95=float(np.quantile(s, 0.95))))
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "26_diarrhea_pathway_summary.csv", index=False)

    if len(df):
        pivot = df.pivot(index="pathway", columns="label", values="mean")
        fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 3, 0.42 * pivot.shape[0] + 1.5))
        vmax = float(np.nanmax(np.abs(pivot.values)))
        im = ax.imshow(pivot.values, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    col = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                            color=col, fontsize=8)
        plt.colorbar(im, ax=ax, label="mean module score (per spot)")
        ax.set_title("Infectious diarrhea pathway score by segment")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "26_diarrhea_pathway_heatmap.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(FIG_DIR / "26_diarrhea_pathway_heatmap.pdf",
                    bbox_inches="tight")
        plt.close(fig)
    print("  wrote 26 diarrhea pathway figures", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _t():
    import time
    return time.strftime("%H:%M:%S")


def main():
    import time
    t0 = time.time()
    print(f"[{_t()}] Loading 4 Visium samples ...", flush=True)
    samples = load_all()
    for sid, adata in samples.items():
        print(f"  {SAMPLES[sid]['label']} ({sid}): {adata.n_obs} spots, "
              f"{adata.n_vars} genes", flush=True)

    steps = [
        ("01-02 tissue landmarks + presence",
            lambda: (fig_tissue_landmarks(samples), fig_rare_celltype_scores(samples))),
        ("03-05 duodenum neuropod",                lambda: fig_duo_neuropod(samples)),
        ("06-08 BEST4 across segments",            lambda: fig_best4_across_segments(samples)),
        ("09-11 endothelial subtypes",             lambda: fig_endothelial_subtypes(samples)),
        ("12-14 pairwise corr + NN-distance",      lambda: fig_neuropod_coexpression(samples)),
        ("15 duo positive overlay",                lambda: fig_duo_double_positive_overlay(samples)),
        ("16 kNN enrichment (compartment null)",   lambda: fig_neighborhood_enrichment(samples)),
        ("17-19 duo deep dive",                    lambda: fig_duo_deep_dive(samples)),
        ("20 BEST4-neuropod (ChAT/MEIS1/ROBO1)",   lambda: fig_best4_neuropod(samples)),
        ("27-29 BEST4 segment markers",            lambda: fig_best4_segment_markers(samples)),
        ("21-23 macrophage subtypes",              lambda: fig_macrophage_subtypes(samples)),
        ("24-25 congenital diarrhea (DRA)",        lambda: fig_congenital_diarrhea(samples)),
        ("26 diarrhea pathway signatures",         lambda: fig_diarrhea_pathways(samples)),
    ]
    selected = os.environ.get("STEPS", "").split(",")
    selected = [s.strip() for s in selected if s.strip()]
    for name, fn in steps:
        if selected and not any(name.startswith(p) for p in selected):
            continue
        print(f"\n[{_t()}] >>> {name}", flush=True)
        st = time.time()
        fn()
        print(f"[{_t()}]   step done in {time.time()-st:.1f}s", flush=True)

    print(f"\n[{_t()}] Done. Total: {time.time()-t0:.1f}s", flush=True)
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
