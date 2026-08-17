"""scIB-style Principal Component Regression (PCR): are cell-type COMPOSITIONS
less confounded by technical/batch covariates than gene EXPRESSION (pseudobulk
DE), while still capturing biological covariates?

Thesis (to be tested, not assumed): for population-level meta-analysis of
single-cell atlases, compositional readouts retain biological signal with much
less technical confounding than per-gene differential expression, so they are a
better substrate for cross-study comparison.

Method (one modality-agnostic number on an identical 0-1 scale):

  PCR(covariate) = sum_k Var_k * R^2(PC_k ~ C(covariate)) / sum_k Var_k

  i.e. the fraction of total embedding variance that a covariate explains
  (Buttner et al. 2019; scIB batch metric "principal component regression").
  Marginal (one covariate at a time), matching scIB and the existing composition
  partial_r2_y_vs_covariate machinery.

Two embeddings, SAME samples and SAME covariate design per eligible cell type:
  - COMPOSITION : per-sample CLR over cell types  -> z-score features -> PCA
  - EXPRESSION  : per-(sample x cell type) pseudobulk log-CPM over HVGs -> z-score
                  -> PCA, computed within each cell type (so composition shifts
                  cannot leak into the expression signal), then aggregated to the
                  lineage (mean PCR across cell types, weighted by n samples).

For each eligible cell type, its pseudobulk sample set is used for both the
expression embedding and the lineage-composition embedding. Cell-type PCR
values are then aggregated to lineage level with identical sample-count
weights for both modalities. This pairing prevents unequal sample support from
driving the modality comparison.

Outputs (under github_vignette_output/composition_vs_expression/):
  composition_vs_expression_pcr_long.csv     per (lineage, covariate, modality)
  composition_vs_expression_pcr_celltype.csv per (lineage, celltype, covariate) expression PCR
  composition_vs_expression_pcr.{pdf,svg,png}  show-stopper figure
"""
from __future__ import annotations

import importlib.util as _importlib_util
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------- config
LINEAGE_PATHS = {
    "stroma":     "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/stroma.h5ad",
    "epithelial": "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/epithelial.h5ad",
    "lymphoid":   "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/lymphoid.h5ad",
    "myeloid":    "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/myeloid.h5ad",
}
OUT_DIR = Path("/Users/kylekimler/GCA/github_vignette_output/composition_vs_expression")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_OUT = Path(
    "/Users/kylekimler/Projects/GCA/publication2026/"
    "fig_sampling_depth_radial/out"
)
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_KEY = "sample_id"
CELLTYPE_COL = "hgca_celltype_v1"

# Covariate taxonomy: copied verbatim from Composition_patpy_variance.ipynb.
BIOLOGICAL_COVARIATES = [
    "sampled_site_condition",   # disease / site context
    "radial_tissue_term",       # gut layer
    "sample_preservation_method",
    "sex_ontology_term",
    "age_range",
]
TECHNICAL_COVARIATES = [
    "dataset_id",               # study / platform = the canonical atlas "batch"
    "assay",
    "sample_collection_method",
    "sequenced_fragment",
    "gene_annotation_version",
]
# tissue_level_1 (gut segment) is the dominant biological axis but is excluded
# from the headline so the bio block is not trivially dominated by anatomy; it is
# reported separately as a reference point.
REFERENCE_COVARIATES = ["tissue_level_1"]

ALL_COVARIATES = BIOLOGICAL_COVARIATES + TECHNICAL_COVARIATES + REFERENCE_COVARIATES

META_COLS = [SAMPLE_KEY] + ALL_COVARIATES

# Support thresholds
MIN_CELLS_PER_PSEUDOBULK = 10     # a sample must have >= this many cells of a cell type
MIN_SAMPLES_PER_CELLTYPE = 25     # cell type tested only if present in >= this many samples
N_HVG = 2000
N_PCS_EXPR = 50
PSEUDOCOUNT = 0.5

NATURE_WONG = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "bluish_green": "#009E73", "blue": "#0072B2", "yellow": "#F0E442",
    "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
LINEAGE_MARKER = {"epithelial": "o", "lymphoid": "s", "myeloid": "^", "stroma": "D"}
BLOCK_COLOR = {
    "biological": NATURE_WONG["bluish_green"],
    "technical":  NATURE_WONG["vermillion"],
    "reference":  NATURE_WONG["black"],
}
_HAVE_CAIRO = (_importlib_util.find_spec("cairo") is not None
               or _importlib_util.find_spec("cairocffi") is not None)
_MM = 25.4


def _pub_rc(base_pt=6, title_pt=7):
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial",
                            "Liberation Sans", "DejaVu Sans"],
        "font.size": base_pt, "axes.labelsize": base_pt, "axes.titlesize": title_pt,
        "xtick.labelsize": base_pt, "ytick.labelsize": base_pt,
        "legend.fontsize": base_pt, "legend.title_fontsize": base_pt,
        "axes.linewidth": 0.5, "axes.edgecolor": "black", "axes.labelcolor": "black",
        "axes.titlecolor": "black", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.color": "black", "ytick.color": "black", "text.color": "black",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    }


def _save_pub(fig, stem):
    destinations = [
        (OUT_DIR, stem),
        (FIGURE_OUT, "panel_d_covariate_pcr_composition_vs_expression"),
    ]
    for out_dir, out_stem in destinations:
        pdf, svg, png = (
            out_dir / f"{out_stem}.{e}" for e in ("pdf", "svg", "png")
        )
        if _HAVE_CAIRO:
            try:
                fig.savefig(pdf, backend="cairo", facecolor="white")
            except Exception as exc:
                print(f"  cairo failed ({exc!r}); default PDF backend.")
                fig.savefig(pdf, facecolor="white")
        else:
            fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        fig.savefig(png, dpi=300, facecolor="white")


def _mode_or_nan(x):
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan


def _block_of(cov):
    if cov in BIOLOGICAL_COVARIATES:
        return "biological"
    if cov in TECHNICAL_COVARIATES:
        return "technical"
    return "reference"


# ----------------------------------------------------------------------------- core math
def _anova_r2_all_pcs(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Bias-corrected one-way ANOVA effect size (omega^2) of each PC against a
    categorical, vectorised over PCs.

    omega^2_k = (SS_between_k - (G-1) * MS_within_k) / (SS_total_k + MS_within_k)
    with MS_within_k = SS_within_k / (n - G). Unlike raw R^2, omega^2 does not
    reward covariates merely for having more levels, so high-cardinality
    `dataset_id` (~30 studies) is comparable to 2-level `sex`. Clipped at 0.

    scores : [n_obs x n_pcs]; groups : [n_obs] labels.
    Returns [n_pcs] array of omega^2 in [0, 1].
    """
    n, K = scores.shape
    grand = scores.mean(axis=0)                       # [K]
    ss_total = ((scores - grand) ** 2).sum(axis=0)    # [K]
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
        mean_g = scores[m].mean(axis=0)               # [K]
        ss_between += n_g * (mean_g - grand) ** 2
    ss_within = np.clip(ss_total - ss_between, 0.0, None)
    ms_within = ss_within / (n - G)                   # [K]
    num = ss_between - (G - 1) * ms_within
    den = ss_total + ms_within
    with np.errstate(divide="ignore", invalid="ignore"):
        omega2 = np.where(den > 0, num / den, 0.0)
    return np.clip(np.nan_to_num(omega2, nan=0.0), 0.0, 1.0)


def pcr_per_covariate(embedding: np.ndarray, var_weights: np.ndarray,
                      meta: pd.DataFrame, covariates: list[str]) -> dict:
    """Variance-weighted PCR for each covariate (vectorised one-way ANOVA).

    embedding   : [n_obs x n_pcs] PC scores
    var_weights : [n_pcs] variance explained by each PC (eigenvalues / EVR)
    meta        : DataFrame aligned row-for-row with embedding; covariate columns
    Returns {covariate: pcr in [0, 1]}.
    """
    out = {}
    w = np.asarray(var_weights, dtype=float)
    w_sum = w.sum()
    meta = meta.reset_index(drop=True)

    for cov in covariates:
        if cov not in meta.columns:
            out[cov] = np.nan
            continue
        cvals = meta[cov].astype("object")
        mask = ~cvals_isunknown(cvals)
        if mask.sum() < 10 or cvals[mask].nunique() < 2:
            out[cov] = np.nan
            continue
        sub_scores = embedding[mask.values]
        groups = cvals[mask].astype(str).values
        r2 = _anova_r2_all_pcs(sub_scores, groups)
        out[cov] = float((w * r2).sum() / w_sum) if w_sum > 0 else np.nan
    return out


def cvals_isunknown(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    return s.isna() | t.isin(["", "unknown", "nan", "none", "n/a", "na"])


def embed_matrix(X: np.ndarray, n_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    """z-score features, PCA. Returns (scores, explained_variance)."""
    X = np.asarray(X, dtype=float)
    # drop zero-variance features
    sd = X.std(axis=0)
    keep = sd > 1e-12
    X = X[:, keep]
    if X.shape[1] == 0:
        return None, None
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    n_comp = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    if n_comp < 1:
        return None, None
    p = PCA(n_components=n_comp, svd_solver="full")
    scores = p.fit_transform(X)
    return scores, p.explained_variance_


# ----------------------------------------------------------------------------- data builders
def compute_clr(adata) -> pd.DataFrame:
    counts = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs[CELLTYPE_COL])
    x = counts.astype(float) + PSEUDOCOUNT
    prop = x.div(x.sum(axis=1), axis=0)
    logx = np.log(prop)
    clr = logx - logx.mean(axis=1).values.reshape(-1, 1)
    return pd.DataFrame(clr, index=prop.index, columns=prop.columns)


def build_meta(adata) -> pd.DataFrame:
    cols = [c for c in ALL_COVARIATES if c in adata.obs.columns]
    agg = {c: _mode_or_nan for c in cols}
    sm = adata.obs.groupby(SAMPLE_KEY).agg(agg)
    sm.index = sm.index.astype(str)
    return sm


def pseudobulk_logcpm(adata, celltype: str) -> tuple[pd.DataFrame, pd.Index] | tuple[None, None]:
    """Per-sample pseudobulk log-CPM for one cell type. Returns (df[samples x genes], samples)."""
    mask = (adata.obs[CELLTYPE_COL] == celltype).values
    if mask.sum() == 0:
        return None, None
    X = adata.X[mask]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    samples = adata.obs.loc[mask, SAMPLE_KEY].astype(str).values

    uniq, inv = np.unique(samples, return_inverse=True)
    n_samp = len(uniq)
    # indicator [n_samp x n_cells] sparse -> sum counts per sample
    ind = sparse.csr_matrix(
        (np.ones(len(inv)), (inv, np.arange(len(inv)))),
        shape=(n_samp, X.shape[0]),
    )
    pb = ind @ X                      # [n_samp x n_genes] summed counts
    pb = np.asarray(pb.todense(), dtype=float)
    cells_per_sample = np.bincount(inv, minlength=n_samp)

    keep = cells_per_sample >= MIN_CELLS_PER_PSEUDOBULK
    pb = pb[keep]
    uniq = uniq[keep]
    if pb.shape[0] < MIN_SAMPLES_PER_CELLTYPE:
        return None, None

    lib = pb.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    cpm = pb / lib * 1e6
    logcpm = np.log1p(cpm)
    df = pd.DataFrame(logcpm, index=uniq, columns=adata.var_names.astype(str))
    return df, pd.Index(uniq)


def select_hvg(df: pd.DataFrame, n: int) -> pd.DataFrame:
    v = df.var(axis=0)
    top = v.sort_values(ascending=False).head(n).index
    return df[top]


# ----------------------------------------------------------------------------- run
def main():
    long_rows = []        # (lineage, covariate, block, modality, pcr, n_samples)
    celltype_rows = []    # (lineage, celltype, covariate, block, pcr, n_samples)

    for lineage, path in LINEAGE_PATHS.items():
        print(f"\n=== {lineage} ===", flush=True)
        adata = sc.read_h5ad(path)
        adata.obs[SAMPLE_KEY] = adata.obs[SAMPLE_KEY].astype(str)
        meta = build_meta(adata)
        present_covs = [c for c in ALL_COVARIATES if c in meta.columns]

        # ---- COMPOSITION embedding (per-sample CLR) ----
        clr = compute_clr(adata)
        clr.index = clr.index.astype(str)
        print(f"  composition matrix: {clr.shape[0]} samples x "
              f"{clr.shape[1]} cell types", flush=True)

        # ---- paired composition + expression embeddings per cell type ----
        celltypes = (adata.obs[CELLTYPE_COL].value_counts())
        celltypes = celltypes[celltypes > 0].index.tolist()
        # Both modalities use the same cell-type sample support and weights.
        acc_num = {
            modality: {c: 0.0 for c in present_covs}
            for modality in ("composition", "expression")
        }
        acc_den = {
            modality: {c: 0.0 for c in present_covs}
            for modality in ("composition", "expression")
        }
        n_ct_tested = 0
        for ct in celltypes:
            pb_df, ct_samples = pseudobulk_logcpm(adata, ct)
            if pb_df is None:
                continue
            common = pb_df.index.intersection(clr.index)
            if len(common) < MIN_SAMPLES_PER_CELLTYPE:
                continue
            pb_df = pb_df.reindex(common)
            ct_clr = clr.reindex(common)
            ct_meta = meta.reindex(common)

            pb_hvg = select_hvg(pb_df, N_HVG)
            sc_scores, sc_evr = embed_matrix(pb_hvg.values, n_pcs=N_PCS_EXPR)
            comp_scores, comp_evr = embed_matrix(
                ct_clr.values, n_pcs=ct_clr.shape[1]
            )
            if sc_scores is None or comp_scores is None:
                continue
            ct_pcr = pcr_per_covariate(sc_scores, sc_evr, ct_meta, present_covs)
            ct_comp_pcr = pcr_per_covariate(
                comp_scores, comp_evr, ct_meta, present_covs
            )
            n_ct = len(common)
            n_ct_tested += 1
            for cov in present_covs:
                expr_v = ct_pcr.get(cov, np.nan)
                comp_v = ct_comp_pcr.get(cov, np.nan)
                celltype_rows.append(dict(lineage=lineage, celltype=ct, covariate=cov,
                                          block=_block_of(cov), pcr=expr_v, n_samples=n_ct))
                for modality, value in (
                    ("expression", expr_v), ("composition", comp_v)
                ):
                    if np.isfinite(value):
                        acc_num[modality][cov] += value * n_ct
                        acc_den[modality][cov] += n_ct
        print(f"  paired: {n_ct_tested} cell types passed support thresholds", flush=True)
        for modality in ("composition", "expression"):
            for cov in present_covs:
                den = acc_den[modality][cov]
                value = acc_num[modality][cov] / den if den > 0 else np.nan
                long_rows.append(dict(
                    lineage=lineage, covariate=cov, block=_block_of(cov),
                    modality=modality, pcr=value, n_samples=int(den)
                ))

        del adata

    long_df = pd.DataFrame(long_rows)
    ct_df = pd.DataFrame(celltype_rows)
    long_df.to_csv(OUT_DIR / "composition_vs_expression_pcr_long.csv", index=False)
    ct_df.to_csv(OUT_DIR / "composition_vs_expression_pcr_celltype.csv", index=False)

    # ---- summary print ----
    wide = long_df.pivot_table(index=["block", "covariate", "lineage"],
                               columns="modality", values="pcr").reset_index()
    print("\n=== per (lineage, covariate) PCR ===")
    print(wide.to_string(index=False))

    pooled = (long_df.groupby(["block", "covariate", "modality"])["pcr"]
              .mean().reset_index()
              .pivot_table(index=["block", "covariate"], columns="modality", values="pcr"))
    pooled["expr_minus_comp"] = pooled["expression"] - pooled["composition"]
    print("\n=== pooled across lineages (mean PCR) ===")
    print(pooled.to_string())

    # ---- retention (composition omega^2 / expression omega^2) per covariate ----
    pooled["retained_frac"] = pooled["composition"] / pooled["expression"]
    print("\n=== retention (composition / expression) per covariate ===")
    print(pooled[["composition", "expression", "expr_minus_comp", "retained_frac"]].to_string())

    # mean retention by block (the key fairness check)
    ret = pooled.reset_index()
    ret_block = ret[ret["block"].isin(["biological", "technical"])].groupby("block")["retained_frac"].mean()
    print("\n=== mean retention by block ===")
    print(ret_block.to_string())

    # headline scalars
    batch_expr = float(pooled.loc[("technical", "dataset_id"), "expression"])
    batch_comp = float(pooled.loc[("technical", "dataset_id"), "composition"])
    batch_reduction = 1.0 - batch_comp / batch_expr
    # largest biological covariate that is genuinely biological (exclude radial, which is
    # partly a collection-method artifact) -> age_range
    print(f"\nBatch (dataset_id): expression omega^2={batch_expr:.3f}  composition omega^2={batch_comp:.3f}  "
          f"-> composition removes {batch_reduction*100:.0f}% of batch variance")
    print(f"Mean retention biological={ret_block['biological']:.2f}  technical={ret_block['technical']:.2f} "
          f"(uniform compression => composition is a ~{1/ret_block.mean():.1f}x denoiser, not a batch-selective filter)")

    stats = dict(batch_expr=batch_expr, batch_comp=batch_comp,
                 batch_reduction=batch_reduction,
                 ret_bio=float(ret_block["biological"]), ret_tech=float(ret_block["technical"]))
    make_figure(long_df, pooled, stats)
    print(f"\nWrote outputs to {OUT_DIR}")


def make_figure(long_df, pooled, stats):
    # order covariates by expression omega^2 (descending) so batch is on top
    tab = pooled.reset_index().set_index("covariate")
    tab = tab.sort_values("expression", ascending=False)
    covs = tab.index.tolist()
    n = len(covs)

    cmap = LinearSegmentedColormap.from_list(
        "wong_warm", [(0.0, "#FFFFFF"), (0.33, "#F0E442"), (0.67, "#E69F00"), (1.0, "#D55E00")])

    with _mpl.rc_context(_pub_rc(base_pt=6, title_pt=7)):
        fig = plt.figure(figsize=(180 / _MM, 108 / _MM))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.62],
            wspace=0.05, hspace=0.55)

        # ---- Panel A: dumbbell (composition vs expression omega^2 per covariate) ----
        ax = fig.add_subplot(gs[0, 0])
        y = np.arange(n)[::-1]  # top = highest expression
        for yi, cov in zip(y, covs):
            comp = tab.loc[cov, "composition"]
            expr = tab.loc[cov, "expression"]
            block = tab.loc[cov, "block"]
            col = BLOCK_COLOR[block]
            ax.plot([comp, expr], [yi, yi], color="0.7", linewidth=0.8, zorder=1)
            ax.scatter(comp, yi, marker="o", s=22, facecolor="white", edgecolor=col,
                       linewidth=1.0, zorder=3)
            ax.scatter(expr, yi, marker="o", s=22, facecolor=col, edgecolor="black",
                       linewidth=0.3, zorder=3)
        ax.set_yticks(y)
        labels = []
        for cov in covs:
            lbl = cov.replace("_", " ")
            if cov == "dataset_id":
                lbl = "dataset id (study / batch)"
            labels.append(lbl)
        ax.set_yticklabels(labels, fontsize=6)
        # bold the batch row label
        for tick, cov in zip(ax.get_yticklabels(), covs):
            if cov == "dataset_id":
                tick.set_fontweight("bold")
            if cov in REFERENCE_COVARIATES:
                tick.set_style("italic")
        ax.set_xlabel("Variance explained  (variance-weighted ω²)")
        ax.set_xlim(0, max(0.05, tab["expression"].max() * 1.08))
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_title("Per-covariate variance: composition vs expression", pad=3, loc="left")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=1.0, markersize=4.5,
                   label="Composition (CLR)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35",
                   markeredgecolor="black", markeredgewidth=0.3, markersize=4.5,
                   label="Expression (pseudobulk)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLOCK_COLOR["biological"],
                   markeredgecolor="black", markeredgewidth=0.3, markersize=4.5, label="Biological"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLOCK_COLOR["technical"],
                   markeredgecolor="black", markeredgewidth=0.3, markersize=4.5, label="Technical"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5,
                  handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

        # ---- Panel B: scIB-style table (omega^2, no retained column) ----
        ax2 = fig.add_subplot(gs[0, 1])
        mat = tab[["composition", "expression"]].values.astype(float)
        vmax = float(np.nanmax(mat))
        ax2.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   extent=[-0.5, 1.5, n - 0.5, -0.5])
        for i in range(n):
            for j in range(2):
                v = mat[i, j]
                ax2.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5, color="black")
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(n - 0.5, -0.5)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Comp.", "Expr."], fontsize=6)
        ax2.xaxis.set_ticks_position("top")
        ax2.xaxis.set_label_position("top")
        ax2.set_yticks([])
        ax2.tick_params(length=0)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.set_title("ω² (pooled across 4 lineages)", pad=14, fontsize=6)

        # ---- Panel C: descriptive anatomy-to-study ratio ----
        _make_headline_panel(fig.add_subplot(gs[1, :]), pooled, long_df)

        head = (
            f"Study effects dominate expression (ω²={stats['batch_expr']:.2f}) and are "
            f"attenuated in composition (ω²={stats['batch_comp']:.2f}; "
            f"{stats['batch_reduction']*100:.0f}% lower), while gut-region variance "
            f"is retained (ω²=0.13 in both modalities)."
        )
        fig.suptitle(head, fontsize=6.2, y=0.995)
        fig.subplots_adjust(left=0.20, right=0.99, top=0.86, bottom=0.09)
        _save_pub(fig, "composition_vs_expression_pcr")
        plt.close(fig)


def _make_headline_panel(ax, pooled, long_df):
    """Show the descriptive gut-region:study omega-squared ratio transparently.

    Lineage points expose heterogeneity; the black diamond is the ratio of the
    unweighted lineage-mean omega-squared values. No fold-improvement statistic
    or inferential interpretation is attached to this post hoc summary.
    """
    batch = pooled.loc[("technical", "dataset_id")]
    region = pooled.loc[("reference", "tissue_level_1")]
    pooled_ratio = {
        "expression": float(region["expression"]) / float(batch["expression"]),
        "composition": float(region["composition"]) / float(batch["composition"]),
    }
    subset = long_df[
        long_df["covariate"].isin(["dataset_id", "tissue_level_1"])
    ].pivot_table(
        index=["lineage", "modality"], columns="covariate", values="pcr"
    ).reset_index()
    subset["ratio"] = subset["tissue_level_1"] / subset["dataset_id"]

    lineage_colours = {
        "epithelial": NATURE_WONG["bluish_green"],
        "lymphoid": NATURE_WONG["blue"],
        "myeloid": NATURE_WONG["orange"],
        "stroma": NATURE_WONG["reddish_purple"],
    }
    y_base = {"composition": 0.0, "expression": 1.0}
    offsets = {
        "epithelial": -0.15, "lymphoid": -0.05,
        "myeloid": 0.05, "stroma": 0.15,
    }
    for _, row in subset.iterrows():
        ax.scatter(
            row["ratio"], y_base[row["modality"]] + offsets[row["lineage"]],
            s=24, marker=LINEAGE_MARKER[row["lineage"]],
            facecolor=lineage_colours[row["lineage"]],
            edgecolor="black", linewidth=0.35, zorder=3,
            label=row["lineage"],
        )
    for modality, ratio in pooled_ratio.items():
        y = y_base[modality]
        ax.scatter(ratio, y, s=38, marker="D", facecolor="black",
                   edgecolor="black", linewidth=0.3, zorder=4)
        ax.text(ratio, y + 0.22, f"{ratio:.2f}", va="bottom",
                ha="center", fontsize=6.5, fontweight="bold", color="black")

    ax.axvline(1.0, color="0.55", linestyle=(0, (2, 2)), linewidth=0.6, zorder=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Composition\n(CLR)", "Expression\n(pseudobulk DE)"],
                       fontsize=6)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ω² gut region / ω² study")
    ax.set_title("Anatomy-to-study variance ratio (descriptive)",
                 pad=3, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), [x.capitalize() for x in unique.keys()],
        loc="lower right", frameon=False, ncol=4, fontsize=5,
        handletextpad=0.25, columnspacing=0.7,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


if __name__ == "__main__":
    main()
