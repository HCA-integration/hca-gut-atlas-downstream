#!/usr/bin/env python
"""
Map the 4 CCC concepts (+ venular ACKR1 sink / MADCAM1 addressin) onto the
TAURUS patient (patpy) map and test association with disease / inflammation /
treatment / remission.

Signal definition per sample (pseudobulk, sender x receiver co-expression):
  For an axis Ligand(sender_ct) -> Receptor(receiver_ct), the per-sample score is
     score = meanexpr(ligand | sender cells in sample) *
             meanexpr(receptor| receiver cells in sample)
  (0 if the sample lacks either cell type). This mirrors the LIANA lr_means
  magnitude at the sample level, so it is directly comparable across patients.

Patient map = precomputed patpy scVI-pseudobulk embedding (216 TAURUS samples,
10 dims) -> 2D UMAP for display. Association tests: Mann-Whitney U on the
sample score between clinical groups (healthy vs UC vs CD; inflamed vs not;
remission vs not; pre vs post treatment).

Outputs -> <OUT>/concept_validation/patientmap/
  sample_scores.csv
  association_stats.csv
  fig_patientmap_<concept>.png   (map coloured by score + boxplots by group)
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse, stats
import matplotlib.pyplot as plt
import gca_plot_style as gps
gps.set_style()
warnings.filterwarnings("ignore")

TAURUS = "/Users/kylekimler/Projects/SETA_paper/taurus/data/TAURUS_raw_counts_annotated_final_annotated_v1.h5ad"
# CLR compositional patient map (cell-type composition, centred-log-ratio) -- the
# map that reflects tissue remodelling rather than pseudobulk expression.
EMB = "/Users/kylekimler/Projects/patient-maps-playground/experiments/results/pipeline_taurus_full/SETA_CLR_embeddings.npy"
EMB_LABEL = "CLR compositional"
META = "/Users/kylekimler/Projects/patient-maps-playground/experiments/results/pipeline_taurus_full/sample_metadata.csv"
OUT = "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA/concept_validation/patientmap"
os.makedirs(OUT, exist_ok=True)
CT_KEY = "predicted_hgca_celltype_v1"
CMAP = gps.SEQ

# axis definitions: (name, ligand, sender_regex, receptor, receiver_regex)
AXES = {
 "C1_serotonin_EC_to_ISC":  ("TPH1","EEC Enterochromaffin","HTR4","Intestinal Stem Cells|Transiently Amplifying"),
 "C1_guanylin_EC_to_BEST4": ("GUCA2B","EEC Enterochromaffin|EEC L|EEC N","GUCY2C","BEST4"),
 "C2_checkpoint_EEC_to_CD8":("PCSK1N","EEC","GPR171","CD8"),
 "C3_tryptase_Mast_to_Epi": ("TPSAB1","Mast Cells","F2RL1","Goblet|Enterocyte|BEST4|Paneth"),
 "C4_glia_hub":             ("NRXN1","Glia","NLGN1","Smooth Muscle|Myofibro"),
 "pillar_ACKR1_sink":       ("CCL21","Medullary Sinus|Lymphatic Endothelial","ACKR1","Venular Endothelial|Pre Venule"),
 "pillar_MADCAM1_addressin":("CCL21","Medullary Sinus|Lymphatic Endothelial","MADCAM1","Venular Endothelial|Medullary Sinus"),
 # C6 perivascular gut-wall wiring (fibrosis / neovascular)
 "C6_perivascular_PDGF":    ("PDGFA","Glia|Fibroblast|Reticular","PDGFRA","Fibroblast|Reticular"),
 "C6_perivascular_NOTCH":   ("JAG1","Arteriolar Endothelial|Capillary Endothelial","NOTCH3","Pericyte|Smooth Muscle|Myofibro"),
 # C7 chemokine recruitment amplifier (pairs with the ACKR1 sink above)
 "C7_chemokine_recruit":    ("CXCL1","Fibroblast|Reticular|Monocyte|Macrophage","CXCR2","Neutrophil"),
}
GENES = sorted({g for a in AXES.values() for g in (a[0], a[2])})

def load():
    print("loading TAURUS backed ...", flush=True)
    A = ad.read_h5ad(TAURUS, backed="r")
    sym = A.var["gene_symbol"].astype(str).values
    gi = {g:int(np.where(sym==g)[0][0]) for g in GENES if g in set(sym)}
    obs = A.obs[[ "sample_id", CT_KEY, "total_counts"]].copy()
    obs.columns=["sample","ct","tot"]
    n=A.n_obs; chunk=60000; mats=[]
    cols=[gi[g] for g in GENES if g in gi]
    for s in range(0,n,chunk):
        e=min(s+chunk,n); x=A.X[s:e]
        if not sparse.issparse(x): x=sparse.csr_matrix(x)
        mats.append(x[:,cols])
    X=sparse.vstack(mats).tocsr()
    tot=obs["tot"].values.astype(float); tot[tot<=0]=1
    Xn=X.multiply(1e4/tot[:,None]).tocsr(); Xn.data=np.log1p(Xn.data)
    genes_present=[g for g in GENES if g in gi]
    print("  matrix", Xn.shape, "genes", len(genes_present), flush=True)
    return Xn, obs.reset_index(drop=True), genes_present

MIN_CT_CELLS = 5   # min sender/receiver cells for a conditional per-cell intensity

def axis_components(Xn, obs, gidx, axis, samples, total):
    """Per-sample decomposition of an LR axis into abundance vs per-cell intensity.

    Returns a DataFrame indexed by sample with:
      n_sender/n_receiver      cell counts of sender/receiver populations
      sender_frac/receiver_frac fraction of the sample that is sender/receiver
      lig_int / rec_int        mean log-norm ligand (in senders) / receptor (in
                               receivers); NaN when that population is ABSENT
      score                    lig_int * rec_int (structural 0 when a pop absent)
    This lets us ask: does per-cell signalling change *beyond* what shifting
    cell-type abundance (fewer epithelial cells in sick tissue) already explains?
    """
    lig, srx, rec, rrx = axis
    def per_sample(ct_regex, gene):
        mask = obs["ct"].astype(str).str.contains(ct_regex, regex=True).values
        idx = np.where(mask)[0]
        sub = obs.iloc[idx]
        vals = np.asarray(Xn[idx][:, gidx[gene]].todense()).ravel()
        d = pd.DataFrame({"sample": sub["sample"].values, "v": vals})
        g = d.groupby("sample")
        return g["v"].mean(), g.size()
    lig_mean, n_send = per_sample(srx, lig)
    rec_mean, n_recv = per_sample(rrx, rec)
    out = pd.DataFrame(index=samples)
    out["total"] = total.reindex(samples).fillna(0)
    out["n_sender"] = n_send.reindex(samples).fillna(0)
    out["n_receiver"] = n_recv.reindex(samples).fillna(0)
    out["sender_frac"] = out["n_sender"] / out["total"].replace(0, np.nan)
    out["receiver_frac"] = out["n_receiver"] / out["total"].replace(0, np.nan)
    out["lig_int"] = lig_mean.reindex(samples)
    out["rec_int"] = rec_mean.reindex(samples)
    # conditional per-cell intensity: only meaningful with >= MIN_CT_CELLS cells
    out.loc[out["n_sender"] < MIN_CT_CELLS, "lig_int"] = np.nan
    out.loc[out["n_receiver"] < MIN_CT_CELLS, "rec_int"] = np.nan
    out["score"] = out["lig_int"].fillna(0) * out["rec_int"].fillna(0)
    return out

Xn, obs, genes = load()
gidx = {g:i for i,g in enumerate(genes)}
samples = sorted(obs["sample"].unique())
total_cells = obs.groupby("sample").size()
COMP = {}                       # per-concept decomposition tables
score_df = pd.DataFrame(index=samples)
for name,(lig,srx,rec,rrx) in AXES.items():
    if lig not in gidx or rec not in gidx:
        print("skip",name,"(gene missing)"); continue
    comp = axis_components(Xn, obs, gidx, (lig,srx,rec,rrx), samples, total_cells)
    COMP[name] = comp
    score_df[name] = comp["score"].values
score_df.index.name="sample_id"

# ---- attach metadata + CLR compositional embedding ----
meta = pd.read_csv(META, index_col=0)
meta = meta.drop_duplicates("sample_id").set_index("sample_id")
emb = np.load(EMB)  # (216, k) aligned to sample_metadata row order
meta = meta.iloc[:emb.shape[0]].copy()
# 2D display via UMAP of the CLR embedding (fallback to first 2 dims)
try:
    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(emb)
except Exception as ex:
    print("umap unavailable, using CLR dims 0/1:", ex); xy = emb[:, :2]
meta["x"], meta["y"] = xy[:, 0], xy[:, 1]
# disease x treatment stratum label
meta["stratum"] = np.where(meta["Disease"].astype(str) == "Healthy", "Healthy",
                           meta["Disease"].astype(str) + "_" + meta["Treatment"].astype(str))
df = meta.join(score_df, on="sample_id")
df.to_csv(f"{OUT}/sample_scores.csv")

# ------------------------------------------------------------------ stats
# Stratified: disease/pre/post-specific. Each comparison is run WITHIN a
# clinical context so that a disease/treatment-specific effect is not diluted.
# Two problems the naive Mann-Whitney ignores and that we now fix:
#   (1) NON-INDEPENDENCE: 216 samples come from 41 patients (median 5/patient),
#       so samples are pseudo-replicates -> patient random intercept.
#   (2) COMPOSITION CONFOUND: sick tissue has fewer epithelial (and rarer EEC)
#       cells, so a lower composite score can be pure abundance. We therefore
#       (a) covariate-adjust the score for sender/receiver fraction, and
#       (b) decompose into abundance (sender/receiver fraction) vs per-cell
#           intensity (ligand-in-senders, receptor-in-receivers) and test each.
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

def mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p, -(1 - 2 * u / (len(a) * len(b)))  # +ve => group A > group B

def mixed_p(data, yvar, covars):
    """Mixed model  y ~ grp (+covars)  with patient random intercept.
    Returns (p_grp, beta_grp, n_used, n_patients). Covariates z-scored."""
    d = data[[yvar, "grp", "Patient"] + covars].dropna().copy()
    if d["grp"].nunique() < 2 or d["Patient"].nunique() < 3 or len(d) < 8:
        return np.nan, np.nan, len(d), d["Patient"].nunique()
    for c in covars:
        sd = d[c].std()
        d[c] = (d[c] - d[c].mean()) / sd if sd > 0 else 0.0
    formula = yvar + " ~ grp" + "".join(f" + {c}" for c in covars)
    try:
        m = smf.mixedlm(formula, d, groups=d["Patient"]).fit(reml=False, method="lbfgs")
        return float(m.pvalues.get("grp", np.nan)), float(m.params.get("grp", np.nan)), \
               len(d), d["Patient"].nunique()
    except Exception as e:
        return np.nan, np.nan, len(d), d["Patient"].nunique()

def sub_mask(colvals):
    m = np.ones(len(df), bool)
    for col, val in colvals:
        m &= (df[col].astype(str) == val).values
    return m

# (context_label, contrast_label, filter, split_col, groupA, groupB)
COMPARISONS = []
# 1) disease onset at baseline (Pre): UC/CD-Pre vs Healthy
for d in ["UC", "CD"]:
    COMPARISONS.append((f"{d}_Pre", f"{d}_Pre vs Healthy",
                        [], "stratum", f"{d}_Pre", "Healthy"))
# 2) inflammation within each disease x treatment stratum
for d in ["UC", "CD"]:
    for t in ["Pre", "Post"]:
        COMPARISONS.append((f"{d}_{t}", "Inflamed vs Non-inflamed",
                            [("Disease", d), ("Treatment", t)],
                            "Inflammation", "Inflamed", "Non_Inflamed"))
# 3) treatment response = remission vs non-remission within POST, per disease
for d in ["UC", "CD"]:
    COMPARISONS.append((f"{d}_Post", "Non-remission vs Remission (response)",
                        [("Disease", d), ("Treatment", "Post")],
                        "Remission_status", "Non_Remission", "Remission"))
# 4) treatment effect = Post vs Pre within each disease
for d in ["UC", "CD"]:
    COMPARISONS.append((d, "Post vs Pre (treatment effect)",
                        [("Disease", d)], "Treatment", "Post", "Pre"))

rows = []
for name in score_df.columns:
    comp = COMP[name].join(df[["Patient", "Disease", "Treatment", "Inflammation",
                               "Remission_status", "stratum"]])
    for ctx, contrast, filt, split_col, ga, gb in COMPARISONS:
        m = sub_mask(filt)
        idx = df.index[m]
        sub = comp.loc[comp.index.intersection(idx)].copy()
        sub = sub[sub[split_col].astype(str).isin([ga, gb])]
        sub["grp"] = (sub[split_col].astype(str) == ga).astype(int)  # ga=1
        a = sub.loc[sub["grp"] == 1, "score"].dropna()
        b = sub.loc[sub["grp"] == 0, "score"].dropna()
        p_naive, eff = mwu(a, b)
        # (1)+(2): patient-random-intercept, composition-adjusted score
        p_adj, beta_adj, n_adj, npat = mixed_p(sub, "score", ["sender_frac", "receiver_frac"])
        # decomposition: does abundance itself shift? does per-cell intensity shift?
        p_sf, b_sf, *_ = mixed_p(sub, "sender_frac", [])
        p_rf, b_rf, *_ = mixed_p(sub, "receiver_frac", [])
        p_li, b_li, n_li, _ = mixed_p(sub, "lig_int", [])
        p_ri, b_ri, n_ri, _ = mixed_p(sub, "rec_int", [])
        rows.append((name, ctx, contrast,
                     round(a.mean(), 4) if len(a) else np.nan,
                     round(b.mean(), 4) if len(b) else np.nan,
                     len(a), len(b), npat, eff, p_naive,
                     p_adj, beta_adj, p_sf, b_sf, p_rf, b_rf,
                     p_li, b_li, n_li, p_ri, b_ri, n_ri))
stat = pd.DataFrame(rows, columns=[
    "concept", "context", "contrast", "mean_groupA", "mean_groupB", "nA", "nB",
    "n_patients", "effect_rankbiserial", "p_naive_mwu",
    "p_score_adj", "beta_score_adj", "p_sender_frac", "beta_sender_frac",
    "p_receiver_frac", "beta_receiver_frac", "p_lig_intensity", "beta_lig_intensity",
    "n_lig_int", "p_rec_intensity", "beta_rec_intensity", "n_rec_int"])
# BH across the primary composition-adjusted test
stat["p_naive_BH"] = np.nan
stat["p_score_adj_BH"] = np.nan
ok = stat["p_naive_mwu"].notna()
stat.loc[ok, "p_naive_BH"] = multipletests(stat.loc[ok, "p_naive_mwu"], method="fdr_bh")[1]
ok2 = stat["p_score_adj"].notna()
stat.loc[ok2, "p_score_adj_BH"] = multipletests(stat.loc[ok2, "p_score_adj"], method="fdr_bh")[1]
stat.to_csv(f"{OUT}/association_stats.csv", index=False)

print("\n=== NAIVE (samples independent) vs COMPOSITION-ADJUSTED MIXED MODEL ===")
show = ["concept", "context", "contrast", "nA", "nB", "n_patients",
        "p_naive_mwu", "p_naive_BH", "p_score_adj", "p_score_adj_BH",
        "p_sender_frac", "p_lig_intensity", "p_rec_intensity"]
comp_tbl = stat.sort_values("p_naive_mwu")[show].head(10)
with pd.option_context("display.width", 220, "display.max_columns", 30):
    print(comp_tbl.to_string(index=False))
print("\nSurvives composition-adjusted mixed model (BH<0.05):")
srv = stat[stat["p_score_adj_BH"] < 0.05].sort_values("p_score_adj_BH")
print(srv[["concept","context","contrast","p_naive_mwu","p_score_adj",
           "p_score_adj_BH","p_lig_intensity","p_rec_intensity"]].to_string(index=False)
      if len(srv) else "  (none)")

# ------------------------------------------------------------------ figures
STRATA = ["Healthy", "UC_Pre", "UC_Post", "CD_Pre", "CD_Post"]
STRAT_COL = {"Healthy": gps.WONG["green"], "UC_Pre": "#F4B183", "UC_Post": "#D55E00",
             "CD_Pre": "#9DC3E6", "CD_Post": "#0072B2"}
INFLAM = ["Non_Inflamed", "Inflamed"]

def box(ax, groups, data, colors, title, positions=None, width=0.62):
    if positions is None:
        positions = list(range(len(groups)))
    bp = ax.boxplot(data, positions=positions, showfliers=False,
                    widths=width, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i]); patch.set_edgecolor("#333333")
        patch.set_linewidth(0.5); patch.set_alpha(0.55)
    for el in ["whiskers", "caps", "medians"]:
        for ln in bp[el]:
            ln.set_color("#333333"); ln.set_linewidth(0.5)
    for i, d in enumerate(data):
        if len(d):
            ax.scatter(np.random.normal(positions[i], 0.07, len(d)), d, s=3,
                       c="#222222", alpha=0.55, edgecolors="none", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title(title)
    gps.open_axes(ax)

for name in score_df.columns:
    fig = plt.figure(figsize=(180 * gps.MM, 62 * gps.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.0], wspace=0.42)
    # panel a: CLR map coloured by score
    ax0 = fig.add_subplot(gs[0, 0])
    v = df[name].values; vmax = np.nanpercentile(v, 98) or 1
    sc = ax0.scatter(df["x"], df["y"], c=v, cmap=CMAP, vmin=0, vmax=vmax, s=14,
                     edgecolors="white", linewidths=0.25)
    ax0.set_title(f"a  {EMB_LABEL} map")
    ax0.set_xlabel("UMAP 1"); ax0.set_ylabel("UMAP 2")
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    cb = fig.colorbar(sc, ax=ax0, fraction=0.045, pad=0.02)
    cb.set_label("Sample LR score"); cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5)
    # panel b: score across disease x treatment strata
    axb = fig.add_subplot(gs[0, 1])
    data_b = [df.loc[df["stratum"] == s, name].dropna().values for s in STRATA]
    box(axb, STRATA, data_b, [STRAT_COL[s] for s in STRATA], "b  Disease × treatment")
    axb.set_ylabel("Sample LR score")
    # panel c: inflamed vs non within each disease x treatment stratum
    # grouped positions: pairs spaced tighter than strata to avoid label collisions
    axc = fig.add_subplot(gs[0, 2])
    strata_c = ["UC_Pre", "UC_Post", "CD_Pre", "CD_Post"]
    groups_c, data_c, cols_c, pos_c = [], [], [], []
    for si, s in enumerate(strata_c):
        base = si * 2.6
        for j, inf in enumerate(INFLAM):
            sel = (df["stratum"] == s) & (df["Inflammation"].astype(str) == inf)
            groups_c.append("Non" if inf == "Non_Inflamed" else "Inf")
            data_c.append(df.loc[sel, name].dropna().values)
            cols_c.append("#DDDDDD" if inf == "Non_Inflamed" else STRAT_COL[s])
            pos_c.append(base + j)
    box(axc, groups_c, data_c, cols_c, "c  Inflammation within stratum",
        positions=pos_c, width=0.72)
    axc.tick_params(axis="x", labelsize=5)
    for si, s in enumerate(strata_c):
        axc.text(si * 2.6 + 0.5, -0.24, s.replace("_", " "),
                 transform=axc.get_xaxis_transform(), ha="center", va="top",
                 fontsize=5.5, fontweight="bold")
    fig.suptitle(f"{name}   ({EMB_LABEL} patient map, TAURUS)", fontsize=7,
                 x=0.01, ha="left", fontweight="bold", y=1.03)
    gps.save(fig, f"{OUT}/fig_patientmap_{name}")
    print("wrote map fig", name)

# ------------------------------------------------------------------ decomposition figures
# Abundance vs per-cell intensity across strata: shows whether a score change is
# "fewer cells" (fractions) or genuine "more signalling per cell" (intensities).
def strat_box(ax, comp, col, ylabel, title, conditional=False):
    data = []
    for s in STRATA:
        vals = comp.loc[comp["stratum"] == s, col]
        data.append(vals.dropna().values)
    box(ax, STRATA, data, [STRAT_COL[s] for s in STRATA], title)
    ax.set_ylabel(ylabel)

for name in score_df.columns:
    lig, srx, rec, rrx = AXES[name]
    comp = COMP[name].join(df[["stratum"]])
    fig = plt.figure(figsize=(180 * gps.MM, 44 * gps.MM))
    gs = fig.add_gridspec(1, 5, wspace=0.6)
    strat_box(fig.add_subplot(gs[0, 0]), comp, "sender_frac",
              "Sender fraction", "a  Sender abundance")
    strat_box(fig.add_subplot(gs[0, 1]), comp, "lig_int",
              f"{lig} log-norm", f"b  {lig} / cell")
    strat_box(fig.add_subplot(gs[0, 2]), comp, "receiver_frac",
              "Receiver fraction", "c  Receiver abundance")
    strat_box(fig.add_subplot(gs[0, 3]), comp, "rec_int",
              f"{rec} log-norm", f"d  {rec} / cell")
    strat_box(fig.add_subplot(gs[0, 4]), comp, "score",
              "Composite LR score", "e  Composite")
    fig.suptitle(f"{name}: abundance vs per-cell intensity  "
                 f"(per-cell panels b, d require ≥ {MIN_CT_CELLS} cells; boxes = "
                 f"Healthy / UC Pre / UC Post / CD Pre / CD Post)", fontsize=6.5,
                 x=0.01, ha="left", fontweight="bold", y=1.08)
    gps.save(fig, f"{OUT}/fig_decomp_{name}")
    print("wrote decomp fig", name)
print("\nDONE ->", OUT)
