#!/usr/bin/env python3
"""Stage 4: publication figures for the Taurus sample-embedding analysis.

Follows ~/Projects/GCA/publication2026/plot_specs.md (Helvetica 5-7 pt, no
gridlines, open L-shaped axes, Wong palette, vector PDF/SVG + 300 dpi PNG).

Panels
------
  latent_<group>_by_state         - MDS of every representation, coloured by
                                     disease state (the "latent space images")
  latent_pretx_response           - pre-treatment samples coloured by response,
                                     across composition / MOFA / GloScope
  predict_auc_heatmap             - representation x tissue, pre-tx response AUC
  predict_roc_<disease>           - ROC of pooled logreg / KNN per disease
  predict_stratified_auc          - AUC by Site / Inflammation stratum
  predict_celltype_coef           - cell types driving non-remission (CD, UC)
"""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

import _common as C

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.edgecolor": "black", "axes.titlesize": 7, "axes.labelsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "figure.dpi": 150,
})
MM = 1 / 25.4


def log(m):
    print(f"[stage4] {m}", flush=True)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, base, w_mm, h_mm):
    fig.set_size_inches(w_mm * MM, h_mm * MM)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    log(f"  wrote {base}.{{pdf,svg,png}}")


# --------------------------------------------------------------------------
def load_group(group):
    mpath = C.group_meta_path(group)
    if not mpath.exists():
        return None
    meta = pd.read_csv(mpath, index_col=0)
    meta.index = meta.index.astype(str)
    return meta


def load_embedding(group, method):
    p = C.repr_embedding_path(group, method, "mds")
    if not p.exists():
        return None
    e = pd.read_csv(p, index_col=0)
    e.index = e.index.astype(str)
    return e


def scatter_states(ax, emb, meta, title):
    states = meta.loc[emb.index, "state"].astype(str)
    for st in C.STATE_LEVELS:
        m = states == st
        if not m.any():
            continue
        ax.scatter(emb.loc[m.values, "MDS1"], emb.loc[m.values, "MDS2"],
                   s=14, c=C.STATE_COLORS[st], edgecolors="white",
                   linewidths=0.3, label=C.STATE_LABELS[st], zorder=3)
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.set_xticks([]); ax.set_yticks([])
    despine(ax)


# --------------------------------------------------------------------------
def fig_latent_by_state(group):
    meta = load_group(group)
    if meta is None:
        return
    methods = [m for m in C.REPR_ORDER if load_embedding(group, m) is not None]
    if not methods:
        return
    ncol = 4
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(180 * MM, 50 * nrow * MM))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(methods):]:
        ax.axis("off")
    for ax, m in zip(axes, methods):
        emb = load_embedding(group, m)
        scatter_states(ax, emb, meta, C.REPR_DISPLAY[m])
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=C.STATE_COLORS[s], markeredgecolor="white",
                      label=C.STATE_LABELS[s])
               for s in C.STATE_LEVELS
               if (meta["state"].astype(str) == s).any()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Taurus {group.replace('_', ' ')} - sample embeddings by disease state",
                 fontsize=8, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_fig(fig, str(C.OUT / f"latent_{group}_by_state"), 180, 50 * nrow + 12)


def fig_pretx_response():
    """Pre-treatment samples coloured by eventual response, key methods."""
    methods = ["composition", "mofa", "gloscope"]
    groups = ["Ileum_CD", "Colon_CD", "Rectum_CD", "Colon_UC", "Rectum_UC"]
    groups = [g for g in groups if load_group(g) is not None]
    fig, axes = plt.subplots(len(groups), len(methods),
                             figsize=(len(methods) * 45 * MM, len(groups) * 45 * MM))
    axes = np.atleast_2d(axes)
    for i, g in enumerate(groups):
        meta = load_group(g)
        pre = meta[(meta[C.TREATMENT_KEY].astype(str) == "Pre")
                   & (meta[C.REMISSION_KEY].astype(str).isin(["Remission", "Non_Remission"]))]
        for j, m in enumerate(methods):
            ax = axes[i, j]
            emb = load_embedding(g, m)
            if emb is None or pre.empty:
                ax.axis("off"); continue
            idx = [s for s in pre.index if s in emb.index]
            sub = emb.loc[idx]
            resp = pre.loc[idx, C.REMISSION_KEY].astype(str)
            for lab, col in C.RESPONSE_COLORS.items():
                mm = resp == lab
                if mm.any():
                    ax.scatter(sub.loc[mm.values, "MDS1"], sub.loc[mm.values, "MDS2"],
                               s=16, c=col, edgecolors="white", linewidths=0.3,
                               label=lab.replace("_", "-"))
            ax.set_xticks([]); ax.set_yticks([]); despine(ax)
            if i == 0:
                ax.set_title(C.REPR_DISPLAY[m], fontsize=7, fontweight="bold")
            if j == 0:
                ax.set_ylabel(g.replace("_", " "), fontsize=7, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=c, markeredgecolor="white", label=l.replace("_", "-"))
               for l, c in C.RESPONSE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Pre-treatment sample embeddings coloured by eventual response",
                 fontsize=8, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_fig(fig, str(C.OUT / "latent_pretx_response"),
             len(methods) * 48, len(groups) * 48 + 14)


# --------------------------------------------------------------------------
def fig_auc_heatmap():
    df = pd.read_csv(C.DATA / "predict_representation_knn.csv")
    if df.empty:
        return
    df["rep"] = df["representation"].map(C.REPR_DISPLAY)
    piv = df.pivot_table(index="rep", columns="group", values="roc_auc")
    piv = piv.reindex([C.REPR_DISPLAY[m] for m in C.REPR_ORDER]).dropna(how="all")
    n_df = df.pivot_table(index="group", values="n", aggfunc="first")

    fig, ax = plt.subplots(figsize=(120 * MM, 70 * MM))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(piv.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c.replace('_', ' ')}\n(n={int(n_df.loc[c, 'n'])})"
                        for c in piv.columns], fontsize=6)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if (v > 0.78 or v < 0.22) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("ROC-AUC (leave-one-out KNN)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.5)
    ax.set_title("Pre-treatment response prediction by representation and tissue",
                 fontsize=7, fontweight="bold", loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save_fig(fig, str(C.OUT / "predict_auc_heatmap"), 130, 75)


def fig_roc():
    oof = pd.read_csv(C.DATA / "predict_oof_predictions.csv")
    if oof.empty:
        return
    model_cols = {"logreg": C.WONG["blue"], "knn": C.WONG["vermillion"]}
    for disease in oof["disease"].unique():
        fig, ax = plt.subplots(figsize=(60 * MM, 60 * MM))
        ax.plot([0, 1], [0, 1], color=C.WONG["grey"], lw=0.5, ls="--")
        for model, col in model_cols.items():
            d = oof[(oof.disease == disease) & (oof.model == model)]
            if d.empty or d["y_true"].nunique() < 2:
                continue
            y = (d["y_true"].astype(str) == "Non_Remission").astype(int)
            fpr, tpr, _ = roc_curve(y, d["p_nonremission"])
            auc = roc_auc_score(y, d["p_nonremission"])
            ax.plot(fpr, tpr, color=col, lw=1.2,
                    label=f"{model} (AUC = {auc:.2f})")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"{disease} pre-treatment response", fontsize=7,
                     fontweight="bold", loc="left")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); despine(ax)
        ax.legend(frameon=False, loc="lower right", fontsize=5.5)
        fig.tight_layout()
        save_fig(fig, str(C.OUT / f"predict_roc_{disease}"), 65, 65)


def fig_stratified():
    df = pd.read_csv(C.DATA / "predict_metrics_stratified.csv")
    if df.empty:
        return
    df = df[df["model"] == "logreg"].copy()
    df["label"] = np.where(df["stratum_var"] == "overall", "Overall",
                           df["stratum"].astype(str).str.replace("_", " "))
    fig, axes = plt.subplots(1, df["disease"].nunique(),
                             figsize=(140 * MM, 55 * MM), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, disease in zip(axes, sorted(df["disease"].unique())):
        d = df[df.disease == disease]
        order = d.drop_duplicates("label")
        colors = [C.WONG["grey"] if lv == "overall" else C.WONG["blue"]
                  for lv in order["stratum_var"]]
        ax.bar(range(len(order)), order["roc_auc"].values, color=colors,
               edgecolor="black", linewidth=0.4, width=0.7)
        ax.axhline(0.5, color=C.WONG["vermillion"], lw=0.6, ls="--")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order["label"], rotation=40, ha="right", fontsize=5.5)
        ax.set_title(f"{disease}", fontsize=7, fontweight="bold", loc="left")
        ax.set_ylim(0, 1); despine(ax)
        ax.text(0.02, 0.92, "chance", color=C.WONG["vermillion"], fontsize=5,
                transform=ax.transAxes)
    axes[0].set_ylabel("ROC-AUC (logreg, LOPO-CV)")
    fig.suptitle("Pre-treatment response prediction, stratified",
                 fontsize=8, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, str(C.OUT / "predict_stratified_auc"), 150, 60)


def fig_coefficients(top_n=12):
    df = pd.read_csv(C.DATA / "predict_logreg_celltype_coef.csv")
    if df.empty:
        return
    for disease in df["disease"].unique():
        d = df[df.disease == disease].sort_values("coef_nonremission")
        sel = pd.concat([d.head(top_n), d.tail(top_n)])
        colors = [C.WONG["blue"] if v < 0 else C.WONG["vermillion"]
                  for v in sel["coef_nonremission"]]
        fig, ax = plt.subplots(figsize=(90 * MM, (0.32 * len(sel) + 1) * 10 * MM))
        ax.barh(range(len(sel)), sel["coef_nonremission"].values, color=colors,
                edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels(sel["celltype"], fontsize=5.5)
        ax.set_xlabel("Logistic coefficient  (+ = higher in non-remission)")
        ax.set_title(f"{disease}: cell types predictive of pre-treatment response",
                     fontsize=7, fontweight="bold", loc="left")
        despine(ax)
        handles = [Line2D([0], [0], marker="s", linestyle="", markersize=5,
                          markerfacecolor=C.WONG["vermillion"], markeredgecolor="black",
                          label="Higher in non-remission"),
                   Line2D([0], [0], marker="s", linestyle="", markersize=5,
                          markerfacecolor=C.WONG["blue"], markeredgecolor="black",
                          label="Higher in remission")]
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5)
        fig.tight_layout()
        save_fig(fig, str(C.OUT / f"predict_celltype_coef_{disease}"),
                 95, 0.32 * len(sel) * 10 + 22)


# --------------------------------------------------------------------------
def main():
    for _, _, label in C.GROUPS:
        fig_latent_by_state(label)
    fig_pretx_response()
    fig_auc_heatmap()
    fig_roc()
    fig_stratified()
    fig_coefficients()
    log("done.")


if __name__ == "__main__":
    main()
