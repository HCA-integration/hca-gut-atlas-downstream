#!/usr/bin/env python3
"""Publication-readable diagnostic modules for reference-uncertainty review.

Focus: epithelial C call, with stroma/myeloid context.
Uses Wong palette matching fig_identity_state_hgca_pangi_v1.
Every figure includes a plain-language 'Reader should learn' caption outside the axes.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGURES, TABLES  # noqa: E402

LOGGER = logging.getLogger("plot_ru")

HGCA = "#0072B2"
PANGI = "#009E73"
GREY = "#666666"
ATLAS_COLORS = {"HGCA": HGCA, "PanGI": PANGI}

mpl.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titlesize": 9,
        "figure.dpi": 150,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.35)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.35)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white", pad_inches=0.35)
    plt.close(fig)
    LOGGER.info("Wrote figures/%s.{png,pdf,svg}", stem)


def _caption(fig: plt.Figure, text: str, y: float = -0.04) -> None:
    """Place caption below axes (negative y in figure coords after tight layout)."""
    fig.text(
        0.5,
        y,
        f"Reader should learn: {text}",
        ha="center",
        va="top",
        fontsize=7,
        color=GREY,
        wrap=True,
        transform=fig.transFigure,
    )


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _load_depth(lineage: str) -> pd.DataFrame:
    if lineage == "stroma":
        p = TABLES / "stable_naming_depth_cells.parquet"
    else:
        p = TABLES / f"{lineage}_stable_naming_depth_cells.parquet"
    df = pd.read_parquet(p)
    return df[np.isclose(df["tau"].astype(float), 0.9)].copy()


def _load_disp(lineage: str) -> pd.DataFrame:
    if lineage == "stroma":
        return pd.read_parquet(TABLES / "sample_aitchison_displacement.parquet")
    return pd.read_parquet(TABLES / f"{lineage}_sample_aitchison_displacement.parquet")


def _load_disp_summary(lineage: str) -> pd.DataFrame:
    if lineage == "stroma":
        return pd.read_csv(TABLES / "sample_aitchison_displacement_summary.csv")
    return pd.read_csv(TABLES / f"{lineage}_sample_aitchison_displacement_summary.csv")


def plot_stable_naming_depth(lineage: str) -> None:
    df = _load_depth(lineage)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.2), sharey=True)
    handles = []
    for ax, atlas in zip(axes, ("HGCA", "PanGI")):
        sub = df.loc[df.atlas == atlas, "stable_naming_depth_norm"].to_numpy()
        ax.hist(sub, bins=25, color=ATLAS_COLORS[atlas], edgecolor="white", linewidth=0.3, range=(0, 1))
        med = float(np.median(sub))
        mean = float(np.mean(sub))
        (h1,) = ax.plot([med, med], [0, 1], color="black", lw=1.0, ls="--", transform=ax.get_xaxis_transform(), label=f"median {med:.2f}")
        (h2,) = ax.plot([mean, mean], [0, 1], color="#D55E00", lw=1.0, ls=":", transform=ax.get_xaxis_transform(), label=f"mean {mean:.2f}")
        handles = [h1, h2]
        ax.set_title(f"{atlas}\nmedian={med:.2f}   mean={mean:.2f}", pad=8, fontsize=9)
        ax.set_xlabel("Stable naming depth (tau=0.90)\n0 = lineage only   1 = leaf")
        ax.set_xlim(0, 1)
        _despine(ax)
    axes[0].set_ylabel(f"TAURUS {lineage} cells")
    fig.suptitle(
        f"{lineage.capitalize()}: how specifically do jackknives agree on cell names?",
        fontsize=10,
        y=0.98,
    )
    fig.legend(handles, ["median", "mean"], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, ncol=2)
    fig.subplots_adjust(top=0.82, bottom=0.28, left=0.10, right=0.98, wspace=0.18)
    _caption(
        fig,
        f"On {lineage}, compare mean (not only median) depth; leaf saturation can hide atlas gaps.",
        y=0.06,
    )
    _save(fig, f"{lineage}_stable_naming_depth_distribution")


def plot_aitchison_violin(lineage: str) -> None:
    df = _load_disp(lineage)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    data = [df.loc[df.atlas == a, "aitchison_to_full_seed0"].to_numpy() for a in ("HGCA", "PanGI")]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.75)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([HGCA, PANGI][i])
        body.set_alpha(0.75)
        body.set_edgecolor("white")
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)
    ymax = max(np.percentile(d, 99) for d in data) * 1.08
    ax.set_ylim(0, ymax)
    for i, a in enumerate(("HGCA", "PanGI"), start=1):
        med = float(np.median(data[i - 1]))
        ax.scatter([i], [med], color="black", s=18, zorder=5)
        ax.text(i, med + 0.04 * ymax, f"{med:.2f}", va="bottom", ha="center", fontsize=7, color=GREY)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["HGCA", "PanGI"])
    ax.set_ylabel("Aitchison distance\n(omit realization vs full/seed0)")
    ax.set_title(
        f"{lineage.capitalize()}: how far do the same samples move\nwhen one shared healthy study is removed?",
        pad=12,
    )
    _despine(ax)
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.16, right=0.96)
    _caption(fig, "Larger distance = more reference-composition sensitivity of sample composition.", y=0.06)
    _save(fig, f"{lineage}_sample_aitchison_displacement")


def plot_paired_study_bars(lineage: str) -> None:
    summ = _load_disp_summary(lineage)
    # column name differs slightly
    dist_col = "median_dist" if "median_dist" in summ.columns else "median_dist"
    if dist_col not in summ.columns:
        # stroma summary uses median_dist from groupby rename
        if "median_dist" not in summ.columns and "aitchison_to_full_seed0" not in summ.columns:
            # sample_aitchison_displacement_summary has median_dist from 04? check
            pass
    if "median_dist" not in summ.columns:
        # stroma file: median_dist from agg in 04 - actually columns are median_dist
        cols = summ.columns.tolist()
        raise KeyError(f"Unexpected columns for {lineage}: {cols}")

    studies = sorted(summ["omitted_study"].astype(str).unique())
    hg = summ[summ.atlas == "HGCA"].set_index("omitted_study")["median_dist"]
    pg = summ[summ.atlas == "PanGI"].set_index("omitted_study")["median_dist"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    # Left: grouped bars
    ax = axes[0]
    x = np.arange(len(studies))
    w = 0.38
    b1 = ax.bar(x - w / 2, [hg.get(s, np.nan) for s in studies], w, color=HGCA, label="HGCA", edgecolor="white", lw=0.3)
    b2 = ax.bar(x + w / 2, [pg.get(s, np.nan) for s in studies], w, color=PANGI, label="PanGI", edgecolor="white", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(studies, rotation=35, ha="right")
    ax.set_ylabel("Median sample Aitchison distance")
    ax.set_title("Paired shared-study omissions", pad=8)
    _despine(ax)

    # Right: PanGI - HGCA deltas
    ax = axes[1]
    delta = np.array([pg.get(s, np.nan) - hg.get(s, np.nan) for s in studies])
    colors = [PANGI if d > 0 else HGCA for d in delta]
    ax.axvline(0, color="#333333", lw=0.8)
    ax.barh(np.arange(len(studies)), delta, color=colors, edgecolor="white", lw=0.3, height=0.7)
    ax.set_yticks(np.arange(len(studies)))
    ax.set_yticklabels(studies)
    ax.set_xlabel("PanGI minus HGCA")
    ax.set_title("Atlas gap per omitted study", pad=8)
    ax.invert_yaxis()
    _despine(ax)

    fig.suptitle(
        f"{lineage.capitalize()}: does every shared-study omit move PanGI more than HGCA?",
        fontsize=10,
        y=0.98,
    )
    fig.legend([b1, b2], ["HGCA", "PanGI"], loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.30, left=0.09, right=0.98, wspace=0.35)
    _caption(
        fig,
        "Positive delta = PanGI more sensitive for that omit; epithelial reverse shows mostly negative deltas.",
        y=0.06,
    )
    _save(fig, f"{lineage}_paired_study_aitchison")


def plot_size_adjusted(lineage: str) -> None:
    if lineage == "stroma":
        size_path = TABLES / "size_adjusted_displacement.json"
        summ = _load_disp_summary(lineage)
        impact = pd.read_csv(ROOT.parent / "manifests" / "stroma_study_omission_impact.csv")
    else:
        size_path = TABLES / f"{lineage}_size_adjusted_displacement.json"
        summ = _load_disp_summary(lineage)
        impact = pd.read_csv(ROOT.parent / "manifests" / f"{lineage}_study_omission_impact.csv")
    size = json.loads(size_path.read_text())

    dist_col = "median_dist" if "median_dist" in summ.columns else None
    if dist_col is None:
        raise KeyError(summ.columns)
    m = summ.merge(
        impact.rename(columns={"study": "omitted_study", "frac_lineage": "frac_lineage_removed"})[
            ["atlas", "omitted_study", "frac_lineage_removed"]
        ],
        on=["atlas", "omitted_study"],
        how="left",
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for atlas, color in ATLAS_COLORS.items():
        sub = m[m.atlas == atlas]
        ax.scatter(
            sub["frac_lineage_removed"] * 100,
            sub["median_dist"],
            s=55,
            color=color,
            label=atlas,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(r["omitted_study"]),
                (r["frac_lineage_removed"] * 100, r["median_dist"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=6,
                color=GREY,
                clip_on=False,
            )
    ax.set_xlabel("% of lineage reference removed")
    ax.set_ylabel("Median sample Aitchison distance")
    ax.set_title(
        f"{lineage.capitalize()}: is atlas gap just omission size?\n"
        f"OLS coef_PanGI = {size.get('coef_PanGI', float('nan')):+.2f}",
        pad=10,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0)
    _despine(ax)
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.12, right=0.78)
    _caption(fig, "If points separate vertically at similar omit sizes, atlas geometry (not only mass) matters.", y=0.06)
    _save(fig, f"{lineage}_displacement_vs_omit_size")


def plot_cross_lineage_summary() -> None:
    rows = []
    for lineage in ("stroma", "myeloid", "epithelial"):
        depth = pd.read_csv(
            TABLES / ("stable_naming_depth_summary.csv" if lineage == "stroma" else f"{lineage}_stable_naming_depth_summary.csv")
        )
        depth = depth[np.isclose(depth.tau.astype(float), 0.9)]
        disp = pd.read_csv(
            TABLES
            / (
                "sample_aitchison_displacement_overall.csv"
                if lineage == "stroma"
                else f"{lineage}_sample_aitchison_displacement_overall.csv"
            )
        )
        for _, r in depth.iterrows():
            rows.append(
                {
                    "lineage": lineage,
                    "atlas": r["atlas"],
                    "mean_depth": r["mean_depth"],
                    "mean_unique": r["mean_unique_leaves"],
                }
            )
        for _, r in disp.iterrows():
            # attach later
            pass
        for atlas in ("HGCA", "PanGI"):
            med = float(disp.loc[disp.atlas == atlas, "median"].iloc[0])
            for row in rows:
                if row["lineage"] == lineage and row["atlas"] == atlas:
                    row["median_aitchison"] = med
    df = pd.DataFrame(rows)
    # fill aitchison if missing from loop quirk
    for lineage in ("stroma", "myeloid", "epithelial"):
        disp = pd.read_csv(
            TABLES
            / (
                "sample_aitchison_displacement_overall.csv"
                if lineage == "stroma"
                else f"{lineage}_sample_aitchison_displacement_overall.csv"
            )
        )
        for atlas in ("HGCA", "PanGI"):
            med = float(disp.loc[disp.atlas == atlas, "median"].iloc[0])
            df.loc[(df.lineage == lineage) & (df.atlas == atlas), "median_aitchison"] = med

    call = {"stroma": "A", "myeloid": "B", "epithelial": "C"}
    order = ["stroma", "myeloid", "epithelial"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    x = np.arange(len(order))
    w = 0.36

    ax = axes[0]
    b1 = ax.bar(x - w / 2, [df.loc[(df.lineage == L) & (df.atlas == "HGCA"), "mean_depth"].iloc[0] for L in order], w, color=HGCA, label="HGCA")
    b2 = ax.bar(x + w / 2, [df.loc[(df.lineage == L) & (df.atlas == "PanGI"), "mean_depth"].iloc[0] for L in order], w, color=PANGI, label="PanGI")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n({call[L]})" for L in order])
    ax.set_ylabel("Mean stable naming depth (tau=0.90)")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Hierarchical naming stability", pad=8)
    _despine(ax)

    ax = axes[1]
    ax.bar(x - w / 2, [df.loc[(df.lineage == L) & (df.atlas == "HGCA"), "median_aitchison"].iloc[0] for L in order], w, color=HGCA, label="HGCA")
    ax.bar(x + w / 2, [df.loc[(df.lineage == L) & (df.atlas == "PanGI"), "median_aitchison"].iloc[0] for L in order], w, color=PANGI, label="PanGI")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n({call[L]})" for L in order])
    ax.set_ylabel("Median sample Aitchison distance")
    ax.set_title("Sample composition sensitivity", pad=8)
    _despine(ax)

    fig.suptitle("Cross-lineage scorecard: LODO-favored lineages do not all jackknife the same way", fontsize=10, y=0.98)
    fig.legend([b1, b2], ["HGCA", "PanGI"], loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.09, right=0.98, wspace=0.30)
    _caption(fig, "Stroma A / myeloid B / epithelial C - epithelial reverses the HGCA advantage on both primary metrics.", y=0.06)
    _save(fig, "cross_lineage_jackknife_scorecard")


def plot_epithelial_seed_vs_jackknife() -> None:
    seed = pd.read_csv(TABLES / "epithelial_seed_leaf_agreement_summary.csv")
    depth = pd.read_csv(TABLES / "epithelial_stable_naming_depth_summary.csv")
    depth = depth[np.isclose(depth.tau.astype(float), 0.9)]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    metrics = ["Seed unique leaves\n(full refs only)", "Jackknife unique leaves\n(shared-study omits)"]
    x = np.arange(2)
    w = 0.36
    hgca_vals = [
        float(seed.loc[seed.atlas == "HGCA", "mean_seed_unique"].iloc[0]),
        float(depth.loc[depth.atlas == "HGCA", "mean_unique_leaves"].iloc[0]),
    ]
    pangi_vals = [
        float(seed.loc[seed.atlas == "PanGI", "mean_seed_unique"].iloc[0]),
        float(depth.loc[depth.atlas == "PanGI", "mean_unique_leaves"].iloc[0]),
    ]
    b1 = ax.bar(x - w / 2, hgca_vals, w, color=HGCA, label="HGCA")
    b2 = ax.bar(x + w / 2, pangi_vals, w, color=PANGI, label="PanGI")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Mean unique predicted leaves per cell")
    ax.set_title("Epithelial: composition change vs ordinary seed noise", pad=10)
    _despine(ax)
    fig.legend([b1, b2], ["HGCA", "PanGI"], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.14, right=0.96)
    _caption(fig, "If jackknife uniqueness is not above seed uniqueness, composition effects are weak relative to training noise.", y=0.06)
    _save(fig, "epithelial_seed_vs_jackknife_uniqueness")


def plot_stroma_reference_vs_patient() -> None:
    path = TABLES / "reference_vs_patient_uncertainty.csv"
    if not path.exists():
        LOGGER.warning("Skip reference_vs_patient: missing %s", path)
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(2)
    w = 0.36
    jack = [float(df.loc[df.atlas == a, "jackknife_median_of_sample_median_aitchison"].iloc[0]) for a in ("HGCA", "PanGI")]
    boot = [float(df.loc[df.atlas == a, "patient_boot_cohort_aitchison_median"].iloc[0]) for a in ("HGCA", "PanGI")]
    # Atlas-colored jackknife; grey bootstrap
    for i, (j, b) in enumerate(zip(jack, boot)):
        ax.bar(i - w / 2, j, w, color=[HGCA, PANGI][i], edgecolor="white", lw=0.3)
        ax.bar(i + w / 2, b, w, color="#999999", edgecolor="white", lw=0.3)
        ratio = float(df.loc[df.atlas == ("HGCA", "PanGI")[i], "reference_sensitivity_ratio_median"].iloc[0])
        ax.text(i, max(j, b) * 1.05, f"ratio {ratio:.1f}", ha="center", fontsize=7, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(["HGCA", "PanGI"])
    ax.set_ylabel("Typical Aitchison displacement")
    ax.set_title("Stroma: reference perturbation vs resampling patients", pad=10)
    fig.legend(
        [Patch(facecolor=HGCA), Patch(facecolor=PANGI), Patch(facecolor="#999999")],
        ["HGCA jackknife", "PanGI jackknife", "Patient bootstrap"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    _despine(ax)
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.14, right=0.96)
    _caption(fig, "Ratio >> 1 means changing reference studies moves analysis more than resampling donors (related scales).", y=0.06)
    _save(fig, "stroma_reference_vs_patient_uncertainty")


def plot_stroma_effect_vectors() -> None:
    path = TABLES / "effect_vector_propagation_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    contrasts = sorted(df["contrast"].unique())
    # short labels
    short = {
        "CD_vs_Healthy": "CD vs Healthy",
        "inflamed_vs_noninflamed": "Inflamed vs not",
        "Pre_vs_Post": "Pre vs Post",
        "Remission_vs_Nonremission_baseline": "Remission vs not",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(contrasts))
    w = 0.36
    b1 = ax.bar(
        x - w / 2,
        [float(df.loc[(df.atlas == "HGCA") & (df.contrast == c), "median_cosine"].iloc[0]) for c in contrasts],
        w,
        color=HGCA,
        label="HGCA",
    )
    b2 = ax.bar(
        x + w / 2,
        [float(df.loc[(df.atlas == "PanGI") & (df.contrast == c), "median_cosine"].iloc[0]) for c in contrasts],
        w,
        color=PANGI,
        label="PanGI",
    )
    ax.axhline(1.0, color="#CCCCCC", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(c, c) for c in contrasts], rotation=20, ha="right")
    ax.set_ylabel("Median cosine to full/seed0 beta")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Stroma: do global disease-effect vectors stay put under jackknives?", pad=10)
    _despine(ax)
    fig.legend([b1, b2], ["HGCA", "PanGI"], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.28, left=0.12, right=0.96)
    _caption(fig, "Higher cosine = global biological interpretation more stable to healthy-reference composition.", y=0.06)
    _save(fig, "stroma_effect_vector_cosine")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Epithelial review set
    plot_stable_naming_depth("epithelial")
    plot_aitchison_violin("epithelial")
    plot_paired_study_bars("epithelial")
    plot_size_adjusted("epithelial")
    plot_epithelial_seed_vs_jackknife()
    # Context
    plot_stable_naming_depth("stroma")
    plot_stable_naming_depth("myeloid")
    plot_aitchison_violin("stroma")
    plot_aitchison_violin("myeloid")
    plot_paired_study_bars("stroma")
    plot_paired_study_bars("myeloid")
    plot_size_adjusted("stroma")
    plot_size_adjusted("myeloid")
    plot_cross_lineage_summary()
    plot_stroma_reference_vs_patient()
    plot_stroma_effect_vectors()
    LOGGER.info("All modules written under %s", FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
