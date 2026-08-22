#!/usr/bin/env python3
from __future__ import annotations

import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C


MM = 1 / 25.4
REGION_COLORS = {
    "Duodenum": "#E9C61D",
    "Jejunum": "#E96475",
    "Ileum": "#208230",
    "Colon": "#3A68AE",
    "Intestine": "#999999",
    "Not reported": "#E0E0E0",
}
SOURCE_COLORS = {"ASC": "#0072B2", "FSC": "#009E73", "PSC": "#D55E00"}
SOURCE_LABELS = {
    "ASC": "Adult stem cell-derived (ASC)",
    "FSC": "Fetal stem cell-derived (FSC)",
    "PSC": "Pluripotent stem cell-derived (PSC)",
}
SOURCE_LABELS_MULTILINE = {
    "ASC": "Adult stem\ncell-derived\n(ASC)",
    "FSC": "Fetal stem\ncell-derived\n(FSC)",
    "PSC": "Pluripotent stem\ncell-derived\n(PSC)",
}
TIME_COLORS = {
    "≤14 d": "#56B4E9",
    "15–55 d": "#0072B2",
    "≥56 d": "#CC79A7",
    "Early": "#F0E442",
    "Late": "#D55E00",
    "Not reported": "#BDBDBD",
}
LIGHT_GREY = "#E0E0E0"
MID_GREY = "#999999"


def source_display(value: str, multiline: bool = False) -> str:
    labels = SOURCE_LABELS_MULTILINE if multiline else SOURCE_LABELS
    return labels.get(str(value), str(value))


def configure_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    matplotlib.rcParams.update(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "legend.title_fontsize": 5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "axes.grid": False,
        }
    )


def despine(ax) -> None:
    sns.despine(ax=ax, top=True, right=True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(fig, name: str, width_mm: float, height_mm: float) -> None:
    fig.set_size_inches(width_mm * MM, height_mm * MM)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(C.OUT / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def taxonomy_order(capability: pd.DataFrame) -> list[str]:
    fields = [
        "hgca_celltype_level2",
        "hgca_celltype_level3",
        "hgca_celltype_level4",
    ]
    frame = capability.copy()
    for field in fields:
        frame[field] = frame[field].fillna("")
    frame["_label"] = frame.index
    return frame.sort_values(fields + ["_label"]).index.tolist()


def source_region_strata(samples: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    desired = [
        ("ASC", "Duodenum"),
        ("ASC", "Ileum"),
        ("ASC", "Colon"),
        ("FSC", "Duodenum"),
        ("FSC", "Ileum"),
        ("PSC", "Colon"),
    ]
    observed = set(zip(samples["source_standardized"], samples["region_broad"]))
    pairs = [pair for pair in desired if pair in observed]
    labels = [f"{source} · {region}" for source, region in pairs]
    regions = {label: region for label, (_, region) in zip(labels, pairs)}
    return labels, regions


def fig5b() -> None:
    celltype = pd.read_csv(
        C.DATA / "fig5b_source_stratified_subtype_segment_identity.csv"
    )
    samples = pd.read_csv(C.DATA / "fig5b_sample_segment_identity.csv", index_col=0)
    capability = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    strata, stratum_region = source_region_strata(samples)
    celltype["stratum"] = (
        celltype["source_standardized"].astype(str)
        + " · "
        + celltype["origin_region"].astype(str)
    )
    samples["stratum"] = (
        samples["source_standardized"].astype(str)
        + " · "
        + samples["region_broad"].astype(str)
    )
    celltype = celltype[
        celltype["stratum"].isin(strata) & (celltype["n_cells"] >= 20)
    ].copy()
    order = [
        label
        for label in taxonomy_order(capability)
        if label in set(celltype["hgca_celltype_v1"])
    ]
    celltype["stratum"] = pd.Categorical(
        celltype["stratum"], categories=strata, ordered=True
    )
    celltype["hgca_celltype_v1"] = pd.Categorical(
        celltype["hgca_celltype_v1"], categories=list(reversed(order)), ordered=True
    )

    fig = plt.figure(figsize=(180 * MM, 122 * MM))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.3, 1], hspace=0.55)
    ax = fig.add_subplot(grid[0])
    sns.scatterplot(
        data=celltype,
        x="stratum",
        y="hgca_celltype_v1",
        hue="segment_match_fraction",
        size="n_cells",
        palette="viridis",
        hue_norm=(0, 1),
        sizes=(8, 70),
        edgecolor="black",
        linewidth=0.2,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mapped HGCA epithelial subtype")
    ax.set_title(
        "Regional fidelity is cell-state specific within organoid source",
        loc="left",
        fontweight="bold",
    )
    ax.tick_params(axis="both", length=0)
    for tick, label in zip(ax.get_xticklabels(), strata):
        tick.set_rotation(0)
        tick.set_color(REGION_COLORS[stratum_region[label]])
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
        ax=ax,
        fraction=0.018,
        pad=0.015,
    )
    colorbar.set_label("Cells nearest the matching HGCA segment")
    colorbar.ax.tick_params(labelsize=5, width=0.5, length=2)
    ax.text(
        1.0,
        1.02,
        "Dot size ∝ mapped cells",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5,
        color=MID_GREY,
    )
    panel_label(ax, "b")

    bx = fig.add_subplot(grid[1])
    plot_samples = samples[samples["stratum"].isin(strata)].copy()
    plot_samples["stratum"] = pd.Categorical(
        plot_samples["stratum"], categories=strata, ordered=True
    )
    palette = {
        stratum: REGION_COLORS[stratum_region[stratum]] for stratum in strata
    }
    sns.boxplot(
        data=plot_samples,
        x="stratum",
        y="segment_identity_fraction",
        hue="stratum",
        order=strata,
        hue_order=strata,
        palette=palette,
        width=0.55,
        linewidth=0.5,
        fliersize=0,
        dodge=False,
        legend=False,
        ax=bx,
    )
    sns.stripplot(
        data=plot_samples,
        x="stratum",
        y="segment_identity_fraction",
        hue="stratum",
        order=strata,
        hue_order=strata,
        palette=palette,
        jitter=0.16,
        dodge=False,
        size=3,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
        legend=False,
        ax=bx,
    )
    bx.set_ylim(-0.03, 1.12)
    bx.set_xlabel("Organoid source and declared segment")
    bx.set_ylabel("Segment-concordant cells")
    bx.set_xticks(range(len(strata)))
    bx.set_xticklabels(
        [
            label.replace(" · ", "\n")
            for label in strata
        ]
    )
    for tick, label in zip(bx.get_xticklabels(), strata):
        tick.set_color(REGION_COLORS[stratum_region[label]])
    counts = plot_samples["stratum"].value_counts()
    for index, stratum in enumerate(strata):
        bx.text(
            index,
            1.04,
            f"n={int(counts.get(stratum, 0))}",
            ha="center",
            va="bottom",
            fontsize=5,
        )
    bx.text(
        1,
        -0.48,
        "Protocol effects are not estimable: only transplant-positive status is recorded (10/98 samples), with no verified control field",
        transform=bx.transAxes,
        ha="right",
        va="top",
        fontsize=5,
        color=MID_GREY,
    )
    despine(bx)
    fig.subplots_adjust(left=0.24, right=0.9, top=0.94, bottom=0.16)
    save_figure(fig, "fig5_b_organoid_segment_identity", 180, 122)


def facet_scatter(
    ax,
    frame: pd.DataFrame,
    field: str,
    palette: dict,
    title: str,
    *,
    direct_labels: bool = False,
) -> None:
    values = frame[field].fillna("Not reported").astype(str)
    plotting = frame.assign(_value=values)
    sns.scatterplot(
        data=plotting,
        x="PCoA1",
        y="PCoA2",
        hue="_value",
        palette=palette,
        s=17,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.9,
        legend=not direct_labels,
        ax=ax,
    )
    if direct_labels:
        for value, group in plotting.groupby("_value", observed=True):
            ax.text(
                group["PCoA1"].median(),
                group["PCoA2"].median(),
                value,
                fontsize=5,
                ha="center",
                va="center",
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                    "pad": 0.25,
                },
            )
    elif ax.get_legend() is not None:
        ax.legend(
            title="",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.78,
            loc="lower right",
            bbox_to_anchor=(0.5, -0.13),
            ncol=1 if field == "source_standardized" else 3,
            handletextpad=0.3,
            columnspacing=0.7,
        )
    ax.set_title(title, loc="left", fontweight="bold")
    despine(ax)


def short_publication(value: str) -> str:
    if value == "HEOCA newly generated":
        return "HEOCA new"
    parts = str(value).replace("_", " ").split()
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1]} {parts[-1]}"
    return " ".join(parts)


def fig5b_defensible() -> None:
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    nearest = pd.read_csv(
        C.DATA / "fig5b_sample_nearest_region_proportions.csv", index_col=0
    )
    subtype = pd.read_csv(
        C.DATA / "fig5b_sample_equal_subtype_segment_identity.csv"
    )
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    publications = (
        metadata.groupby("publication_display", observed=True)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fields = [
        ("source_standardized", "Source"),
        ("region_broad", "Region"),
        ("time_class", "Time"),
        ("gel", "Matrix"),
        ("molecular", "Molecular"),
        ("protocol", "Transplant"),
    ]
    category_palette = {
        **SOURCE_COLORS,
        **REGION_COLORS,
        "Mixed": "#000000",
        "Complete": "#0072B2",
        "Partial": "#56B4E9",
        "Missing": "#E0E0E0",
    }

    fig = plt.figure(figsize=(180 * MM, 128 * MM))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.72, 1.3], wspace=0.9)
    ax = fig.add_subplot(grid[0])
    for y, publication in enumerate(publications):
        group = metadata[metadata["publication_display"] == publication]
        for x, (field, _) in enumerate(fields):
            values = group[field].dropna().astype(str).unique()
            if field in {"source_standardized", "region_broad"}:
                value = values[0] if len(values) == 1 else "Mixed"
                text = {
                    "ASC": "A",
                    "FSC": "F",
                    "PSC": "P",
                    "Duodenum": "D",
                    "Jejunum": "J",
                    "Ileum": "I",
                    "Colon": "C",
                    "Intestine": "X",
                    "Mixed": "M",
                }.get(value, "–")
            else:
                fraction = group[field].notna().mean()
                value = "Missing" if fraction == 0 else ("Complete" if fraction == 1 else "Partial")
                text = "–" if value == "Missing" else ("✓" if value == "Complete" else "◐")
            ax.scatter(
                x,
                y,
                marker="s",
                s=42,
                color=category_palette.get(value, MID_GREY),
                edgecolor="white",
                linewidth=0.3,
            )
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=5,
                color="white" if value not in {"Missing", "Duodenum"} else "black",
                fontweight="bold",
            )
    labels = [
        f"{short_publication(publication)} (n={int((metadata['publication_display'] == publication).sum())})"
        for publication in publications
    ]
    ax.set_yticks(range(len(publications)), labels)
    ax.set_xticks(range(len(fields)), [display for _, display in fields], rotation=45, ha="right")
    ax.set_ylim(len(publications) - 0.5, -0.5)
    ax.set_xlim(-0.6, len(fields) - 0.4)
    ax.tick_params(length=0, labelsize=5)
    ax.set_title("Design overlap and effective sample size", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0,
        -0.16,
        "Source: A adult stem cell-derived (ASC) · F fetal stem cell-derived (FSC) · P pluripotent stem cell-derived (PSC) · M mixed\n"
        "Region: D duodenum · J jejunum · I ileum · C colon · X nonspecific · M mixed\n"
        "Metadata coverage: ✓ complete · ◐ partial · – missing",
        transform=ax.transAxes,
        fontsize=5,
        linespacing=1.35,
        va="top",
    )
    panel_label(ax, "b")

    bx = fig.add_subplot(grid[1])
    sample_subtype = pd.read_csv(
        C.DATA / "fig5b_sample_subtype_segment_identity.csv"
    )
    eligible = (
        sample_subtype.groupby(
            [
                "source_standardized",
                "origin_region",
                "hgca_celltype_v1",
            ],
            observed=True,
        )
        .agg(
            n_samples=("sample_id", "nunique"),
            n_publications=("publication_display", "nunique"),
            median_concordance=("segment_match_fraction", "median"),
        )
        .reset_index()
    )
    eligible = eligible[
        (eligible["n_samples"] >= 3) & (eligible["n_publications"] >= 2)
    ]
    selected = (
        eligible.sort_values(
            ["origin_region", "median_concordance"],
            ascending=[True, False],
        )
        .groupby(["source_standardized", "origin_region"], observed=True)
        .head(1)
    )
    selected_pairs = set(
        zip(
            selected["source_standardized"],
            selected["origin_region"],
            selected["hgca_celltype_v1"],
        )
    )
    focused = sample_subtype[
        [
            (source, region, subtype) in selected_pairs
            for source, region, subtype in zip(
                sample_subtype["source_standardized"],
                sample_subtype["origin_region"],
                sample_subtype["hgca_celltype_v1"],
            )
        ]
    ].copy()
    focused["pair"] = (
        focused["source_standardized"]
        + " "
        + focused["origin_region"].str.slice(0, 1)
        + " · "
        + focused["hgca_celltype_v1"]
    )
    pair_order = (
        selected.sort_values(
            ["origin_region", "median_concordance"],
            ascending=[True, False],
        )
        .assign(
            pair=lambda frame: frame["source_standardized"]
            + " "
            + frame["origin_region"].str.slice(0, 1)
            + " · "
            + frame["hgca_celltype_v1"]
        )["pair"]
        .tolist()
    )
    pair_palette = {
        pair: REGION_COLORS[
            focused.loc[focused["pair"] == pair, "origin_region"].iloc[0]
        ]
        for pair in pair_order
    }
    sns.boxplot(
        data=focused,
        y="pair",
        x="segment_match_fraction",
        hue="pair",
        order=pair_order,
        hue_order=pair_order,
        palette=pair_palette,
        linewidth=0.5,
        fliersize=0,
        dodge=False,
        legend=False,
        ax=bx,
    )
    sns.stripplot(
        data=focused,
        y="pair",
        x="segment_match_fraction",
        hue="pair",
        order=pair_order,
        hue_order=pair_order,
        palette=pair_palette,
        size=2.2,
        jitter=0.14,
        edgecolor="white",
        linewidth=0.2,
        dodge=False,
        legend=False,
        ax=bx,
    )
    bx.set_xlim(-0.03, 1.03)
    bx.set_xlabel("Cells nearest declared segment")
    bx.set_ylabel("")
    bx.set_title(
        "Top source-stratified cell states",
        loc="left",
        fontweight="bold",
    )
    bx.text(
        0,
        -0.12,
        "D = duodenum, I = ileum, C = colon; boxes summarize samples",
        transform=bx.transAxes,
        fontsize=5,
        color=MID_GREY,
        va="top",
    )
    despine(bx)

    cx = fig.add_subplot(grid[2])
    subtype = subtype[
        subtype["origin_region"].isin(["Ileum", "Colon"])
        & (subtype["n_samples"] >= 5)
        & (subtype["n_publications"] >= 3)
    ].copy()
    matrix = subtype.pivot(
        index="hgca_celltype_v1",
        columns="origin_region",
        values="median_sample_concordance",
    ).reindex(columns=["Ileum", "Colon"])
    order = [
        label
        for label in taxonomy_order(hierarchy)
        if label in matrix.index
    ]
    matrix = matrix.reindex(order)
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Median sample concordance", "shrink": 0.45},
        ax=cx,
    )
    cx.set_xlabel("Declared segment")
    cx.set_ylabel("")
    cx.set_title(
        "Cell states carrying resolvable\nregional signal",
        loc="left",
        fontweight="bold",
    )
    cx.tick_params(axis="y", labelsize=5, length=0, pad=1)
    for tick, region in zip(cx.get_xticklabels(), ["Ileum", "Colon"]):
        tick.set_color(REGION_COLORS[region])
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.2)
    save_figure(fig, "fig5_b_organoid_segment_identity", 180, 128)


def fig5b_origin_proximity() -> None:
    summary = pd.read_csv(C.DATA / "fig5b_origin_proximity_summary.csv")
    cells = pd.read_csv(
        C.DATA / "fig5b_origin_proximity_rarefied_cells.csv.gz"
    )
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    summary = summary[
        (summary["n_samples"] >= 3) & (summary["n_publications"] >= 2)
    ].copy()
    subtype_order = [
        subtype
        for subtype in taxonomy_order(hierarchy)
        if subtype in set(summary["hgca_celltype_v1"])
    ]
    region_order = ["Duodenum", "Ileum", "Colon"]
    sources = ["ASC", "FSC", "PSC"]
    cmap = sns.color_palette("vlag", as_cmap=True)
    limit = float(
        np.nanquantile(
            np.abs(summary["median_relative_origin_proximity"]), 0.98
        )
    )
    limit = max(limit, 0.1)

    fig = plt.figure(figsize=(180 * MM, 178 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.42)
    top = outer[0].subgridspec(1, 3, wspace=0.18)
    top_axes = []
    for source_index, source in enumerate(sources):
        ax = fig.add_subplot(top[0, source_index])
        top_axes.append(ax)
        subset = summary[summary["source_standardized"] == source].copy()
        subset["region_x"] = subset["origin_region"].map(
            {value: index for index, value in enumerate(region_order)}
        )
        subset["subtype_y"] = subset["hgca_celltype_v1"].map(
            {value: index for index, value in enumerate(subtype_order)}
        )
        sns.scatterplot(
            data=subset,
            x="region_x",
            y="subtype_y",
            hue="median_relative_origin_proximity",
            size="fraction_cells_origin_closer",
            palette=cmap,
            hue_norm=(-limit, limit),
            sizes=(5, 70),
            edgecolor="black",
            linewidth=0.2,
            legend=False,
            ax=ax,
        )
        ax.set_xticks(
            range(len(region_order)),
            [value.replace("Duodenum", "Duod.") for value in region_order],
        )
        for tick, region in zip(ax.get_xticklabels(), region_order):
            tick.set_color(REGION_COLORS[region])
        ax.set_yticks(range(len(subtype_order)))
        if source_index == 0:
            ax.set_yticklabels(subtype_order)
            ax.set_ylabel("HGCA epithelial subtype")
            panel_label(ax, "b")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.set_ylim(len(subtype_order) - 0.5, -0.5)
        ax.set_xlim(-0.55, len(region_order) - 0.45)
        ax.set_xlabel("Declared segment of origin")
        ax.set_title(source_display(source, multiline=True), fontweight="bold")
        ax.tick_params(length=0, labelsize=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.1,
        0.965,
        "Origin-specific proximity of organoid cell states to healthy HGCA epithelium",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        0.1,
        0.942,
        "Fifty cells per eligible sample–subtype; positive values mean the declared origin is closer than the best alternative tissue",
        fontsize=5,
        color=MID_GREY,
    )
    cbar_axis = fig.add_axes([0.91, 0.61, 0.012, 0.2])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            cmap=cmap,
        ),
        cax=cbar_axis,
    )
    colorbar.set_label(
        "Median relative origin proximity\n(positive = origin closer)",
        labelpad=6,
    )
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=size,
            label=label,
        )
        for size, label in [(2.5, "25%"), (4.5, "50%"), (7, "100%")]
    ]
    fig.legend(
        handles=size_handles,
        title="Cells with declared\norigin closer",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.89, 0.55),
        ncol=1,
    )

    conditions = (
        summary[["source_standardized", "origin_region"]]
        .drop_duplicates()
        .assign(
            source_order=lambda frame: frame["source_standardized"].map(
                {value: index for index, value in enumerate(sources)}
            ),
            region_order=lambda frame: frame["origin_region"].map(
                {value: index for index, value in enumerate(region_order)}
            ),
        )
        .sort_values(["source_order", "region_order"])
        .head(6)
    )
    bottom = outer[1].subgridspec(2, 3, hspace=0.65, wspace=0.55)
    low = float(cells["relative_origin_proximity"].quantile(0.01))
    high = float(cells["relative_origin_proximity"].quantile(0.99))
    for facet_index, (_, condition) in enumerate(conditions.iterrows()):
        ax = fig.add_subplot(bottom[facet_index // 3, facet_index % 3])
        source = condition["source_standardized"]
        region = condition["origin_region"]
        condition_summary = summary[
            (summary["source_standardized"] == source)
            & (summary["origin_region"] == region)
        ].nlargest(2, "median_relative_origin_proximity")
        selected = condition_summary["hgca_celltype_v1"].tolist()
        plot_cells = cells[
            (cells["source_standardized"] == source)
            & (cells["origin_region"] == region)
            & cells["hgca_celltype_v1"].isin(selected)
        ]
        sns.boxenplot(
            data=plot_cells,
            x="relative_origin_proximity",
            y="hgca_celltype_v1",
            order=selected,
            color=REGION_COLORS[region],
            linewidth=0.5,
            showfliers=False,
            ax=ax,
        )
        ax.axvline(0, color="black", lw=0.5, ls="--")
        ax.set_xlim(low, high)
        ax.set_xlabel("Relative origin proximity")
        ax.set_ylabel("")
        ax.set_title(
            f"{source_display(source)}\n{region}",
            loc="left",
            fontweight="bold",
            color=REGION_COLORS[region],
        )
        ax.tick_params(axis="y", labelsize=5)
        despine(ax)
    fig.subplots_adjust(left=0.2, right=0.88, top=0.87, bottom=0.08)
    save_figure(fig, "fig5_b_organoid_segment_identity", 180, 178)


def fig5b_compact_origin_proximity() -> None:
    summary = pd.read_csv(C.DATA / "fig5b_origin_proximity_summary.csv")
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    summary = summary[
        (summary["n_samples"] >= 3) & (summary["n_publications"] >= 2)
    ].copy()
    subtype_order = [
        subtype
        for subtype in taxonomy_order(hierarchy)
        if subtype in set(summary["hgca_celltype_v1"])
    ]
    columns = [
        ("ASC", "Duodenum"),
        ("ASC", "Ileum"),
        ("ASC", "Colon"),
        ("FSC", "Duodenum"),
        ("FSC", "Ileum"),
        ("PSC", "Colon"),
    ]
    column_lookup = {
        pair: index for index, pair in enumerate(columns)
    }
    summary = summary[
        [
            (source, region) in column_lookup
            for source, region in zip(
                summary["source_standardized"], summary["origin_region"]
            )
        ]
    ].copy()
    summary["x"] = [
        column_lookup[(source, region)]
        for source, region in zip(
            summary["source_standardized"], summary["origin_region"]
        )
    ]
    summary["y"] = summary["hgca_celltype_v1"].map(
        {value: index for index, value in enumerate(subtype_order)}
    )
    cmap = sns.color_palette("vlag", as_cmap=True)
    limit = max(
        float(
            np.nanquantile(
                np.abs(summary["median_relative_origin_proximity"]), 0.98
            )
        ),
        0.1,
    )
    fig, ax = plt.subplots(figsize=(180 * MM, 76 * MM))
    sns.scatterplot(
        data=summary,
        x="x",
        y="y",
        hue="median_relative_origin_proximity",
        size="fraction_cells_origin_closer",
        palette=cmap,
        hue_norm=(-limit, limit),
        sizes=(5, 65),
        edgecolor="black",
        linewidth=0.2,
        legend=False,
        ax=ax,
    )
    ax.set_xticks(
        range(len(columns)),
        [region.replace("Duodenum", "Duod.") for _, region in columns],
    )
    for tick, (_, region) in zip(ax.get_xticklabels(), columns):
        tick.set_color(REGION_COLORS[region])
    ax.set_yticks(range(len(subtype_order)), subtype_order)
    ax.set_ylim(len(subtype_order) - 0.5, -1.25)
    ax.set_xlim(-0.55, len(columns) - 0.45)
    ax.set_xlabel("Declared segment of origin")
    ax.set_ylabel("HGCA epithelial subtype")
    ax.tick_params(length=0, labelsize=5)
    for boundary in [2.5, 4.5]:
        ax.axvline(boundary, color=LIGHT_GREY, lw=0.7)
    for center, source in [(1, "ASC"), (3.5, "FSC"), (5, "PSC")]:
        ax.text(
            center,
            -0.82,
            source_display(source),
            ha="center",
            va="center",
            fontsize=5,
            fontweight="bold",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "b")
    colorbar_axis = fig.add_axes([0.89, 0.3, 0.012, 0.42])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            cmap=cmap,
        ),
        cax=colorbar_axis,
    )
    colorbar.set_label(
        "Median relative origin proximity\n(positive = origin closer)",
        labelpad=5,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.3,
                markersize=size,
                label=label,
            )
            for size, label in [(2.5, "25%"), (4.5, "50%"), (7, "100%")]
        ],
        title="Cells with declared\norigin closer",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.875, 0.08),
        ncol=1,
    )
    fig.text(
        0.14,
        0.97,
        "Origin proximity by cell state, source and segment",
        fontsize=7,
        fontweight="bold",
    )
    fig.text(
        0.14,
        0.925,
        "Dot area = fraction of balanced cells closer to declared origin; blank = insufficient independent samples",
        fontsize=5,
        color=MID_GREY,
    )
    fig.subplots_adjust(left=0.21, right=0.86, top=0.84, bottom=0.18)
    save_figure(fig, "fig5_b_compact_origin_proximity", 180, 76)


def supplemental_design_overlap() -> None:
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    publications = (
        metadata.groupby("publication_display", observed=True)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fields = [
        ("source_standardized", "Source"),
        ("region_broad", "Region"),
        ("time_class", "Time"),
        ("gel", "Matrix"),
        ("molecular", "Molecular"),
        ("protocol", "Transplant"),
    ]
    palette = {
        **SOURCE_COLORS,
        **REGION_COLORS,
        "Mixed": "#000000",
        "Complete": "#0072B2",
        "Partial": "#56B4E9",
        "Missing": "#E0E0E0",
    }
    fig, (ax, bx) = plt.subplots(
        1,
        2,
        figsize=(180 * MM, 108 * MM),
        gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.48},
    )
    for y, publication in enumerate(publications):
        group = metadata[metadata["publication_display"] == publication]
        for x, (field, _) in enumerate(fields):
            values = group[field].dropna().astype(str).unique()
            if field in {"source_standardized", "region_broad"}:
                value = values[0] if len(values) == 1 else "Mixed"
                text = {
                    "ASC": "A",
                    "FSC": "F",
                    "PSC": "P",
                    "Duodenum": "D",
                    "Jejunum": "J",
                    "Ileum": "I",
                    "Colon": "C",
                    "Intestine": "X",
                    "Mixed": "M",
                }.get(value, "–")
            else:
                fraction = group[field].notna().mean()
                value = (
                    "Missing"
                    if fraction == 0
                    else ("Complete" if fraction == 1 else "Partial")
                )
                text = (
                    "–"
                    if value == "Missing"
                    else ("✓" if value == "Complete" else "◐")
                )
            ax.scatter(
                x,
                y,
                marker="s",
                s=42,
                color=palette.get(value, MID_GREY),
                edgecolor="white",
                linewidth=0.3,
            )
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=5,
                color="white" if value not in {"Missing", "Duodenum"} else "black",
                fontweight="bold",
            )
    ax.set_yticks(
        range(len(publications)),
        [
            f"{short_publication(publication)} (n={(metadata['publication_display'] == publication).sum()})"
            for publication in publications
        ],
    )
    ax.set_xticks(
        range(len(fields)),
        [display for _, display in fields],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(len(publications) - 0.5, -0.5)
    ax.set_xlim(-0.6, len(fields) - 0.4)
    ax.tick_params(length=0, labelsize=5)
    ax.set_title("Publication-level design overlap", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0,
        -0.16,
        "Source: A adult stem cell-derived (ASC) · F fetal stem cell-derived (FSC) · P pluripotent stem cell-derived (PSC) · M mixed\n"
        "Region: D duodenum · J jejunum · I ileum · C colon · X nonspecific · M mixed\n"
        "Coverage: ✓ complete · ◐ partial · – missing",
        transform=ax.transAxes,
        fontsize=5,
        linespacing=1.35,
        va="top",
    )
    panel_label(ax, "a")

    completeness_fields = [
        ("source_standardized", "Organoid source"),
        ("region_detail", "Specific region"),
        ("time_class", "Maturation/time"),
        ("gel", "Matrix"),
        ("molecular", "Molecular condition"),
        ("protocol", "Transplant protocol"),
        ("assay", "Assay"),
        ("donor_id", "Donor/sample identifier"),
    ]
    completeness = pd.DataFrame(
        [
            {
                "field": display,
                "percent": 100 * metadata[field].notna().mean(),
            }
            for field, display in completeness_fields
        ]
    ).sort_values("percent")
    sns.barplot(
        data=completeness,
        y="field",
        x="percent",
        color="#0072B2",
        edgecolor="none",
        ax=bx,
    )
    for patch, value in zip(bx.patches, completeness["percent"]):
        bx.text(
            min(value + 2, 102),
            patch.get_y() + patch.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            fontsize=5,
        )
    bx.set_xlim(0, 108)
    bx.set_xlabel("Samples with populated field")
    bx.set_ylabel("")
    bx.set_title("Metadata completeness", loc="left", fontweight="bold")
    despine(bx)
    panel_label(bx, "b")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.92, bottom=0.2)
    save_figure(fig, "fig5_supp3_metadata_design_overlap", 180, 108)


def fig5c_defensible() -> None:
    coords = pd.read_csv(C.DATA / "fig5c_clr_pca_coordinates.csv", index_col=0)
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    variance = pd.read_csv(C.DATA / "fig5c_variance_partition.csv")
    partial_r2 = pd.read_csv(C.DATA / "fig5c_pca_partial_r2.csv")
    frame = coords.join(metadata)
    publications = (
        frame["publication_display"].value_counts().sort_values(ascending=False).index.tolist()
    )
    r2 = variance.set_index("field")["descriptive_r2"].to_dict()
    xpad = 0.04 * (frame["PC1"].max() - frame["PC1"].min())
    ypad = 0.04 * (frame["PC2"].max() - frame["PC2"].min())
    xlim = (frame["PC1"].min() - xpad, frame["PC1"].max() + xpad)
    ylim = (frame["PC2"].min() - ypad, frame["PC2"].max() + ypad)

    fig = plt.figure(figsize=(180 * MM, 138 * MM))
    grid = fig.add_gridspec(
        5, 6, height_ratios=[1, 1, 1, 1, 1.55], hspace=0.42, wspace=0.2
    )
    for index in range(24):
        ax = fig.add_subplot(grid[index // 6, index % 6])
        if index >= len(publications):
            ax.axis("off")
            continue
        publication = publications[index]
        active = frame["publication_display"] == publication
        ax.scatter(
            frame["PC1"],
            frame["PC2"],
            s=4,
            color=LIGHT_GREY,
            edgecolor="none",
        )
        ax.scatter(
            frame.loc[active, "PC1"],
            frame.loc[active, "PC2"],
            s=7,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.15,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(short_publication(publication), fontsize=5, pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index == 0:
            panel_label(ax, "c")
    fig.text(
        0.07,
        0.965,
        f"Publication fingerprints on shared CLR-PCA coordinates (descriptive R²={r2['publication_display']:.2f})",
        fontsize=7,
        fontweight="bold",
        ha="left",
    )

    bottom_fields = [
        ("time_class", TIME_COLORS, f"Maturation/time (n=63; R²={r2['time_class']:.2f})"),
        ("region_broad", REGION_COLORS, f"Declared region (R²={r2['region_broad']:.2f})"),
        ("source_standardized", SOURCE_COLORS, f"Organoid source (R²={r2['source_standardized']:.2f})"),
    ]
    bottom_grid = grid[4, :].subgridspec(1, 4, wspace=0.45)
    for panel, (field, base_palette, title) in enumerate(bottom_fields):
        ax = fig.add_subplot(bottom_grid[0, panel])
        raw_values = frame[field].fillna("Not reported").astype(str)
        if field == "source_standardized":
            values = raw_values.map(source_display)
            palette = {
                source_display(value): color
                for value, color in SOURCE_COLORS.items()
            }
            palette["Not reported"] = MID_GREY
        else:
            values = raw_values
            palette = {
                value: base_palette.get(value, MID_GREY)
                for value in values.unique()
            }
        sns.scatterplot(
            data=frame.assign(_value=values),
            x="PC1",
            y="PC2",
            hue="_value",
            palette=palette,
            s=12,
            edgecolor="white",
            linewidth=0.2,
            legend=True,
            ax=ax,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)
        ax.set_xlabel("CLR PC1")
        ax.set_ylabel("CLR PC2" if panel == 0 else "")
        ax.set_title(title, loc="left", fontsize=6, fontweight="bold")
        ax.legend(
            title="",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=1 if field == "source_standardized" else 3,
            fontsize=4.5 if field == "source_standardized" else 5,
            handletextpad=0.25,
            columnspacing=0.55,
        )
        despine(ax)
    ax = fig.add_subplot(bottom_grid[0, 3])
    partial_plot = partial_r2.sort_values(
        "partial_r2_all_nonzero_pcs", ascending=True
    )
    sns.barplot(
        data=partial_plot,
        y="display",
        x="partial_r2_all_nonzero_pcs",
        color="#0072B2",
        edgecolor="none",
        ax=ax,
    )
    for patch, n_samples in zip(ax.patches, partial_plot["n_samples"]):
        ax.text(
            patch.get_width() + 0.01,
            patch.get_y() + patch.get_height() / 2,
            f"n={int(n_samples)}",
            va="center",
            fontsize=4,
        )
    ax.set_xlabel("Partial R²")
    ax.set_ylabel("")
    ax.set_title(
        "Covariate-adjusted partial R²\n"
        f"All CLR PCs (n={int(partial_r2['n_nonzero_pcs'].iloc[0])}; "
        "100% variance)",
        loc="left",
        fontsize=6,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=4.5)
    ax.set_box_aspect(1)
    despine(ax)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.23)
    save_figure(fig, "fig5_c_organoid_composition_embedding", 180, 138)


def fig5c() -> None:
    coords = pd.read_csv(C.DATA / "fig5c_pcoa_coordinates.csv", index_col=0)
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    variance = pd.read_csv(C.DATA / "fig5c_variance_partition.csv")
    frame = coords.join(metadata)
    publication_order = (
        frame["publication_display"].value_counts().sort_values(ascending=False).index
    )
    publication_key = pd.DataFrame(
        {
            "publication_display": publication_order,
            "publication_id": [
                f"P{index + 1:02d}" for index in range(len(publication_order))
            ],
        }
    )
    publication_key.to_csv(C.DATA / "fig5c_publication_key.csv", index=False)
    publication_lookup = dict(
        zip(publication_key["publication_display"], publication_key["publication_id"])
    )
    frame["publication_id"] = frame["publication_display"].map(publication_lookup)
    publication_palette_values = list(sns.color_palette("tab20", 20)) + list(
        sns.color_palette("Set2", max(3, len(publication_order) - 20))
    )
    publication_palette = {
        publication_lookup[value]: publication_palette_values[index]
        for index, value in enumerate(publication_order)
    }
    source_palette = {**SOURCE_COLORS, "Not reported": MID_GREY}
    region_palette = {
        value: REGION_COLORS[value]
        for value in frame["region_broad"].fillna("Not reported").unique()
    }
    time_palette = {
        value: TIME_COLORS.get(value, MID_GREY)
        for value in frame["time_class"].fillna("Not reported").unique()
    }
    r2 = variance.set_index("field")["descriptive_r2"].to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 108 * MM), sharex=True, sharey=True)
    facet_scatter(
        axes[0, 0],
        frame,
        "publication_id",
        publication_palette,
        f"Publication (R²={r2['publication_display']:.2f})",
        direct_labels=True,
    )
    facet_scatter(
        axes[0, 1],
        frame,
        "time_class",
        time_palette,
        f"Maturation/time (n=63; R²={r2['time_class']:.2f})",
    )
    time_order = ["Early", "≤14 d", "15–55 d", "Late", "≥56 d"]
    time_rank = {value: index for index, value in enumerate(time_order)}
    trajectory = (
        frame.dropna(subset=["time_class"])
        .groupby(["publication_display", "time_class"], observed=True)[
            ["PCoA1", "PCoA2"]
        ]
        .mean()
        .reset_index()
    )
    for _, group in trajectory.groupby("publication_display", observed=True):
        group = group[group["time_class"].isin(time_rank)].copy()
        if group["time_class"].nunique() < 2:
            continue
        group["_order"] = group["time_class"].map(time_rank)
        group = group.sort_values("_order")
        axes[0, 1].plot(
            group["PCoA1"],
            group["PCoA2"],
            color=MID_GREY,
            lw=0.45,
            alpha=0.55,
            zorder=1,
        )
    time_legend = axes[0, 1].get_legend()
    if time_legend is not None:
        handles = time_legend.legend_handles + [
            Line2D(
                [0],
                [0],
                color=MID_GREY,
                lw=0.6,
                label="Study time-centroid path",
            )
        ]
        labels = [text.get_text() for text in time_legend.get_texts()] + [
            "Study time-centroid path"
        ]
        axes[0, 1].legend(
            handles,
            labels,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.78,
            loc="lower right",
            fontsize=5,
            ncol=3,
        )
    facet_scatter(
        axes[1, 0],
        frame,
        "region_broad",
        region_palette,
        f"Declared region (R²={r2['region_broad']:.2f})",
    )
    facet_scatter(
        axes[1, 1],
        frame,
        "source_standardized",
        source_palette,
        f"Organoid source (R²={r2['source_standardized']:.2f})",
    )
    for row in range(2):
        for column in range(2):
            axes[row, column].set_xlabel("")
            axes[row, column].set_ylabel("")
    panel_label(axes[0, 0], "c")
    fig.supxlabel("PCoA 1", x=0.52, y=0.025, fontsize=6)
    fig.supylabel("PCoA 2", x=0.025, y=0.53, fontsize=6)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.22, hspace=0.48, wspace=0.22)
    save_figure(fig, "fig5_c_organoid_composition_embedding", 180, 108)


def supplemental_capability_qc() -> None:
    capability = pd.read_csv(
        C.DATA / "supp_protocol_subtype_capability_long.csv"
    )
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    per_cell = pd.read_csv(C.DATA / "per_cell_mapping_qc_flags.csv.gz")
    sample_qc = pd.read_csv(C.DATA / "supp_mapping_qc_by_sample.csv", index_col=0)
    sensitivity = pd.read_csv(C.DATA / "supp_embedding_sensitivity.csv")

    field_order = [
        "source_standardized",
        "time_class",
        "gel",
        "molecular",
        "protocol",
    ]
    capability = capability[capability["field"].isin(field_order)].copy()
    value_order = {
        "source_standardized": ["ASC", "FSC", "PSC"],
        "time_class": ["≤14 d", "15–55 d", "≥56 d", "Early", "Late", "Not reported"],
        "gel": ["Matrigel", "Suspension", "Not reported"],
        "protocol": ["transplant", "Not reported"],
    }
    columns_by_field = {}
    for field in field_order:
        values = capability.loc[capability["field"] == field, "value"].unique()
        ordered = [x for x in value_order.get(field, []) if x in values]
        ordered += sorted(set(values) - set(ordered))
        columns_by_field[field] = ordered
    order = taxonomy_order(hierarchy)
    capability["y"] = capability["hgca_celltype_v1"].map(
        {label: len(order) - 1 - index for index, label in enumerate(order)}
    )

    fig = plt.figure(figsize=(180 * MM, 170 * MM))
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1.7, 1], width_ratios=[1.15, 1, 1], hspace=0.62, wspace=0.42
    )
    top = grid[0, :].subgridspec(
        1,
        len(field_order),
        width_ratios=[
            max(2, len(columns_by_field[field])) for field in field_order
        ],
        wspace=0.28,
    )
    top_axes = []
    vmax = min(0.25, capability["median_abundance"].max())
    display_names = {
        "source_standardized": "Source",
        "time_class": "Maturation/time",
        "gel": "Matrix",
        "molecular": "Molecular condition",
        "protocol": "Transplant field",
    }
    for field_index, field in enumerate(field_order):
        ax = fig.add_subplot(top[0, field_index])
        top_axes.append(ax)
        values = columns_by_field[field]
        subset = capability[capability["field"] == field].copy()
        subset["x"] = subset["value"].map(
            {value: index for index, value in enumerate(values)}
        )
        sns.scatterplot(
            data=subset,
            x="x",
            y="y",
            size="detection_rate_rarefied",
            hue="median_abundance",
            palette="mako",
            hue_norm=(0, vmax),
            sizes=(1, 34),
            edgecolor="black",
            linewidth=0.1,
            legend=False,
            ax=ax,
        )
        labels = []
        for value in values:
            count = int(
                subset.loc[subset["value"] == value, "n_samples"].iloc[0]
            )
            display_value = (
                source_display(value)
                if field == "source_standardized"
                else value
            )
            labels.append(f"{display_value}\n(n={count})")
        ax.set_xticks(range(len(values)), labels, rotation=90)
        ax.set_yticks(range(len(order)))
        if field_index == 0:
            ax.set_yticklabels(list(reversed(order)))
            ax.set_ylabel("HGCA epithelial subtype")
            ax.text(
                -0.22,
                1.04,
                "a",
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                va="bottom",
            )
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(display_names[field], loc="left", fontweight="bold")
        ax.tick_params(length=0, labelsize=5)
        ax.set_ylim(-1, len(order) + 0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar_axis = fig.add_axes([0.905, 0.55, 0.012, 0.25])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=Normalize(0, vmax),
            cmap="mako",
        ),
        cax=colorbar_axis,
    )
    colorbar.set_label("Median subtype proportion when detected", labelpad=6)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#4C78A8",
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=np.sqrt(size),
            label=label,
        )
        for size, label in [(9, "25%"), (25, "50%"), (49, "100%")]
    ]
    fig.legend(
        handles=size_handles,
        title="Rarefied sample\ndetection rate",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.9, 0.48),
        ncol=1,
    )

    bx = fig.add_subplot(grid[1, 0])
    sampled = per_cell.sample(
        n=min(60_000, len(per_cell)), random_state=20260718
    ).copy()
    sampled["distance_robust_z_within_label"] = sampled[
        "distance_robust_z_within_label"
    ].clip(-4, 10)
    sns.histplot(
        data=sampled,
        x="hgca_pred_conf_sysvi_knn",
        y="distance_robust_z_within_label",
        bins=55,
        pthresh=0.02,
        cmap="mako",
        cbar=False,
        ax=bx,
    )
    bx.axvline(0.5, color="#D55E00", lw=0.7, ls="--")
    bx.axhline(3.5, color="#D55E00", lw=0.7, ls="--")
    bx.set_xlabel("Per-cell sysVI confidence")
    bx.set_ylabel("Reference-distance outlier score\n(within mapped cell type)")
    bx.set_title("Per-cell mapping quality", loc="left", fontweight="bold")
    despine(bx)
    panel_label(bx, "b")

    cx = fig.add_subplot(grid[1, 1])
    sample_qc["region_broad"] = sample_qc["region_broad"].fillna("Not reported")
    sns.boxplot(
        data=sample_qc,
        x="source_standardized",
        y="fraction_strict_mapping_pass",
        hue="region_broad",
        order=["ASC", "FSC", "PSC"],
        palette=REGION_COLORS,
        linewidth=0.5,
        fliersize=0,
        ax=cx,
    )
    sns.stripplot(
        data=sample_qc,
        x="source_standardized",
        y="fraction_strict_mapping_pass",
        hue="region_broad",
        order=["ASC", "FSC", "PSC"],
        palette=REGION_COLORS,
        dodge=True,
        jitter=0.12,
        size=2.2,
        edgecolor="white",
        linewidth=0.2,
        ax=cx,
    )
    handles, labels = cx.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    cx.legend(
        unique.values(),
        unique.keys(),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.75,
        loc="upper right",
        ncol=1,
        fontsize=4,
    )
    cx.set_xlabel("")
    cx.set_xticks(range(3))
    cx.set_xticklabels(
        [source_display(value, multiline=True) for value in ["ASC", "FSC", "PSC"]]
    )
    cx.tick_params(axis="x", labelsize=4.5)
    cx.set_ylabel("Cells passing confidence + distance QC")
    cx.set_title("Sample-level mapping QC", loc="left", fontweight="bold")
    despine(cx)
    panel_label(cx, "c")

    dx = fig.add_subplot(grid[1, 2])
    fine = sensitivity[
        (sensitivity["annotation_level"] == "hgca_celltype_v1")
        & (~sensitivity["rare_subtypes_excluded"].astype(bool))
    ].copy()
    sns.lineplot(
        data=fine,
        x="min_confident_cells",
        y="distance_spearman_vs_primary",
        hue="pseudocount",
        marker="o",
        linewidth=0.8,
        markersize=3,
        palette="colorblind",
        ax=dx,
    )
    strict = sensitivity[
        sensitivity["annotation_level"] == "strict_confidence_plus_distance"
    ]
    if not strict.empty:
        dx.scatter(
            strict["min_confident_cells"],
            strict["distance_spearman_vs_primary"],
            marker="s",
            s=12,
            color="black",
            label="Strict cell QC",
            zorder=4,
        )
    dx.set_ylim(0.5, 1.02)
    dx.set_xlabel("Minimum confident cells")
    dx.set_ylabel("Distance correlation with primary")
    dx.set_title("CLR geometry is QC-robust", loc="left", fontweight="bold")
    dx.legend(frameon=False, loc="lower right")
    despine(dx)
    panel_label(dx, "d")
    fig.subplots_adjust(left=0.18, right=0.88, top=0.94, bottom=0.12)
    save_figure(fig, "fig5_supp1_protocol_capability_mapping_qc", 180, 170)


def supplemental_metadata_prediction() -> None:
    prediction = pd.read_csv(C.DATA / "supp_metadata_prediction_summary.csv")
    null = pd.read_csv(C.DATA / "supp_metadata_prediction_null.csv")
    candidates = pd.read_csv(
        C.DATA / "supp_missing_metadata_prediction_candidates.csv"
    )
    heldout = pd.read_csv(
        C.DATA / "supp_metadata_prediction_heldout_samples.csv"
    )
    variance = pd.read_csv(C.DATA / "fig5c_variance_partition.csv")
    metadata = pd.read_csv(C.DATA / "sample_metadata.csv", index_col=0)
    label_map = {
        "source_standardized": "Source",
        "region_detail": "Region",
        "time_class": "Maturation/time",
        "gel": "Matrix",
    }
    prediction["display"] = prediction["target"].map(label_map)
    null["display"] = null["target"].map(label_map)

    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 118 * MM))
    fig.suptitle(
        "Can CLR composition predict metadata in an unseen publication?",
        x=0.09,
        y=0.985,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    ax = axes[0, 0]
    sns.violinplot(
        data=null,
        x="display",
        y="balanced_accuracy",
        color="#BFD7EA",
        inner="quartile",
        cut=0,
        linewidth=0.4,
        ax=ax,
    )
    sns.scatterplot(
        data=prediction,
        x="display",
        y="balanced_accuracy",
        color="#0072B2",
        marker="_",
        s=90,
        linewidth=1.2,
        zorder=4,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Balanced accuracy in held-out publications")
    ax.set_title("Maturation signal is exploratory and study-limited", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    despine(ax)
    panel_label(ax, "a")

    bx = axes[0, 1]
    heldout["display"] = heldout["target"].map(label_map)
    sns.boxplot(
        data=heldout,
        x="display",
        y="maximum_probability",
        hue="correct",
        palette={True: "#0072B2", False: MID_GREY},
        linewidth=0.5,
        fliersize=0,
        ax=bx,
    )
    sns.stripplot(
        data=heldout,
        x="display",
        y="maximum_probability",
        hue="correct",
        palette={True: "#0072B2", False: MID_GREY},
        dodge=True,
        jitter=0.12,
        size=2,
        alpha=0.6,
        ax=bx,
    )
    handles, labels = bx.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    bx.legend(
        unique.values(),
        ["Correct" if value == "True" else "Incorrect" for value in unique],
        frameon=False,
        loc="lower left",
        ncol=2,
    )
    bx.set_xlabel("Metadata target")
    bx.set_ylabel("Maximum held-publication probability")
    bx.set_title(
        "High probabilities do not imply cross-dataset accuracy",
        loc="left",
        fontweight="bold",
    )
    bx.tick_params(axis="x", rotation=20)
    despine(bx)
    panel_label(bx, "b")

    cx = axes[1, 0]
    selected = variance[
        variance["field"].isin(
            [
                "publication_display",
                "time_class",
                "region_broad",
                "source_standardized",
                "tech",
                "gel",
                "molecular",
            ]
        )
    ].copy()
    descriptive = (
        selected[["display", "descriptive_r2"]]
        .rename(columns={"descriptive_r2": "r2"})
        .assign(component="Descriptive")
    )
    incremental = selected[
        [
            "display",
            "incremental_r2_after_publication",
            "within_publication_permutation_p",
        ]
    ].rename(columns={"incremental_r2_after_publication": "r2"})
    incremental.loc[incremental["display"] == "Publication", "r2"] = np.nan
    incremental["component"] = np.where(
        (incremental["within_publication_permutation_p"] < 0.05)
        & (incremental["display"] != "Maturation/time"),
        "Increment, supported",
        "Increment, unsupported",
    )
    long = pd.concat([descriptive, incremental], ignore_index=True)
    sns.barplot(
        data=long,
        y="display",
        x="r2",
        hue="component",
        palette={
            "Descriptive": LIGHT_GREY,
            "Increment, supported": "#0072B2",
            "Increment, unsupported": MID_GREY,
        },
        edgecolor="none",
        ax=cx,
    )
    cx.set_xlabel("Fraction of CLR variation explained")
    cx.set_ylabel("")
    cx.set_title("Publication absorbs much measured structure", loc="left", fontweight="bold")
    cx.legend(frameon=False, loc="lower right")
    despine(cx)
    panel_label(cx, "c")

    dx = axes[1, 1]
    missingness_rows = []
    target_fields = {
        "Source": "source_standardized",
        "Region": "region_detail",
        "Maturation/time": "time_class",
        "Matrix": "gel",
        "Molecular": "molecular",
        "Protocol": "protocol",
    }
    prediction_lookup = prediction.set_index("display")
    for display, field in target_fields.items():
        populated = int(metadata[field].notna().sum())
        row = {
            "display": display,
            "percent_populated": 100 * populated / len(metadata),
            "performance_above_null": np.nan,
            "validated": False,
        }
        if display in prediction_lookup.index:
            record = prediction_lookup.loc[display]
            row["performance_above_null"] = (
                record["balanced_accuracy"]
                - record["null_balanced_accuracy_median"]
            )
            row["validated"] = bool(
                record["balanced_accuracy"]
                > record["null_balanced_accuracy_q975"]
                and record["within_publication_permutation_p"] < 0.05
                and record["n_publications_within_study_variation"] >= 3
            )
        missingness_rows.append(row)
    missingness = pd.DataFrame(missingness_rows)
    sns.scatterplot(
        data=missingness,
        x="percent_populated",
        y="performance_above_null",
        hue="validated",
        palette={True: "#0072B2", False: MID_GREY},
        s=38,
        edgecolor="white",
        linewidth=0.3,
        legend=False,
        ax=dx,
    )
    for _, row in missingness.dropna(subset=["performance_above_null"]).iterrows():
        dx.text(
            row["percent_populated"] + 1.5,
            row["performance_above_null"],
            row["display"],
            fontsize=5,
            va="center",
        )
    dx.axhline(0, color="black", lw=0.4)
    dx.set_xlim(0, 105)
    dx.set_xlabel("Samples with recorded metadata (%)")
    dx.set_ylabel("Balanced accuracy above null median")
    dx.set_title("Missing metadata cannot generally be recovered", loc="left", fontweight="bold")
    despine(dx)
    panel_label(dx, "d")
    fig.text(
        0.09,
        0.025,
        "For every fold, one publication is held out completely; models are trained only on labeled samples from the remaining publications.",
        fontsize=5,
        color=MID_GREY,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.15, hspace=0.62, wspace=0.35)
    save_figure(fig, "fig5_supp2_metadata_prediction", 180, 118)


def fig5e_clr_pca_loadings() -> None:
    loadings = pd.read_csv(
        C.DATA / "fig5c_clr_pca_loadings.csv", index_col=0
    )
    variance = pd.read_csv(C.DATA / "fig5c_clr_pca_variance.csv").set_index(
        "axis"
    )
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 68 * MM))
    for index, axis in enumerate(["PC1", "PC2"]):
        selected = (
            loadings[axis]
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        frame = (
            loadings.loc[selected, [axis]]
            .sort_values(axis)
            .reset_index()
            .rename(columns={axis: "loading"})
        )
        frame["direction"] = np.where(
            frame["loading"] >= 0, "Positive", "Negative"
        )
        sns.barplot(
            data=frame,
            y="hgca_celltype_v1",
            x="loading",
            hue="direction",
            palette={"Positive": "#0072B2", "Negative": MID_GREY},
            dodge=False,
            edgecolor="none",
            legend=False,
            ax=axes[index],
        )
        axes[index].axvline(0, color="black", lw=0.4)
        axes[index].set_xlabel("CLR-PCA loading")
        axes[index].set_ylabel("")
        axes[index].set_title(
            f"{axis} ({100 * variance.loc[axis, 'explained_variance_fraction']:.1f}% variance)",
            loc="left",
            fontweight="bold",
        )
        despine(axes[index])
    panel_label(axes[0], "e")
    fig.legend(
        handles=[
            Patch(facecolor="#0072B2", edgecolor="none", label="Positive"),
            Patch(facecolor=MID_GREY, edgecolor="none", label="Negative"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.86, bottom=0.18, wspace=0.62)
    save_figure(fig, "fig5_e_clr_pca_loadings", 150, 68)


def fig5f_covariate_enrichment() -> None:
    enrichment = pd.read_csv(
        C.DATA / "supp_covariate_subtype_adjusted_enrichment.csv"
    )
    residuals = pd.read_csv(
        C.DATA / "supp_covariate_subtype_adjusted_residuals.csv.gz"
    )
    hierarchy = pd.read_csv(
        C.DATA / "supp_subtype_capability_table.csv", index_col=0
    )
    fields = [
        "source_standardized",
        "region_broad",
        "time_class",
        "gel",
        "molecular",
    ]
    display = {
        "source_standardized": "Source\nadjusted for\nregion + maturation",
        "region_broad": "Region\nadjusted for\nsource + maturation",
        "time_class": "Maturation/time\nadjusted for\nsource + region",
        "gel": "Matrix\nadjusted for\nsource + region\n+ maturation",
        "molecular": "Molecular condition\nadjusted for\nsource + region\n+ maturation + matrix",
    }
    value_order = {
        "source_standardized": ["ASC", "FSC", "PSC"],
        "region_broad": ["Duodenum", "Jejunum", "Ileum", "Colon", "Intestine"],
        "time_class": ["≤14 d", "15–55 d", "≥56 d", "Early", "Late"],
        "gel": ["Matrigel", "Suspension"],
    }
    subtype_order = taxonomy_order(hierarchy)
    diverging_cmap = sns.color_palette("vlag", as_cmap=True)
    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=0.48)
    top = outer[0].subgridspec(
        1,
        len(fields),
        width_ratios=[
            max(
                2,
                enrichment.loc[enrichment["field"] == field, "category"].nunique(),
            )
            for field in fields
        ],
        wspace=0.3,
    )
    heat_axes = []
    for field_index, field in enumerate(fields):
        ax = fig.add_subplot(top[0, field_index])
        heat_axes.append(ax)
        subset = enrichment[enrichment["field"] == field]
        categories = [
            value
            for value in value_order.get(field, [])
            if value in set(subset["category"])
        ]
        categories += sorted(set(subset["category"]) - set(categories))
        matrix = subset.pivot(
            index="hgca_celltype_v1",
            columns="category",
            values="adjusted_standardized_enrichment",
        ).reindex(index=subtype_order, columns=categories)
        sns.heatmap(
            matrix,
            cmap=diverging_cmap,
            center=0,
            vmin=-2,
            vmax=2,
            cbar=False,
            yticklabels=field_index == 0,
            linewidths=0.15,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(
            display[field],
            loc="left",
            fontweight="bold",
            fontsize=5,
            pad=3,
        )
        ax.set_xlabel("")
        ax.set_ylabel("HGCA epithelial subtype" if field_index == 0 else "")
        if field == "source_standardized":
            ax.set_xticklabels(
                [source_display(value, multiline=True) for value in categories]
            )
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.tick_params(axis="y", labelsize=5, length=0)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-2, 2), cmap=diverging_cmap),
        ax=heat_axes,
        fraction=0.014,
        pad=0.012,
    )
    colorbar.set_label(
        "Covariate-adjusted standardized CLR enrichment",
        labelpad=6,
    )
    heat_axes[0].text(
        -0.18,
        1.04,
        "f",
        transform=heat_axes[0].transAxes,
        fontsize=7,
        fontweight="bold",
        va="bottom",
    )

    candidates = enrichment[
        enrichment["adjusted_standardized_enrichment"].notna()
        & (enrichment["n_category_samples"] >= 3)
    ].copy()
    candidates["absolute_effect"] = candidates[
        "adjusted_standardized_enrichment"
    ].abs()
    selected_rows = []
    used_subtypes = set()
    for _, record in candidates.sort_values(
        "absolute_effect", ascending=False
    ).iterrows():
        if record["hgca_celltype_v1"] in used_subtypes:
            continue
        selected_rows.append(record)
        used_subtypes.add(record["hgca_celltype_v1"])
        if len(selected_rows) == 6:
            break
    bottom = outer[1].subgridspec(2, 3, hspace=0.7, wspace=0.5)
    for index, record in enumerate(selected_rows):
        ax = fig.add_subplot(bottom[index // 3, index % 3])
        field = record["field"]
        category = str(record["category"])
        category_label = (
            source_display(category) if field == "source_standardized" else category
        )
        subtype = record["hgca_celltype_v1"]
        adjusted = residuals[residuals["target"] == field].copy()
        frame = pd.DataFrame(
            {
                "adjusted_clr": adjusted[subtype],
                "group": np.where(
                    adjusted["target_value"].astype(str).eq(category),
                    category,
                    "Other reported values",
                ),
            }
        )
        group_order = ["Other reported values", category]
        sns.boxplot(
            data=frame,
            x="group",
            y="adjusted_clr",
            order=group_order,
            color="#D9EAF4",
            linewidth=0.5,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=frame,
            x="group",
            y="adjusted_clr",
            order=group_order,
            color="#0072B2",
            size=2.2,
            jitter=0.14,
            alpha=0.8,
            ax=ax,
        )
        ax.axhline(0, color=MID_GREY, lw=0.4)
        ax.set_xlabel("")
        ax.set_ylabel("Adjusted CLR residual (SD)" if index % 3 == 0 else "")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Other", textwrap.shorten(category_label, width=22)]
        )
        ax.set_title(
            f"{display[field].splitlines()[0]} · {textwrap.shorten(category_label, width=28)}\n"
            f"{textwrap.shorten(subtype, width=30)}",
            loc="left",
            fontsize=5.5,
            fontweight="bold",
        )
        despine(ax)
    fig.subplots_adjust(left=0.16, right=0.91, top=0.95, bottom=0.09)
    save_figure(fig, "fig5_f_covariate_subtype_enrichment", 180, 165)


def main() -> None:
    configure_style()
    logger = C.setup_logging("04_render_revised_figures")
    fig5b_origin_proximity()
    fig5b_compact_origin_proximity()
    fig5c_defensible()
    fig5e_clr_pca_loadings()
    fig5f_covariate_enrichment()
    supplemental_capability_qc()
    supplemental_metadata_prediction()
    supplemental_design_overlap()
    logger.info("Wrote revised Figure 5b/5c and two supplemental composites")


if __name__ == "__main__":
    main()
