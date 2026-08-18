#!/usr/bin/env python3
"""Stage 2: patpy benchmark of the sample representations, per tissue x disease.

For each group we rebuild a sample-level AnnData from the Stage 1 distance
matrices, then score every representation with the patpy SPARE metrics:

  * Information retention  - kNN prediction of *relevant* biology
                             (disease state severity, response, inflammation,
                             disease, treatment), calibrated F1 / |Spearman|.
  * Batch mixing           - 1 - kNN prediction of *technical* covariates
                             (batch, library type, n_cells); higher = better.
  * Trajectory             - |Spearman| between diffusion pseudotime and the
                             ordinal disease-severity trajectory.
  * Total                  - (2*Info + 2*Trajectory + Batch) / 5  (SPARE weights)

Outputs (data/):
  benchmark_scores_long.csv     - one row per group x representation x covariate
  benchmark_summary.csv         - one row per group x representation (the table)
Outputs (out/):
  benchmark_table_<group>.{pdf,png}   - the styled plottable table()
  benchmark_table_mean.{pdf,png}      - mean across groups

Reproduces docs/tutorials/notebooks/benchmarking_sample_representation_methods
from the patpy tutorials, adapted to the Taurus per-tissue design.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc

import _common as C

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

N_NEIGHBORS = 5


def log(m):
    print(f"[stage2] {m}", flush=True)


# --------------------------------------------------------------------------
def available_methods(group: str) -> list[str]:
    return [k for k in C.REPR_ORDER if C.repr_distance_path(group, k).exists()]


def load_group(group: str):
    meta_path = C.group_meta_path(group)
    if not meta_path.exists():
        return None
    meta = pd.read_csv(meta_path, index_col=0)
    meta.index = meta.index.astype(str)

    methods = available_methods(group)
    if not methods:
        return None

    dists = {}
    for m in methods:
        d = pd.read_csv(C.repr_distance_path(group, m), index_col=0)
        d.index = d.index.astype(str)
        d.columns = d.columns.astype(str)
        dists[m] = d

    # Common samples present in metadata and every distance matrix.
    common = list(meta.index)
    for d in dists.values():
        common = [s for s in common if s in d.index]
    meta = meta.loc[common]
    dists = {m: d.loc[common, common] for m, d in dists.items()}
    return meta, dists, methods, common


def build_meta_adata(meta: pd.DataFrame, dists: dict, methods: list[str]):
    n = meta.shape[0]
    ad = sc.AnnData(X=np.zeros((n, 1), dtype="float32"), obs=meta.copy())
    ad.obs_names = meta.index.astype(str)
    ad.obs["state_severity"] = pd.to_numeric(ad.obs["state_severity"], errors="coerce")
    for num in ("Age", "Disease_duration", "Inflammation_score", "n_cells"):
        if num in ad.obs:
            ad.obs[num] = pd.to_numeric(ad.obs[num], errors="coerce")

    n_nbrs = int(min(15, max(2, n - 1)))
    for m in methods:
        ad.obsm[f"{m}_distances"] = dists[m].values.astype(float)
        sc.pp.neighbors(ad, use_rep=f"{m}_distances", key_added=f"{m}_neighbors",
                        metric="precomputed", n_neighbors=n_nbrs)
    ad.uns["sample_representations"] = list(methods)
    return ad


def usable_schema(meta: pd.DataFrame) -> dict:
    """Drop covariates that are missing or have <2 observed categories/values."""
    out = {}
    for ctype, cov_tasks in C.BENCHMARK_SCHEMA.items():
        kept = {}
        for col, task in cov_tasks.items():
            if col not in meta.columns:
                continue
            s = meta[col].replace(["None", "nan", "Not_avail"], np.nan)
            if task in ("classification",):
                if s.dropna().nunique() >= 2:
                    kept[col] = task
            else:  # regression / ranking
                if pd.to_numeric(s, errors="coerce").dropna().nunique() >= 3:
                    kept[col] = task
        if kept:
            out[ctype] = kept
    return out


def score_group(group: str):
    loaded = load_group(group)
    if loaded is None:
        log(f"  {group}: no data, skipping")
        return None, None
    meta, dists, methods, common = loaded
    if len(common) < 5:
        log(f"  {group}: only {len(common)} samples; benchmark underpowered, skipping")
        return None, None

    log(f"  {group}: {len(common)} samples, methods={methods}")
    ad = build_meta_adata(meta, dists, methods)
    schema = usable_schema(meta)

    from patpy.tl.evaluation import knn_prediction_score, trajectory_correlation

    long_df = knn_prediction_score(
        ad, schema, representations=methods,
        n_neighbors=N_NEIGHBORS, reverse_technical_score=True,
    )
    long_df.insert(0, "group", group)

    # Trajectory: diffusion pseudotime vs ordinal severity, rooted at the
    # lowest-severity (healthiest) sample.
    traj = pd.Series(np.nan, index=methods)
    sev = ad.obs["state_severity"]
    if sev.notna().sum() >= 4 and sev.dropna().nunique() >= 3:
        root = sev.idxmin()
        try:
            tdf = trajectory_correlation(
                meta_adata=ad, root_sample=root,
                trajectory_variable="state_severity",
                representations=methods, inverse_trajectory=False,
            )
            traj = tdf["correlation"].abs().reindex(methods)
        except Exception as e:  # noqa: BLE001
            log(f"    trajectory failed: {e}")

    # Per-method summary (SPARE buckets).
    def bucket_mean(m, ctype):
        sub = long_df[(long_df["representation"] == m)
                      & (long_df["covariate_type"] == ctype)]
        return sub["score"].mean() if len(sub) else np.nan

    rows = []
    for m in methods:
        info = bucket_mean(m, "relevant")
        batch = bucket_mean(m, "technical")
        tr = traj.get(m, np.nan)
        total = (2 * np.nan_to_num(info) + 2 * np.nan_to_num(tr)
                 + np.nan_to_num(batch)) / 5
        rows.append({
            "group": group, "representation": m,
            "Information retention": info, "Batch mixing": batch,
            "Trajectory": tr, "Total": total,
        })
    summary = pd.DataFrame(rows).sort_values("Total", ascending=False)
    return long_df, summary


# --------------------------------------------------------------------------
def styled_table(summary: pd.DataFrame, title: str, path_base):
    """Render the SPARE-style plottable Table (the tutorial table())."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar

    df = summary.copy()
    df["representation"] = df["representation"].map(C.REPR_DISPLAY).fillna(df["representation"])
    df = df.set_index("representation")
    cols = ["Total", "Information retention", "Trajectory", "Batch mixing"]
    df = df[cols].sort_values("Total", ascending=False).round(3)

    cmap_bar = LinearSegmentedColormap.from_list(
        "bugw", ["#FF9693", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256)
    bar_kw = {"cmap": cmap_bar, "plot_bg_bar": True, "annotate": True,
              "height": 0.5, "lw": 0.5, "formatter": lambda x: round(x, 2)}
    col_defs = [
        ColumnDefinition("Total", width=1.0, plot_fn=bar, plot_kw=bar_kw),
        ColumnDefinition("Information retention", width=1.1, plot_fn=bar,
                         plot_kw=bar_kw, group="aggregate scores"),
        ColumnDefinition("Trajectory", width=1.0, plot_fn=bar, plot_kw=bar_kw,
                         group="aggregate scores"),
        ColumnDefinition("Batch mixing", width=1.0, plot_fn=bar, plot_kw=bar_kw,
                         group="aggregate scores"),
    ]
    fig, ax = plt.subplots(figsize=(11, 0.7 * len(df) + 1.6))
    plt.rcParams["font.family"] = "Helvetica"
    Table(df, column_definitions=col_defs, ax=ax,
          row_dividers=True, footer_divider=True,
          textprops={"fontsize": 11, "ha": "center"},
          column_border_kw={"linewidth": 0.5, "color": "black"})
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote table -> {path_base}.{{pdf,png}}")


def main():
    long_all, summary_all = [], []
    for _, _, label in C.GROUPS:
        long_df, summary = score_group(label)
        if long_df is not None:
            long_all.append(long_df)
            summary_all.append(summary)
            styled_table(summary, f"Sample-representation benchmark - {label}",
                         C.OUT / f"benchmark_table_{label}")

    if not summary_all:
        log("no groups scored; aborting")
        return

    long_cat = pd.concat(long_all, ignore_index=True)
    summary_cat = pd.concat(summary_all, ignore_index=True)
    long_cat.to_csv(C.DATA / "benchmark_scores_long.csv", index=False)
    summary_cat.to_csv(C.DATA / "benchmark_summary.csv", index=False)
    log(f"wrote benchmark_scores_long.csv ({len(long_cat)} rows) and benchmark_summary.csv")

    # Mean across groups.
    mean_summary = (summary_cat.groupby("representation")[
        ["Information retention", "Trajectory", "Batch mixing", "Total"]]
        .mean().reset_index().sort_values("Total", ascending=False))
    mean_summary.to_csv(C.DATA / "benchmark_summary_mean.csv", index=False)
    styled_table(mean_summary, "Sample-representation benchmark - mean across tissues",
                 C.OUT / "benchmark_table_mean")
    log("done.")


if __name__ == "__main__":
    main()
