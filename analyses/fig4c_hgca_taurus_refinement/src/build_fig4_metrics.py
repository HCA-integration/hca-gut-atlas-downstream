"""
Build all auditable metrics tables for HGCA manuscript Fig 4a
(Taurus HGCA-transferred label refinement, cohort support, transfer certainty).

Inputs
------
* Taurus per-cell obs (light CSV export of the annotated v1 h5ad, or the
  h5ad itself if the CSV is missing). Pass ``--obs`` or set ``TAURUS_OBS``.
* HGCA taxonomy CSV with full ``hgca_celltype_level{1..5}`` + ``hgca_celltype_v1``
  paths. Default: the bundled ``data/demo/GCA_taxonomy_2026_CAP.csv``.

Outputs (all under ``publication2026/fig4_hgca_taurus_refinement/data/``)
-----------------------------------------------------------------------
* ``fig4a_lineage_label_counts.csv``    Author vs HGCA label counts overall
                                        and per lineage at every author
                                        resolution (major / minor /
                                        final_analysis / closest_GCA_celltype).
                                        For final_analysis rows also reports
                                        ``n_hgca_shared`` + ``n_hgca_novel``
                                        (must sum to ``n_hgca_v1_labels``).
* ``fig4a_refinement_by_cell.csv.gz``   Per-cell hierarchical classification:
                                        same_resolution_as_author /
                                        atlas_increased_resolution /
                                        atlas_reduced_resolution /
                                        minor_reassignment_from_author /
                                        moderate_reassignment_from_author /
                                        major_reassignment_from_author /
                                        low_confidence (entropy overlay) /
                                        no-crosswalk / absent-from-author-tax.
                                        Includes author_level{1,2,3}_match
                                        flags, lca_depth, path_dist, and
                                        taxonomy paths.
* ``fig4a_refinement_summary.csv``      Overall and per-lineage counts of each
                                        refinement class. Reports numerator
                                        AND denominator with each percent.
* ``fig4a_hgca_identity_evidence.csv``  One row per HGCA v1 identity seen in
                                        Taurus. Columns include lineage,
                                        n_cells, pct_within_lineage,
                                        n_samples, n_donors, sample_prev @
                                        (5, 10, 20) cells, donor_prev @
                                        (5, 10, 20), present-in-Healthy /
                                        -Pre / -Post flags, median max-post,
                                        median entropy, IQR entropy,
                                        pct_confident (@ entropy<0.5),
                                        single-donor / single-condition flags,
                                        author parent + refinement class of
                                        the majority of its cells.
* ``fig4a_headline_metrics.json``       Cohort-wide headline numbers with all
                                        thresholds explicit.

Notes on safeguards
-------------------
* Percent label increase is *reported*, never used as a headline claim of
  improvement; the panel foregrounds cell-level refinement + evidence.
* Prevalence at multiple cell thresholds (5/10/20) is a sensitivity check.
* We deliberately keep "same resolution as author" and "atlas increased
  resolution" separate (Fig 2 nomenclature). A cell whose HGCA path equals
  its author-crosswalked GCA path is *not* counted as increased resolution,
  even though the labelling system changed.
* PanGI comparison columns are intentionally absent; the panel scaffold has a
  reserved sidecar slot that can be populated once the cluster PanGI run
  becomes available. This module never claims HGCA improves on PanGI.

Usage
-----
python publication2026/fig4_hgca_taurus_refinement/src/build_fig4_metrics.py \
    [--obs PATH] [--taxonomy PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOG = logging.getLogger("build_fig4_metrics")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Author resolutions we compare against HGCA v1. `final_analysis` is the
# authors' finest published annotation; `closest_GCA_celltype` is their own
# manual crosswalk to GCA vocabulary and doubles as the semantic hook for the
# hierarchical refinement classification.
AUTHOR_RESOLUTIONS: List[str] = [
    "major",
    "minor",
    "final_analysis",
    "closest_GCA_celltype",
]

# We exclude these tokens from label-count denominators consistently.
LABEL_EXCLUDE_TOKENS: Tuple[str, ...] = (
    "unknown",
    "unassigned",
    "doublet",
    "low quality",
    "low_quality",
    "low-quality",
    "not predicted",
    "not_predicted",
    "nan",
    "",
)

# Two entropy thresholds on the SCANVI softmax entropy (nats):
#   * ``CONFIDENCE_ENTROPY_MAX`` — primary threshold used to decide whether
#     a per-cell hierarchical class is overridden with the ``low_confidence``
#     overlay label. At 1.0 nats the model has ~60% mass on its top pick;
#     at ≤ 1.0 the top class is a *clear plurality*, which is a defensible
#     bar for keeping the model's call.
#   * ``HIGH_CONFIDENCE_ENTROPY_MAX`` — strict tier for "the model is
#     confident" reporting; below 0.5 nats corresponds to ~80% mass on the
#     top pick. Kept as a sensitivity number, not used as an overlay.
CONFIDENCE_ENTROPY_MAX = 1.0
HIGH_CONFIDENCE_ENTROPY_MAX = 0.5

# Sample-prevalence cell-count thresholds (main + sensitivity check).
SAMPLE_PREVALENCE_THRESHOLDS: Tuple[int, ...] = (5, 10, 20)

# Support thresholds for the low-support flag (main threshold).
MIN_CELLS_FOR_SUPPORT = 10
MIN_DONORS_FOR_SUPPORT = 2
MIN_SAMPLES_FOR_SUPPORT = 2

# Column expected on the Taurus obs table.
REQUIRED_OBS_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "Patient",
    "Disease",
    "Site",
    "Treatment",
    "Inflammation",
    "assigned_lineage",
    "mapping_lineage",
    "predicted_hgca_celltype_v1",
    "uncertainty_hgca_celltype_v1",
    "closest_GCA_celltype",
    "final_analysis",
    "major",
    "minor",
)

LINEAGES_ORDERED: Tuple[str, ...] = ("lymphoid", "stroma", "epithelial", "myeloid")
# Lineage scopes for Fig4 follow the lineage-restricted transfer strata
# (which pickle/model labeled the cell), not author assigned_lineage tags.
LINEAGE_SCOPE_COL = "mapping_lineage"

# Hierarchical classes. Resolution classes match Fig 2; former
# ``changed_branch_from_author`` is split by LCA depth into three
# reassignment severities (see ``_reassignment_class``).
# Display names (for plots / legends) live in render_fig4_panel.py.
HIERARCHY_CLASSES: Tuple[str, ...] = (
    "atlas_increased_resolution",
    "same_resolution_as_author",
    "atlas_reduced_resolution",
    "minor_reassignment_from_author",
    "moderate_reassignment_from_author",
    "major_reassignment_from_author",
    "low_confidence",
    "uncertainty_unavailable",
    "no_author_crosswalk",
    "author_absent_from_taxonomy",
    "predicted_absent_from_taxonomy",
    "absent_from_taxonomy",
    "no_prediction",
    "unmapped",
)

# Classes that used to be lumped as "changed branch".
REASSIGNMENT_CLASSES: Tuple[str, ...] = (
    "minor_reassignment_from_author",
    "moderate_reassignment_from_author",
    "major_reassignment_from_author",
)

# Legacy Fig 4 / Fig 2 names → current names (kept for reading older CSVs).
LEGACY_CLASS_TO_FIG2: Dict[str, str] = {
    "refined": "atlas_increased_resolution",
    "concordant": "same_resolution_as_author",
    "coarsened": "atlas_reduced_resolution",
    "relabelled": "major_reassignment_from_author",
    "changed_branch_from_author": "major_reassignment_from_author",
}

# Reassignment severity from taxonomy *path distance* (edges between labels),
# not absolute LCA depth. Absolute LCA depth mis-grades shallow trees: S1 vs S3
# fibroblasts share only level-2 "Fibroblasts" (LCA=2) but are siblings.
#   path_dist = (len(author_path) - lca_depth) + (len(pred_path) - lca_depth)
#   minor    ≤ 3  — siblings (2) or one-step aunt/niece (3), e.g. S1↔S3,
#                   mid↔tip villus, arteriolar↔PAC
#   moderate = 4  — cousins under the same mid-level family, e.g. lower-villus
#                   enterocyte ↔ mid-crypt colonocyte
#   major    ≥ 5  — early fork, e.g. TA (Proliferative) ↔ colonocyte progenitor
#                   (Absorptive), sharing only lineage
MINOR_REASSIGNMENT_MAX_PATH_DIST = 3
MODERATE_REASSIGNMENT_PATH_DIST = 4


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _normalize_label(x: object) -> str:
    """Strip whitespace / newlines so taxonomy lookups match Fig 2."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split())

def _configure_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )


def _is_excluded_label(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    s = str(x).strip().lower()
    return s in LABEL_EXCLUDE_TOKENS


def _eligible_labels(series: pd.Series) -> pd.Series:
    """Series → eligible label strings with newlines collapsed (Fig2-consistent)."""
    normalized = series.map(_normalize_label)
    mask = ~normalized.map(_is_excluded_label)
    return normalized[mask]


def _load_obs(obs_path: Path) -> pd.DataFrame:
    LOG.info("loading obs table: %s", obs_path)
    if obs_path.suffix.lower() == ".h5ad":
        import anndata as ad
        a = ad.read_h5ad(obs_path, backed="r")
        df = a.obs[list(REQUIRED_OBS_COLUMNS)].copy()
        a.file.close()
    else:
        df = pd.read_csv(
            obs_path,
            usecols=list(REQUIRED_OBS_COLUMNS),
            low_memory=False,
        )
    # Collapse whitespace/newlines in label columns so vocabulary arithmetic
    # cannot double-count "Foo\\n(Bar)" vs "Foo (Bar)".
    for col in (
        "predicted_hgca_celltype_v1",
        "closest_GCA_celltype",
        "final_analysis",
        "major",
        "minor",
    ):
        if col in df.columns:
            df[col] = df[col].map(_normalize_label)
    LOG.info("loaded %s cells with %d columns", f"{len(df):,}", df.shape[1])
    return df


def _load_taxonomy(tax_path: Path) -> pd.DataFrame:
    LOG.info("loading HGCA taxonomy: %s", tax_path)
    tax = pd.read_csv(tax_path, low_memory=False)
    keep = [c for c in tax.columns if c.startswith("hgca_celltype_level")] + [
        "hgca_celltype_v1",
    ]
    if "hgca_celltype_v0" in tax.columns:
        keep.append("hgca_celltype_v0")
    tax = tax[keep].copy()
    for c in tax.columns:
        tax[c] = tax[c].astype("string").str.strip()
    LOG.info("taxonomy: %d rows, level cols = %s", len(tax), keep)
    return tax


# -----------------------------------------------------------------------------
# Taxonomy path utilities
# -----------------------------------------------------------------------------

@dataclass
class TaxonomyLookup:
    """Lookup helpers for HGCA hierarchical paths.

    ``paths_by_label`` maps *any* label seen at any level to the deepest tuple
    of level-columns whose deepest cell equals that label. For labels reused at
    multiple depths (rare in this taxonomy but possible), the deepest is used.
    ``level_of_label`` records the depth at which each label sits (1..5).
    """
    level_cols: List[str]
    paths_by_label: Dict[str, Tuple[str, ...]]
    level_of_label: Dict[str, int]
    v1_to_path: Dict[str, Tuple[str, ...]]


def _longest_common_prefix(paths: List[Tuple[str, ...]]) -> Tuple[str, ...]:
    """Shared ancestral path across several taxonomy paths (Fig 2 logic)."""
    if not paths:
        return ()
    prefix = list(paths[0])
    for path in paths[1:]:
        shared = 0
        for left, right in zip(prefix, path):
            if left != right:
                break
            shared += 1
        prefix = prefix[:shared]
        if not prefix:
            break
    return tuple(prefix)


def _build_taxonomy_lookup(tax: pd.DataFrame) -> TaxonomyLookup:
    """Build hierarchical paths aligned with Fig 2 ``_taxonomy_paths``.

    Author / coarse terms (v0 or any non-leaf label) resolve to the **longest
    common prefix** of all taxonomy rows that carry that label. HGCA v1 leaves
    resolve to their full level1…level5 path. Labels are newline-normalized so
    taxonomy strings like ``Monocyte Derived Dendritic Cells\\n(MO DC)`` match.
    """
    level_cols = [c for c in tax.columns if c.startswith("hgca_celltype_level")]
    level_cols = sorted(level_cols, key=lambda c: int(c.replace("hgca_celltype_level", "")))

    row_paths: List[Tuple[str, ...]] = []
    for _, row in tax.iterrows():
        levels: List[str] = []
        for c in level_cols:
            v = row[c]
            if pd.isna(v):
                break
            levels.append(_normalize_label(v))
        row_paths.append(tuple(levels))

    # v1 leaf → full path
    v1_to_path: Dict[str, Tuple[str, ...]] = {}
    for (_, row), path in zip(tax.iterrows(), row_paths):
        v1 = _normalize_label(row.get("hgca_celltype_v1"))
        if not v1 or not path:
            continue
        v1_to_path.setdefault(v1, path)

    # Any label → LCP of all rows containing that label (Fig 2 author-path rule)
    label_to_paths: Dict[str, List[Tuple[str, ...]]] = {}
    for path in row_paths:
        for depth in range(1, len(path) + 1):
            label_to_paths.setdefault(path[depth - 1], []).append(path[:depth])
    for v1, path in v1_to_path.items():
        label_to_paths.setdefault(v1, []).append(path)
    if "hgca_celltype_v0" in tax.columns:
        for (_, row), path in zip(tax.iterrows(), row_paths):
            v0 = _normalize_label(row.get("hgca_celltype_v0"))
            if v0 and path:
                label_to_paths.setdefault(v0, []).append(path)

    paths: Dict[str, Tuple[str, ...]] = {
        label: _longest_common_prefix(plist)
        for label, plist in label_to_paths.items()
        if plist
    }
    # Prefer the dedicated v1 path when available (leaf, not LCP-collapsed).
    paths.update(v1_to_path)
    depth = {label: len(path) for label, path in paths.items()}

    LOG.info("taxonomy lookup: %d labels indexed, %d v1 leaves",
             len(paths), len(v1_to_path))
    return TaxonomyLookup(
        level_cols=level_cols,
        paths_by_label=paths,
        level_of_label=depth,
        v1_to_path=v1_to_path,
    )


def _path_distance(author_path: Tuple[str, ...], predicted_path: Tuple[str, ...],
                   lca_depth: int) -> int:
    """Number of taxonomy edges between two labels via their LCA."""
    return (len(author_path) - lca_depth) + (len(predicted_path) - lca_depth)


def _reassignment_class(path_dist: int) -> str:
    """Grade a true branch change by tree path distance (not absolute LCA depth).

    * minor — path_dist ≤ 3 (siblings or one-step aunt/niece)
    * moderate — path_dist == 4 (cousins in the same mid-level family)
    * major — path_dist ≥ 5 (early fork near the lineage root)
    """
    if path_dist <= MINOR_REASSIGNMENT_MAX_PATH_DIST:
        return "minor_reassignment_from_author"
    if path_dist == MODERATE_REASSIGNMENT_PATH_DIST:
        return "moderate_reassignment_from_author"
    return "major_reassignment_from_author"


def _classify_pair(
    author_gca: str,
    predicted_v1: str,
    tax: TaxonomyLookup,
) -> Tuple[str, str, str, Optional[int], Optional[int]]:
    """Classify one (author-crosswalk, HGCA-v1) pair.

    Returns:
        hierarchy_class,
        author_path_joined ("|"-separated),
        predicted_path_joined ("|"-separated),
        lca_depth (shared ancestral levels; None when paths unavailable),
        path_dist (edges between labels; None when paths unavailable)
    """
    author = _normalize_label(author_gca)
    predicted = _normalize_label(predicted_v1)
    if _is_excluded_label(author):
        return ("no_author_crosswalk", "", "", None, None)
    if _is_excluded_label(predicted):
        return ("no_prediction", "", "", None, None)

    author_path = tax.paths_by_label.get(author)
    predicted_path = tax.v1_to_path.get(predicted) or tax.paths_by_label.get(predicted)

    if author_path is None and predicted_path is None:
        return ("absent_from_taxonomy", "", "", None, None)
    if author_path is None:
        return ("author_absent_from_taxonomy", "", "|".join(predicted_path), None, None)
    if predicted_path is None:
        return ("predicted_absent_from_taxonomy", "|".join(author_path), "", None, None)

    ap = "|".join(author_path)
    pp = "|".join(predicted_path)
    common = len(_longest_common_prefix([author_path, predicted_path]))
    dist = _path_distance(author_path, predicted_path, common)
    if author_path == predicted_path:
        return ("same_resolution_as_author", ap, pp, common, dist)
    if (
        len(author_path) < len(predicted_path)
        and common == len(author_path)
    ):
        return ("atlas_increased_resolution", ap, pp, common, dist)
    if (
        len(predicted_path) < len(author_path)
        and common == len(predicted_path)
    ):
        return ("atlas_reduced_resolution", ap, pp, common, dist)
    return (_reassignment_class(dist), ap, pp, common, dist)


def _level_matches(
    author_path: Tuple[str, ...],
    predicted_path: Tuple[str, ...],
    max_level: int = 3,
) -> Dict[str, Optional[float]]:
    """Fig 2 author-match-level flags (NaN when either path is too shallow)."""
    out: Dict[str, Optional[float]] = {}
    for level in range(1, max_level + 1):
        key = f"author_level{level}_match"
        if len(author_path) < level or len(predicted_path) < level:
            out[key] = None
        else:
            out[key] = float(author_path[level - 1] == predicted_path[level - 1])
    return out


# -----------------------------------------------------------------------------
# Metric builders
# -----------------------------------------------------------------------------

def build_label_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Author vs HGCA label counts, overall and per lineage, at each author
    resolution.

    For ``final_analysis`` rows, also report the HGCA vocabulary split:
    * ``n_hgca_shared`` — HGCA v1 labels already present in the author
      ``closest_GCA_celltype`` crosswalk vocabulary (global cohort set)
    * ``n_hgca_novel`` — HGCA v1 labels absent from that crosswalk

    Invariant enforced for every final_analysis row:
    ``n_hgca_shared + n_hgca_novel == n_hgca_v1_labels``.
    """
    if LINEAGE_SCOPE_COL not in df.columns:
        raise KeyError(
            f"obs is missing {LINEAGE_SCOPE_COL!r}; attach lineage-pickle "
            "mapping_lineage before rebuilding Fig4"
        )

    rows: List[Dict[str, object]] = []
    # Overall includes every cell; lineage scopes are transfer strata only.
    mapped = df[df[LINEAGE_SCOPE_COL].notna() & ~df[LINEAGE_SCOPE_COL].map(_is_excluded_label)]
    scopes: List[Tuple[str, pd.DataFrame]] = [("overall", mapped)]
    for lineage, sub in mapped.groupby(LINEAGE_SCOPE_COL, dropna=False):
        if _is_excluded_label(lineage):
            continue
        if str(lineage) not in LINEAGES_ORDERED:
            LOG.warning("skipping unexpected mapping lineage scope: %s", lineage)
            continue
        scopes.append((str(lineage), sub))

    # Novel/shared is defined against the cohort-wide author GCA crosswalk.
    author_gca_vocab = set(_eligible_labels(mapped["closest_GCA_celltype"]).unique())
    LOG.info(
        "author closest_GCA vocabulary for novel/shared split: %d terms",
        len(author_gca_vocab),
    )
    LOG.info(
        "lineage scopes use %s (%d / %d cells mapped)",
        LINEAGE_SCOPE_COL,
        len(mapped),
        len(df),
    )

    for scope_name, sub in scopes:
        hgca_series = _eligible_labels(sub["predicted_hgca_celltype_v1"])
        hgca_set = set(hgca_series.unique())
        n_hgca_labels = int(len(hgca_set))
        n_hgca_cells = int(len(hgca_series))
        n_hgca_shared = int(len(hgca_set & author_gca_vocab))
        n_hgca_novel = int(len(hgca_set - author_gca_vocab))
        if n_hgca_shared + n_hgca_novel != n_hgca_labels:
            raise AssertionError(
                f"{scope_name}: shared ({n_hgca_shared}) + novel "
                f"({n_hgca_novel}) != HGCA total ({n_hgca_labels})"
            )
        for author_res in AUTHOR_RESOLUTIONS:
            author_series = _eligible_labels(sub[author_res])
            n_author_labels = int(author_series.nunique())
            n_author_cells = int(len(author_series))
            pct = np.nan
            if n_author_labels > 0:
                pct = 100.0 * (n_hgca_labels - n_author_labels) / n_author_labels
            rows.append(
                {
                    "scope": scope_name,
                    "author_resolution": author_res,
                    "lineage_scope": LINEAGE_SCOPE_COL,
                    "n_author_labels": n_author_labels,
                    "n_hgca_v1_labels": n_hgca_labels,
                    "n_hgca_shared": n_hgca_shared,
                    "n_hgca_novel": n_hgca_novel,
                    "n_author_cells": n_author_cells,
                    "n_hgca_cells": n_hgca_cells,
                    "pct_label_change": pct,
                    "hgca_shannon_effective":
                        _effective_shannon(hgca_series),
                    "author_shannon_effective":
                        _effective_shannon(author_series),
                }
            )
    out = pd.DataFrame(rows)
    # Transfer strata are disjoint: lineage HGCA uniques must sum to overall.
    fa = out.query("author_resolution == 'final_analysis'")
    overall = int(fa.loc[fa["scope"] == "overall", "n_hgca_v1_labels"].iloc[0])
    lineage_sum = int(
        fa.loc[fa["scope"].isin(LINEAGES_ORDERED), "n_hgca_v1_labels"].sum()
    )
    if lineage_sum != overall:
        raise AssertionError(
            f"mapping-lineage HGCA counts must be disjoint: "
            f"sum(lineages)={lineage_sum} != overall={overall}"
        )
    LOG.info(
        "HGCA vocabulary disjoint across %s: overall=%d = sum(lineages)=%d",
        LINEAGE_SCOPE_COL,
        overall,
        lineage_sum,
    )
    return out


def _effective_shannon(series: pd.Series) -> float:
    """Exp of Shannon entropy = effective number of labels."""
    vals = _eligible_labels(series)
    if len(vals) == 0:
        return np.nan
    p = vals.value_counts(normalize=True).values
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(p * np.log(p, where=p > 0))
    return float(np.exp(H))


def build_refinement_per_cell(
    df: pd.DataFrame,
    tax: TaxonomyLookup,
) -> pd.DataFrame:
    """Per-cell hierarchical classification (Fig 2 nomenclature).

    Low-confidence cells (entropy > threshold) get overridden to
    ``low_confidence`` for accounting; the underlying hierarchical class is
    still stored in ``refinement_class_taxonomy`` for audit. Also stores
    Fig 2 ``author_level{1,2,3}_match`` flags per cell.
    """
    LOG.info("classifying %s cells", f"{len(df):,}")

    # unique (author_gca, predicted_v1) pairs → classify once, then join back
    df = df.copy()
    df["_pair"] = list(zip(
        df["closest_GCA_celltype"].astype("string"),
        df["predicted_hgca_celltype_v1"].astype("string"),
    ))

    pair_index = df["_pair"].drop_duplicates()
    classified: Dict[
        Tuple[str, str], Tuple[str, str, str, Optional[int], Optional[int]]
    ] = {}
    level_by_pair: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for pair in pair_index:
        a = "" if pd.isna(pair[0]) else str(pair[0])
        p = "" if pd.isna(pair[1]) else str(pair[1])
        cls, ap, pp, lca, dist = _classify_pair(a, p, tax)
        classified[pair] = (cls, ap, pp, lca, dist)
        author_path = tuple(ap.split("|")) if ap else ()
        pred_path = tuple(pp.split("|")) if pp else ()
        level_by_pair[pair] = _level_matches(author_path, pred_path)

    cls_series = df["_pair"].map(lambda k: classified[k])
    df["refinement_class_taxonomy"] = [c[0] for c in cls_series]
    df["author_taxonomy_path"] = [c[1] for c in cls_series]
    df["predicted_taxonomy_path"] = [c[2] for c in cls_series]
    df["lca_depth"] = [c[3] for c in cls_series]
    df["path_dist"] = [c[4] for c in cls_series]
    for level in (1, 2, 3):
        key = f"author_level{level}_match"
        df[key] = df["_pair"].map(lambda k, key=key: level_by_pair[k][key])

    entropy = pd.to_numeric(df["uncertainty_hgca_celltype_v1"], errors="coerce")
    df["uncertainty_missing"] = entropy.isna()
    # Only cells with exported entropy can enter the low-confidence overlay.
    df["low_confidence"] = (entropy > CONFIDENCE_ENTROPY_MAX) & ~df["uncertainty_missing"]
    df["high_confidence"] = (
        (entropy <= HIGH_CONFIDENCE_ENTROPY_MAX) & ~df["uncertainty_missing"]
    )

    # Display class: Fig2 taxonomy, with low-confidence overlay when entropy
    # exists. If entropy is missing (new hard-label export), keep the taxonomy
    # class visible and account for the gap via uncertainty_missing.
    df["refinement_class"] = np.where(
        df["low_confidence"], "low_confidence", df["refinement_class_taxonomy"]
    )
    df.drop(columns=["_pair"], inplace=True)
    return df


def build_refinement_summary(cell_class: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell hierarchy classes overall and per lineage. Every
    percent has its numerator + denominator (n_cells) reported."""
    lineage_col = (
        LINEAGE_SCOPE_COL if LINEAGE_SCOPE_COL in cell_class.columns
        else "assigned_lineage"
    )
    mapped = cell_class[
        cell_class[lineage_col].notna() & ~cell_class[lineage_col].map(_is_excluded_label)
    ]
    scopes: List[Tuple[str, pd.DataFrame]] = [("overall", mapped)]
    for lineage, sub in mapped.groupby(lineage_col, dropna=False):
        if _is_excluded_label(lineage):
            continue
        scopes.append((str(lineage), sub))

    rows: List[Dict[str, object]] = []
    for scope_name, sub in scopes:
        n_total = len(sub)
        counts = sub["refinement_class"].value_counts()
        # ensure fixed schema for readability (Fig 2 nomenclature)
        for cls in [
            "atlas_increased_resolution",
            "same_resolution_as_author",
            "atlas_reduced_resolution",
            "minor_reassignment_from_author",
            "moderate_reassignment_from_author",
            "major_reassignment_from_author",
            "low_confidence",
            "no_author_crosswalk",
            "author_absent_from_taxonomy",
            "predicted_absent_from_taxonomy",
            "absent_from_taxonomy",
            "no_prediction",
            "unmapped",
        ]:
            n = int(counts.get(cls, 0))
            rows.append(
                {
                    "scope": scope_name,
                    "refinement_class": cls,
                    "n_cells": n,
                    "n_cells_in_scope": n_total,
                    "pct_of_scope": (100.0 * n / n_total) if n_total else 0.0,
                }
            )
        n_uncert_missing = int(sub["uncertainty_missing"].sum()) if (
            "uncertainty_missing" in sub.columns
        ) else 0
        rows.append(
            {
                "scope": scope_name,
                "refinement_class": "uncertainty_unavailable",
                "n_cells": n_uncert_missing,
                "n_cells_in_scope": n_total,
                "pct_of_scope": (
                    (100.0 * n_uncert_missing / n_total) if n_total else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_identity_evidence(cell_class: pd.DataFrame) -> pd.DataFrame:
    """One row per (lineage, HGCA v1 identity) actually seen in Taurus.

    Columns are: cohort support, transfer certainty, disease-state presence,
    robustness flags, majority refinement class + majority author parent,
    plus ``novel_hgca_identity`` = True when the HGCA v1 label was *never*
    used by the Taurus authors' own crosswalk vocabulary
    (``closest_GCA_celltype``). Novel identities are the headline evidence
    that HGCA label transfer *names* cell types Taurus did not.
    """
    LOG.info("building per-identity evidence table")
    df = cell_class.copy()
    lineage_col = (
        LINEAGE_SCOPE_COL if LINEAGE_SCOPE_COL in df.columns else "assigned_lineage"
    )
    df["predicted_hgca_celltype_v1"] = df["predicted_hgca_celltype_v1"].astype(str)
    df = df[~df["predicted_hgca_celltype_v1"].map(_is_excluded_label)]
    df = df[df[lineage_col].notna() & ~df[lineage_col].map(_is_excluded_label)]
    df["uncertainty_hgca_celltype_v1"] = pd.to_numeric(
        df["uncertainty_hgca_celltype_v1"], errors="coerce"
    )
    # Evidence rows are keyed by transfer stratum.
    df["lineage"] = df[lineage_col].astype(str)

    # author vocabulary = set of terms the Taurus authors' own crosswalk maps to
    author_vocab = set(_eligible_labels(df["closest_GCA_celltype"]).unique())
    LOG.info("  Taurus author vocabulary (closest_GCA_celltype): %d unique terms",
             len(author_vocab))
    LOG.info("  identity evidence lineage key: %s", lineage_col)

    # per-sample cell counts once (used to compute prevalence at multiple thresholds)
    LOG.info("  computing per-sample per-identity counts")
    per_sample = (
        df.groupby(["lineage", "predicted_hgca_celltype_v1", "sample_id"],
                   observed=True)
        .size()
        .rename("cells_in_sample")
        .reset_index()
    )
    per_donor = (
        df.groupby(["lineage", "predicted_hgca_celltype_v1", "Patient"],
                   observed=True)
        .size()
        .rename("cells_in_donor")
        .reset_index()
    )
    total_samples_per_lineage = (
        df.groupby("lineage", observed=True)["sample_id"].nunique()
    )
    total_donors_per_lineage = (
        df.groupby("lineage", observed=True)["Patient"].nunique()
    )
    cells_per_lineage = df.groupby("lineage", observed=True).size()

    LOG.info("  computing per-identity aggregates")
    grouped = df.groupby(["lineage", "predicted_hgca_celltype_v1"], observed=True)

    rows: List[Dict[str, object]] = []
    for (lineage, identity), sub in grouped:
        if _is_excluded_label(lineage) or _is_excluded_label(identity):
            continue

        n_cells = len(sub)
        n_samples = int(sub["sample_id"].nunique())
        n_donors = int(sub["Patient"].nunique())

        # prevalence @ threshold: fraction of samples/donors with >=k cells
        per_sample_this = per_sample[
            (per_sample["lineage"] == lineage)
            & (per_sample["predicted_hgca_celltype_v1"] == identity)
        ]["cells_in_sample"]
        per_donor_this = per_donor[
            (per_donor["lineage"] == lineage)
            & (per_donor["predicted_hgca_celltype_v1"] == identity)
        ]["cells_in_donor"]
        prevalence: Dict[str, object] = {}
        for k in SAMPLE_PREVALENCE_THRESHOLDS:
            n_s_over_k = int((per_sample_this >= k).sum())
            n_d_over_k = int((per_donor_this >= k).sum())
            prevalence[f"n_samples_ge{k}"] = n_s_over_k
            prevalence[f"n_donors_ge{k}"] = n_d_over_k
            prevalence[f"sample_prevalence_ge{k}"] = (
                n_s_over_k / int(total_samples_per_lineage[lineage])
                if total_samples_per_lineage[lineage] else np.nan
            )
            prevalence[f"donor_prevalence_ge{k}"] = (
                n_d_over_k / int(total_donors_per_lineage[lineage])
                if total_donors_per_lineage[lineage] else np.nan
            )

        # transfer certainty aggregates
        ent = sub["uncertainty_hgca_celltype_v1"].dropna()
        med_ent = float(ent.median()) if len(ent) else np.nan
        iqr_ent = (
            float(ent.quantile(0.75) - ent.quantile(0.25)) if len(ent) else np.nan
        )
        pct_confident = (
            100.0 * float((ent <= CONFIDENCE_ENTROPY_MAX).mean())
            if len(ent) else np.nan
        )
        pct_high_confident = (
            100.0 * float((ent <= HIGH_CONFIDENCE_ENTROPY_MAX).mean())
            if len(ent) else np.nan
        )

        # disease-state presence
        disease_counts = sub["Disease"].value_counts()
        treatment_counts = sub["Treatment"].value_counts()
        present_healthy = int(disease_counts.get("Healthy", 0)) > 0
        present_pre = int(
            sub[(sub["Disease"].isin(("CD", "UC"))) & (sub["Treatment"] == "Pre")]
            .shape[0]
        ) > 0
        present_post = int(
            sub[(sub["Disease"].isin(("CD", "UC"))) & (sub["Treatment"] == "Post")]
            .shape[0]
        ) > 0

        # majority refinement class + author parent
        refinement_counts = sub["refinement_class"].value_counts()
        majority_refinement = str(refinement_counts.idxmax())
        # author parent = most common non-empty author_taxonomy_path[-1]
        parents = sub.loc[
            sub["author_taxonomy_path"].astype(str) != "", "author_taxonomy_path"
        ].astype(str)
        if len(parents):
            top_parent_path = parents.value_counts().idxmax()
            majority_author_parent = str(top_parent_path).split("|")[-1]
        else:
            majority_author_parent = ""

        # robustness flags
        low_support = (
            n_cells < MIN_CELLS_FOR_SUPPORT
            or n_donors < MIN_DONORS_FOR_SUPPORT
            or n_samples < MIN_SAMPLES_FOR_SUPPORT
        )
        single_donor = n_donors <= 1
        single_condition = (int(present_healthy) + int(present_pre) + int(present_post)) <= 1

        pct_within_lineage = (
            100.0 * n_cells / int(cells_per_lineage[lineage])
            if cells_per_lineage[lineage] else np.nan
        )

        novel_flag = str(identity).strip() not in author_vocab
        row = {
            "lineage": lineage,
            "hgca_celltype_v1": identity,
            "novel_hgca_identity": bool(novel_flag),
            "n_cells": n_cells,
            "pct_within_lineage": pct_within_lineage,
            "n_samples": n_samples,
            "n_donors": n_donors,
            "median_entropy": med_ent,
            "iqr_entropy": iqr_ent,
            "pct_confident": pct_confident,
            "pct_high_confident": pct_high_confident,
            "present_healthy": bool(present_healthy),
            "present_pre_treatment": bool(present_pre),
            "present_post_treatment": bool(present_post),
            "majority_refinement_class": majority_refinement,
            "majority_author_parent": majority_author_parent,
            "low_support_flag": bool(low_support),
            "single_donor_flag": bool(single_donor),
            "single_condition_flag": bool(single_condition),
        }
        row.update(prevalence)
        rows.append(row)

    ordered_cols = [
        "lineage",
        "hgca_celltype_v1",
        "novel_hgca_identity",
        "majority_refinement_class",
        "majority_author_parent",
        "n_cells",
        "pct_within_lineage",
        "n_samples",
        "n_donors",
        *[f"n_samples_ge{k}" for k in SAMPLE_PREVALENCE_THRESHOLDS],
        *[f"n_donors_ge{k}" for k in SAMPLE_PREVALENCE_THRESHOLDS],
        *[f"sample_prevalence_ge{k}" for k in SAMPLE_PREVALENCE_THRESHOLDS],
        *[f"donor_prevalence_ge{k}" for k in SAMPLE_PREVALENCE_THRESHOLDS],
        "median_entropy",
        "iqr_entropy",
        "pct_confident",
        "pct_high_confident",
        "present_healthy",
        "present_pre_treatment",
        "present_post_treatment",
        "low_support_flag",
        "single_donor_flag",
        "single_condition_flag",
    ]
    out = pd.DataFrame(rows)[ordered_cols].sort_values(
        ["lineage", "n_cells"], ascending=[True, False]
    )
    return out


def build_headline_metrics(
    label_counts: pd.DataFrame,
    ref_summary: pd.DataFrame,
    identity_evidence: pd.DataFrame,
    cell_class: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Concise, headline-ready numbers with denominators + thresholds."""
    def _pct_row(scope: str, increased_only: bool = True) -> Dict[str, float | int]:
        sub = ref_summary[ref_summary["scope"] == scope]
        total = int(sub["n_cells_in_scope"].iloc[0]) if len(sub) else 0
        classes = (
            ("atlas_increased_resolution",)
            if increased_only
            else ("atlas_increased_resolution", "same_resolution_as_author")
        )
        n = int(sub[sub["refinement_class"].isin(classes)]["n_cells"].sum())
        return {"n": n, "denominator": total,
                "percent": (100.0 * n / total) if total else 0.0}

    identity_overall_multi_donor = int(
        (identity_evidence["n_donors"] >= MIN_DONORS_FOR_SUPPORT).sum()
    )
    identity_overall_multi_sample = int(
        (identity_evidence["n_samples"] >= MIN_SAMPLES_FOR_SUPPORT).sum()
    )
    identity_overall_low_support = int(identity_evidence["low_support_flag"].sum())

    per_lineage_label_gain = (
        label_counts.query("author_resolution == 'final_analysis'")
        .set_index("scope")[
            [
                "n_author_labels",
                "n_hgca_v1_labels",
                "n_hgca_shared",
                "n_hgca_novel",
                "pct_label_change",
            ]
        ]
        .to_dict(orient="index")
    )

    per_lineage_multi_donor_multi_sample = (
        identity_evidence.groupby("lineage")
        .apply(lambda g: {
            "n_identities": int(len(g)),
            "n_novel_from_hgca": int(g["novel_hgca_identity"].sum()),
            "n_shared_with_author_crosswalk": int((~g["novel_hgca_identity"]).sum()),
            "n_multi_donor": int((g["n_donors"] >= MIN_DONORS_FOR_SUPPORT).sum()),
            "n_multi_sample": int((g["n_samples"] >= MIN_SAMPLES_FOR_SUPPORT).sum()),
            "n_low_support": int(g["low_support_flag"].sum()),
            "n_single_donor": int(g["single_donor_flag"].sum()),
            "n_single_condition": int(g["single_condition_flag"].sum()),
        })
        .to_dict()
    )

    n_hgca_seen = int(identity_evidence["hgca_celltype_v1"].nunique())
    # Unique HGCA labels, not lineage×identity rows (labels can recur across lineages).
    n_novel_overall = int(
        identity_evidence.loc[
            identity_evidence["novel_hgca_identity"], "hgca_celltype_v1"
        ].nunique()
    )
    n_shared_overall = int(
        identity_evidence.loc[
            ~identity_evidence["novel_hgca_identity"], "hgca_celltype_v1"
        ].nunique()
    )
    if n_shared_overall + n_novel_overall != n_hgca_seen:
        raise AssertionError(
            f"unique shared ({n_shared_overall}) + novel ({n_novel_overall}) "
            f"!= HGCA seen ({n_hgca_seen})"
        )
    # top-N novel identities (largest cohorts) for the manuscript legend
    top_novel = (
        identity_evidence[identity_evidence["novel_hgca_identity"]]
        .sort_values("n_cells", ascending=False)
        .head(12)
        [["lineage", "hgca_celltype_v1", "n_cells", "n_donors", "n_samples"]]
        .to_dict(orient="records")
    )

    headline = {
        "config": {
            "confidence_entropy_max": CONFIDENCE_ENTROPY_MAX,
            "high_confidence_entropy_max": HIGH_CONFIDENCE_ENTROPY_MAX,
            "sample_prevalence_thresholds": list(SAMPLE_PREVALENCE_THRESHOLDS),
            "min_cells_for_support": MIN_CELLS_FOR_SUPPORT,
            "min_donors_for_support": MIN_DONORS_FOR_SUPPORT,
            "min_samples_for_support": MIN_SAMPLES_FOR_SUPPORT,
            "author_resolutions_reported": AUTHOR_RESOLUTIONS,
            "primary_author_resolution": "final_analysis",
        },
        "per_lineage_label_gain_final_analysis_vs_hgca_v1": per_lineage_label_gain,
        "atlas_increased_resolution_cells_overall_vs_scope":
            _pct_row("overall", increased_only=True),
        "same_or_increased_resolution_cells_overall_vs_scope":
            _pct_row("overall", increased_only=False),
        # level-match rates (Fig 2 sidecar metrics), among cells with evaluable depth
        "author_level1_match_rate": (
            float(pd.to_numeric(cell_class["author_level1_match"], errors="coerce").mean())
            if cell_class is not None and "author_level1_match" in cell_class.columns
            else None
        ),
        "author_level2_match_rate": (
            float(pd.to_numeric(cell_class["author_level2_match"], errors="coerce").mean())
            if cell_class is not None and "author_level2_match" in cell_class.columns
            else None
        ),
        "author_level3_match_rate": (
            float(pd.to_numeric(cell_class["author_level3_match"], errors="coerce").mean())
            if cell_class is not None and "author_level3_match" in cell_class.columns
            else None
        ),
        "n_hgca_identities_seen_in_taurus": n_hgca_seen,
        "n_hgca_shared_with_author_closest_gca": n_shared_overall,
        "n_novel_hgca_identities_not_in_taurus_vocabulary": n_novel_overall,
        "pct_novel_hgca_identities":
            100.0 * n_novel_overall / n_hgca_seen if n_hgca_seen else 0.0,
        "top_novel_hgca_identities_by_cohort": top_novel,
        "n_identities_multi_donor_support":
            identity_overall_multi_donor,
        "n_identities_multi_sample_support":
            identity_overall_multi_sample,
        "n_identities_failing_low_support_threshold":
            identity_overall_low_support,
        "per_lineage_identity_evidence_counts":
            per_lineage_multi_donor_multi_sample,
    }
    return headline


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--obs",
        default=Path(os.environ["TAURUS_OBS"]) if os.environ.get("TAURUS_OBS") else None,
        type=Path,
    )
    p.add_argument(
        "--taxonomy",
        default=Path(__file__).resolve().parents[3]
        / "data"
        / "demo"
        / "GCA_taxonomy_2026_CAP.csv",
        type=Path,
    )
    p.add_argument(
        "--out-dir",
        default=Path(__file__).resolve().parents[1] / "data",
        type=Path,
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging()
    if args.obs is None:
        raise SystemExit("Pass --obs /path/to/taurus_obs.csv or set TAURUS_OBS.")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = _load_obs(args.obs)
    for c in REQUIRED_OBS_COLUMNS:
        if c not in df.columns:
            LOG.error("missing required column on obs: %s", c)
            return 1
    tax = _build_taxonomy_lookup(_load_taxonomy(args.taxonomy))

    LOG.info("== 1. label count table (author vs HGCA v1) ==")
    label_counts = build_label_count_table(df)
    label_counts.to_csv(out / "fig4a_lineage_label_counts.csv", index=False)
    LOG.info("  wrote %s", out / "fig4a_lineage_label_counts.csv")

    LOG.info("== 2. per-cell refinement classification ==")
    cell_class = build_refinement_per_cell(df, tax)
    keep_cols = [
        "sample_id", "Patient", "Disease", "Treatment",
        "assigned_lineage", "mapping_lineage",
        "final_analysis", "closest_GCA_celltype", "predicted_hgca_celltype_v1",
        "uncertainty_hgca_celltype_v1", "refinement_class",
        "refinement_class_taxonomy", "low_confidence",
        "author_taxonomy_path", "predicted_taxonomy_path",
        "lca_depth", "path_dist",
        "author_level1_match", "author_level2_match", "author_level3_match",
    ]
    cell_class[keep_cols].to_csv(
        out / "fig4a_refinement_by_cell.csv.gz", index=False, compression="gzip"
    )
    LOG.info("  wrote %s", out / "fig4a_refinement_by_cell.csv.gz")

    LOG.info("== 3. refinement summary ==")
    ref_summary = build_refinement_summary(cell_class)
    ref_summary.to_csv(out / "fig4a_refinement_summary.csv", index=False)
    LOG.info("  wrote %s", out / "fig4a_refinement_summary.csv")

    LOG.info("== 4. per-identity evidence ==")
    identity_evidence = build_identity_evidence(cell_class)
    identity_evidence.to_csv(out / "fig4a_hgca_identity_evidence.csv", index=False)
    LOG.info("  wrote %s", out / "fig4a_hgca_identity_evidence.csv")

    LOG.info("== 5. headline metrics ==")
    headline = build_headline_metrics(
        label_counts, ref_summary, identity_evidence, cell_class=cell_class
    )
    with open(out / "fig4a_headline_metrics.json", "w") as fh:
        json.dump(headline, fh, indent=2, default=str)
    LOG.info("  wrote %s", out / "fig4a_headline_metrics.json")

    print(json.dumps(headline, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
