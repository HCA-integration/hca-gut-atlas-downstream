"""Shared configuration for the Taurus per-tissue sample-embedding analysis.

Defines paths, the manuscript Wong palette (matching the other sample-level
figures), the tissue x disease grouping, and small IO helpers used across the
three pipeline stages:

  stage1_run_representations.py  - compute patpy sample representations
  stage2_benchmark.py            - patpy metrics table (the tutorial table())
  stage3_predict_response.py     - pre-treatment response prediction

Supplementary Figure 16a,c (patpy embeddings and pretreatment remission
AUC). SampleCLR attention (16b) is not in this directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Cell-type label set
# --------------------------------------------------------------------------
# The whole pipeline can be run on different cell-type annotations. Select with
# the LABELSET environment variable; outputs land in label-set-specific
# sub-folders so the two analyses never collide.
#   hgca_v1               - HGCA v1 predicted labels (label transfer), ~93 types
#   author_final_analysis - original highest-resolution author labels, ~109 types
#   pangi                 - PanGI SCANVI level_3 labels projected onto TAURUS
LABELSETS = {
    "hgca_v1": "predicted_hgca_celltype_v1",
    "author_final_analysis": "final_analysis",
    "pangi": "predicted_pangi_level_3_annot",
}
LABELSET = os.environ.get("LABELSET", "hgca_v1")
if LABELSET not in LABELSETS:
    raise ValueError(
        f"Unknown LABELSET={LABELSET!r}; choose one of {list(LABELSETS)}")

# --------------------------------------------------------------------------
# Paths (per label set)
# --------------------------------------------------------------------------
ANALYSIS = Path(__file__).resolve().parents[1]
PUB_ROOT = Path(os.environ.get("SFIG16_ROOT", str(ANALYSIS)))
DATA = PUB_ROOT / "data" / LABELSET
OUT = PUB_ROOT / "out" / LABELSET
REPR_DIR = DATA / "representations"
for _d in (DATA, OUT, REPR_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "")
    return Path(value) if value else None


def _first_existing(*candidates: Path | None) -> Path:
    found = [path for path in candidates if path is not None and path.is_file()]
    if found:
        return found[0]
    return Path("")


TAURUS_H5AD = _first_existing(
    _env_path("TAURUS_H5AD"),
    Path(""),
)
TAURUS_OBS_CSV = _first_existing(
    _env_path("TAURUS_OBS"),
    Path(""),
)
HGCA_V1_REMAP_SIDECAR = _first_existing(
    _env_path("HGCA_V1_REMAP_SIDECAR"),
    Path(""),
)
PANGI_LABEL_SIDECAR = _first_existing(
    _env_path("PANGI_LABEL_SIDECAR"),
    Path(""),
)
PANGI_SAMPLECLR_EMBEDDING = _first_existing(
    _env_path("PANGI_SAMPLECLR_EMBEDDING"),
    Path(""),
)
HGCA_SAMPLECLR_EMBEDDING = _first_existing(
    _env_path("HGCA_SAMPLECLR_EMBEDDING"),
    Path(""),
)

# --------------------------------------------------------------------------
# Column names in the Taurus obs
# --------------------------------------------------------------------------
SAMPLE_KEY = "sample_id"
PATIENT_KEY = "Patient"
CELLTYPE_KEY = LABELSETS[LABELSET]   # active cell-type annotation column
TISSUE_KEY = "Ileum_vs_Colon"        # coarse tissue: Ileum / Colon / Rectum
DISEASE_KEY = "Disease"              # CD / UC / Healthy
TREATMENT_KEY = "Treatment"          # Pre / Post / NaN (healthy)
REMISSION_KEY = "Remission_status"   # Remission / Non_Remission / ...

# Sample-level covariates carried through every stage.
META_COLS = [
    PATIENT_KEY, DISEASE_KEY, "Site", TISSUE_KEY, TREATMENT_KEY,
    REMISSION_KEY, "Inflammation", "Inflammation_score", "Disease_duration",
    "Age", "Gender", "Batch", "Match", "LibraryType",
]

# --------------------------------------------------------------------------
# Tissue x disease grouping
# --------------------------------------------------------------------------
TISSUES = ["Ileum", "Colon", "Rectum"]
DISEASES = ["CD", "UC"]


def group_name(tissue: str, disease: str) -> str:
    return f"{tissue}_{disease}"


# All analysis groups: (tissue, disease, label). Healthy samples from the same
# tissue are added to each group as reference anchors (state "Healthy").
GROUPS = [(t, d, group_name(t, d)) for t in TISSUES for d in DISEASES]

# Pooled multi-segment cohorts (Stage 1 + SPARE). Specs are consumed by
# stage1_run_reps.build_special_group_adata().
#   AllSeg_CD_Healthy — all tissues, CD (Pre+Post) + Healthy controls
#   AllSeg_CD_Pre     — all tissues, CD pretreatment only + Healthy controls
SPECIAL_GROUPS = {
    "AllSeg_CD_Healthy": {
        "tissues": TISSUES,
        "diseases": ["CD", "Healthy"],
        "treatments": None,  # all treatments (Healthy has NaN)
        "description": "All segments: CD (Pre+Post) + Healthy",
    },
    "AllSeg_CD_Pre": {
        "tissues": TISSUES,
        "diseases": ["CD", "Healthy"],
        "treatments": ["Pre"],  # CD Pre only; Healthy kept via disease==Healthy
        "description": "All segments: CD pretreatment + Healthy",
    },
}

# --------------------------------------------------------------------------
# Disease-state labelling (shared with fig_cd_ileum_compositional_pca)
# --------------------------------------------------------------------------
STATE_LEVELS = [
    "Healthy", "Post_Remission", "Pre_Remission",
    "Pre_Non_Remission", "Post_Non_Remission", "Other",
]
STATE_LABELS = {
    "Healthy": "Healthy",
    "Post_Remission": "Post, remission",
    "Pre_Remission": "Pre, remission",
    "Pre_Non_Remission": "Pre, non-remission",
    "Post_Non_Remission": "Post, non-remission",
    "Other": "Other",
}
# Severity ordering (used for the trajectory-preservation metric).
SEVERITY = {
    "Healthy": 1, "Post_Remission": 2, "Pre_Remission": 3,
    "Pre_Non_Remission": 4, "Post_Non_Remission": 5,
}


def assign_state(disease, treatment, remission):
    """Map (Disease, Treatment, Remission_status) -> disease state string."""
    if disease == "Healthy":
        return "Healthy"
    if disease in ("CD", "UC"):
        if treatment == "Pre" and remission == "Non_Remission":
            return "Pre_Non_Remission"
        if treatment == "Pre" and remission == "Remission":
            return "Pre_Remission"
        if treatment == "Post" and remission == "Non_Remission":
            return "Post_Non_Remission"
        if treatment == "Post" and remission == "Remission":
            return "Post_Remission"
    return "Other"


# --------------------------------------------------------------------------
# Wong colorblind-safe palette (plot_specs.md §9). Matches the other
# sample-level figures so colours mean the same thing across panels.
# --------------------------------------------------------------------------
WONG = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",       # HCA blue
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    "light_grey": "#E0E0E0",
}

# Disease-state colours (cool = inactive/healthy, warm = active CD/UC).
STATE_COLORS = {
    "Healthy": WONG["bluish_green"],
    "Post_Remission": WONG["blue"],
    "Pre_Remission": WONG["reddish_purple"],
    "Pre_Non_Remission": WONG["vermillion"],
    "Post_Non_Remission": WONG["black"],
    "Other": WONG["grey"],
}

# Binary response colours for the prediction stage.
RESPONSE_COLORS = {
    "Remission": WONG["blue"],
    "Non_Remission": WONG["vermillion"],
}

# Tissue and disease colours (categorical covariates on the embeddings).
TISSUE_COLORS = {
    "Ileum": WONG["orange"],
    "Colon": WONG["blue"],
    "Rectum": WONG["bluish_green"],
}
DISEASE_COLORS = {
    "CD": WONG["vermillion"],
    "UC": WONG["sky_blue"],
    "Healthy": WONG["bluish_green"],
}
INFLAMMATION_COLORS = {
    "Inflamed": WONG["vermillion"],
    "Non_Inflamed": WONG["blue"],
    "Healthy": WONG["bluish_green"],
}

# --------------------------------------------------------------------------
# Representation methods (CPU-feasible set chosen for this dataset)
# --------------------------------------------------------------------------
# Display name -> short key used in filenames / table rows.
# MixMIL is a *supervised* multi-instance method (Engelmann et al. 2024); it is
# trained to classify disease status (CD/UC vs Healthy), a donor-level target
# orthogonal to the treatment-response outcome the trajectory stage evaluates,
# so it remains a fair, comparable representation rather than a response oracle.
REPR_METHODS = {
    "Cell-type composition (CLR)": "composition",
    "Pseudobulk": "pseudobulk",
    "CT pseudobulk": "ct_pseudobulk",
    "MOFA": "mofa",
    "GloScope": "gloscope",
    "PILOT": "pilot",
    "MixMIL": "mixmil",
    "SampleCLR (remission)": "sampleclr_remission",
    "Random vector": "random",
}
# Order for plotting / tables (best baselines first, random last).
REPR_ORDER = [
    "composition", "ct_pseudobulk", "pseudobulk",
    "mofa", "gloscope", "pilot", "mixmil", "sampleclr_remission", "random",
]
REPR_DISPLAY = {v: k for k, v in REPR_METHODS.items()}

# --------------------------------------------------------------------------
# patpy benchmark schema: which covariates are relevant (biology we want the
# embedding to capture), technical (batch effects we want removed), and
# contextual. task is the prediction type used by knn_prediction_score.
# --------------------------------------------------------------------------
BENCHMARK_SCHEMA = {
    "relevant": {
        "state_severity": "ranking",      # ordinal disease severity
        REMISSION_KEY: "classification",  # response
        "Inflammation": "classification",
        DISEASE_KEY: "classification",
        TREATMENT_KEY: "classification",
        # Present for pooled multi-segment cohorts; dropped by usable_schema
        # when constant within a single-tissue group.
        TISSUE_KEY: "classification",
    },
    "technical": {
        "Batch": "classification",
        "n_cells": "regression",
        "LibraryType": "classification",
    },
    "contextual": {
        "Age": "regression",
        "Gender": "classification",
        "Disease_duration": "regression",
    },
}


def repr_distance_path(group: str, method_key: str) -> Path:
    return REPR_DIR / f"{group}__{method_key}__distances.csv"


def repr_embedding_path(group: str, method_key: str, embed: str = "mds") -> Path:
    return REPR_DIR / f"{group}__{method_key}__{embed}.csv"


def group_meta_path(group: str) -> Path:
    return REPR_DIR / f"{group}__sample_metadata.csv"


def group_composition_path(group: str) -> Path:
    return REPR_DIR / f"{group}__composition.csv"


# Minimum cells for a sample to enter the representation analysis.
MIN_CELLS_PER_SAMPLE = 50

# Author vs HGCA comparison colors (Wong; match composite-figure convention).
AUTHOR_LABEL_COLOR = WONG["orange"]   # final_analysis
HGCA_LABEL_COLOR = WONG["blue"]       # predicted_hgca_celltype_v1
PANGI_LABEL_COLOR = WONG["bluish_green"]  # predicted_pangi_level_3_annot


def apply_remapped_hgca_v1_labels(obs):
    """Overlay the 2026-07-22 lineage-mapping hard labels onto an obs frame.

    The source h5ad still carries the prior transfer column; analyses that
    claim the new mapping must call this after load. Returns a copy.
    """
    import pandas as pd

    if not HGCA_V1_REMAP_SIDECAR.exists():
        raise FileNotFoundError(
            f"Missing remapped HGCA v1 sidecar: {HGCA_V1_REMAP_SIDECAR}"
        )
    labels = pd.read_csv(HGCA_V1_REMAP_SIDECAR, index_col=0, compression="gzip")
    out = obs.copy()
    shared = out.index.astype(str).intersection(labels.index.astype(str))
    # h5ad stores the prior transfer as categorical; widen before overwrite.
    out["predicted_hgca_celltype_v1"] = out["predicted_hgca_celltype_v1"].astype(
        object
    )
    out.loc[shared, "predicted_hgca_celltype_v1"] = labels.loc[
        shared, "predicted_hgca_celltype_v1"
    ].astype(str).to_numpy()
    if "uncertainty_hgca_celltype_v1" in out.columns:
        out["uncertainty_hgca_celltype_v1"] = pd.to_numeric(
            out["uncertainty_hgca_celltype_v1"], errors="coerce"
        )
        remapped_entropy = pd.to_numeric(
            labels.loc[shared, "uncertainty_hgca_celltype_v1"], errors="coerce"
        )
        out.loc[shared, "uncertainty_hgca_celltype_v1"] = remapped_entropy.to_numpy()
    return out


def apply_pangi_labels(obs):
    """Attach PanGI SCANVI hard labels from the projectable-reference mapping."""
    import pandas as pd

    if not PANGI_LABEL_SIDECAR.exists():
        raise FileNotFoundError(
            f"Missing PanGI label sidecar: {PANGI_LABEL_SIDECAR}"
        )
    labels = pd.read_csv(PANGI_LABEL_SIDECAR, compression="gzip")
    if "barcode" not in labels.columns:
        raise KeyError("PanGI sidecar must contain a barcode column")
    labels = labels.drop_duplicates("barcode").set_index("barcode")
    out = obs.copy()
    shared = out.index.astype(str).intersection(labels.index.astype(str))
    if len(shared) == 0:
        raise RuntimeError(
            "No overlapping barcodes between TAURUS obs and PanGI sidecar"
        )
    out["predicted_pangi_level_3_annot"] = "Unmapped"
    out.loc[shared, "predicted_pangi_level_3_annot"] = labels.loc[
        shared, "predicted_pangi_level_3_annot"
    ].astype(str).to_numpy()
    if "predicted_pangi_common_celltype" in labels.columns:
        out["predicted_pangi_common_celltype"] = "Unmapped"
        out.loc[shared, "predicted_pangi_common_celltype"] = labels.loc[
            shared, "predicted_pangi_common_celltype"
        ].astype(str).to_numpy()
    return out
