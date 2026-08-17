#!/usr/bin/env python3
"""Add HCA-style chemical_fractionation metadata + fix Martin2019 radial.

Ontology (controlled vocabulary, lowercase like sample_collection_method):
  chemical_fractionation ∈ {unfractionated, fractionated, unknown}

Definitions (methods-based, not purely radial):
  unfractionated — tissue digested in one compartment-agnostic step (no prior
                   EDTA/DTT epithelial strip that separates EPI from LP)
  fractionated   — EDTA/DTT (or equivalent) strip used to separate epithelium
                   from lamina propria before digest / library construction
  unknown        — public methods insufficient to assign

radial_tissue_term remains the anatomical-compartment label. Martin2019 is
recoded EPI_LP → LP (LP-focused prep after EDTA strip; Cell 2019).

Writes:
  - sample-level annotation table under fig_sampling_depth_radial/data/
  - patched clr_long.csv (radial + chemical_fractionation)
  - new lineage + all_cells h5ads under meta_datasets/integrated-objects-chemfrac/
  - optional update of concatenated_sample_metadata.csv
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG_DATA = HERE.parent / "data"
OBJ_IN = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else Path()
OBJ_OUT = Path(os.environ["HGCA_OBJECTS_OUT"]) if os.environ.get("HGCA_OBJECTS_OUT") else Path()
SAMPLE_META = Path(os.environ["HGCA_SAMPLE_META"]) if os.environ.get("HGCA_SAMPLE_META") else Path()

LINEAGE_FILES = ["epithelial.h5ad", "lymphoid.h5ad", "myeloid.h5ad", "stroma.h5ad"]
ALL_CELLS = "hgca_all_lineages_v1.h5ad"

# Dataset-level overrides from STAR Methods / protocols (priority over radial).
# See rare_cell_defensibility/data/dissociation_method_notes.csv and paper audit.
FRACTIONATED_DATASETS = {
    "Martin2019",       # EDTA strip → collagenase LP; epithelium excluded
    "Kong2023_v2",      # warm EDTA strip separate from Liberase LP
    "Kong2023_v3",      # same Kong workflow
    "James2020_3p",     # EDTA/DTT then Liberase (epi supernatant not kept)
    "James2020_5p",     # same James protocol
    "Zheng2024",        # EDTA epithelial separation then Liberase LP
    "Jaeger2021",       # IEL/LP enzymatic isolation from resection
    "Uzzan2022",        # HBSS-EDTA then collagenase IV (STAR Methods)
}

UNFRACTIONATED_DATASETS = {
    "Krzak2023",        # cold whole-biopsy one-step
    "Maddipatla2023",   # cold whole-biopsy
    "Elmentaite2020",   # Liberase whole-tissue, no EDTA strip
    "Luoma2020",        # Miltenyi tumor kit one-step whole biopsy
    "Lee2020",          # unsorted whole surgical digest
    "Egozi2023",        # collagenase + gentleMACS; No selection = whole
    "Huang2019",        # whole mucosal suspension (then CD45 FACS)
    "Kinchen2018",      # mesenchyme-oriented but one-compartment digest
    "Dominguez2022",    # Liberase pan-tissue (no EPI/LP split)
    "He2020",
    "Yu2021",
    "Wells2025",
    "Wells2025_2",
    "BasuHelmsley",     # WM whole-mount biopsy encoding
}

UNKNOWN_DATASETS = {
    "ArendsHelmsley",
    "HamiltonHelmsley",
}

ALLOWED = ("unfractionated", "fractionated", "unknown")


def assign_chemical_fractionation(dataset_id: str, radial: str) -> str:
    ds = str(dataset_id)
    r = str(radial).upper()
    if ds in FRACTIONATED_DATASETS:
        return "fractionated"
    if ds in UNFRACTIONATED_DATASETS:
        return "unfractionated"
    if ds in UNKNOWN_DATASETS:
        # Hamilton has separate EPI vs LP libraries → fractionated by construction
        if r in {"EPI", "LP"}:
            return "fractionated"
        return "unknown"
    # Radial fallback for datasets without curated methods notes
    if r in {"EPI", "LP"}:
        return "fractionated"
    if r in {"EPI_LP", "EPI_LP_MUSC", "WM", "LP_MUSC", "MUSC"}:
        return "unfractionated"
    return "unknown"


def build_sample_table(clr_path: Path | None = None) -> pd.DataFrame:
    """Sample-level table from clr_long (unique sample_id rows)."""
    clr_path = clr_path or (FIG_DATA / "clr_long.csv")
    clr = pd.read_csv(clr_path)
    meta_cols = [
        c for c in [
            "sample_id", "donor_id", "dataset_id", "tissue_level_1",
            "sampled_site_condition", "radial_tissue_term",
            "sample_collection_method", "sample_preservation_method",
            "sex_ontology_term", "age_range", "assay",
        ] if c in clr.columns
    ]
    meta = clr[meta_cols].drop_duplicates("sample_id").copy()
    meta["radial_tissue_term_original"] = meta["radial_tissue_term"].astype(str)

    # Martin2019 fix: LP-focused prep, not EPI_LP
    martin = meta["dataset_id"].astype(str).eq("Martin2019")
    meta.loc[martin, "radial_tissue_term"] = "LP"
    meta["radial_tissue_term_corrected"] = martin

    meta["chemical_fractionation"] = [
        assign_chemical_fractionation(d, r)
        for d, r in zip(meta["dataset_id"], meta["radial_tissue_term"])
    ]
    # After Martin→LP, fractionation must be fractionated (also in FRACTIONATED_DATASETS)
    assert meta.loc[martin, "chemical_fractionation"].eq("fractionated").all()

    bad = ~meta["chemical_fractionation"].isin(ALLOWED)
    if bad.any():
        raise ValueError(meta.loc[bad, ["sample_id", "chemical_fractionation"]])
    return meta


def patch_clr_long(sample_meta: pd.DataFrame) -> None:
    clr_path = FIG_DATA / "clr_long.csv"
    clr = pd.read_csv(clr_path)
    m = sample_meta.set_index("sample_id")
    clr["radial_tissue_term"] = clr["sample_id"].map(m["radial_tissue_term"])
    clr["chemical_fractionation"] = clr["sample_id"].map(m["chemical_fractionation"])
    clr.to_csv(clr_path, index=False)
    print(f"Patched {clr_path} (+ chemical_fractionation; Martin→LP)")


def patch_sample_metadata_csv(sample_meta: pd.DataFrame) -> None:
    if not SAMPLE_META.exists():
        print(f"Skip sample metadata (missing): {SAMPLE_META}")
        return
    df = pd.read_csv(SAMPLE_META)
    m = sample_meta.set_index("sample_id")
    # Only patch rows present in atlas CLR table; leave others unchanged unless Martin
    overlap = df["sample_id"].isin(m.index)
    df.loc[overlap, "radial_tissue_term"] = df.loc[overlap, "sample_id"].map(
        m["radial_tissue_term"]
    )
    # Martin even if not in clr (should be)
    martin = df["dataset_id"].astype(str).eq("Martin2019")
    df.loc[martin, "radial_tissue_term"] = "LP"

    # Assign fractionation for all rows in the CSV
    df["chemical_fractionation"] = [
        assign_chemical_fractionation(d, r)
        for d, r in zip(df["dataset_id"], df["radial_tissue_term"])
    ]
    df.to_csv(SAMPLE_META, index=False)
    print(f"Updated {SAMPLE_META}")


def _set_obs_string_column(adata: ad.AnnData, col: str, values: pd.Series) -> None:
    """Write a string/categorical obs column aligned to adata.obs_names."""
    # values indexed by sample_id → broadcast to cells
    if "sample_id" not in adata.obs.columns:
        raise KeyError("sample_id missing from obs")
    mapped = adata.obs["sample_id"].astype(str).map(values)
    adata.obs[col] = pd.Categorical(mapped.fillna("unknown"), categories=list(ALLOWED))


def write_h5ads(sample_meta: pd.DataFrame, write_all_cells: bool = True) -> None:
    OBJ_OUT.mkdir(parents=True, exist_ok=True)
    m = sample_meta.set_index("sample_id")
    radial_map = m["radial_tissue_term"]

    files = list(LINEAGE_FILES)
    if write_all_cells:
        files.append(ALL_CELLS)

    for name in files:
        src = OBJ_IN / name
        dst = OBJ_OUT / name
        if not src.exists():
            print(f"MISSING {src}")
            continue
        print(f"Loading {src} …", flush=True)
        adata = ad.read_h5ad(src)
        # Martin + sample-table radial corrections
        sid = adata.obs["sample_id"].astype(str)
        new_radial = sid.map(radial_map)
        adata.obs["radial_tissue_term"] = np.where(
            new_radial.notna(), new_radial, adata.obs["radial_tissue_term"].astype(str)
        )
        martin_cells = adata.obs["dataset_id"].astype(str).eq("Martin2019")
        if martin_cells.any():
            adata.obs.loc[martin_cells, "radial_tissue_term"] = "LP"

        # Assign fractionation from dataset + (corrected) radial for every cell
        frac = [
            assign_chemical_fractionation(d, r)
            for d, r in zip(
                adata.obs["dataset_id"].astype(str),
                adata.obs["radial_tissue_term"].astype(str),
            )
        ]
        adata.obs["chemical_fractionation"] = pd.Categorical(
            frac, categories=list(ALLOWED)
        )

        print(f"  writing {dst} …", flush=True)
        adata.write_h5ad(dst, compression="gzip")
        with h5py.File(dst, "r") as f:
            assert "chemical_fractionation" in f["obs"]
        print(
            f"  OK {name}: n_obs={adata.n_obs:,}; Martin cells={int(martin_cells.sum()):,}; "
            f"frac={adata.obs['chemical_fractionation'].value_counts().to_dict()}",
            flush=True,
        )
        del adata


def write_rules_table() -> None:
    rows = []
    for ds in sorted(FRACTIONATED_DATASETS):
        rows.append(dict(dataset_id=ds, chemical_fractionation="fractionated",
                         source="curated_methods"))
    for ds in sorted(UNFRACTIONATED_DATASETS):
        rows.append(dict(dataset_id=ds, chemical_fractionation="unfractionated",
                         source="curated_methods"))
    for ds in sorted(UNKNOWN_DATASETS):
        rows.append(dict(dataset_id=ds, chemical_fractionation="unknown",
                         source="curated_methods",
                         note="radial EPI/LP still forces fractionated"))
    pd.DataFrame(rows).to_csv(FIG_DATA / "chemical_fractionation_dataset_rules.csv",
                              index=False)


def summarize(sample_meta: pd.DataFrame) -> None:
    print("\n=== chemical_fractionation × collection × radial (all CLR samples) ===")
    ct = (
        sample_meta.assign(
            collection=sample_meta["sample_collection_method"].astype(str).str.lower(),
            radial=sample_meta["radial_tissue_term"].astype(str),
        )
        .groupby(["chemical_fractionation", "collection", "radial"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["chemical_fractionation", "collection", "n"], ascending=[True, True, False])
    )
    print(ct.to_string(index=False))
    print("\nMartin2019 after fix:")
    print(
        sample_meta[sample_meta.dataset_id.eq("Martin2019")][
            ["sample_id", "radial_tissue_term_original", "radial_tissue_term",
             "chemical_fractionation"]
        ].to_string(index=False)
    )


def main(write_objects: bool = True, write_all_cells: bool = True):
    write_rules_table()
    sample_meta = build_sample_table()
    sample_meta.to_csv(FIG_DATA / "sample_chemical_fractionation.csv", index=False)
    print(f"Wrote {FIG_DATA / 'sample_chemical_fractionation.csv'} "
          f"({len(sample_meta)} samples)")
    summarize(sample_meta)
    patch_clr_long(sample_meta)
    patch_sample_metadata_csv(sample_meta)
    if write_objects:
        write_h5ads(sample_meta, write_all_cells=write_all_cells)
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-objects", action="store_true",
                   help="Only write CSV / clr_long patches (no h5ad copies)")
    p.add_argument("--skip-all-cells", action="store_true",
                   help="Write lineage h5ads only (skip 15GB all-cells)")
    args = p.parse_args()
    main(write_objects=not args.skip_objects,
         write_all_cells=not args.skip_all_cells)
