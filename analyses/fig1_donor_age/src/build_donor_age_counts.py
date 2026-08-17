#!/usr/bin/env python3
"""Count unique donors by age_range for ileum and colon from the integrated atlas."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import pandas as pd

HERE = Path(__file__).resolve()
FIGURE_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DEFAULT_H5AD = REPO_ROOT / "data" / "demo" / "hgca_all_lineages_v1_demo.h5ad"
EXPECTED_FIG1 = REPO_ROOT / "data" / "demo" / "expected" / "fig1"
AGE_ORDER = [
    "0-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "unknown",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=FIGURE_DIR / "data")
    args = parser.parse_args()
    h5ad = args.h5ad or Path(os.environ.get("HGCA_H5AD", str(DEFAULT_H5AD)))
    data_dir = args.outdir
    if "demo" in h5ad.name and data_dir == FIGURE_DIR / "data":
        data_dir = EXPECTED_FIG1
        print(f"Demo input: writing to {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    if "demo" in h5ad.name:
        print("DEMO MODE: results are for software checking, not manuscript figures.")
    print(f"Reading {h5ad}")
    adata = ad.read_h5ad(h5ad, backed="r")
    obs = adata.obs[["donor_id", "age_range", "tissue_level_1"]].copy()
    for col in obs.columns:
        obs[col] = obs[col].astype(str).str.strip()

    ages = sorted(set(obs["age_range"]))
    if "1-5" in ages:
        raise SystemExit(
            "Integrated object still contains a '1-5' age_range bin; "
            "refuse to plot until metadata are corrected."
        )
    unexpected = [age for age in ages if age not in AGE_ORDER]
    if unexpected:
        raise SystemExit(f"Unexpected age_range values: {unexpected}")

    rows: list[dict[str, object]] = []
    for tissue in ("ileum", "colon"):
        sub = obs.loc[obs["tissue_level_1"] == tissue]
        counts = (
            sub.drop_duplicates(["donor_id", "age_range"])
            .groupby("age_range")["donor_id"]
            .nunique()
        )
        for age in AGE_ORDER:
            rows.append(
                {
                    "tissue": tissue,
                    "age_range": age,
                    "n_donors": int(counts.get(age, 0)),
                }
            )

    out = pd.DataFrame(rows)
    out_path = data_dir / "donor_age_by_tissue.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    for tissue in ("ileum", "colon"):
        n = obs.loc[obs["tissue_level_1"] == tissue, "donor_id"].nunique()
        print(f"{tissue}: {n} unique donors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
