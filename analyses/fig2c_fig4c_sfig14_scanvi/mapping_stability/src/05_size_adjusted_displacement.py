#!/usr/bin/env python3
"""Test whether sample displacement is explained by omission size."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import MANIFESTS, TABLES  # noqa: E402


def main() -> int:
    disp = pd.read_parquet(TABLES / "sample_aitchison_displacement.parquet")
    # mean distance per atlas x omit x seed
    per = (
        disp.groupby(["atlas", "omitted_study", "model_seed"], as_index=False)[
            "aitchison_to_full_seed0"
        ]
        .mean()
        .rename(columns={"aitchison_to_full_seed0": "mean_sample_dist"})
    )
    impact = pd.read_csv(MANIFESTS / "stroma_study_omission_impact.csv")
    impact = impact.rename(columns={"study": "omitted_study", "frac_lineage": "frac_lineage_removed"})
    per = per.merge(
        impact[["atlas", "omitted_study", "frac_lineage_removed", "n_cells", "n_labels_lost_entirely"]],
        on=["atlas", "omitted_study"],
        how="left",
    )
    per.to_csv(TABLES / "displacement_vs_omission_size.csv", index=False)

    rows = []
    for atlas, g in per.groupby("atlas"):
        r, p = stats.spearmanr(g["frac_lineage_removed"], g["mean_sample_dist"])
        rows.append({"atlas": atlas, "spearman_r": r, "spearman_p": p, "n": len(g)})
    # partial: residualize distance on frac within each atlas, then compare atlases
    # simple ANCOVA-like: OLS mean_dist ~ frac + atlas
    # encode atlas
    y = per["mean_sample_dist"].to_numpy()
    x_frac = per["frac_lineage_removed"].to_numpy()
    x_atlas = (per["atlas"] == "PanGI").astype(float).to_numpy()
    X = np.column_stack([np.ones(len(y)), x_frac, x_atlas])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # paired shared-study contrast of mean distance (average over seeds)
    shared = (
        per.groupby(["atlas", "omitted_study"], as_index=False)["mean_sample_dist"]
        .mean()
        .pivot(index="omitted_study", columns="atlas", values="mean_sample_dist")
    )
    shared["PanGI_minus_HGCA"] = shared["PanGI"] - shared["HGCA"]
    shared = shared.merge(
        impact[impact.atlas == "HGCA"][["omitted_study", "frac_lineage_removed"]].rename(
            columns={"frac_lineage_removed": "frac_HGCA"}
        ),
        on="omitted_study",
    )
    shared = shared.merge(
        impact[impact.atlas == "PanGI"][["omitted_study", "frac_lineage_removed"]].rename(
            columns={"frac_lineage_removed": "frac_PanGI"}
        ),
        on="omitted_study",
    )
    shared.to_csv(TABLES / "paired_shared_study_displacement.csv")

    out = {
        "within_atlas_spearman": rows,
        "ols_mean_dist_~_1_frac_atlasPanGI": {
            "intercept": float(coef[0]),
            "coef_frac_removed": float(coef[1]),
            "coef_PanGI": float(coef[2]),
            "interpretation": "Positive coef_PanGI means higher displacement after adjusting for frac removed",
        },
        "paired_median_PanGI_minus_HGCA": float(shared["PanGI_minus_HGCA"].median()),
        "paired_mean_PanGI_minus_HGCA": float(shared["PanGI_minus_HGCA"].mean()),
    }
    (TABLES / "size_adjusted_displacement.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(shared.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
