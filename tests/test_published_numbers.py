#!/usr/bin/env python3
"""Recover a few manuscript-facing numbers from the full composition tables.

These checks use `data/composition/clr_long.csv` (502 samples × 94 types),
not the 3,185-cell demo slice. Follicle rates and the colon full-thickness
Mann–Whitney are recomputed here. Figure 3b values are checksums of the
production heatmap table (a naive one-way R² on all samples does not
reproduce those four numbers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[1]
CLR = REPO / "data" / "composition" / "clr_long.csv"
FIG3B = REPO / "data" / "fig3" / "celltype_metadata_sensitivity_top2.csv"
COLON_FT = (
    REPO
    / "data"
    / "composition"
    / "by_tissue"
    / "colon"
    / "clr_wilcoxon_full_thickness_colon.csv"
)
LODO_PREP = (
    REPO
    / "analyses"
    / "fig2c_fig4c_sfig14_scanvi"
    / "lodo"
    / "src"
    / "data"
    / "preparation.py"
)
LODO_TRAINER = (
    REPO
    / "analyses"
    / "fig2c_fig4c_sfig14_scanvi"
    / "lodo"
    / "src"
    / "models"
    / "scanvi_benchmark.py"
)
GC = ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"]
MIN_SAMPLES_PER_GROUP = 5
REST_LAYERS = {"epi", "epi_lp", "lp", "wm"}


def _clr() -> pd.DataFrame:
    return pd.read_csv(CLR)


def test_lodo_source_modules_are_tracked() -> None:
    assert LODO_PREP.is_file(), (
        "LODO DataPreparation is missing. Check .gitignore: "
        "analyses/**/data/ must not hide lodo/src/data/."
    )
    assert LODO_TRAINER.is_file(), (
        "LODO SCANVITrainer is missing. Check .gitignore: "
        "analyses/**/models/ must not hide lodo/src/models/."
    )


def test_atlas_census_from_clr_long() -> None:
    clr = _clr()
    total = int(clr["n_cells"].sum())
    epithelial = int(clr.loc[clr["celltype"] == "Epithelial", "n_cells"].sum())
    assert clr["sample_id"].nunique() == 502
    assert clr["donor_id"].nunique() == 265
    assert clr["celltype"].nunique() == 94
    assert clr["dataset_id"].nunique() == 27
    assert total == 944_502
    assert epithelial == 104
    # Manuscript 944,390 is the intended total after dropping the catch-all
    # "Epithelial" label. The shipped table is 8 cells short of that.
    assert total - epithelial == 944_398


def test_follicle_or_k3_rates_recomputed() -> None:
    """Recover Fig. 3i rates from counts: (LZ >= 3) OR (DZ >= 3)."""
    clr = _clr()
    piv = (
        clr[clr["celltype"].isin(GC)]
        .pivot_table(
            index="sample_id",
            columns="celltype",
            values="n_cells",
            aggfunc="sum",
            fill_value=0,
        )
    )
    meta = clr.groupby("sample_id").agg(
        segment=("tissue_level_1", "first"),
        collection=("sample_collection_method", "first"),
    )
    m = meta.join(piv, how="left").fillna(0)
    m["flag"] = (m[GC[0]] >= 3) | (m[GC[1]] >= 3)

    expected = {
        ("ileum", "biopsy"): 0.439,
        ("colon", "biopsy"): 0.398,
        ("ileum", "surgical resection"): 0.158,
        ("colon", "surgical resection"): 0.130,
    }
    for (seg, coll), rate in expected.items():
        sub = m[(m["segment"] == seg) & (m["collection"] == coll)]
        got = float(sub["flag"].mean())
        assert abs(got - rate) < 0.005, (seg, coll, got, rate)


def _colon_full_thickness_mwu() -> pd.DataFrame:
    """Same contrast as recompute_clr_tables.py: rest vs EPI_LP_MUSC, colon."""
    clr = _clr()
    colon = clr[clr["tissue_level_1"].astype(str).str.strip().str.lower() == "colon"].copy()
    rad = colon["radial_tissue_term"].astype(str).str.strip().str.lower()
    colon["_g"] = "other"
    colon.loc[rad == "epi_lp_musc", "_g"] = "full_thickness"
    colon.loc[rad.isin(REST_LAYERS), "_g"] = "rest"
    rows = []
    for ct, sub in colon.groupby("celltype"):
        ya = pd.to_numeric(sub.loc[sub["_g"] == "rest", "clr"], errors="coerce").dropna()
        yb = pd.to_numeric(
            sub.loc[sub["_g"] == "full_thickness", "clr"], errors="coerce"
        ).dropna()
        if len(ya) < MIN_SAMPLES_PER_GROUP or len(yb) < MIN_SAMPLES_PER_GROUP:
            continue
        _, p = mannwhitneyu(ya.to_numpy(), yb.to_numpy(), alternative="two-sided")
        rows.append(
            {
                "celltype": ct,
                "n_A": int(len(ya)),
                "n_B": int(len(yb)),
                "delta_CLR_B_minus_A": float(yb.mean() - ya.mean()),
                "p_value": float(p),
            }
        )
    res = pd.DataFrame(rows)
    _, res["p_adj"], _, _ = multipletests(res["p_value"], method="fdr_bh", alpha=0.05)
    return res


def test_fig3f_colon_full_thickness_recomputed() -> None:
    """Recover Fig. 3f colon counts and two headline ΔCLR values from clr_long."""
    res = _colon_full_thickness_mwu()
    sig = res[res["p_adj"] < 0.05]
    assert len(res) == 94
    assert len(sig) == 46
    assert int((sig["delta_CLR_B_minus_A"] > 0).sum()) == 23
    assert int((sig["delta_CLR_B_minus_A"] < 0).sum()) == 23

    s3 = res.set_index("celltype").loc["Submucosal Fibroblasts (S3)"]
    pv = res.set_index("celltype").loc["Perivascular Resident Macrophages"]
    assert abs(s3["delta_CLR_B_minus_A"] - 2.5965) < 0.002
    assert abs(pv["delta_CLR_B_minus_A"] - 1.1075) < 0.002
    assert s3["n_A"] == 47 and s3["n_B"] == 32

    shipped = pd.read_csv(COLON_FT)
    merged = res.merge(shipped, on="celltype", suffixes=("_got", "_shipped"))
    assert len(merged) == 94
    assert np.allclose(
        merged["delta_CLR_B_minus_A_got"],
        merged["delta_CLR_B_minus_A_shipped"],
        atol=1e-12,
    )


def test_fig3b_production_table_checksum() -> None:
    """These four values match the submitted heatmap table, not a recompute."""
    top2 = pd.read_csv(FIG3B, index_col=0)
    assert abs(top2.loc["Submucosal Fibroblasts (S3)", "radial tissue term"] - 0.562) < 0.002
    assert abs(
        top2.loc["Post Arteriole Capillary Endothelial (PAC)", "radial tissue term"] - 0.453
    ) < 0.002
    assert abs(top2.loc["Gamma Delta T Cells", "age range"] - 0.360) < 0.002
    assert abs(
        top2.loc["Colonocyte Progenitors", "sample preservation method"] - 0.285
    ) < 0.002


if __name__ == "__main__":
    test_lodo_source_modules_are_tracked()
    test_atlas_census_from_clr_long()
    test_follicle_or_k3_rates_recomputed()
    test_fig3f_colon_full_thickness_recomputed()
    test_fig3b_production_table_checksum()
    print("ok: published-number checks passed")
