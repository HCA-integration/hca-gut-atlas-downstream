#!/usr/bin/env python3
"""Test whether cold dissociation associates with higher follicle/TLS niche
capture *within* each radial layer (EPI_LP, LP, full thickness).

Scope:
  - tissue_level_1 ∈ {duodenum, jejunum, ileum, colon}
  - sampled_site_condition ∈ {healthy, adjacent}
  - Protocol class from published methods (cold = primary digest on ice /
    cold-active protease; warm = 37°C Liberase/collagenase/EDTA strip)

Outputs under ../data/ and a Nature-style panel under ../out/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "out"
PARENT = HERE.parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

CANONICAL = ["duodenum", "jejunum", "ileum", "colon"]
LAYERS = ["EPI_LP", "LP", "EPI_LP_MUSC"]
LAYER_LAB = {
    "EPI_LP": "Epi+LP",
    "LP": "LP",
    "EPI_LP_MUSC": "Full thickness",
}

# Curated from STAR Methods / protocols.io (see evidence column).
# cold = primary tissue digest on ice / cold-active B. licheniformis protease
# warm = primary enzymatic steps at 37°C (Liberase/collagenase and/or warm EDTA)
# unknown = unpublished or methods insufficient to assign temperature class
PROTOCOL = {
    "Krzak2023": {
        "class": "cold",
        "evidence": "On-ice B. licheniformis protease + brief RT collagenase IV (protocols.io / Nat Genet)",
    },
    "Maddipatla2023": {
        "class": "cold",
        "evidence": "Minced on ice; B. licheniformis protease digest on ice 30 min (medRxiv 2022.05.19.22275263)",
    },
    "Elmentaite2020": {
        "class": "warm",
        "evidence": "Liberase TL/DH + DNase ± hyaluronidase up to 30 min at 37°C (Nature 2021)",
    },
    "Kong2023_v2": {
        "class": "warm",
        "evidence": "PBS/10 mM EDTA 15 min 37°C epithelial strip; Liberase TM LP 45 min 37°C (Immunity 2023)",
    },
    "Kong2023_v3": {
        "class": "warm",
        "evidence": "Same Kong et al. warm EDTA + Liberase workflow (Immunity 2023)",
    },
    "Luoma2020": {
        "class": "warm",
        "evidence": "Miltenyi Human Tumor Dissociation Kit collagenase 25–30 min at 37°C (Cell 2020)",
    },
    "Uzzan2022": {
        "class": "warm",
        "evidence": "HBSS-EDTA 20 min 37°C then collagenase IV + DNase 40 min 37°C (Nat Med 2022 STAR)",
    },
    "Huang2019": {
        "class": "warm",
        "evidence": "Colonic biopsy single-cell suspension for FACS/10x; standard warm mucosal processing (Cell 2019)",
    },
    "Martin2019": {
        "class": "warm",
        "evidence": "Ileal resection/lesion LP-focused enzymatic dissociation (Cell 2019)",
    },
    "Lee2020": {
        "class": "warm",
        "evidence": "CRC surgical tissue enzymatic dissociation (Nat Genet 2020)",
    },
    "James2020_3p": {
        "class": "warm",
        "evidence": "EDTA/DTT 37°C epithelial strip then Liberase TL 37°C (Nat Immunol 2020)",
    },
    "James2020_5p": {
        "class": "warm",
        "evidence": "Same James et al. warm EDTA/DTT + Liberase protocol (Nat Immunol 2020)",
    },
    "Dominguez2022": {
        "class": "warm",
        "evidence": "Pan-tissue Liberase TL immune isolation at 37°C (Science 2022)",
    },
    "He2020": {
        "class": "warm",
        "evidence": "Multi-organ enzymatic dissociation (Genome Biol 2020)",
    },
    "Zheng2024": {
        "class": "warm",
        "evidence": "EDTA epithelial separation 15 min 37°C; Liberase TM LP 30 min 37°C (eLife 2024 methods)",
    },
    "Wells2025": {
        "class": "warm",
        "evidence": "Cross-tissue immune atlas Liberase-style warm digest (Nat Immunol 2025)",
    },
    "Wells2025_2": {
        "class": "warm",
        "evidence": "Same Wells et al. warm Liberase-family processing (Nat Immunol 2025)",
    },
    "Jaeger2021": {
        "class": "warm",
        "evidence": "IEL/LP enzymatic isolation from resection (Nat Commun 2021)",
    },
    "Burclaff2022": {
        "class": "warm",
        "evidence": "Epithelial survey enzymatic dissociation (CMGH 2022)",
    },
    "Wang2020": {
        "class": "warm",
        "evidence": "Intestinal epithelial dissociation (JEM 2020)",
    },
    "Kinchen2018": {
        "class": "warm",
        "evidence": "Colonic mesenchyme enzymatic dissociation (Cell 2018)",
    },
    "Egozi2023": {
        "class": "warm",
        "evidence": "Neonatal intestine enzymatic dissociation (PLOS Biol 2023)",
    },
    "Yu2021": {
        "class": "warm",
        "evidence": "Multi-organ atlas enzymatic dissociation (Cell 2021)",
    },
    # Unpublished / insufficient methods detail for temperature class
    "ArendsHelmsley": {
        "class": "unknown",
        "evidence": "Unpublished Helmsley dataset; no public STAR Methods",
    },
    "BasuHelmsley": {
        "class": "unknown",
        "evidence": "Unpublished Helmsley dataset; no public STAR Methods",
    },
    "HamiltonHelmsley": {
        "class": "unknown",
        "evidence": "Mentions optimized epithelial-preserving protocol but temperature not explicit in available text",
    },
}


def load_samples() -> pd.DataFrame:
    niche = pd.read_csv(DATA / "niche_capture_samples.csv")
    clr = pd.read_csv(PARENT / "clr_long.csv")
    meta = clr.drop_duplicates("sample_id")[
        ["sample_id", "sampled_site_condition"]
    ]
    d = niche.merge(meta, on="sample_id", how="left")
    d = d[d["segment"].isin(CANONICAL)].copy()
    d = d[d["sampled_site_condition"].isin(["healthy", "adjacent"])].copy()
    d["protocol_class"] = d["dataset_id"].map(
        lambda x: PROTOCOL.get(x, {}).get("class", "unknown")
    )
    d["protocol_evidence"] = d["dataset_id"].map(
        lambda x: PROTOCOL.get(x, {}).get("evidence", "")
    )
    return d


def fisher_or(tab: np.ndarray):
    """Odds ratio cold vs warm for niche+; OR>1 means cold higher capture."""
    # rows: cold/warm; cols: niche+/niche-
    odds, p = fisher_exact(tab, alternative="two-sided")
    return float(odds), float(p)


def analyze_layer(d: pd.DataFrame, layer: str) -> dict:
    sub = d[d["radial_layer"] == layer].copy()
    known = sub[sub["protocol_class"].isin(["cold", "warm"])]
    cold = known[known["protocol_class"] == "cold"]
    warm = known[known["protocol_class"] == "warm"]

    out = {
        "radial_layer": layer,
        "layer_label": LAYER_LAB[layer],
        "n_samples_total": int(len(sub)),
        "n_cold": int(len(cold)),
        "n_warm": int(len(warm)),
        "n_unknown": int((sub["protocol_class"] == "unknown").sum()),
        "n_cold_studies": int(cold["dataset_id"].nunique()),
        "n_warm_studies": int(warm["dataset_id"].nunique()),
        "cold_studies": ",".join(sorted(cold["dataset_id"].unique())),
        "warm_studies": ",".join(sorted(warm["dataset_id"].unique())),
        "rate_cold": float(cold["niche_primary"].mean()) if len(cold) else np.nan,
        "rate_warm": float(warm["niche_primary"].mean()) if len(warm) else np.nan,
        "n_pos_cold": int(cold["niche_primary"].sum()) if len(cold) else 0,
        "n_pos_warm": int(warm["niche_primary"].sum()) if len(warm) else 0,
        "testable": bool(len(cold) >= 3 and len(warm) >= 3),
        "odds_ratio_cold_vs_warm": np.nan,
        "fisher_p": np.nan,
        "study_mw_p": np.nan,
        "note": "",
    }

    if out["n_cold_studies"] == 0:
        out["note"] = "No confirmed cold-dissociation studies in this layer"
        out["testable"] = False
        return out
    if out["n_warm_studies"] == 0:
        out["note"] = "No confirmed warm-dissociation studies in this layer"
        out["testable"] = False
        return out

    # Sample-level 2x2
    a = out["n_pos_cold"]
    b = out["n_cold"] - a
    c = out["n_pos_warm"]
    dneg = out["n_warm"] - c
    tab = np.array([[a, b], [c, dneg]])
    if tab.min() >= 0 and tab.sum() > 0 and out["testable"]:
        or_, p = fisher_or(tab)
        out["odds_ratio_cold_vs_warm"] = or_
        out["fisher_p"] = p

    # Study-level rates (equal weight per study with n>=2)
    def study_rates(x: pd.DataFrame) -> pd.Series:
        g = x.groupby("dataset_id").agg(
            n=("sample_id", "nunique"), pos=("niche_primary", "sum")
        )
        g = g[g["n"] >= 2]
        return g["pos"] / g["n"]

    rc, rw = study_rates(cold), study_rates(warm)
    if len(rc) >= 2 and len(rw) >= 2:
        try:
            out["study_mw_p"] = float(
                mannwhitneyu(rc, rw, alternative="two-sided").pvalue
            )
        except ValueError:
            out["study_mw_p"] = np.nan
    elif out["n_cold_studies"] == 1:
        out["note"] = (
            "Cold arm is a single study; sample-level OR confounds protocol with dataset"
        )
    return out


def main() -> None:
    d = load_samples()
    d.to_csv(DATA / "niche_samples_protocol_class.csv", index=False)

    # Protocol lookup table
    proto_rows = []
    for ds, info in PROTOCOL.items():
        proto_rows.append(
            {"dataset_id": ds, "protocol_class": info["class"], "evidence": info["evidence"]}
        )
    pd.DataFrame(proto_rows).sort_values(["protocol_class", "dataset_id"]).to_csv(
        DATA / "protocol_temperature_class.csv", index=False
    )

    layer_rows = [analyze_layer(d, layer) for layer in LAYERS]
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(DATA / "cold_vs_warm_by_layer.csv", index=False)

    # Study × layer rates for plotting
    plot_rows = []
    for layer in LAYERS:
        sub = d[
            (d["radial_layer"] == layer)
            & (d["protocol_class"].isin(["cold", "warm"]))
        ]
        for (ds, proto), g in sub.groupby(["dataset_id", "protocol_class"]):
            n = int(g["sample_id"].nunique())
            pos = int(g["niche_primary"].sum())
            rate = pos / n if n else np.nan
            lo, hi = (
                proportion_confint(pos, n, method="wilson") if n else (np.nan, np.nan)
            )
            plot_rows.append(
                {
                    "radial_layer": layer,
                    "layer_label": LAYER_LAB[layer],
                    "dataset_id": ds,
                    "protocol_class": proto,
                    "n_samples": n,
                    "n_pos": pos,
                    "capture_rate": rate,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(DATA / "cold_vs_warm_study_rates.csv", index=False)

    # EPI_LP leave-one-cold-study-out (robustness)
    epi = d[
        (d["radial_layer"] == "EPI_LP")
        & (d["protocol_class"].isin(["cold", "warm"]))
    ]
    lodo_rows = []
    cold_studies = sorted(epi.loc[epi.protocol_class == "cold", "dataset_id"].unique())
    for leave in ["none"] + cold_studies:
        x = epi if leave == "none" else epi[epi["dataset_id"] != leave]
        cold = x[x["protocol_class"] == "cold"]
        warm = x[x["protocol_class"] == "warm"]
        if len(cold) < 1 or len(warm) < 1:
            continue
        tab = np.array(
            [
                [int(cold.niche_primary.sum()), int((~cold.niche_primary).sum())],
                [int(warm.niche_primary.sum()), int((~warm.niche_primary).sum())],
            ]
        )
        or_, p = fisher_or(tab)
        lodo_rows.append(
            {
                "left_out_cold_study": leave,
                "n_cold": len(cold),
                "n_warm": len(warm),
                "rate_cold": float(cold.niche_primary.mean()),
                "rate_warm": float(warm.niche_primary.mean()),
                "odds_ratio": or_,
                "fisher_p": p,
            }
        )
    pd.DataFrame(lodo_rows).to_csv(DATA / "cold_vs_warm_epilp_lodo.csv", index=False)

    print("=== Cold vs warm niche capture (healthy/adjacent; canonical segments) ===\n")
    for _, r in layer_df.iterrows():
        print(f"{r['layer_label']} ({r['radial_layer']})")
        print(
            f"  cold: {r['n_pos_cold']}/{r['n_cold']} = {r['rate_cold']:.3f} "
            f"({r['n_cold_studies']} studies: {r['cold_studies'] or '—'})"
        )
        print(
            f"  warm: {r['n_pos_warm']}/{r['n_warm']} = {r['rate_warm']:.3f} "
            f"({r['n_warm_studies']} studies: {r['warm_studies'] or '—'})"
        )
        print(
            f"  unknown samples excluded from test: {r['n_unknown']}"
        )
        if r["testable"]:
            print(
                f"  Fisher OR(cold vs warm)={r['odds_ratio_cold_vs_warm']:.3f}, "
                f"p={r['fisher_p']:.3g}; study MW p={r['study_mw_p']}"
            )
        if r["note"]:
            print(f"  NOTE: {r['note']}")
        print()

    print("EPI_LP leave-one-cold-study-out:")
    print(pd.DataFrame(lodo_rows).to_string(index=False))


if __name__ == "__main__":
    main()
