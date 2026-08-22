#!/usr/bin/env python3
"""Extract compact curated edge tables for Supplementary Figure 13 panels b–e."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMBINED = Path(
    os.environ.get("LIANA_COMBINED_CSV")
    or os.environ.get("CCC_EDGE_CSV")
    or ""
)
if not COMBINED.is_file():
    raise SystemExit(
        "Set LIANA_COMBINED_CSV or CCC_EDGE_CSV to "
        "combined_lr_per_tissue_level_1.csv"
    )
SEG_ORDER = ["duodenum", "jejunum", "ileum", "colon"]
BAN_SENDERS = ("Neutrophil", "Eosinophil")

PVC_VEN = {
    "Pre Venule Capillary Endothelial (PVC)",
    "Venular Endothelial",
}
LYMPH = {"Lymphatic Endothelial", "Medullary Sinus Endothelial"}
MACS = {
    "Perivascular Resident Macrophages",
    "Follicle Associated Resident Macrophages",
    "Homeostatic Macrophages",
    "M0 Macrophages",
    "Cycling Macrophages",
}


def ensemble(mag: pd.Series) -> pd.Series:
    return -np.log10(mag.clip(lower=1e-300))


def read_filtered(keep_fn, usecols=None) -> pd.DataFrame:
    cols = usecols or [
        "tissue_level_1",
        "source",
        "target",
        "ligand_complex",
        "receptor_complex",
        "lr_means",
        "magnitude_rank",
        "specificity_rank",
    ]
    rows = []
    for chunk in pd.read_csv(COMBINED, usecols=cols, chunksize=750_000):
        m = chunk.loc[keep_fn(chunk)]
        if len(m):
            rows.append(m)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    # Panel b: ACKR1 sink on PVC / venular (exclude granulocyte senders)
    ack = read_filtered(
        lambda d: d["target"].isin(PVC_VEN)
        & (d["receptor_complex"] == "ACKR1")
        & d["tissue_level_1"].isin(SEG_ORDER)
    )
    ack = ack[~ack["source"].str.contains("|".join(BAN_SENDERS), case=False, na=False)]
    ack["ensemble_score"] = ensemble(ack["magnitude_rank"])
    ack["lr_pair"] = ack["ligand_complex"] + "->" + ack["receptor_complex"]
    # Best segment per (source, target, lr), then diversify: top per ligand,
    # then fill remaining slots by score (avoids CCL5-only CD8 stack).
    ack_uniq = (
        ack.sort_values("ensemble_score", ascending=False)
        .groupby(["source", "target", "lr_pair"], as_index=False)
        .first()
    )
    per_lig = (
        ack_uniq.sort_values("ensemble_score", ascending=False)
        .groupby("ligand_complex", as_index=False)
        .first()
    )
    rest = ack_uniq[~ack_uniq.index.isin(per_lig.index)].sort_values(
        "ensemble_score", ascending=False
    )
    # rebuild rest without the already-chosen rows
    chosen_keys = set(
        zip(per_lig["source"], per_lig["target"], per_lig["lr_pair"])
    )
    rest = ack_uniq[
        ~ack_uniq.apply(
            lambda r: (r["source"], r["target"], r["lr_pair"]) in chosen_keys,
            axis=1,
        )
    ].sort_values("ensemble_score", ascending=False)
    ack_best = pd.concat([per_lig, rest], ignore_index=True).head(14)
    ack.to_csv(DATA / "ackr1_pvc_venular_all.csv", index=False)
    ack_best.to_csv(DATA / "panel_b_ackr1_sink_top.csv", index=False)

    # Panel c: lymphatic / sinus CCL21 → CCR7
    ccr7 = read_filtered(
        lambda d: d["source"].isin(LYMPH)
        & (d["ligand_complex"] == "CCL21")
        & (d["receptor_complex"] == "CCR7")
        & d["tissue_level_1"].isin(SEG_ORDER)
    )
    ccr7["ensemble_score"] = ensemble(ccr7["magnitude_rank"])
    ccr7["lr_pair"] = "CCL21->CCR7"
    ccr7_best = (
        ccr7.sort_values("ensemble_score", ascending=False)
        .groupby(["source", "target"], as_index=False)
        .first()
        .sort_values("ensemble_score", ascending=False)
        .head(12)
    )
    ccr7.to_csv(DATA / "ccl21_ccr7_lymphatic_all.csv", index=False)
    ccr7_best.to_csv(DATA / "panel_c_lymphatic_ccl21_ccr7_top.csv", index=False)

    # Panel d: PV vs FARM niche contrast — prefer curated axes; ban sticky hits
    # C1QA/B→CD93: recent work indicates C1q does not bind CD93 — exclude.
    BAN_LR = {
        "APP->CD74",
        "RPS19->C5AR1",
        "B2M->KLRD1",
        "VIM->CD44",
        "CALM1->INSR",
        "NRG1->HLA-DPB1",
        "TNFSF13B->HLA-DPB1",
        "S100A10->CFTR",
        "C1QA->CD93",
        "C1QB->CD93",
        "C1QC->CD93",
    }
    PV_PREF_LR = {
        "CXCL8->ACKR1",
        "CXCL2->ACKR1",
        "CXCL3->ACKR1",
        "CCL18->ACKR1",
        "CCL13->ACKR1",
        "CCL8->ACKR1",
        "CCL2->ACKR1",
        "S100A9->CD36",
        "S100A8->CD36",
        "PDGFC->FLT1",
        "PDGFC->FLT4",
        "F13A1->ITGA9",
        "F13A1->ITGB1",
        "MMP9->RECK",
        "FN1->FLT4",
        "VEGFC->FLT4",
        "EDN1->ADGRL4",
        "CCN2->ITGAM",
        "GAS6->MERTK",
        "PROS1->MERTK",
    }
    FARM_PREF_LR = {
        "CXCL13->CXCR5",
        "CCL19->CCR7",
        "CCL21->CCR7",
        "C3->CR2",
        "FCER2->CR2",
        "TNFSF13B->TNFRSF13C",
        "CD40LG->CD40",
        "LGALS3->LAG3",
        "LGALS9->HAVCR2",
        "CXCL8->ACKR1",
        "CXCL2->ACKR1",
        "F13A1->ITGA9",
        "F13A1->ITGB1",
        "MMP9->RECK",
        "PDGFC->FLT4",
        "S100A9->CD36",
        "FN1->FLT4",
    }
    PV_PARTNERS = (
        "Endothelial",
        "Sinus",
        "PAC",
        "PVC",
        "Arteriolar",
        "Venular",
        "Capillary",
        "Lymphatic",
    )
    FARM_PARTNERS = (
        "Endothelial",
        "Sinus",
        "PAC",
        "PVC",
        "fDC",
        "FRC",
        "mLTo",
        "Tfh",
        "Tfr",
        "Memory B",
        "GC B",
        "migDC",
        "BEST4",
        "MRC",
    )

    mac = read_filtered(
        lambda d: (
            (
                d["source"].isin(
                    {
                        "Perivascular Resident Macrophages",
                        "Follicle Associated Resident Macrophages",
                    }
                )
                | d["target"].isin(
                    {
                        "Perivascular Resident Macrophages",
                        "Follicle Associated Resident Macrophages",
                    }
                )
            )
            & d["tissue_level_1"].isin(SEG_ORDER)
        )
    )
    mac["ensemble_score"] = ensemble(mac["magnitude_rank"])
    mac["lr_pair"] = mac["ligand_complex"] + "->" + mac["receptor_complex"]
    mac = mac[~mac["lr_pair"].isin(BAN_LR)]
    mac = mac[
        ~mac["source"].str.contains("|".join(BAN_SENDERS), case=False, na=False)
        & ~mac["target"].str.contains("|".join(BAN_SENDERS), case=False, na=False)
    ]

    def pick_focus(ct: str, partner_tokens: tuple[str, ...], pref_lr: set[str], n: int = 8):
        sub = mac[(mac["source"] == ct) | (mac["target"] == ct)].copy()
        sub["focus"] = ct
        sub["partner"] = np.where(sub["source"] == ct, sub["target"], sub["source"])
        sub["direction"] = np.where(sub["source"] == ct, "outgoing", "incoming")
        sub = sub[sub["partner"].apply(lambda s: any(t in s for t in partner_tokens))]
        best = (
            sub.sort_values("ensemble_score", ascending=False)
            .groupby(["focus", "direction", "partner", "lr_pair"], as_index=False)
            .first()
        )
        pref = best[best["lr_pair"].isin(pref_lr)].sort_values(
            "ensemble_score", ascending=False
        )
        other = best[~best["lr_pair"].isin(pref_lr)].sort_values(
            "ensemble_score", ascending=False
        )
        out = pd.concat([pref, other], ignore_index=True).head(n)
        return out

    mac_panel = pd.concat(
        [
            pick_focus(
                "Perivascular Resident Macrophages", PV_PARTNERS, PV_PREF_LR, 8
            ),
            pick_focus(
                "Follicle Associated Resident Macrophages",
                FARM_PARTNERS,
                FARM_PREF_LR,
                8,
            ),
        ],
        ignore_index=True,
    )
    mac_panel.to_csv(DATA / "panel_d_mac_subtype_niche_top.csv", index=False)

    print("Wrote curated tables to", DATA)
    print("  panel_b:", len(ack_best))
    print("  panel_c:", len(ccr7_best))
    print("  panel_d:", len(mac_panel))


if __name__ == "__main__":
    main()
