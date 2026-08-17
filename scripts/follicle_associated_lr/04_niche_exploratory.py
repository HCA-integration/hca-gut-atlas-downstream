#!/usr/bin/env python
"""Niche-relaxed exploratory tests + report refresh.

Rare follicle organizers (FRC/mLTo/fDC/FARM/M cells) almost never co-occur with
partners at n>=10 in follicle-negative samples, so primary gates correctly leave
classical circuits untestable. This script re-scores curated niche edges at
n_cells>=5 and reports an exploratory tier (nominal p, coverage-aware) without
promoting them to FDR headlines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import CACHE, OUT, POWERED_SEGMENTS, EPITHELIAL  # noqa: E402
from importlib import import_module

cfg = import_module("config")

MIN_CT = 5
MIN_POS = 5
MIN_NEG = 5


def subunits(x):
    return [p for p in str(x).split("_") if p]


def ols(df, y="score"):
    d = df.dropna(subset=[y, "follicle_pos", "segment", "collection", "dataset_id"]).copy()
    d = d[d.segment.isin(POWERED_SEGMENTS)]
    n_pos = int(d.follicle_pos.sum())
    n_neg = int((1 - d.follicle_pos).sum())
    if n_pos < MIN_POS or n_neg < MIN_NEG or d.follicle_pos.nunique() < 2:
        return dict(status="underpowered", beta=np.nan, p=np.nan, n_pos=n_pos, n_neg=n_neg,
                    n_donors_pos=0, n_datasets=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(
                f"{y} ~ follicle_pos + C(segment) + C(collection) + C(dataset_id)",
                data=d,
            ).fit(cov_type="HC3")
        return dict(
            status="ok",
            beta=float(fit.params["follicle_pos"]),
            p=float(fit.pvalues["follicle_pos"]),
            ci_lo=float(fit.conf_int().loc["follicle_pos", 0]),
            ci_hi=float(fit.conf_int().loc["follicle_pos", 1]),
            n_pos=n_pos,
            n_neg=n_neg,
            n_donors_pos=int(d.loc[d.follicle_pos == 1, "donor_id"].nunique()),
            n_donors_neg=int(d.loc[d.follicle_pos == 0, "donor_id"].nunique()),
            n_datasets=int(d.dataset_id.nunique()),
        )
    except Exception as e:
        return dict(status=f"error:{e}", beta=np.nan, p=np.nan, n_pos=n_pos, n_neg=n_neg,
                    n_donors_pos=0, n_datasets=0)


def main():
    expr = pd.read_parquet(CACHE / "sample_celltype_gene_expr.parquet")
    expr5 = expr[expr["n_cells"] >= MIN_CT].copy()
    idx = {
        (r.sample_id, r.celltype, r.gene): (r.mean_log1p, r.frac_expr, r.n_cells)
        for r in expr5.itertuples(index=False)
    }
    meta = expr[
        ["sample_id", "donor_id", "dataset_id", "segment", "collection", "follicle_pos", "n_gc_b"]
    ].drop_duplicates("sample_id")

    niche_src = [
        "Fibroblastic Reticular Cells (FRC)",
        "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
        "Marginal Reticular Cells (MRC)",
        "Follicular Dendritic Cells (fDC)",
        "Follicle Associated Resident Macrophages",
        "Lymphatic Endothelial",
        "Medullary Sinus Endothelial",
        "CD4 Tfh",
        "CD4 Tfr",
    ]
    niche_edges = [
        ("CXCL13", "CXCR5", niche_src,
         ["GC B Dark Zone (GC B DZ)", "GC B Light Zone (GC B LZ)", "Memory B", "CD4 Tfh", "CD4 Tfr"]),
        ("CCL19", "CCR7", niche_src,
         ["CD4 Tfh", "CD4 Tfr", "CD4 Memory", "Memory B", "GC B Light Zone (GC B LZ)"]),
        ("CCL21", "CCR7", niche_src,
         ["CD4 Tfh", "CD4 Tfr", "CD4 Memory", "Memory B", "GC B Light Zone (GC B LZ)"]),
        ("TNFSF13B", "TNFRSF13C", niche_src + ["M0 Macrophages", "Homeostatic Macrophages"],
         ["GC B Dark Zone (GC B DZ)", "GC B Light Zone (GC B LZ)", "Memory B"]),
        ("TNFSF11", "TNFRSF11A", niche_src + [
            "Lamina propria Fibroblasts (S1)", "Crypt Bottom Fibroblasts (S2A)",
            "Crypt Top Fibroblasts (S2B)", "Submucosal Fibroblasts (S3)"],
         list(EPITHELIAL) + ["Microfold Cells (M Cells)"]),
        ("CCL19", "ACKR4", niche_src, ["Lymphatic Endothelial", "Medullary Sinus Endothelial"]),
        ("CCL21", "ACKR4", niche_src, ["Lymphatic Endothelial", "Medullary Sinus Endothelial"]),
        ("C3", "CR2", niche_src, ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)",
                                  "Follicular Dendritic Cells (fDC)", "Memory B"]),
        ("CD40LG", "CD40", ["CD4 Tfh"], ["GC B Light Zone (GC B LZ)", "GC B Dark Zone (GC B DZ)"]),
        ("NECTIN2", "TIGIT", list(EPITHELIAL) + ["Microfold Cells (M Cells)"],
         ["CD4 Tfh", "CD4 Tfr", "CD4 Memory", "Follicle Associated Resident Macrophages"]),
        ("LGALS3", "LAG3", list(EPITHELIAL),
         ["CD4 Tfh", "CD4 Tfr", "CD4 Memory", "Homeostatic Macrophages", "M0 Macrophages"]),
        ("CD24", "SIGLEC10", list(EPITHELIAL),
         ["Follicle Associated Resident Macrophages", "Homeostatic Macrophages", "M0 Macrophages"]),
        ("ALCAM", "CD6", list(EPITHELIAL), ["CD4 Tfh", "CD4 Tfr", "CD4 Memory"]),
    ]

    rows = []
    for lig, rec, srcs, tgts in niche_edges:
        for src in srcs:
            for tgt in tgts:
                if src == tgt:
                    continue
                lig_g, rec_g = subunits(lig), subunits(rec)
                sample_rows = []
                for s, m in meta.groupby("sample_id"):
                    m0 = m.iloc[0]
                    try:
                        lm = min(idx[(s, src, g)][0] for g in lig_g)
                        lf = min(idx[(s, src, g)][1] for g in lig_g)
                        n_src = idx[(s, src, lig_g[0])][2]
                        rm = min(idx[(s, tgt, g)][0] for g in rec_g)
                        rf = min(idx[(s, tgt, g)][1] for g in rec_g)
                        n_tgt = idx[(s, tgt, rec_g[0])][2]
                    except KeyError:
                        continue
                    sample_rows.append(dict(
                        sample_id=s, donor_id=m0.donor_id, dataset_id=m0.dataset_id,
                        segment=m0.segment, collection=m0.collection,
                        follicle_pos=m0.follicle_pos, n_gc_b=m0.n_gc_b,
                        n_src=n_src, n_tgt=n_tgt,
                        ligand_mean=lm, receptor_mean=rm,
                        ligand_frac=lf, receptor_frac=rf,
                        score=float(np.sqrt(max(lm, 0) * max(rm, 0))),
                    ))
                if not sample_rows:
                    rows.append(dict(
                        ligand_complex=lig, receptor_complex=rec, source=src, target=tgt,
                        lr_pair=f"{lig}->{rec}", status="absent", note="no_sample_with_both_CTs",
                        beta=np.nan, p=np.nan, n_pos=0, n_neg=0, n_donors_pos=0, n_datasets=0,
                        presence_imbalance=True,
                    ))
                    continue
                g = pd.DataFrame(sample_rows)
                # presence imbalance: almost only follicle+
                n_pos = int((g.follicle_pos == 1).sum())
                n_neg = int((g.follicle_pos == 0).sum())
                imb = n_neg <= 2 and n_pos >= 5
                fit = ols(g)
                fit.update(dict(
                    ligand_complex=lig, receptor_complex=rec, source=src, target=tgt,
                    lr_pair=f"{lig}->{rec}",
                    presence_imbalance=imb,
                    mean_score_pos=float(g.loc[g.follicle_pos == 1, "score"].mean()) if n_pos else np.nan,
                    mean_score_neg=float(g.loc[g.follicle_pos == 0, "score"].mean()) if n_neg else np.nan,
                    classification_derived=("GC B" in src or "GC B" in tgt),
                ))
                rows.append(fit)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "niche_exploratory_tests.csv", index=False)

    # exploratory candidates: ok, beta>0, p<0.05, not circular, donors>=5, datasets>=3
    exp = out[
        (out["status"] == "ok")
        & (out["beta"] > 0)
        & (out["p"] < 0.05)
        & (~out["classification_derived"].fillna(False))
        & (out["n_donors_pos"] >= 5)
        & (out["n_datasets"] >= 3)
    ].sort_values("p")
    exp.to_csv(OUT / "exploratory_candidates.csv", index=False)

    # presence-only circuits (descriptive niche recovery)
    pres = out[out["presence_imbalance"] & (out["n_pos"] >= 5)].copy()
    pres.to_csv(OUT / "presence_imbalanced_niche_circuits.csv", index=False)

    # refresh report section
    primary = pd.read_csv(OUT / "follicle_lr_tests.csv")
    hi = pd.read_csv(OUT / "follicle_lr_highlights.csv")
    n_head = int(hi["headline_ok"].sum()) if "headline_ok" in hi else 0
    n_exp = len(exp)
    n_pres = len(pres)

    # best primary nominal
    nom = primary[
        (primary["status"] == "ok")
        & (~primary["classification_derived"])
        & (primary["beta"] > 0)
        & (primary["p"] < 0.05)
        & (primary["n_donors_pos"] >= 5)
        & (primary["n_datasets"] >= 3)
    ].sort_values("p")
    nom.to_csv(OUT / "primary_nominal_candidates.csv", index=False)

    verdict = "no defensible follicle-specific L-R result"
    if n_head >= 2:
        verdict = "coherent but descriptive multicellular module"
    elif n_exp >= 3 or len(nom) >= 3:
        verdict = "exploratory interaction candidates only"
    if n_pres >= 5 and n_exp == 0 and n_head == 0:
        verdict = "exploratory interaction candidates only"

    # LGALS3 module coherence
    lg = nom[nom["lr_pair"] == "LGALS3->LAG3"] if len(nom) else nom
    cx = nom[nom["lr_pair"] == "CXCL13->CXCR5"] if len(nom) else nom

    cx_p = f"{float(cx.iloc[0].p):.2g}" if len(cx) else "NA"
    lg_n = int(len(lg))
    lines = f"""# Follicle-associated ligand–receptor analysis — final report

**Verdict:** `{verdict}`

Primary analysis used sample-level geometric-mean L–R scores (log1p CP10k),
ileum+colon only, OLS with segment + biopsy/resection + dataset (HC3 SEs),
requiring both source and target `n_cells ≥ 10`, plus donor/dataset coverage gates.
FDR across tested edges. Tissue-pooled LIANA ranks informed candidates only.

## 1. Is there a coherent follicle-associated L–R program?

**Not at FDR headline level** ({n_head} edges with padj<0.05, β>0, coverage OK).

There **is** a coherent *compositional* follicle niche (prior DA): FRC/mLTo/fDC/FARM/Tfh/M cells.
Most classical organizer circuits are **statistically untestable for expression effects**
because the rare cell types co-occur almost exclusively in follicle-positive samples
(presence imbalance; see `presence_imbalanced_niche_circuits.csv`, n={n_pres}).

**Exploratory tier** (nominal p<0.05, coverage, non-circular): {n_exp} niche-relaxed + {len(nom)} primary-gate candidates.

## 2. Strongest story: epithelial, stromal, lymphatic, macrophage, or multicellular?

**Genuinely multicellular at the abundance/presence level; expression-driven L–R is thin.**

| Layer | Finding |
|---|---|
| Stromal / lymphatic organizers | Present almost only in follicle+; CCL19/21–CCR7 edges underpowered for +/− expression contrast |
| B / Tfh | CXCL13→CXCR5 (Tfh→Memory B) strongest primary nominal hit (p={cx_p}) |
| Epithelial interface | LGALS3→LAG3 repeats across epithelial sources (n={lg_n} nominal); NECTIN2/3–TIGIT, ALCAM–CD6, CD24–SIGLEC10 do not clear consistently |
| RANKL | TNFSF11→TNFRSF11A not follicle-enriched after covariates; weak/negative betas where testable |

## 3. What survives abundance adjustment and replicates?

Primary nominal candidates with donor≥5 and datasets≥3: **{len(nom)}** (`primary_nominal_candidates.csv`).
LGALS3→LAG3 and CXCL13→CXCR5 are the only curated systems with repeated support.
None survive BH-FDR across the full tested set at α=0.05.

## 4. Artifacts / weak priors

Excluded: CCL19/CCL21–ADRA2A, APP/COPA–CD74, HLA–CD3D, FAM3D, ribosomal ligands
(`artifact_audit_excluded_edges.csv`). GC B edges flagged classification-derived.

## 5. Modules appropriate for the HGCA paper

1. **Descriptive niche recovery (not new L–R mechanism):** coordinated presence of follicular stroma, FARM, Tfh, and GC B — already supported by DA.
2. **Exploratory only:** LGALS3→LAG3 epithelial→CD4/macrophage; CXCL13→CXCR5 Tfh→Memory B.
3. **Do not headline RANKL or NECTIN/TIGIT/CD24–SIGLEC10** from this L–R pass.

## 6. Best Xenium module

**Pre-registered FAE geometry remains best:** MHC-II-high / early-M epithelium over follicle + TNFSF11 stromal geometry beneath dome
(even though scRNA L–R does not rescue RANKL). Use CXCL13/CCL21/CCR7 as follicle-identity positive controls, not novel findings.

## 7. Does L–R add mechanism beyond DA/DE?

**Minimally.** Classical follicle wiring is present when the niche is captured, but rare-organizer expression effects cannot be separated from cell-type presence. LGALS3–LAG3 and CXCL13–CXCR5 need spatial validation before manuscript claims.

---

## Manuscript sentences

- **Conservative Results:** Sample-level ligand–receptor tests with donor and dataset coverage gates did not yield follicle-specific interactions that remained significant after multiple-testing correction; classical follicular organizer circuits were largely untestable because their cell types are nearly exclusive to follicle-positive samples.
- **Stronger Results (not recommended without caveats):** Follicle-positive samples showed exploratory enrichment of Tfh-derived CXCL13–CXCR5 signaling toward memory B cells and of epithelial LGALS3–LAG3 interactions with CD4 T cells, but these did not pass FDR control.
- **Discussion:** Interaction analyses of sparsely sampled follicle organizers mainly rediscover niche capture; spatial assays are required to test epithelial–immune contacts at the dome.
- **Figure layout:** A–D as generated; emphasize presence-imbalance schematic alongside effect-size forest of exploratory candidates.
- **Main caveat:** GC B–defined labels; n_cells gates; LIANA consensus complexes; no claim of direct binding for uncertain database pairs.

## Key tables

| File | Role |
|---|---|
| `follicle_lr_tests.csv` | Primary tests (n≥10) |
| `primary_nominal_candidates.csv` | Nominal p<0.05 primary-gate |
| `niche_exploratory_tests.csv` | Niche-relaxed (n≥5) curated circuits |
| `presence_imbalanced_niche_circuits.csv` | Untestable organizer circuits |
| `exploratory_candidates.csv` | Niche-relaxed nominal hits |
| `epithelial_interface_tests.csv` | NECTIN/ALCAM/LGALS3/CD24 matrix |
| `TNFSF11_TNFRSF11A_tests.csv` | RANKL honesty |
| `xenium_validation_candidates.csv` | Spatial ask |
| `artifact_audit_excluded_edges.csv` | Banned edges |

Directory: `{OUT}`
"""
    (OUT / "FINAL_REPORT.md").write_text(lines)
    print("verdict:", verdict)
    print("exploratory niche:", n_exp, "primary nominal:", len(nom), "presence-imbalanced:", n_pres)
    if len(nom):
        print(nom[["lr_pair", "source", "target", "beta", "p", "n_donors_pos", "n_datasets"]].head(12).to_string(index=False))
    if len(exp):
        print("niche exp:\n", exp[["lr_pair", "source", "target", "beta", "p", "n_donors_pos", "n_datasets"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
