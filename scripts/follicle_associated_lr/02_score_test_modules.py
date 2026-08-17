#!/usr/bin/env python
"""Score sample-level L–R interactions, test follicle+/− effects, build modules.

Statistical unit = sample. Primary model (ileum+colon):
  score ~ follicle_pos + C(segment) + C(collection) + C(dataset_id)
with HC3 robust SEs (same framework as follicle DA).

Also reports abundance-adjusted models and expression-only scores.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (  # noqa: E402
    BANNED_CT_SUBSTR,
    BANNED_EXACT,
    BANNED_LIGAND_PREFIX,
    CACHE,
    CLASSIFICATION_DERIVED,
    CURATED,
    ECOSYSTEM,
    EPITHELIAL,
    EXPR_PROP_SOFT,
    FAE_MODULE_SAMPLE,
    FDR_ALPHA,
    LIANA_COMBINED,
    MIN_CT_CELLS,
    MIN_DATASETS,
    MIN_DONORS_NEG,
    MIN_DONORS_POS,
    MIN_SAMPLES_NEG,
    MIN_SAMPLES_POS,
    MODULE_COLORS,
    OUT,
    POWERED_SEGMENTS,
    WONG,
)

EPS = 1e-9


def subunits(x: str) -> list[str]:
    return [p for p in str(x).split("_") if p]


def complex_mean(expr_ct: pd.DataFrame, genes: list[str], col: str) -> float:
    vals = []
    for g in genes:
        hit = expr_ct.loc[expr_ct["gene"] == g, col]
        if hit.empty:
            return np.nan
        vals.append(float(hit.iloc[0]))
    return float(np.min(vals))  # LIANA-like: complex limited by weakest subunit


def is_banned(lig: str, rec: str, src: str, tgt: str) -> tuple[bool, str]:
    if (lig, rec) in BANNED_EXACT:
        return True, f"banned_exact:{lig}->{rec}"
    for p in BANNED_LIGAND_PREFIX:
        if lig.startswith(p) or any(s.startswith(p) for s in subunits(lig)):
            return True, f"banned_ligand_prefix:{p}"
    for s in BANNED_CT_SUBSTR:
        if s in src or s in tgt:
            return True, f"banned_celltype:{s}"
    # generic HLA to CD3 subunit
    if lig.startswith("HLA-") and rec.startswith("CD3"):
        return True, "hla_to_cd3_subunit"
    return False, ""


def evidence_class(lig: str, rec: str) -> str:
    established = {
        ("CXCL13", "CXCR5"), ("CCL19", "CCR7"), ("CCL21", "CCR7"),
        ("TNFSF13B", "TNFRSF13C"), ("TNFSF11", "TNFRSF11A"),
        ("LTB", "LTBR"), ("CD40LG", "CD40"), ("CCL25", "CCR9"),
        ("VCAM1", "ITGA4_ITGB7"), ("ICAM1", "ITGAL_ITGB2"),
        ("NECTIN2", "TIGIT"), ("ALCAM", "CD6"), ("CD24", "SIGLEC10"),
    }
    atypical = {
        ("CCL19", "ACKR4"), ("CCL21", "ACKR4"), ("CCL2", "ACKR2"),
        ("CCL5", "ACKR2"), ("CCL14", "ACKR1"), ("CCL2", "ACKR1"),
    }
    plausible = {
        ("LGALS3", "LAG3"), ("NECTIN3", "TIGIT"), ("C3", "CR2"),
        ("C1QA", "CR1"), ("GAS6", "MERTK"), ("PROS1", "MERTK"),
        ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B"),
    }
    key = (lig, rec)
    if key in established:
        return "well_established_direct"
    if key in atypical:
        return "atypical_chemokine_handling"
    if key in plausible:
        return "plausible_extracellular_or_adhesion"
    if key in BANNED_EXACT:
        return "likely_artifact"
    return "database_supported_uncertain"


def assign_module(lig: str, rec: str, src: str, tgt: str) -> str:
    if (lig, rec) in {("TNFSF11", "TNFRSF11A"), ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B")}:
        return "mcell_induction"
    if (lig, rec) in {("CXCL13", "CXCR5"), ("TNFSF13B", "TNFRSF13C"), ("TNFSF13B", "TNFRSF17"),
                      ("TNFSF13", "TNFRSF13B"), ("CD40LG", "CD40"), ("C3", "CR2"), ("FCER2", "CR2")}:
        return "bcell_recruitment_retention"
    if (lig, rec) in {("CCL19", "CCR7"), ("CCL21", "CCR7")} and (
        "Tfh" in tgt or "Tfr" in tgt or "Tfh" in src or "Tfr" in src
        or "FRC" in src or "mLTo" in src or "MRC" in src
    ):
        return "tfh_tfr_positioning"
    if (lig, rec) in {("CCL19", "CCR7"), ("CCL21", "CCR7"), ("CXCL13", "CXCR5"),
                      ("LTB", "LTBR"), ("LTA", "LTBR"), ("VCAM1", "ITGA4_ITGB1"),
                      ("VCAM1", "ITGA4_ITGB7"), ("ICAM1", "ITGAL_ITGB2")}:
        return "follicular_stromal_organization"
    if (lig, rec) in {("CCL19", "ACKR4"), ("CCL21", "ACKR4"), ("CCL2", "ACKR2"),
                      ("CCL5", "ACKR2"), ("CCL25", "CCR9")} or "Lymphatic" in src or "Sinus" in src:
        if lig.startswith("CCL") or lig.startswith("CXCL"):
            return "lymphatic_chemokine_gradient"
    if (lig, rec) in {("NECTIN2", "TIGIT"), ("NECTIN3", "TIGIT"), ("ALCAM", "CD6"),
                      ("LGALS3", "LAG3"), ("CD24", "SIGLEC10")}:
        return "epithelial_regulatory_adhesion"
    if src in EPITHELIAL and ("HLA" in lig or lig in {"CD74", "CIITA"}):
        return "epithelial_ag_presentation_interface"
    if "Macrophage" in src or "Macrophage" in tgt or lig.startswith("C1Q") or lig in {"C3", "GAS6", "PROS1"}:
        return "macrophage_antigen_handling"
    if src in EPITHELIAL:
        return "epithelial_ag_presentation_interface"
    return "follicular_stromal_organization"


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def ols_follicle(df: pd.DataFrame, y: str, abundance: bool = False) -> dict:
    d = df.dropna(subset=[y, "follicle_pos", "segment", "collection", "dataset_id"]).copy()
    d = d[d["segment"].isin(POWERED_SEGMENTS)]
    if d["follicle_pos"].nunique() < 2 or len(d) < 20:
        return dict(status="underpowered", beta=np.nan, se=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p=np.nan, n=len(d), n_pos=int(d.follicle_pos.sum()) if len(d) else 0,
                    n_neg=int((1 - d.follicle_pos).sum()) if len(d) else 0,
                    n_donors_pos=0, n_donors_neg=0, n_datasets=0, model="skipped")
    n_pos = int(d.follicle_pos.sum())
    n_neg = int((1 - d.follicle_pos).sum())
    n_don_pos = d.loc[d.follicle_pos == 1, "donor_id"].nunique()
    n_don_neg = d.loc[d.follicle_pos == 0, "donor_id"].nunique()
    n_ds = d.dataset_id.nunique()
    if (n_pos < MIN_SAMPLES_POS or n_neg < MIN_SAMPLES_NEG
            or n_don_pos < MIN_DONORS_POS or n_don_neg < MIN_DONORS_NEG
            or n_ds < MIN_DATASETS):
        return dict(status="coverage_fail", beta=np.nan, se=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, p=np.nan, n=len(d), n_pos=n_pos, n_neg=n_neg,
                    n_donors_pos=n_don_pos, n_donors_neg=n_don_neg, n_datasets=n_ds,
                    model="skipped_coverage")
    try:
        if abundance:
            formula = (
                f"{y} ~ follicle_pos + C(segment) + C(collection) + C(dataset_id) "
                "+ log1p_n_src + log1p_n_tgt"
            )
            model = "ols_abundance_adjusted"
        else:
            formula = f"{y} ~ follicle_pos + C(segment) + C(collection) + C(dataset_id)"
            model = "ols_adjusted"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(formula, data=d).fit(cov_type="HC3")
        if "follicle_pos" not in fit.params.index:
            return dict(status="coef_missing", beta=np.nan, se=np.nan, ci_lo=np.nan,
                        ci_hi=np.nan, p=np.nan, n=len(d), n_pos=n_pos, n_neg=n_neg,
                        n_donors_pos=n_don_pos, n_donors_neg=n_don_neg, n_datasets=n_ds,
                        model=model)
        beta = float(fit.params["follicle_pos"])
        se = float(fit.bse["follicle_pos"])
        ci = fit.conf_int().loc["follicle_pos"]
        return dict(status="ok", beta=beta, se=se, ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                    p=float(fit.pvalues["follicle_pos"]), n=len(d), n_pos=n_pos, n_neg=n_neg,
                    n_donors_pos=n_don_pos, n_donors_neg=n_don_neg, n_datasets=n_ds,
                    model=model)
    except Exception as e:
        return dict(status=f"error:{type(e).__name__}", beta=np.nan, se=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n=len(d), n_pos=n_pos,
                    n_neg=n_neg, n_donors_pos=n_don_pos, n_donors_neg=n_don_neg,
                    n_datasets=n_ds, model="failed")


def build_candidate_edges(expr: pd.DataFrame) -> pd.DataFrame:
    """Curated + LIANA-priority edges restricted to observed ecosystem CTs."""
    rows = []
    for lig, rec in CURATED:
        rows.append(dict(ligand_complex=lig, receptor_complex=rec, priority="curated"))
    pri = CACHE / "liana_priority_lr_pairs.csv"
    if pri.exists():
        p = pd.read_csv(pri)
        for _, r in p.iterrows():
            rows.append(dict(
                ligand_complex=r["ligand_complex"],
                receptor_complex=r["receptor_complex"],
                priority="liana_top",
            ))
    cands = pd.DataFrame(rows).drop_duplicates(["ligand_complex", "receptor_complex"])
    # Expand to source/target pairs present in expr with usable cells in ≥ few samples
    usable = expr.loc[expr["usable"], ["sample_id", "celltype"]].drop_duplicates()
    ct_counts = usable["celltype"].value_counts()
    cts = [c for c in ECOSYSTEM if ct_counts.get(c, 0) >= MIN_SAMPLES_POS]
    # Limit discovery pairs: curated for all src/tgt in module-relevant sets;
    # liana_top only for pairs that appear in tissue-pooled LIANA ecosystem edges
    edges = []
    liana_edges = set()
    if LIANA_COMBINED.exists():
        usecols = ["source", "target", "ligand_complex", "receptor_complex", "magnitude_rank"]
        for chunk in pd.read_csv(LIANA_COMBINED, usecols=usecols, chunksize=500_000):
            m = chunk[
                chunk["source"].isin(cts)
                & chunk["target"].isin(cts)
                & (chunk["magnitude_rank"] <= 0.05)
            ]
            for _, r in m.iterrows():
                liana_edges.add((r["ligand_complex"], r["receptor_complex"], r["source"], r["target"]))
    # curated: targeted partner sets
    partner_map = {
        ("TNFSF11", "TNFRSF11A"): (
            [c for c in cts if "Fibroblast" in c or "FRC" in c or "mLTo" in c or "MRC" in c or "fDC" in c],
            [c for c in cts if c in EPITHELIAL],
        ),
        ("CXCL13", "CXCR5"): (
            [c for c in cts if any(x in c for x in ("fDC", "FRC", "mLTo", "MRC", "Tfh"))],
            [c for c in cts if any(x in c for x in ("GC B", "Tfh", "Tfr", "Memory B"))],
        ),
        ("CCL19", "CCR7"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "Lymphatic", "Sinus"))],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "Memory B", "GC B"))],
        ),
        ("CCL21", "CCR7"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "Lymphatic", "Sinus"))],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "Memory B", "GC B"))],
        ),
        ("NECTIN2", "TIGIT"): (
            [c for c in cts if c in EPITHELIAL],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "Macrophage"))],
        ),
        ("NECTIN3", "TIGIT"): (
            [c for c in cts if c in EPITHELIAL],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "Macrophage"))],
        ),
        ("ALCAM", "CD6"): (
            [c for c in cts if c in EPITHELIAL],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4"))],
        ),
        ("LGALS3", "LAG3"): (
            [c for c in cts if c in EPITHELIAL],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "Macrophage"))],
        ),
        ("CD24", "SIGLEC10"): (
            [c for c in cts if c in EPITHELIAL],
            [c for c in cts if "Macrophage" in c],
        ),
        ("TNFSF13B", "TNFRSF13C"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "fDC", "Macrophage"))],
            [c for c in cts if any(x in c for x in ("GC B", "Memory B"))],
        ),
        ("TNFSF13B", "TNFRSF17"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "fDC", "Macrophage"))],
            [c for c in cts if any(x in c for x in ("GC B", "Memory B"))],
        ),
        ("LTB", "LTBR"): (
            [c for c in cts if any(x in c for x in ("GC B", "Tfh", "Tfr", "Memory B"))],
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "fDC"))],
        ),
        ("LTA", "LTBR"): (
            [c for c in cts if any(x in c for x in ("GC B", "Tfh", "Tfr"))],
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "fDC"))],
        ),
        ("CCL19", "ACKR4"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "Lymphatic", "Sinus"))],
            [c for c in cts if any(x in c for x in ("Lymphatic", "Sinus", "Endothelial"))],
        ),
        ("CCL21", "ACKR4"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "Lymphatic", "Sinus"))],
            [c for c in cts if any(x in c for x in ("Lymphatic", "Sinus", "Endothelial"))],
        ),
        ("CCL25", "CCR9"): (
            [c for c in cts if c in EPITHELIAL or "Lymphatic" in c],
            [c for c in cts if "CD4" in c or "Memory B" in c or "GC B" in c],
        ),
        ("C3", "CR2"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "MRC", "fDC", "Macrophage"))],
            [c for c in cts if any(x in c for x in ("GC B", "Memory B", "fDC"))],
        ),
        ("C1QA", "CR1"): (
            [c for c in cts if "Macrophage" in c],
            [c for c in cts if any(x in c for x in ("GC B", "fDC", "Endothelial"))],
        ),
        ("GAS6", "MERTK"): (
            [c for c in cts if any(x in c for x in ("Endothelial", "Fibroblast", "FRC"))],
            [c for c in cts if "Macrophage" in c],
        ),
        ("ICAM1", "ITGAL_ITGB2"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "Endothelial", "fDC"))],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "GC B"))],
        ),
        ("VCAM1", "ITGA4_ITGB7"): (
            [c for c in cts if any(x in c for x in ("FRC", "mLTo", "Endothelial", "MRC"))],
            [c for c in cts if any(x in c for x in ("Tfh", "Tfr", "CD4", "Memory B", "GC B"))],
        ),
    }
    for _, r in cands.iterrows():
        lig, rec = r["ligand_complex"], r["receptor_complex"]
        if r["priority"] == "curated" and (lig, rec) in partner_map:
            srcs, tgts = partner_map[(lig, rec)]
            for s in srcs:
                for t in tgts:
                    if s != t:
                        edges.append(dict(ligand_complex=lig, receptor_complex=rec,
                                          source=s, target=t, priority="curated"))
        elif r["priority"] == "curated":
            # curated without explicit map: only LIANA-supported concrete edges
            for lig2, rec2, s, t in liana_edges:
                if lig2 == lig and rec2 == rec:
                    edges.append(dict(ligand_complex=lig, receptor_complex=rec,
                                      source=s, target=t, priority="curated_liana"))
    # liana-supported concrete edges
    for lig, rec, s, t in liana_edges:
        edges.append(dict(ligand_complex=lig, receptor_complex=rec,
                          source=s, target=t, priority="liana_edge"))
    ed = pd.DataFrame(edges).drop_duplicates(["ligand_complex", "receptor_complex", "source", "target"])
    # drop banned
    keep = []
    audit = []
    for _, r in ed.iterrows():
        ban, why = is_banned(r["ligand_complex"], r["receptor_complex"], r["source"], r["target"])
        if ban:
            audit.append({**r.to_dict(), "reason": why})
        else:
            keep.append(r)
    pd.DataFrame(audit).to_csv(OUT / "artifact_audit_excluded_edges.csv", index=False)
    ed = pd.DataFrame(keep)
    # Prefer curated + limit liana discovery to top N by joining ranks
    if len(ed) > 2500:
        ed["_cur"] = ed["priority"].str.startswith("curated").astype(int)
        ed = ed.sort_values("_cur", ascending=False).head(2500).drop(columns="_cur")
    ed.to_csv(CACHE / "candidate_edges.csv", index=False)
    print(f"Candidate edges: {len(ed):,}  (audit excluded {len(audit)})")
    return ed


def score_edges(expr: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample scores for each edge."""
    # index expr for fast lookup
    expr_u = expr.loc[expr["usable"]].copy()
    # dict: (sample, ct, gene) -> (mean, frac, n)
    print("Indexing expression…")
    idx = {}
    for r in expr_u.itertuples(index=False):
        idx[(r.sample_id, r.celltype, r.gene)] = (r.mean_log1p, r.frac_expr, r.n_cells)

    meta_cols = [
        "sample_id", "donor_id", "dataset_id", "segment", "collection",
        "follicle_pos", "n_gc_b", "gc_abundance_frac",
    ]
    meta = expr[meta_cols].drop_duplicates("sample_id")
    samples = meta["sample_id"].tolist()

    out_rows = []
    t0 = time.time()
    for i, e in enumerate(edges.itertuples(index=False)):
        if i and i % 200 == 0:
            print(f"  scored {i}/{len(edges)} edges…")
        lig_g = subunits(e.ligand_complex)
        rec_g = subunits(e.receptor_complex)
        for s in samples:
            # source ligand
            lig_means, lig_fracs, n_src = [], [], None
            ok = True
            for g in lig_g:
                key = (s, e.source, g)
                if key not in idx:
                    ok = False
                    break
                m, f, n = idx[key]
                lig_means.append(m)
                lig_fracs.append(f)
                n_src = n
            if not ok:
                continue
            rec_means, rec_fracs, n_tgt = [], [], None
            for g in rec_g:
                key = (s, e.target, g)
                if key not in idx:
                    ok = False
                    break
                m, f, n = idx[key]
                rec_means.append(m)
                rec_fracs.append(f)
                n_tgt = n
            if not ok:
                continue
            lig_m = float(np.min(lig_means))
            rec_m = float(np.min(rec_means))
            lig_f = float(np.min(lig_fracs))
            rec_f = float(np.min(rec_fracs))
            # prevalence soft gate
            if lig_f < EXPR_PROP_SOFT and rec_f < EXPR_PROP_SOFT:
                # still record but mark low
                pass
            score = float(np.sqrt(max(lig_m, 0) * max(rec_m, 0)))
            out_rows.append(dict(
                sample_id=s,
                source=e.source,
                target=e.target,
                ligand_complex=e.ligand_complex,
                receptor_complex=e.receptor_complex,
                priority=e.priority,
                n_src=n_src,
                n_tgt=n_tgt,
                ligand_mean=lig_m,
                receptor_mean=rec_m,
                ligand_frac=lig_f,
                receptor_frac=rec_f,
                score=score,
                score_expr=score,  # same; abundance handled in model
                present=int(lig_f >= EXPR_PROP_SOFT and rec_f >= EXPR_PROP_SOFT),
            ))
    scores = pd.DataFrame(out_rows)
    scores = scores.merge(meta, on="sample_id", how="left")
    scores["log1p_n_src"] = np.log1p(scores["n_src"])
    scores["log1p_n_tgt"] = np.log1p(scores["n_tgt"])
    scores["lr_pair"] = scores["ligand_complex"] + "->" + scores["receptor_complex"]
    scores["edge_id"] = (
        scores["source"] + "||" + scores["target"] + "||" + scores["lr_pair"]
    )
    scores.to_parquet(CACHE / "sample_edge_scores.parquet", index=False)
    print(f"Score rows: {len(scores):,} in {(time.time()-t0)/60:.1f} min")
    return scores


def test_edges(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    edge_ids = scores["edge_id"].unique()
    print(f"Testing {len(edge_ids):,} edges…")
    for i, eid in enumerate(edge_ids):
        if i and i % 100 == 0:
            print(f"  tested {i}/{len(edge_ids)}")
        g = scores[scores["edge_id"] == eid].copy()
        # require both arms have some present scores optionally; use all usable
        # restrict to samples where both CTs usable (already true)
        base = g.iloc[0]
        ban, why = is_banned(base.ligand_complex, base.receptor_complex,
                             base.source, base.target)
        clas = evidence_class(base.ligand_complex, base.receptor_complex)
        mod = assign_module(base.ligand_complex, base.receptor_complex,
                            base.source, base.target)
        # primary score model
        m1 = ols_follicle(g, "score", abundance=False)
        m2 = ols_follicle(g, "score", abundance=True)
        # ligand/receptor expression alone in source/target (follicle effect)
        # build mini frames
        lig_df = g[["sample_id", "donor_id", "dataset_id", "segment", "collection",
                    "follicle_pos", "ligand_mean"]].drop_duplicates("sample_id")
        lig_df = lig_df.rename(columns={"ligand_mean": "y"})
        rec_df = g[["sample_id", "donor_id", "dataset_id", "segment", "collection",
                    "follicle_pos", "receptor_mean"]].drop_duplicates("sample_id")
        rec_df = rec_df.rename(columns={"receptor_mean": "y"})
        # reuse ols with y
        def _ols_y(dfy):
            d = dfy.rename(columns={"y": "score"})
            return ols_follicle(d, "score", abundance=False)

        ml = _ols_y(lig_df)
        mr = _ols_y(rec_df)
        # abundance of source/target
        src_ab = g[["sample_id", "donor_id", "dataset_id", "segment", "collection",
                    "follicle_pos", "log1p_n_src"]].drop_duplicates("sample_id")
        src_ab = src_ab.rename(columns={"log1p_n_src": "score"})
        tgt_ab = g[["sample_id", "donor_id", "dataset_id", "segment", "collection",
                    "follicle_pos", "log1p_n_tgt"]].drop_duplicates("sample_id")
        tgt_ab = tgt_ab.rename(columns={"log1p_n_tgt": "score"})
        ms = ols_follicle(src_ab, "score", abundance=False)
        mt = ols_follicle(tgt_ab, "score", abundance=False)

        # segment specificity: beta in ileum vs colon separately (simple)
        seg_notes = {}
        for seg in POWERED_SEGMENTS:
            gs = g[g["segment"] == seg]
            if gs["follicle_pos"].nunique() < 2 or len(gs) < 15:
                seg_notes[seg] = np.nan
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = smf.ols(
                            "score ~ follicle_pos + C(collection) + C(dataset_id)",
                            data=gs,
                        ).fit(cov_type="HC3")
                    seg_notes[seg] = float(fit.params.get("follicle_pos", np.nan))
                except Exception:
                    seg_notes[seg] = np.nan

        # collection sensitivity
        coll_betas = {}
        for coll in ("biopsy", "resection"):
            gc = g[g["collection"] == coll]
            if gc["follicle_pos"].nunique() < 2 or len(gc) < 15:
                coll_betas[coll] = np.nan
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = smf.ols(
                            "score ~ follicle_pos + C(segment) + C(dataset_id)",
                            data=gc,
                        ).fit(cov_type="HC3")
                    coll_betas[coll] = float(fit.params.get("follicle_pos", np.nan))
                except Exception:
                    coll_betas[coll] = np.nan

        circ = (
            base.source in CLASSIFICATION_DERIVED
            or base.target in CLASSIFICATION_DERIVED
        )
        rows.append(dict(
            edge_id=eid,
            source=base.source,
            target=base.target,
            ligand_complex=base.ligand_complex,
            receptor_complex=base.receptor_complex,
            lr_pair=base.lr_pair,
            priority=base.priority,
            module=mod,
            evidence_class=clas,
            classification_derived=circ,
            banned=ban,
            ban_reason=why,
            mean_score_pos=float(g.loc[g.follicle_pos == 1, "score"].mean()),
            mean_score_neg=float(g.loc[g.follicle_pos == 0, "score"].mean()),
            mean_n_src_pos=float(g.loc[g.follicle_pos == 1, "n_src"].mean()),
            mean_n_src_neg=float(g.loc[g.follicle_pos == 0, "n_src"].mean()),
            mean_n_tgt_pos=float(g.loc[g.follicle_pos == 1, "n_tgt"].mean()),
            mean_n_tgt_neg=float(g.loc[g.follicle_pos == 0, "n_tgt"].mean()),
            mean_lig_frac_pos=float(g.loc[g.follicle_pos == 1, "ligand_frac"].mean()),
            mean_rec_frac_pos=float(g.loc[g.follicle_pos == 1, "receptor_frac"].mean()),
            beta=m1["beta"], se=m1["se"], ci_lo=m1["ci_lo"], ci_hi=m1["ci_hi"],
            p=m1["p"], status=m1["status"], model=m1["model"],
            n=m1["n"], n_pos=m1["n_pos"], n_neg=m1["n_neg"],
            n_donors_pos=m1["n_donors_pos"], n_donors_neg=m1["n_donors_neg"],
            n_datasets=m1["n_datasets"],
            beta_abund_adj=m2["beta"], p_abund_adj=m2["p"], status_abund_adj=m2["status"],
            beta_ligand_expr=ml["beta"], p_ligand_expr=ml["p"],
            beta_receptor_expr=mr["beta"], p_receptor_expr=mr["p"],
            beta_src_abundance=ms["beta"], p_src_abundance=ms["p"],
            beta_tgt_abundance=mt["beta"], p_tgt_abundance=mt["p"],
            beta_ileum=seg_notes.get("ileum", np.nan),
            beta_colon=seg_notes.get("colon", np.nan),
            beta_biopsy=coll_betas.get("biopsy", np.nan),
            beta_resection=coll_betas.get("resection", np.nan),
        ))
    res = pd.DataFrame(rows)
    res["padj"] = bh(res["p"].values)
    res["padj_abund_adj"] = bh(res["p_abund_adj"].values)
    # decompose call
    def decomp(r):
        if not np.isfinite(r.beta) or r.status != "ok":
            return "untested"
        abund_driven = (
            np.isfinite(r.beta_abund_adj)
            and abs(r.beta_abund_adj) < 0.5 * abs(r.beta)
            and (r.p_src_abundance < 0.05 or r.p_tgt_abundance < 0.05)
        )
        expr_driven = (
            np.isfinite(r.beta_abund_adj)
            and r.p_abund_adj < 0.05
            and (r.p_ligand_expr < 0.05 or r.p_receptor_expr < 0.05)
        )
        presence_only = (
            (r.mean_n_src_neg < MIN_CT_CELLS or r.mean_n_tgt_neg < MIN_CT_CELLS)
            and r.mean_n_src_pos >= MIN_CT_CELLS
            and r.mean_n_tgt_pos >= MIN_CT_CELLS
        )
        if presence_only:
            return "rare_cell_presence"
        if abund_driven and not expr_driven:
            return "abundance_driven"
        if expr_driven and not abund_driven:
            return "expression_driven"
        if expr_driven and abund_driven:
            return "coordinated_abundance_and_expression"
        if np.isfinite(r.beta_abund_adj) and r.p_abund_adj < 0.05:
            return "survives_abundance_adjustment"
        return "weak_or_mixed"

    res["effect_decomposition"] = res.apply(decomp, axis=1)
    res.to_csv(OUT / "follicle_lr_tests.csv", index=False)
    return res


def main():
    t0 = time.time()
    expr_path = CACHE / "sample_celltype_gene_expr.parquet"
    if not expr_path.exists():
        raise SystemExit("Run 01_build_sample_expr.py first")
    expr = pd.read_parquet(expr_path)
    print(f"Loaded expr: {len(expr):,}")

    edges = build_candidate_edges(expr)
    # Prefer curated edges first for scoring if huge
    curated_edges = edges[edges["priority"].str.startswith("curated")]
    liana_edges = edges[edges["priority"] == "liana_edge"]
    # cap liana edges
    if len(liana_edges) > 800:
        # keep those with both CTs having decent sample coverage
        cov = expr.loc[expr["usable"]].groupby("celltype")["sample_id"].nunique()
        liana_edges = liana_edges[
            liana_edges["source"].map(cov).fillna(0).ge(MIN_SAMPLES_POS)
            & liana_edges["target"].map(cov).fillna(0).ge(MIN_SAMPLES_POS)
        ].head(800)
    edges = pd.concat([curated_edges, liana_edges], ignore_index=True).drop_duplicates(
        ["ligand_complex", "receptor_complex", "source", "target"]
    )
    print(f"Scoring {len(edges)} edges…")
    scores = score_edges(expr, edges)
    res = test_edges(scores)

    # Highlight table: non-circular, not banned, status ok, padj or strong curated
    hi = res[
        (~res["classification_derived"])
        & (res["status"] == "ok")
        & (res["evidence_class"] != "likely_artifact")
    ].copy()
    hi["headline_ok"] = (
        (hi["padj"] < FDR_ALPHA)
        & (hi["n_donors_pos"] >= MIN_DONORS_POS)
        & (hi["n_datasets"] >= MIN_DATASETS)
        & (hi["beta"] > 0)
    )
    hi = hi.sort_values(["headline_ok", "padj", "beta"], ascending=[False, True, False])
    hi.to_csv(OUT / "follicle_lr_highlights.csv", index=False)

    # Module summary
    mod_rows = []
    for mod, g in hi.groupby("module"):
        g2 = g[g["headline_ok"]]
        mod_rows.append(dict(
            module=mod,
            n_edges_tested=len(g),
            n_headline=len(g2),
            median_beta=float(g["beta"].median()) if len(g) else np.nan,
            best_edge=g2.iloc[0]["edge_id"] if len(g2) else (g.iloc[0]["edge_id"] if len(g) else ""),
            best_padj=float(g2.iloc[0]["padj"]) if len(g2) else np.nan,
            mean_n_datasets=float(g["n_datasets"].mean()) if len(g) else np.nan,
            color=MODULE_COLORS.get(mod, WONG["grey"]),
        ))
    pd.DataFrame(mod_rows).sort_values("n_headline", ascending=False).to_csv(
        OUT / "module_summary.csv", index=False
    )

    # Epithelial interface matrix ingredients
    iface = ["NECTIN2->TIGIT", "NECTIN3->TIGIT", "ALCAM->CD6",
             "LGALS3->LAG3", "CD24->SIGLEC10"]
    epi_mat = res[res["lr_pair"].isin(iface) & res["source"].isin(EPITHELIAL)].copy()
    epi_mat.to_csv(OUT / "epithelial_interface_tests.csv", index=False)

    # MHC-II correlation: join FAE module scores
    if FAE_MODULE_SAMPLE.exists():
        fae = pd.read_csv(FAE_MODULE_SAMPLE)
        # sample-level mhc2
        if "mhc2_antigen_presentation" in fae.columns:
            mhc = fae[["sample_id", "mhc2_antigen_presentation", "mcell_early"]].drop_duplicates("sample_id")
            # correlate ligand expression of interface genes in epithelial CTs with mhc2
            ligs = ["NECTIN2", "NECTIN3", "ALCAM", "LGALS3", "CD24"]
            corr_rows = []
            for ct in sorted(EPITHELIAL):
                for lig in ligs:
                    e = expr[(expr["celltype"] == ct) & (expr["gene"] == lig) & expr["usable"]]
                    if len(e) < 20:
                        continue
                    m = e.merge(mhc, on="sample_id")
                    if m["mhc2_antigen_presentation"].nunique() < 5:
                        continue
                    corr_rows.append(dict(
                        celltype=ct, gene=lig,
                        r_mhc2=float(m["mean_log1p"].corr(m["mhc2_antigen_presentation"])),
                        r_mcell_early=float(m["mean_log1p"].corr(m["mcell_early"])),
                        n=len(m),
                    ))
            pd.DataFrame(corr_rows).to_csv(OUT / "epithelial_ligand_vs_mhc2.csv", index=False)

    # TNFSF11 honesty table
    rankl = res[
        (res["ligand_complex"] == "TNFSF11") & (res["receptor_complex"] == "TNFRSF11A")
    ].sort_values("padj")
    rankl.to_csv(OUT / "TNFSF11_TNFRSF11A_tests.csv", index=False)

    summary = dict(
        n_edges_tested=int(len(res)),
        n_status_ok=int((res["status"] == "ok").sum()),
        n_headline=int(hi["headline_ok"].sum()),
        n_classification_derived=int(res["classification_derived"].sum()),
        runtime_min=round((time.time() - t0) / 60, 2),
    )
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Wrote tests to", OUT)


if __name__ == "__main__":
    main()
