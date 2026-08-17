#!/usr/bin/env python3
"""Analytical DESeq2 power for S7 panel g (published vs +contributed).

Current recovery script uses design ``~ dataset_id + seg`` only. Sample is the
pseudobulk *unit of observation*, not a covariate — so there is no
``~ sample_id + seg`` model. The natural alternative is:

  A) ``~ dataset_id + seg``  — batch-adjusted (preferred for multi-study atlas)
  B) ``~ seg``               — samples as biological replicates (no dataset FE)

Power model (Wald, two-sided):
  1) Fit DESeq2 on the full atlas → β̂ (log2FC) and SE_full per gene
  2) Treat genes with padj < FDR and |β̂| ≥ LFC_MIN as the target set
  3) Scale SE to a new balanced sample size:
        SE(n) = SE_full * sqrt(n_eff_full / n)
     where n_eff_full = harmonic mean of (n_ileum, n_colon) in the fit
  4) Power_g(n) = Φ(|β̂|/SE(n) − z_{1−α/2}) + Φ(−|β̂|/SE(n) − z_{1−α/2})
  5) Report mean power over the target gene set at n_pub and n_all

This is smooth, monotone, and not tautological (full-set power < 1).
Requires patpy (pydeseq2).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

LOG = logging.getLogger("deseq2_analytical_power")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OBJ = Path(os.environ["HGCA_OBJECTS"]) if os.environ.get("HGCA_OBJECTS") else None

PREPUB_IDS = {
    "ArendsHelmsley",
    "DominguezUnpub2",
    "DominguezUnpub",
    "BasuGCARNA",
    "BasuHelmsley",
    "HamiltonHelmsley",
    "KarakashevaHelmlsey",
}

# Mix of clear power gains + near-saturated published-only examples.
# Note: cDC2 has essentially no batch-adjusted ileum–colon DE under
# ``~ dataset_id + seg`` (≤1 gene), so it only appears for ``seg_only``.
CELLTYPES = [
    ("cDC2", "cDC2", "myeloid"),
    ("Homeostatic Macrophages", "Hom. mac", "myeloid"),  # near-saturated w/o prepub
    ("CD8 IEL", "CD8 IEL", "lymphoid"),
    ("Memory B", "Memory B", "lymphoid"),
    ("Goblet Cells", "Goblet", "epithelial"),
]

DESIGNS = {
    "dataset_seg": "~ dataset_id + seg",
    "seg_only": "~ seg",
}

SEG_A = "ileum"
SEG_B = "colon"
MIN_CELLS_PB = 10
MIN_SAMPLES = 8
FDR = 0.05
LFC_MIN = 0.5  # |log2FC| threshold for target genes
ALPHA = 0.05  # Wald α for power (unadjusted; standard in RNA-seq power calcs)
N_GRID = [8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120]


def _clean_ct(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()


def load_lineage(lineage: str):
    if OBJ is None:
        raise SystemExit("Set HGCA_OBJECTS or pass --objects.")
    path = OBJ / f"{lineage}.h5ad"
    LOG.info("loading %s", path)
    ad = sc.read_h5ad(path)
    ad.obs["ct"] = _clean_ct(ad.obs["hgca_celltype_v1"])
    ad.obs["seg"] = ad.obs["tissue_level_1"].astype(str)
    ad.obs["prov"] = np.where(
        ad.obs["dataset_id"].astype(str).isin(PREPUB_IDS),
        "Consortium contributed",
        "Published studies",
    )
    return ad


def build_pseudobulk(ad, celltype: str):
    mask = (ad.obs["ct"] == celltype) & ad.obs["seg"].isin([SEG_A, SEG_B])
    sub = ad[mask]
    if sub.n_obs == 0:
        return None, None
    X = sub.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    samples = sub.obs["sample_id"].astype(str).to_numpy()
    uniq, inv = np.unique(samples, return_inverse=True)
    ind = sparse.csr_matrix(
        (np.ones(len(inv)), (inv, np.arange(len(inv)))),
        shape=(len(uniq), X.shape[0]),
    )
    pb = np.asarray((ind @ X).todense(), dtype=float)
    cps = np.bincount(inv, minlength=len(uniq))
    meta = (
        sub.obs.groupby("sample_id", observed=True)
        .agg(
            seg=("seg", "first"),
            dataset_id=("dataset_id", "first"),
            prov=("prov", "first"),
        )
        .loc[uniq]
    )
    meta["dataset_id"] = meta["dataset_id"].astype(str)
    meta["seg"] = meta["seg"].astype(str)
    keep = cps >= MIN_CELLS_PB
    pb, meta = pb[keep], meta.iloc[keep].copy()
    if pb.shape[0] < 2 * MIN_SAMPLES:
        return None, None
    keep_g = (pb >= 10).sum(axis=0) >= 5
    pb = np.maximum(np.round(pb[:, keep_g]), 0).astype(np.int32)
    counts = pd.DataFrame(pb, index=meta.index.astype(str))
    counts.columns = ad.var_names[keep_g].astype(str)
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    meta = meta.loc[counts.index]
    return counts, meta


def spanning_datasets(meta: pd.DataFrame) -> list[str]:
    return [
        str(d)
        for d, sub in meta.groupby("dataset_id")
        if set(sub["seg"]) == {SEG_A, SEG_B}
    ]


def n_eff(meta: pd.DataFrame) -> float:
    """Harmonic mean of segment sample counts (balanced effective n)."""
    n_a = int((meta["seg"] == SEG_A).sum())
    n_b = int((meta["seg"] == SEG_B).sum())
    if n_a < 1 or n_b < 1:
        return float("nan")
    return 2.0 / (1.0 / n_a + 1.0 / n_b)


def run_deseq2(counts: pd.DataFrame, meta: pd.DataFrame, design: str) -> pd.DataFrame | None:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    md = meta.copy()
    md["dataset_id"] = md["dataset_id"].astype(str)
    md["seg"] = pd.Categorical(md["seg"].astype(str), categories=[SEG_B, SEG_A])
    c = counts.loc[:, counts.sum(axis=0) > 0]
    if c.shape[1] < 50 or c.shape[0] < 2 * MIN_SAMPLES:
        return None
    if "dataset_id" in design and not spanning_datasets(md):
        LOG.warning("no spanning dataset for design %s", design)
        return None
    # Drop unused dataset levels
    if "dataset_id" in design:
        md["dataset_id"] = md["dataset_id"].astype("category")
        md["dataset_id"] = md["dataset_id"].cat.remove_unused_categories()
        if md["dataset_id"].nunique() < 2:
            return None
    try:
        dds = DeseqDataSet(
            counts=c,
            metadata=md,
            design=design,
            refit_cooks=False,
            quiet=True,
            n_cpus=2,
        )
        dds.deseq2()
        stat = DeseqStats(dds, contrast=["seg", SEG_A, SEG_B], quiet=True)
        stat.summary()
        return stat.results_df
    except Exception as exc:
        LOG.warning("DESeq2 failed (%s): %s", design, exc)
        return None


def wald_power(beta: np.ndarray, se: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Two-sided Wald power for H0: β=0 given true effect β and SE."""
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    mu = np.abs(beta) / se
    # power = P(Z > z) + P(Z < -z) under N(mu, 1) for |effect| parameterization
    return stats.norm.sf(z - mu) + stats.norm.cdf(-z - mu)


def power_summary_at_n(
    beta: np.ndarray,
    se_full: np.ndarray,
    n_eff_full: float,
    n: float,
    target: float = 0.8,
) -> dict[str, float]:
    """Mean gene-level Wald power ± SE (and 95% CI of the mean).

    Per-gene power is a derived quantity from β̂ and SE(β̂). The band is the
    SE / Wald CI of the *mean power across the target gene set* — the usual
    way to show uncertainty on an average power curve.
    """
    se_n = se_full * np.sqrt(n_eff_full / n)
    p = wald_power(beta, se_n)
    p = p[np.isfinite(p)]
    k = len(p)
    if k == 0:
        return {
            "power_mean": float("nan"),
            "power_se": float("nan"),
            "power_ci_lo": float("nan"),
            "power_ci_hi": float("nan"),
            "frac_power_ge_0.8": float("nan"),
        }
    mean = float(np.mean(p))
    sd = float(np.std(p, ddof=1)) if k > 1 else 0.0
    se = float(sd / np.sqrt(k)) if k > 1 else 0.0
    z = stats.norm.ppf(0.975)
    # Ribbon for plots: mean ± s.d. across genes (SE-of-mean CI is too narrow
    # to see with hundreds of target genes). SE / Wald CI kept for tables.
    return {
        "power_mean": mean,
        "power_sd": sd,
        "power_se": se,
        "power_ci_lo": float(max(0.0, mean - z * se)),
        "power_ci_hi": float(min(1.0, mean + z * se)),
        "power_sd_lo": float(max(0.0, mean - sd)),
        "power_sd_hi": float(min(1.0, mean + sd)),
        "frac_power_ge_0.8": float(np.mean(p >= target)),
    }


def compute_celltype(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    ct_full: str,
    ct_short: str,
    design_key: str,
):
    design = DESIGNS[design_key]
    pub = meta["prov"].astype(str) == "Published studies"
    n_pub = int(
        min((meta.loc[pub, "seg"] == SEG_A).sum(), (meta.loc[pub, "seg"] == SEG_B).sum())
    )
    n_all = int(min((meta["seg"] == SEG_A).sum(), (meta["seg"] == SEG_B).sum()))
    ne_all = n_eff(meta)
    ne_pub = n_eff(meta.loc[pub]) if pub.any() else float("nan")
    LOG.info(
        "%s [%s]: n_pub=%d n_all=%d  n_eff_pub=%.1f n_eff_all=%.1f",
        ct_short,
        design_key,
        n_pub,
        n_all,
        ne_pub,
        ne_all,
    )
    if n_all < MIN_SAMPLES:
        return pd.DataFrame(), pd.DataFrame()

    LOG.info("%s [%s]: fitting full atlas", ct_short, design_key)
    res = run_deseq2(counts, meta, design)
    if res is None or "lfcSE" not in res.columns:
        LOG.warning("%s [%s]: no results", ct_short, design_key)
        return pd.DataFrame(), pd.DataFrame()

    ok = (
        res["padj"].fillna(1.0).lt(FDR)
        & res["log2FoldChange"].abs().ge(LFC_MIN)
        & res["lfcSE"].gt(0)
        & np.isfinite(res["log2FoldChange"])
        & np.isfinite(res["lfcSE"])
    )
    tgt = res.loc[ok]
    if len(tgt) < 10:
        LOG.warning("%s [%s]: only %d target genes — skip", ct_short, design_key, len(tgt))
        return pd.DataFrame(), pd.DataFrame()

    beta = tgt["log2FoldChange"].to_numpy(float)
    se = tgt["lfcSE"].to_numpy(float)
    LOG.info("%s [%s]: %d target DE genes (|LFC|≥%.2g, FDR<%.2g)", ct_short, design_key, len(tgt), LFC_MIN, FDR)

    # Persist target gene table
    out_genes = tgt[["log2FoldChange", "lfcSE", "padj", "baseMean"]].copy()
    out_genes.insert(0, "gene", out_genes.index.astype(str))
    out_genes.to_csv(
        DATA / f"deseq2_power_targets_{design_key}_{ct_short.replace(' ', '_')}.csv",
        index=False,
    )

    # Curve over balanced n (use n_eff_full as the SE reference scale)
    n_grid = sorted({n for n in N_GRID if MIN_SAMPLES <= n <= max(n_all, n_pub)} | {n_pub, n_all})
    curve_rows = []
    for n in n_grid:
        summ = power_summary_at_n(beta, se, ne_all, float(n))
        curve_rows.append(
            {
                "celltype": ct_full,
                "celltype_short": ct_short,
                "design": design_key,
                "design_formula": design,
                "n_per_segment": int(n),
                **summ,
                "n_target_genes": len(tgt),
                "n_eff_full": ne_all,
            }
        )
    curve = pd.DataFrame(curve_rows)

    marks = []
    for label, n_mark, ne_mark in [
        ("Published only", n_pub, ne_pub if np.isfinite(ne_pub) else float(n_pub)),
        ("Published + contributed", n_all, ne_all),
    ]:
        # Use balanced n_mark for x; power scaled from full-fit SE via n_eff
        # Operating point uses the cohort's own effective n when available
        n_for_power = float(ne_mark) if np.isfinite(ne_mark) else float(n_mark)
        summ = power_summary_at_n(beta, se, ne_all, n_for_power)
        marks.append(
            {
                "celltype": ct_full,
                "celltype_short": ct_short,
                "design": design_key,
                "design_formula": design,
                "point": label,
                "n_per_segment": int(n_mark),
                "n_eff": float(n_for_power),
                **summ,
                "n_target_genes": len(tgt),
            }
        )
        LOG.info(
            "  %s: n_bal=%d n_eff=%.1f → mean power=%.3f ± %.3f (95%% CI)  frac≥0.8=%.3f",
            label,
            n_mark,
            n_for_power,
            summ["power_mean"],
            summ["power_se"],
            summ["frac_power_ge_0.8"],
        )
    return curve, pd.DataFrame(marks)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--design",
        action="append",
        choices=sorted(DESIGNS),
        help="Design key to run (repeatable). Default: both.",
    )
    p.add_argument("--objects", type=Path, default=None)
    args = p.parse_args(argv)
    global OBJ
    if args.objects is not None:
        OBJ = args.objects
    designs = args.design or list(DESIGNS)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "deseq2_analytical_power.log"),
        ],
    )
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    loaded: dict = {}
    curves, marks = [], []
    for design_key in designs:
        for ct_full, ct_short, lineage in CELLTYPES:
            if lineage not in loaded:
                loaded[lineage] = load_lineage(lineage)
            counts, meta = build_pseudobulk(loaded[lineage], ct_full)
            if counts is None:
                LOG.warning("skip %s — no pseudobulk", ct_full)
                continue
            c, m = compute_celltype(counts, meta, ct_full, ct_short, design_key)
            if len(c):
                curves.append(c)
            if len(m):
                marks.append(m)

    if not marks:
        LOG.error("no analytical power results")
        return 1

    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    mark = pd.concat(marks, ignore_index=True)
    curve.to_csv(DATA / "de_power_deseq2_analytical.csv", index=False)
    mark.to_csv(DATA / "de_power_deseq2_analytical_markers.csv", index=False)
    LOG.info("wrote data/de_power_deseq2_analytical*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
