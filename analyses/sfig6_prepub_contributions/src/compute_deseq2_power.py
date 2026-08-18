#!/usr/bin/env python3
"""Pseudobulk DESeq2 power curves for S7 panel g (multiple cell-type variants).

For each variant, aggregates cells to sample × segment pseudobulks (HGCA v1),
downsamples balanced samples/arm, and runs pydeseq2 with:

    design = ~ dataset_id + seg

Operating points (open/filled circles) are taken from the same balanced
downsampling curve at n_published and n_all so they lie on the lines.

Variants emphasize cell types with high segment / tissue_level_1 ω² (fig3 /
supp composition omega tables), not an arbitrary macrophage set.

Requires: patpy env (pydeseq2).
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
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

LOG = logging.getLogger("deseq2_power")

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

# (full hgca_celltype_v1 name, short label, lineage h5ad stem)
# Feasible = enough ileum+colon pseudobulks + ≥1 spanning dataset.
VARIANTS: dict[str, list[tuple[str, str, str]]] = {
    # High segment-ω² epithelial programs with powered ileum+colon pseudobulks
    # (colonocyte/enterocyte subtypes are nearly segment-private, so excluded)
    "epithelial_omega": [
        ("Goblet Cells", "Goblet", "epithelial"),
        ("Transiently Amplifying Cells (TA)", "TA", "epithelial"),
    ],
    # Top lymphoid types by segment ω² with powered ileum/colon pseudobulks
    "lymphoid_t": [
        ("CD8 IEL", "CD8 IEL", "lymphoid"),
        ("CD8 Effector Memory", "CD8 TEM", "lymphoid"),
        ("CD4 Tfh", "CD4 Tfh", "lymphoid"),
    ],
    "lymphoid_b": [
        ("Memory B", "Memory B", "lymphoid"),
        ("Plasma IGA", "Plasma IgA", "lymphoid"),
        ("GC B Light Zone (GC B LZ)", "GC B LZ", "lymphoid"),
    ],
    # Myeloid without the M0/Hom mac pair; keeps DC/mast from the original story
    "myeloid_dc_mast": [
        ("cDC2", "cDC2", "myeloid"),
        ("Mast Cells", "Mast", "myeloid"),
        ("Classical Monocytes", "cMono", "myeloid"),
    ],
}

SEG_A = "ileum"
SEG_B = "colon"
MIN_CELLS_PB = 10
MIN_SAMPLES = 6
FDR = 0.05
N_SEEDS = 3
N_GRID = [6, 8, 10, 12, 15, 18, 20, 23, 25, 28, 30, 33, 37, 41, 49, 58, 64, 74, 83, 91]


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
    meta["n_cells"] = cps

    keep = cps >= MIN_CELLS_PB
    pb = pb[keep]
    meta = meta.iloc[keep].copy()
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
    out = []
    for d, sub in meta.groupby("dataset_id"):
        if set(sub["seg"]) == {SEG_A, SEG_B}:
            out.append(str(d))
    return out


def choose_balanced_indices(
    meta: pd.DataFrame, n_per: int, rng: np.random.Generator
) -> np.ndarray | None:
    ia = np.where(meta["seg"].to_numpy() == SEG_A)[0]
    ib = np.where(meta["seg"].to_numpy() == SEG_B)[0]
    if len(ia) < n_per or len(ib) < n_per:
        return None
    span = spanning_datasets(meta)
    if not span:
        return None

    forced: list[int] = []
    for seg, pool in [(SEG_A, ia), (SEG_B, ib)]:
        cand = [
            int(i)
            for i in pool
            if meta.iloc[int(i)]["dataset_id"] in span and int(i) not in forced
        ]
        if not cand:
            return None
        forced.append(int(rng.choice(cand)))

    n_a = sum(meta.iloc[i]["seg"] == SEG_A for i in forced)
    n_b = sum(meta.iloc[i]["seg"] == SEG_B for i in forced)
    need_a, need_b = n_per - n_a, n_per - n_b
    rest_a = [int(i) for i in ia if int(i) not in forced]
    rest_b = [int(i) for i in ib if int(i) not in forced]
    if need_a > len(rest_a) or need_b > len(rest_b):
        return None
    pick_a = list(rng.choice(rest_a, need_a, replace=False)) if need_a else []
    pick_b = list(rng.choice(rest_b, need_b, replace=False)) if need_b else []
    return np.array(forced + pick_a + pick_b, dtype=int)


def n_de_deseq2(counts: pd.DataFrame, meta: pd.DataFrame) -> float:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    md = meta.copy()
    md["dataset_id"] = md["dataset_id"].astype(str)
    md["seg"] = pd.Categorical(md["seg"].astype(str), categories=[SEG_B, SEG_A])
    c = counts.loc[:, counts.sum(axis=0) > 0]
    if c.shape[1] < 50 or c.shape[0] < 2 * MIN_SAMPLES:
        return np.nan
    if not spanning_datasets(md):
        return np.nan
    try:
        dds = DeseqDataSet(
            counts=c,
            metadata=md,
            design="~ dataset_id + seg",
            refit_cooks=False,
            quiet=True,
            n_cpus=2,
        )
        dds.deseq2()
        stat = DeseqStats(dds, contrast=["seg", SEG_A, SEG_B], quiet=True)
        stat.summary()
        res = stat.results_df
        return float((res["padj"] < FDR).fillna(False).sum())
    except Exception as exc:
        LOG.warning("DESeq2 failed: %s", exc)
        return np.nan


def compute_for_celltype(
    counts: pd.DataFrame, meta: pd.DataFrame, ct_full: str, ct_short: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pub = meta["prov"].astype(str) == "Published studies"
    n_pub = int(
        min((meta.loc[pub, "seg"] == SEG_A).sum(), (meta.loc[pub, "seg"] == SEG_B).sum())
    )
    n_all = int(min((meta["seg"] == SEG_A).sum(), (meta["seg"] == SEG_B).sum()))
    LOG.info("%s: balanced samples published=%d all=%d", ct_short, n_pub, n_all)

    if not spanning_datasets(meta):
        LOG.warning("%s: no spanning dataset — skip", ct_short)
        return pd.DataFrame(), pd.DataFrame()
    if n_all < MIN_SAMPLES:
        LOG.warning("%s: bal_all=%d < %d — skip", ct_short, n_all, MIN_SAMPLES)
        return pd.DataFrame(), pd.DataFrame()

    curve_rows = []
    raw = sorted(
        {n for n in N_GRID if MIN_SAMPLES <= n <= n_all} | {n_pub, n_all}
    )
    raw = [n for n in raw if n >= MIN_SAMPLES]
    # Cap grid size so large-n lymphoid types stay tractable (~10 DESeq2 n-points).
    if len(raw) > 10:
        keep_idx = np.linspace(0, len(raw) - 1, 10).round().astype(int)
        n_grid = sorted({raw[i] for i in keep_idx} | {n_pub, n_all})
        n_grid = [n for n in n_grid if n >= MIN_SAMPLES]
    else:
        n_grid = raw

    by_n: dict[int, float] = {}
    for n in n_grid:
        vals = []
        for seed in range(N_SEEDS):
            v = np.nan
            for attempt in range(8):
                idx = choose_balanced_indices(
                    meta, n, np.random.default_rng(seed * 1009 + attempt + 17 * n)
                )
                if idx is None:
                    continue
                v = n_de_deseq2(counts.iloc[idx], meta.iloc[idx])
                if np.isfinite(v):
                    break
            vals.append(v)
        vals_f = [x for x in vals if np.isfinite(x)]
        if not vals_f:
            LOG.info("  n=%d → no successful fits", n)
            continue
        mean_de = float(np.mean(vals_f))
        by_n[int(n)] = mean_de
        curve_rows.append(
            {
                "celltype": ct_full,
                "celltype_short": ct_short,
                "n_per_segment": int(n),
                "n_de_mean": mean_de,
                "n_de_sd": float(np.std(vals_f, ddof=1)) if len(vals_f) > 1 else 0.0,
                "n_reps": len(vals_f),
            }
        )
        LOG.info(
            "  n=%d → DE=%.1f ± %.1f (%d ok)",
            n,
            mean_de,
            np.std(vals_f) if len(vals_f) > 1 else 0,
            len(vals_f),
        )

    # Markers = curve values at operating n so points sit on the lines.
    mark_rows = []
    for label, n_mark in [
        ("Published only", n_pub),
        ("Published + contributed", n_all),
    ]:
        if n_mark not in by_n:
            continue
        mark_rows.append(
            {
                "celltype": ct_full,
                "celltype_short": ct_short,
                "point": label,
                "n_per_segment": int(n_mark),
                "n_de_mean": float(by_n[n_mark]),
            }
        )

    return pd.DataFrame(curve_rows), pd.DataFrame(mark_rows)


def run_variant(name: str, specs: list[tuple[str, str, str]]) -> None:
    LOG.info("######## variant %s ########", name)
    # Cache loaded lineages
    loaded: dict[str, object] = {}
    curves, marks = [], []

    for ct_full, ct_short, lineage in specs:
        if lineage not in loaded:
            loaded[lineage] = load_lineage(lineage)
        ad = loaded[lineage]
        LOG.info("=== %s (%s) ===", ct_full, lineage)
        counts, meta = build_pseudobulk(ad, ct_full)
        if counts is None:
            LOG.warning("skip %s — insufficient pseudobulks", ct_full)
            continue
        # Villus tip often has tiny colon n; allow if bal_all >= 5 by relaxing
        bal_all = int(min((meta["seg"] == SEG_A).sum(), (meta["seg"] == SEG_B).sum()))
        if bal_all < MIN_SAMPLES:
            LOG.warning(
                "skip %s — bal_all=%d (need ≥%d for stable DESeq2)",
                ct_short,
                bal_all,
                MIN_SAMPLES,
            )
            continue
        meta.to_csv(DATA / f"pseudobulk_meta_{name}_{ct_short.replace(' ', '_')}.csv")
        LOG.info(
            "pseudobulk %s: %d samples × %d genes; spanning=%s",
            ct_short,
            counts.shape[0],
            counts.shape[1],
            spanning_datasets(meta),
        )
        c, m = compute_for_celltype(counts, meta, ct_full, ct_short)
        if len(c):
            curves.append(c)
        if len(m):
            marks.append(m)

    for ad in loaded.values():
        del ad

    if not curves:
        LOG.error("variant %s produced no curves", name)
        return

    curve = pd.concat(curves, ignore_index=True)
    mark = pd.concat(marks, ignore_index=True) if marks else pd.DataFrame()
    curve.to_csv(DATA / f"de_power_deseq2_{name}.csv", index=False)
    mark.to_csv(DATA / f"de_power_deseq2_{name}_markers.csv", index=False)
    LOG.info("wrote data/de_power_deseq2_%s*.csv", name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS) + ["all"],
        help="Variant to compute (repeatable). Default: all",
    )
    p.add_argument("--objects", type=Path, default=None)
    args = p.parse_args(argv)
    global OBJ
    if args.objects is not None:
        OBJ = args.objects

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "deseq2_power.log"),
        ],
    )
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    wanted = args.variant or ["all"]
    names = list(VARIANTS) if "all" in wanted else wanted
    for name in names:
        run_variant(name, VARIANTS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
