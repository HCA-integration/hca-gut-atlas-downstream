#!/usr/bin/env python3
"""Classic recovery-power curves for S7 (pseudobulk DESeq2).

Why not "DE gene count vs n"?
  With ~dataset_id + seg and small sample n, the discovery count is dominated by
  which batches land in a draw — hence jumpy lines and huge CIs.

This script instead:
  1) Fits DESeq2 on the full available sample set → reference DE gene set
  2) At each balanced n, asks: what fraction of those genes are recovered?
     power = |DE(n) ∩ reference| / |reference|

That is a standard power / sensitivity curve (y ∈ [0, 1]), usually smooth and
monotone, and the published-only vs +contributed operating points sit on it.

Design: ~ dataset_id + seg  (pydeseq2). Requires patpy.
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

LOG = logging.getLogger("deseq2_recovery")

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

# High segment-ω² types with enough ileum+colon pseudobulks + spanning datasets
# and a non-trivial full-set DE call under ~ dataset_id + seg.
CELLTYPES = [
    ("Goblet Cells", "Goblet", "epithelial"),
    ("Transiently Amplifying Cells (TA)", "TA", "epithelial"),
    ("CD8 IEL", "CD8 IEL", "lymphoid"),
    ("Memory B", "Memory B", "lymphoid"),
    ("Plasma IGA", "Plasma IgA", "lymphoid"),
]

SEG_A = "ileum"
SEG_B = "colon"
MIN_CELLS_PB = 10
MIN_SAMPLES = 8
FDR = 0.05
MIN_REF_DE = 20  # skip types with almost no batch-adjusted DE
N_SEEDS = 4
# Light balanced grid (optional curve). Markers use actual sample sets, not this.
N_GRID = [8, 12, 16, 20, 30, 40, 50, 60, 80]


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


def run_deseq2(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame | None:
    """Return results_df or None on failure."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    md = meta.copy()
    md["dataset_id"] = md["dataset_id"].astype(str)
    md["seg"] = pd.Categorical(md["seg"].astype(str), categories=[SEG_B, SEG_A])
    c = counts.loc[:, counts.sum(axis=0) > 0]
    if c.shape[1] < 50 or c.shape[0] < 2 * MIN_SAMPLES:
        return None
    if not spanning_datasets(md):
        return None
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
        return stat.results_df
    except Exception as exc:
        LOG.warning("DESeq2 failed: %s", exc)
        return None


def de_genes(res: pd.DataFrame, fdr: float = FDR) -> set[str]:
    hit = res["padj"].fillna(1.0) < fdr
    return set(res.index[hit].astype(str))


def compute_celltype(counts, meta, ct_full, ct_short):
    pub_mask = meta["prov"].astype(str) == "Published studies"
    n_pub = int(
        min(
            (meta.loc[pub_mask, "seg"] == SEG_A).sum(),
            (meta.loc[pub_mask, "seg"] == SEG_B).sum(),
        )
    )
    n_all = int(min((meta["seg"] == SEG_A).sum(), (meta["seg"] == SEG_B).sum()))
    n_pub_total = int(pub_mask.sum())
    n_all_total = int(len(meta))
    LOG.info(
        "%s: balanced n_pub=%d n_all=%d | actual samples pub=%d all=%d",
        ct_short,
        n_pub,
        n_all,
        n_pub_total,
        n_all_total,
    )
    if n_all < MIN_SAMPLES or not spanning_datasets(meta):
        LOG.warning("%s: not estimable — skip", ct_short)
        return pd.DataFrame(), pd.DataFrame(), set()

    # Reference = full available sample set (actual atlas call)
    LOG.info("%s: fitting full-set reference DE", ct_short)
    res_full = run_deseq2(counts, meta)
    if res_full is None:
        return pd.DataFrame(), pd.DataFrame(), set()
    ref = de_genes(res_full)
    LOG.info("%s: reference DE genes = %d", ct_short, len(ref))
    if len(ref) < MIN_REF_DE:
        LOG.warning("%s: only %d ref DE (< %d) — skip", ct_short, len(ref), MIN_REF_DE)
        return pd.DataFrame(), pd.DataFrame(), ref

    pd.Series(sorted(ref), name="gene").to_csv(
        DATA / f"deseq2_ref_de_{ct_short.replace(' ', '_')}.csv", index=False
    )

    # --- Operating-point markers: ACTUAL sample sets (preferred panel-g view) ---
    # Full set recovers 100% by definition. Published-only = sensitivity of the
    # pre-contribution atlas to that same reference gene set.
    marks = [
        {
            "celltype": ct_full,
            "celltype_short": ct_short,
            "point": "Published + contributed",
            "n_per_segment": int(n_all),
            "n_samples": n_all_total,
            "power_mean": 1.0,
            "n_de": len(ref),
            "n_ref_de": len(ref),
        }
    ]
    power_pub = np.nan
    n_de_pub = 0
    if n_pub >= MIN_SAMPLES and spanning_datasets(meta.loc[pub_mask]):
        LOG.info("%s: fitting published-only DE (actual samples)", ct_short)
        res_pub = run_deseq2(counts.loc[pub_mask], meta.loc[pub_mask])
        if res_pub is not None:
            hit_pub = de_genes(res_pub)
            power_pub = len(hit_pub & ref) / len(ref)
            n_de_pub = len(hit_pub)
            LOG.info(
                "%s: published-only recovery = %.2f (%d / %d ref DE)",
                ct_short,
                power_pub,
                len(hit_pub & ref),
                len(ref),
            )
    if np.isfinite(power_pub):
        marks.append(
            {
                "celltype": ct_full,
                "celltype_short": ct_short,
                "point": "Published only",
                "n_per_segment": int(n_pub),
                "n_samples": n_pub_total,
                "power_mean": float(power_pub),
                "n_de": int(n_de_pub),
                "n_ref_de": len(ref),
            }
        )
    else:
        LOG.warning("%s: published-only DESeq2 not estimable", ct_short)

    # --- Optional balanced curve (sample-size shape; can be noisy) ---
    raw = sorted(n for n in N_GRID if MIN_SAMPLES <= n <= n_all)
    if len(raw) > 7:
        idx = np.linspace(0, len(raw) - 1, 7).round().astype(int)
        n_grid = sorted({raw[i] for i in idx})
    else:
        n_grid = raw

    curve_rows = []
    for n in n_grid:
        powers = []
        for seed in range(N_SEEDS):
            pwr = np.nan
            for attempt in range(6):
                idx = choose_balanced_indices(
                    meta, int(n), np.random.default_rng(seed * 1009 + attempt + 13 * n)
                )
                if idx is None:
                    continue
                res = run_deseq2(counts.iloc[idx], meta.iloc[idx])
                if res is None:
                    continue
                hit = de_genes(res)
                pwr = len(hit & ref) / len(ref)
                break
            powers.append(pwr)
        ok = [p for p in powers if np.isfinite(p)]
        if not ok:
            LOG.info("  balanced n=%d → no fits", n)
            continue
        mean_p = float(np.mean(ok))
        curve_rows.append(
            {
                "celltype": ct_full,
                "celltype_short": ct_short,
                "n_per_segment": int(n),
                "power_mean": mean_p,
                "power_sd": float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0,
                "n_ref_de": len(ref),
                "n_reps": len(ok),
            }
        )
        LOG.info(
            "  balanced n=%d → power=%.2f ± %.2f (%d ok)",
            n,
            mean_p,
            np.std(ok) if len(ok) > 1 else 0,
            len(ok),
        )

    return pd.DataFrame(curve_rows), pd.DataFrame(marks), ref


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--markers-only",
        action="store_true",
        help="Only compute actual-set operating points (skip balanced curve; fast)",
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
            logging.FileHandler(ROOT / "logs" / "deseq2_recovery.log"),
        ],
    )
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # Monkey-patch grid empty when markers-only
    global N_GRID
    if args.markers_only:
        N_GRID = []

    loaded: dict = {}
    curves, marks = [], []
    for ct_full, ct_short, lineage in CELLTYPES:
        if lineage not in loaded:
            loaded[lineage] = load_lineage(lineage)
        counts, meta = build_pseudobulk(loaded[lineage], ct_full)
        if counts is None:
            LOG.warning("skip %s — no pseudobulk", ct_full)
            continue
        c, m, _ = compute_celltype(counts, meta, ct_full, ct_short)
        if len(c):
            curves.append(c)
        if len(m):
            marks.append(m)

    if not marks:
        LOG.error("no recovery markers produced")
        return 1

    mark = pd.concat(marks, ignore_index=True)
    mark.to_csv(DATA / "de_power_deseq2_recovery_markers.csv", index=False)
    if curves:
        curve = pd.concat(curves, ignore_index=True)
        curve.to_csv(DATA / "de_power_deseq2_recovery.csv", index=False)
    elif args.markers_only:
        # Empty curve file so render still prefers the slope chart
        pd.DataFrame(
            columns=[
                "celltype",
                "celltype_short",
                "n_per_segment",
                "power_mean",
                "power_sd",
                "n_ref_de",
                "n_reps",
            ]
        ).to_csv(DATA / "de_power_deseq2_recovery.csv", index=False)
    LOG.info("wrote data/de_power_deseq2_recovery*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
