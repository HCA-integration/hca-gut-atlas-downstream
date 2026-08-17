#!/usr/bin/env python
"""
Run LIANA (rank_aggregate, consensus resource) per gut tissue group on the HGCA
all-lineages v1 object, using `hgca_celltype_v1` as the cell label.

Outputs one CSV per tissue group (long-form LR table) plus combined CSVs:
  - per_tissue_level_1/HCA_<tissue>.csv         (one per tissue_level_1 group)
  - per_highres_tissue/HCA_<tissue>.csv         (one per highres_tissue_ontology group)
  - combined_lr_per_tissue_level_1.csv          (all rows + tissue_level_1 column)
  - combined_lr_per_highres_tissue.csv          (all rows + highres_tissue_ontology column)

The output CSVs feed scripts/ccc_centrality_gut_axis.R.

Notes
-----
- X is treated as raw counts; we log1p-normalize a copy per subset before LIANA.
- Each tissue subset is cell-type-stratified subsampled to MAX_CELLS_PER_TISSUE
  to keep runtimes tractable.
- Disease==normal cells only.
- Mesentery / accessory tissues are excluded from tissue_level_1 by default.

Env:
  LIANA_AD_PATH        AnnData .h5ad path (default integrated-objects/hgca_all_lineages_v1.h5ad)
  LIANA_OUTPUT_DIR     output dir (default github_vignette_output/LIANA)
  LIANA_GROUP_KEY      "hgca_celltype_v1" (default)
  LIANA_MAX_CELLS      cap per tissue subset (default 25000)
  LIANA_MIN_CELLS      skip groups smaller than this (default 1500)
  LIANA_METHOD         "rank_aggregate" (default; true LIANA ensemble) | "cellphonedb"
  LIANA_N_PERMS        LIANA n_perms (default 1000 for rank_aggregate)
  LIANA_EXPR_PROP      LIANA expr_prop (default 0.1)
  LIANA_TISSUE_LEVEL_1 "0"/"1" toggle (default 1)
  LIANA_HIGHRES        "0"/"1" toggle (default 1)
  LIANA_SKIP_EXISTING  "0"/"1" skip if CSV already exists (default 1)
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import liana as li


def env_str(k: str, d: str) -> str:
    v = os.environ.get(k, "")
    return v if v else d


def env_int(k: str, d: int) -> int:
    v = os.environ.get(k, "")
    return int(v) if v else d


def env_float(k: str, d: float) -> float:
    v = os.environ.get(k, "")
    return float(v) if v else d


def env_bool(k: str, d: bool) -> bool:
    v = os.environ.get(k, "")
    if not v:
        return d
    return v not in ("0", "false", "False", "FALSE", "no", "")


AD_PATH = env_str(
    "LIANA_AD_PATH",
    "/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/hgca_all_lineages_v1.h5ad",
)
OUT_DIR = Path(env_str(
    "LIANA_OUTPUT_DIR",
    "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA",
))
GROUP_KEY = env_str("LIANA_GROUP_KEY", "hgca_celltype_v1")
MAX_CELLS = env_int("LIANA_MAX_CELLS", 12000)
MIN_CELLS = env_int("LIANA_MIN_CELLS", 1500)
LIANA_METHOD = env_str("LIANA_METHOD", "rank_aggregate")
_default_perms = 0 if LIANA_METHOD == "cellphonedb" else 1000
N_PERMS = env_int("LIANA_N_PERMS", _default_perms)
N_JOBS = env_int("LIANA_N_JOBS", 4)
EXPR_PROP = env_float("LIANA_EXPR_PROP", 0.1)
DO_LEVEL1 = env_bool("LIANA_TISSUE_LEVEL_1", True)
DO_HIGHRES = env_bool("LIANA_HIGHRES", True)
SKIP_EXIST = env_bool("LIANA_SKIP_EXISTING", True)

EXCLUDE_LEVEL1 = {"mesentery", "accessory", "stomach"}
EXCLUDE_HIGHRES_PATTERNS = (
    "mesentery", "appendix", "renal medulla",
)


def _safe_filename(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_").replace("'", "").replace(",", "")


def stratified_subsample(obs: pd.DataFrame, group_key: str, n_max: int,
                         seed: int = 0) -> np.ndarray:
    """Cell-type stratified subsample of obs index, capped at n_max total."""
    if len(obs) <= n_max:
        return np.asarray(obs.index)
    rng = np.random.default_rng(seed)
    counts = obs[group_key].value_counts()
    n_groups = len(counts)
    quota = max(50, n_max // max(1, n_groups))
    keep_idx = []
    for grp, ix in obs.groupby(group_key, observed=True).groups.items():
        ix = np.asarray(ix)
        take = min(len(ix), quota)
        if take == 0:
            continue
        keep_idx.append(rng.choice(ix, size=take, replace=False))
    out = np.concatenate(keep_idx) if keep_idx else np.asarray(obs.index)
    if len(out) > n_max:
        out = rng.choice(out, size=n_max, replace=False)
    return out


def prepare_subset(a_full, mask: np.ndarray, group_key: str) -> ad.AnnData:
    sub = a_full[mask].to_memory().copy()
    if "gene_symbol" in sub.var.columns:
        sub.var_names = sub.var["gene_symbol"].astype(str).values
        sub.var_names_make_unique()
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    keep_genes = np.asarray((sub.X > 0).sum(axis=0)).ravel() >= 5
    sub = sub[:, keep_genes].copy()
    sub.obs[group_key] = sub.obs[group_key].astype(str)
    return sub


def run_liana_one(subset_label: str, sub_ad: ad.AnnData, out_csv: Path,
                  group_key: str) -> bool:
    n_groups = sub_ad.obs[group_key].nunique()
    if n_groups < 2:
        print(f"  [skip] {subset_label}: only {n_groups} cell type")
        return False
    print(f"  running LIANA[{LIANA_METHOD}, n_perms={N_PERMS}] on "
          f"{sub_ad.shape[0]} cells x {sub_ad.shape[1]} genes "
          f"({n_groups} cell types) ...", flush=True)
    t0 = time.time()
    method_fn = getattr(li.mt, LIANA_METHOD)
    call_kwargs = dict(
        groupby=group_key,
        resource_name="consensus",
        use_raw=False,
        expr_prop=EXPR_PROP,
        n_perms=N_PERMS,
        verbose=True,
        inplace=False,
    )
    # rank_aggregate / several methods accept n_jobs
    if LIANA_METHOD == "rank_aggregate":
        call_kwargs["n_jobs"] = N_JOBS
        call_kwargs["aggregate_method"] = env_str("LIANA_AGGREGATE_METHOD", "rra")
    df = method_fn(sub_ad, **call_kwargs)
    df = df.copy()
    df.insert(0, "tissue_subset", subset_label)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    dt = time.time() - t0
    print(f"  -> wrote {out_csv}  ({len(df):,} rows, {dt/60:.1f} min)",
          flush=True)
    return True


def run_per_group(a_full, group_col: str, out_subdir: str, label: str,
                  exclude_set=None, exclude_patterns=()):
    out_dir = OUT_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = a_full.obs
    obs_normal = obs[obs.get("disease", "normal").astype(str) == "normal"]
    if not len(obs_normal):
        print(f"[{label}] no 'normal' rows; skipping")
        return []
    obs_normal = obs_normal.copy()
    obs_normal["_subset"] = obs_normal[group_col].astype(str)
    counts = obs_normal["_subset"].value_counts()
    cands = []
    for k, n in counts.items():
        if k in (None, "", "nan", "NaN"):
            continue
        if exclude_set and k.lower() in exclude_set:
            continue
        if any(p in k.lower() for p in exclude_patterns):
            continue
        if n < MIN_CELLS:
            continue
        cands.append(k)
    print(f"[{label}] {len(cands)} groups to run: {cands}", flush=True)
    out_csvs = []
    for ks in cands:
        out_csv = out_dir / f"HCA_{_safe_filename(ks)}.csv"
        if SKIP_EXIST and out_csv.exists():
            print(f"  [skip-exist] {out_csv.name}")
            out_csvs.append(out_csv)
            continue
        gix = obs_normal.index[obs_normal["_subset"] == ks]
        keep = stratified_subsample(obs_normal.loc[gix], GROUP_KEY,
                                    MAX_CELLS, seed=hash(ks) & 0xFFFF)
        mask = a_full.obs.index.isin(keep)
        try:
            sub = prepare_subset(a_full, mask, GROUP_KEY)
            if run_liana_one(ks, sub, out_csv, GROUP_KEY):
                out_csvs.append(out_csv)
        except Exception:
            print(f"  [error] {ks}:")
            traceback.print_exc()
        finally:
            try:
                del sub
            except Exception:
                pass
            gc.collect()
    return out_csvs


def combine_outputs(out_csvs: list[Path], group_col: str, out_csv: Path):
    if not out_csvs:
        print(f"  [combine] nothing to combine for {out_csv.name}")
        return
    rows = []
    for f in out_csvs:
        try:
            df = pd.read_csv(f)
        except Exception:
            print(f"  [combine] skip unreadable {f}")
            continue
        df = df.rename(columns={"tissue_subset": group_col})
        rows.append(df)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  [combine] wrote {out_csv}  ({len(df):,} rows from {len(rows)} files)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"AD_PATH:      {AD_PATH}")
    print(f"OUT_DIR:      {OUT_DIR}")
    print(f"GROUP_KEY:    {GROUP_KEY}")
    print(f"MAX_CELLS:    {MAX_CELLS}")
    print(f"MIN_CELLS:    {MIN_CELLS}")
    print(f"METHOD:       {LIANA_METHOD}")
    print(f"N_PERMS:      {N_PERMS}")
    print(f"EXPR_PROP:    {EXPR_PROP}")
    print(f"DO_LEVEL1:    {DO_LEVEL1}  | DO_HIGHRES: {DO_HIGHRES}")
    print(f"SKIP_EXIST:   {SKIP_EXIST}")
    sys.stdout.flush()

    print("Loading AnnData (backed='r')...")
    t0 = time.time()
    a = ad.read_h5ad(AD_PATH, backed="r")
    print(f"  loaded shape={a.shape}  in {time.time()-t0:.1f}s")

    if DO_LEVEL1:
        print("\n=== tissue_level_1 ===")
        out_csvs = run_per_group(
            a, "tissue_level_1",
            out_subdir="per_tissue_level_1",
            label="tissue_level_1",
            exclude_set=EXCLUDE_LEVEL1,
        )
        combine_outputs(out_csvs, "tissue_level_1",
                        OUT_DIR / "combined_lr_per_tissue_level_1.csv")

    if DO_HIGHRES:
        print("\n=== highres_tissue_ontology ===")
        out_csvs = run_per_group(
            a, "highres_tissue_ontology",
            out_subdir="per_highres_tissue",
            label="highres_tissue_ontology",
            exclude_patterns=EXCLUDE_HIGHRES_PATTERNS,
        )
        combine_outputs(out_csvs, "highres_tissue_ontology",
                        OUT_DIR / "combined_lr_per_highres_tissue.csv")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
