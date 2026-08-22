"""
Run [mapQC](https://github.com/theislab/mapqc) on a joint SCANVI latent space after reference vs query (train vs test) mapping.

Filtering defaults follow the mapQC detailed notebook: lenient ``k_max``, ``min_n_cells``, and
``exclude_same_study`` when the reference is atlas-like. Optionally retries until neighborhood
pass rate reaches at least the SCANVI test accuracy (benchmark-aligned target).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)

try:
    import mapqc
    from mapqc import evaluate as mapqc_evaluate
    from mapqc import run_mapqc
except ImportError:
    mapqc = None
    run_mapqc = None  # type: ignore
    mapqc_evaluate = None  # type: ignore


def mapqc_available() -> bool:
    return run_mapqc is not None


def _nhood_pass_rate(adata: ad.AnnData) -> Tuple[int, int, float]:
    """Return (n_pass, n_total, fraction) for neighborhood filtering."""
    if "mapqc_params" not in adata.uns or "mapqc_nhood_filtering" not in adata.obs.columns:
        return 0, 0, 0.0
    n_total = int(adata.uns["mapqc_params"].get("n_nhoods", 0))
    n_pass = int((adata.obs["mapqc_nhood_filtering"] == "pass").sum())
    frac = (n_pass / n_total) if n_total else 0.0
    return n_pass, n_total, frac


def _build_joint_adata(
    model,
    at: ad.AnnData,
    aq: ad.AnnData,
    emb_key: str,
) -> ad.AnnData:
    adata_full = ad.concat([at, aq], merge="same")
    latent_train = model.get_latent_representation()
    latent_test = model.get_latent_representation(aq)
    adata_full.obsm["X_scanvi"] = np.vstack([latent_train, latent_test])
    if emb_key != "X_scanvi":
        adata_full.obsm[emb_key] = adata_full.obsm["X_scanvi"]
    return adata_full


def run_mapqc_after_scanvi(
    model,
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    config: Dict[str, Any],
    output_dir: Path,
    *,
    label_key: str,
    scanvi_accuracy: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run mapQC with optional retry until neighborhood pass fraction ≥ target (default: SCANVI accuracy).

    Config ``mapqc``:
      - ``match_scanvi_accuracy`` (default True): target pass rate = ``scanvi_accuracy``.
      - ``target_nhood_pass_frac``: override target in (0, 1].
      - ``max_mapqc_attempts`` (default 8): retry with looser filtering.
    """
    mq = dict(config.get("mapqc") or {})
    if not mq.get("enabled", True):
        logger.info("mapqc.enabled is false — skipping mapQC")
        return None
    if not mapqc_available():
        logger.warning(
            "mapqc is not installed. pip install mapqc "
            "(https://github.com/theislab/mapqc) — skipping mapQC."
        )
        return None

    out = Path(output_dir) / "mapqc"
    out.mkdir(parents=True, exist_ok=True)

    at = adata_train.copy()
    aq = adata_test.copy()
    at.obs["mapqc_refq"] = "reference"
    aq.obs["mapqc_refq"] = "query"
    batch_key = config.get("batch_key") or "sample_id"
    if batch_key not in at.obs.columns or batch_key not in aq.obs.columns:
        raise KeyError(
            f"mapQC needs batch_key={batch_key!r} on train and test obs. Set batch_key in config."
        )
    at.obs["mapqc_sample"] = at.obs[batch_key].astype(str) + "__ref"
    aq.obs["mapqc_sample"] = aq.obs[batch_key].astype(str) + "__query"

    study_key_cfg = mq.get("study_key")
    if study_key_cfg is None:
        study_key_cfg = "dataset_id" if "dataset_id" in at.obs.columns else None

    if study_key_cfg and study_key_cfg in at.obs.columns:
        at.obs["mapqc_study"] = at.obs[study_key_cfg].astype(str)
        aq.obs["mapqc_study"] = aq.obs[study_key_cfg].astype(str)
        use_study = True
    else:
        at.obs["mapqc_study"] = "study_placeholder"
        aq.obs["mapqc_study"] = "study_placeholder"
        use_study = False

    at.obs["mapqc_cc"] = "reference"
    aq.obs["mapqc_cc"] = mq.get("query_control_label", "query_holdout")
    q_label = mq.get("query_control_label", "query_holdout")

    emb_key = mq.get("adata_emb_loc", "X_scanvi")
    n_query = aq.n_obs

    target = mq.get("target_nhood_pass_frac")
    if target is None and mq.get("match_scanvi_accuracy", True) and scanvi_accuracy is not None:
        target = float(scanvi_accuracy)
    if target is None:
        target = 0.5
    target = max(0.01, min(1.0, float(target)))

    max_attempts = int(mq.get("max_mapqc_attempts", 8))

    # Base params (YAML); notebook: k_max ~5–10× k_min, lenient min_n_cells, exclude_same_study false for small refs
    base = {
        "n_nhoods": int(mq.get("n_nhoods", min(200, max(30, n_query - 1)))),
        "k_min": int(mq.get("k_min", 20)),
        "k_max": int(mq.get("k_max", 2500)),
        "min_n_cells": int(mq.get("min_n_cells", 5)),
        "min_n_samples_r": int(mq.get("min_n_samples_r", 3)),
        "exclude_same_study": bool(mq.get("exclude_same_study", False)),
        "seed": mq.get("seed", 42),
    }

    n_nhoods = base["n_nhoods"]
    if n_nhoods >= n_query:
        n_nhoods = max(5, n_query // 5)
        base["n_nhoods"] = n_nhoods
        logger.warning("Adjusted n_nhoods to %s (< n_query=%s)", n_nhoods, n_query)

    attempt_log: List[Dict[str, Any]] = []
    adata_full: Optional[ad.AnnData] = None
    last_frac = 0.0

    for attempt in range(max_attempts):
        p = copy.deepcopy(base)
        # Progressive lenience (mapqc_detailed.ipynb: raise k_max first)
        if attempt > 0:
            p["k_max"] = min(int(p["k_max"] * 1.6), max(500, (at.n_obs + aq.n_obs) - 2))
            p["k_min"] = max(8, int(p["k_min"] * 0.9))
            p["min_n_cells"] = max(5, p["min_n_cells"] - 1)
            p["exclude_same_study"] = False

        adata_full = _build_joint_adata(model, at, aq, emb_key)

        exclude_same = p["exclude_same_study"] and use_study
        if not use_study:
            exclude_same = False

        run_mapqc(
            adata_full,
            adata_emb_loc=emb_key,
            ref_q_key="mapqc_refq",
            q_cat="query",
            r_cat="reference",
            sample_key="mapqc_sample",
            n_nhoods=p["n_nhoods"],
            k_min=p["k_min"],
            k_max=p["k_max"],
            min_n_cells=p["min_n_cells"],
            min_n_samples_r=p["min_n_samples_r"],
            study_key="mapqc_study" if exclude_same else None,
            exclude_same_study=exclude_same,
            grouping_key=None,
            distance_metric=mq.get("distance_metric", "energy_distance"),
            seed=int(p["seed"]) if p["seed"] is not None else None,
            overwrite=True,
            return_nhood_info_df=False,
            verbose=attempt == 0,
        )

        n_pass, n_tot, frac = _nhood_pass_rate(adata_full)
        last_frac = frac
        attempt_log.append(
            {
                "attempt": attempt + 1,
                "params": p,
                "n_nhoods_pass": n_pass,
                "n_nhoods_total": n_tot,
                "frac_nhoods_pass": round(frac, 4),
                "target": target,
            }
        )
        logger.info(
            "mapQC attempt %s: %.1f%% neighborhoods passed (%s/%s), target ≥ %.1f%%",
            attempt + 1,
            100 * frac,
            n_pass,
            n_tot,
            100 * target,
        )
        if frac >= target:
            break

    with open(out / "mapqc_attempts.json", "w") as f:
        json.dump(
            {
                "target_nhood_pass_frac": target,
                "scanvi_accuracy_used": scanvi_accuracy,
                "attempts": attempt_log,
            },
            f,
            indent=2,
            default=str,
        )

    assert adata_full is not None
    q_mask = adata_full.obs["mapqc_refq"] == "query"

    stats: Dict[str, Any] = {}
    try:
        stats = mapqc_evaluate(
            adata_full,
            case_control_key="mapqc_cc",
            case_cats=[],
            control_cats=[q_label],
        )
    except Exception as e:
        logger.warning("mapqc.evaluate failed (scores in obs): %s", e)
        stats = {"evaluate_error": str(e)}

    if "case_control" not in adata_full.obs.columns:
        adata_full.obs["case_control"] = "Reference"
        adata_full.obs.loc[q_mask, "case_control"] = f"Control ({q_label})"

    with open(out / "mapqc_evaluate_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    cols = [
        "mapqc_score",
        "mapqc_filtering",
        "mapqc_cc",
        "mapqc_refq",
        label_key,
    ]
    cols = [c for c in cols if c in adata_full.obs.columns]
    adata_full.obs.loc[q_mask, cols].to_csv(out / "mapqc_scores_query_cells.csv")

    if "mapqc_nhood_number" in adata_full.obs.columns:
        nh = adata_full.obs["mapqc_nhood_number"].notna()
        adata_full.obs.loc[nh, [c for c in adata_full.obs.columns if c.startswith("mapqc_")]].to_csv(
            out / "mapqc_nhood_info.csv"
        )

    nn = min(int(mq.get("n_neighbors_umap", 15)), max(2, adata_full.n_obs - 1))
    sc.pp.neighbors(adata_full, use_rep=emb_key, n_neighbors=nn)
    sc.tl.umap(adata_full)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    import matplotlib.pyplot as plt

    try:
        fig = mapqc.pl.umap.mapqc_scores(adata_full, return_fig=True, vmin=-4, vmax=4)
        fig.savefig(fig_dir / "mapqc_scores_umap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning("mapqc UMAP score plot failed: %s", e)

    try:
        fig2 = mapqc.pl.umap.neighborhood_filtering(adata_full, return_fig=True)
        fig2.savefig(fig_dir / "mapqc_neighborhood_filtering_umap.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        logger.warning("mapqc neighborhood filtering plot failed: %s", e)

    try:
        scores = adata_full.obs.loc[q_mask, "mapqc_score"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scores.dropna(), bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("mapQC score")
        ax.set_ylabel("Query cells")
        ax.set_title("mapQC query distribution (SCANVI latent)")
        fig.savefig(fig_dir / "mapqc_score_histogram_query.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning("mapqc histogram failed: %s", e)

    return {
        "stats": stats,
        "output_dir": str(out),
        "frac_nhoods_pass": last_frac,
        "target_nhood_pass_frac": target,
    }
