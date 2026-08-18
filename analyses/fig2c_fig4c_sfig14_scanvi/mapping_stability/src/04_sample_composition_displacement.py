#!/usr/bin/env python3
"""Aitchison displacement of sample x leaf compositions across jackknives.

Fixed CLR basis from each atlas's full/seed0 composition.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paths import FIGURES, PREDICTIONS, TABLES  # noqa: E402

LOGGER = logging.getLogger("sample_displacement")
PSEUDO = 0.5


def _clr(mat: np.ndarray) -> np.ndarray:
    # mat: samples x labels, counts or fractions
    x = mat + PSEUDO
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def _compositions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pred_path, columns=["sample_id", "leaf_prediction"])
    tab = pd.crosstab(df["sample_id"], df["leaf_prediction"])
    return tab


def atlas_displacement(atlas: str) -> pd.DataFrame:
    base = PREDICTIONS / atlas / "stroma"
    full0 = base / "full" / "seed0" / "predictions.parquet"
    ref = _compositions(full0)
    labels = list(ref.columns)
    samples = list(ref.index)
    ref_mat = ref.reindex(index=samples, columns=labels, fill_value=0).to_numpy(dtype=float)
    ref_clr = _clr(ref_mat)

    rows = []
    for d in sorted(base.glob("omit_*/seed*")):
        tab = _compositions(d / "predictions.parquet")
        mat = tab.reindex(index=samples, columns=labels, fill_value=0).to_numpy(dtype=float)
        clr = _clr(mat)
        # Aitchison distance per sample
        dist = np.sqrt(((clr - ref_clr) ** 2).sum(axis=1))
        omit = d.parent.name.replace("omit_", "")
        seed = int(d.name.replace("seed", ""))
        for sid, di in zip(samples, dist):
            rows.append(
                {
                    "atlas": atlas,
                    "sample_id": sid,
                    "omitted_study": omit,
                    "model_seed": seed,
                    "aitchison_to_full_seed0": float(di),
                }
            )
    out = pd.DataFrame(rows)
    # attach disease from full predictions
    meta = pd.read_parquet(full0, columns=["sample_id", "Disease", "Patient"]).drop_duplicates("sample_id")
    out = out.merge(meta, on="sample_id", how="left")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    frames = [atlas_displacement(a) for a in ("HGCA", "PanGI")]
    all_d = pd.concat(frames, ignore_index=True)
    all_d.to_parquet(TABLES / "sample_aitchison_displacement.parquet", index=False)

    summary = (
        all_d.groupby(["atlas", "omitted_study"])
        .agg(
            n=("sample_id", "count"),
            median_dist=("aitchison_to_full_seed0", "median"),
            mean_dist=("aitchison_to_full_seed0", "mean"),
        )
        .reset_index()
        .sort_values(["atlas", "median_dist"], ascending=[True, False])
    )
    summary.to_csv(TABLES / "sample_aitchison_displacement_summary.csv", index=False)
    LOGGER.info("\n%s", summary.to_string(index=False))

    overall = all_d.groupby("atlas")["aitchison_to_full_seed0"].agg(["median", "mean"]).reset_index()
    overall.to_csv(TABLES / "sample_aitchison_displacement_overall.csv", index=False)
    LOGGER.info("Overall:\n%s", overall.to_string(index=False))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    data = [all_d.loc[all_d.atlas == a, "aitchison_to_full_seed0"].values for a in ("HGCA", "PanGI")]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(["#0072B2", "#009E73"][i])
        b.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["HGCA", "PanGI"])
    ax.set_ylabel("Aitchison distance to full/seed0 composition")
    ax.set_title("How much do the same samples move when one shared study is removed?")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"sample_reference_clouds_distance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    (TABLES / "sample_displacement_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "pseudocount": PSEUDO,
                "reference": "full/seed0 CLR basis per atlas",
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
