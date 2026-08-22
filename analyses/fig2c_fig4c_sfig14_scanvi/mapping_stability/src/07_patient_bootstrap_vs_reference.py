#!/usr/bin/env python3
"""Compare reference-jackknife Aitchison displacement vs patient bootstrap.

Reference fixed at full/seed0 composition. Bootstrap TAURUS patients with
replacement (1000 replicates). Same CLR/Aitchison representation as
04_sample_composition_displacement.py.
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
from paths import PREDICTIONS, TABLES  # noqa: E402

LOGGER = logging.getLogger("patient_bootstrap")
PSEUDO = 0.5
N_BOOT = 1000


def _clr(mat: np.ndarray) -> np.ndarray:
    x = mat + PSEUDO
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def _compositions_from_pred(pred: pd.DataFrame, labels: list[str], samples: list[str]) -> np.ndarray:
    tab = pd.crosstab(pred["sample_id"], pred["leaf_prediction"])
    return tab.reindex(index=samples, columns=labels, fill_value=0).to_numpy(dtype=float)


def atlas_analysis(atlas: str, rng: np.random.Generator) -> dict:
    base = PREDICTIONS / atlas / "stroma"
    full0 = pd.read_parquet(base / "full" / "seed0" / "predictions.parquet")
    meta = full0[["sample_id", "Patient", "Disease"]].drop_duplicates("sample_id")
    labels = sorted(full0["leaf_prediction"].astype(str).unique())
    samples = meta["sample_id"].astype(str).tolist()
    sample_to_patient = dict(zip(meta.sample_id.astype(str), meta.Patient.astype(str)))

    ref_mat = _compositions_from_pred(full0, labels, samples)
    ref_clr = _clr(ref_mat)

    # Jackknife distribution: median Aitchison across samples per realization
    jack_medians = []
    jack_all = []
    for d in sorted(base.glob("omit_*/seed*")):
        pred = pd.read_parquet(d / "predictions.parquet")
        mat = _compositions_from_pred(pred, labels, samples)
        clr = _clr(mat)
        dist = np.sqrt(((clr - ref_clr) ** 2).sum(axis=1))
        jack_medians.append(float(np.median(dist)))
        jack_all.extend(dist.tolist())

    # Patient bootstrap: resample patients, rebuild cohort composition using
    # the FIXED full/seed0 cell predictions (reference held fixed).
    # For each bootstrap replicate, compute sample compositions from the
    # resampled patient set's samples, then Aitchison distance of each
    # retained sample's composition to the original full-cohort sample
    # composition in the same CLR basis.
    #
    # Comparable scalar: median over samples of ||clr_boot(sample) - clr_ref(sample)||
    # where clr_boot uses the same cells for that sample (sample composition is
    # invariant to which other patients are present). That would make bootstrap
    # displacement ~0 for within-sample composition.
    #
    # Correct interpretation for cohort-level composition: compare the
    # *cohort mean CLR composition* under patient resampling vs the observed
    # cohort mean - OR compare pairwise sample geometry.
    #
    # Spec: "same composition representation" and "biological sampling
    # uncertainty on a directly comparable scale" to sample Aitchison
    # displacement under reference jackknife.
    #
    # We use: for each bootstrap of patients, take all samples belonging to
    # drawn patients (with multiplicity if patient drawn multiple times -
    # duplicate sample rows), recompute the *cohort-level* label composition
    # (pooled cells), CLR-transform that single composition vector, and
    # measure Aitchison distance to the full-cohort CLR composition.
    # Additionally, for sample-level comparability: within each bootstrap,
    # for each original sample still present, the sample composition is
    # unchanged; therefore sample-level bootstrap displacement is zero.
    # The meaningful patient-sampling uncertainty for cohort geometry is the
    # cohort-composition distance. We also compute a sample-level analogue by
    # measuring how much each sample's *neighborhood* changes - deferred.
    #
    # Primary comparable scalar pair:
    #   REF: median_sample Aitchison(omit_ref, full_ref)  [existing]
    #   BIO: Aitchison(cohort_comp under patient bootstrap, full cohort_comp)
    #
    # Also report mean sample-level jackknife median for ratio.

    # Cohort composition (pooled cells)
    def cohort_clr(pred: pd.DataFrame) -> np.ndarray:
        counts = pred["leaf_prediction"].astype(str).value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
        # single-row CLR
        return _clr(counts.reshape(1, -1))[0]

    full_cohort_clr = cohort_clr(full0)
    patients = meta["Patient"].astype(str).unique()
    patient_to_samples = meta.groupby(meta.Patient.astype(str))["sample_id"].apply(
        lambda s: s.astype(str).tolist()
    ).to_dict()
    # cell-level patient map
    cell_patient = full0["Patient"].astype(str).to_numpy()
    cell_leaf = full0["leaf_prediction"].astype(str).to_numpy()

    boot_cohort_dist = []
    # Sample-level bootstrap via patient-resampled *relative abundance*
    # contribution: for each bootstrap, reweight is not applied; instead we
    # compute leave-in composition for each sample unchanged and measure
    # distance of each sample to the bootstrap *cohort mean composition*
    # vs distance to full cohort mean - as a sensitivity of sample-to-cohort
    # geometry. Primary ratio uses cohort Aitchison as above.

    for _ in range(N_BOOT):
        draw = rng.choice(patients, size=len(patients), replace=True)
        # pool cells from drawn patients (with multiplicity)
        masks = []
        for p in draw:
            masks.append(cell_patient == p)
        # multiplicity: concatenate counts
        counts = np.zeros(len(labels), dtype=float)
        lab_index = {lab: i for i, lab in enumerate(labels)}
        for p in draw:
            leaves = cell_leaf[cell_patient == p]
            for lab, n in pd.Series(leaves).value_counts().items():
                counts[lab_index[str(lab)]] += float(n)
        b_clr = _clr(counts.reshape(1, -1))[0]
        dist = float(np.sqrt(np.sum((b_clr - full_cohort_clr) ** 2)))
        boot_cohort_dist.append(dist)

    jack_med = float(np.median(jack_medians))
    boot_med = float(np.median(boot_cohort_dist))
    ratio = jack_med / boot_med if boot_med > 0 else np.nan

    # Uncertainty on ratio via bootstrap of jackknife study medians? Use
    # percentile CI on boot distribution and jackknife distribution separately.
    out = {
        "atlas": atlas,
        "n_jackknife_realizations": len(jack_medians),
        "n_bootstrap": N_BOOT,
        "jackknife_median_of_sample_median_aitchison": jack_med,
        "jackknife_mean_of_sample_median_aitchison": float(np.mean(jack_medians)),
        "jackknife_p025": float(np.percentile(jack_medians, 2.5)),
        "jackknife_p975": float(np.percentile(jack_medians, 97.5)),
        "patient_boot_cohort_aitchison_median": boot_med,
        "patient_boot_cohort_aitchison_mean": float(np.mean(boot_cohort_dist)),
        "patient_boot_p025": float(np.percentile(boot_cohort_dist, 2.5)),
        "patient_boot_p975": float(np.percentile(boot_cohort_dist, 97.5)),
        "reference_sensitivity_ratio_median": float(ratio),
        "note": (
            "Ratio = typical jackknife sample-median Aitchison (omit vs full seed0) "
            "/ typical patient-bootstrap cohort-composition Aitchison. "
            "Scales are related but not identical units; interpret cautiously."
        ),
    }

    # Also: sample-level jackknife overall median (matches report 04)
    existing = TABLES / "sample_aitchison_displacement.parquet"
    if existing.exists():
        d = pd.read_parquet(existing)
        d = d[d.atlas == atlas]
        out["jackknife_overall_sample_median_aitchison"] = float(d["aitchison_to_full_seed0"].median())
        out["reference_sensitivity_ratio_using_overall_sample_median"] = float(
            d["aitchison_to_full_seed0"].median() / boot_med if boot_med > 0 else np.nan
        )

    pd.Series(boot_cohort_dist, name="cohort_aitchison").to_frame().assign(atlas=atlas).to_parquet(
        TABLES / f"patient_bootstrap_cohort_aitchison_{atlas}.parquet", index=False
    )
    pd.Series(jack_medians, name="sample_median_aitchison").to_frame().assign(atlas=atlas).to_parquet(
        TABLES / f"jackknife_realization_medians_{atlas}.parquet", index=False
    )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = np.random.default_rng(0)
    rows = [atlas_analysis(a, rng) for a in ("HGCA", "PanGI")]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "reference_vs_patient_uncertainty.csv", index=False)
    (TABLES / "reference_vs_patient_uncertainty.json").write_text(
        json.dumps(
            {"created_utc": datetime.now(timezone.utc).isoformat(), "results": rows},
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
