#!/usr/bin/env python3
"""Queue myeloid screening (default) or backfill trainings. Skips complete checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "train_lineage_reference.py"
SHARED = ROOT.parent / "manifests" / "myeloid_shared_studies_exact_name.csv"
from paths import MODELS  # noqa: E402
MODELS = MODELS


def complete(atlas: str, omit: str, seed: int) -> bool:
    omit_tag = "full" if omit == "full" else f"omit_{omit}"
    d = MODELS / atlas / "myeloid" / omit_tag / f"seed{seed}"
    return (d / "training_manifest.json").exists() and (d / "model.pt").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument(
        "--phase",
        choices=["screen", "backfill", "all"],
        default="screen",
        help="screen: 3 full seeds + 1 seed/omit; backfill: 10 full + 3 seeds/omit",
    )
    ap.add_argument("--atlases", nargs="+", default=["HGCA", "PanGI"])
    args = ap.parse_args()

    shared = pd.read_csv(SHARED)["study"].astype(str).tolist()
    jobs: list[tuple[str, str, int]] = []

    if args.phase in ("screen", "all"):
        for atlas in args.atlases:
            for seed in range(3):
                jobs.append((atlas, "full", seed))
            for study in shared:
                jobs.append((atlas, study, 0))

    if args.phase in ("backfill", "all"):
        for atlas in args.atlases:
            for seed in range(10):
                jobs.append((atlas, "full", seed))
            for study in shared:
                for seed in range(3):
                    jobs.append((atlas, study, seed))

    # dedupe preserving order
    seen = set()
    uniq = []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            uniq.append(j)
    jobs = uniq

    pending = [j for j in jobs if not complete(*j)]
    print(f"Queued {len(jobs)} jobs; {len(pending)} pending; {len(jobs)-len(pending)} complete")
    for atlas, omit, seed in pending:
        cmd = [
            sys.executable,
            str(TRAIN),
            "--lineage",
            "myeloid",
            "--atlas",
            atlas,
            "--omit",
            omit,
            "--seed",
            str(seed),
        ]
        print(" ".join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
