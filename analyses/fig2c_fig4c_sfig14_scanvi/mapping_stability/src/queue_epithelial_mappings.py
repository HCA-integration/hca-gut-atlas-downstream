#!/usr/bin/env python3
"""Queue epithelial TAURUS mappings for complete checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "src" / "map_taurus_lineage_realization.py"
from paths import MODELS, PREDICTIONS  # noqa: E402
PREDS = PREDICTIONS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--atlases", nargs="+", default=["HGCA", "PanGI"])
    args = ap.parse_args()
    jobs = []
    for atlas in args.atlases:
        base = MODELS / atlas / "epithelial"
        if not base.exists():
            continue
        for man in sorted(base.glob("*/seed*/training_manifest.json")):
            model_dir = man.parent
            rel = model_dir.relative_to(MODELS)
            if (PREDS / rel / "predictions_manifest.json").exists():
                continue
            jobs.append(model_dir)
    print(f"{len(jobs)} epithelial mappings pending")
    for model_dir in jobs:
        cmd = [
            sys.executable,
            str(MAP),
            "--lineage",
            "epithelial",
            "--model-dir",
            str(model_dir),
        ]
        print(" ".join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
