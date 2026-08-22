#!/usr/bin/env python3
"""Queue TAURUS stroma mappings for all saved SCANVI realizations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from paths import MODELS, PREDICTIONS  # noqa: E402
PREDS = PREDICTIONS
MAP = ROOT / "src" / "map_taurus_stroma_realization.py"


def complete(model_dir: Path) -> bool:
    rel = model_dir.relative_to(MODELS)
    d = PREDS / rel
    return (d / "predictions_manifest.json").exists() and (d / "predictions.parquet").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--atlases", nargs="+", default=["HGCA", "PanGI"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = []
    for atlas in args.atlases:
        for model_dir in sorted((MODELS / atlas / "stroma").glob("*/seed*")):
            if "_smoke" in str(model_dir):
                continue
            if not (model_dir / "model.pt").exists():
                continue
            jobs.append(model_dir)

    pending = [j for j in jobs if not complete(j)]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Total realizations={len(jobs)}; pending={len(pending)}; complete={len(jobs)-len(pending)}")
    for model_dir in pending:
        cmd = [sys.executable, str(MAP), "--model-dir", str(model_dir)]
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
