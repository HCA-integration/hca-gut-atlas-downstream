"""Run the diarrhea ionic programs / pediatric analysis as a standalone script.

This is a CLI wrapper around the Diarrhea_Ionic_Programs_Pediatric.ipynb notebook
that bakes in a headless matplotlib backend so no figure popups can interrupt the
run. Plots are saved as PDF + SVG + PNG to:

    ~/GCA/github_vignette_output/diarrhea_ionic_programs/

Usage:

    /Users/kylekimler/miniforge3/envs/scanpy/bin/python \
        scripts/run_diarrhea_ionic_programs.py

The script extracts code cells from the notebook at runtime, so notebook edits
flow through automatically.
"""
import json
import os
import sys
from pathlib import Path

# Force headless backend BEFORE importing matplotlib anywhere downstream.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

NOTEBOOK = Path(__file__).resolve().parent.parent / "vignettes" / "Diarrhea_Ionic_Programs_Pediatric.ipynb"
if not NOTEBOOK.exists():
    print(f"ERROR: notebook not found: {NOTEBOOK}", file=sys.stderr)
    sys.exit(1)

print(f"Running headless: {NOTEBOOK}")

with NOTEBOOK.open("r") as fh:
    nb = json.load(fh)

cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
print(f"Code cells: {len(cells)}")

namespace = {"__name__": "__main__"}

for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    print(f"\n=== cell {i + 1} / {len(cells)} ===")
    try:
        exec(compile(src, f"<cell-{i + 1}>", "exec"), namespace)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nFailed at cell {i + 1}: {exc}")
        sys.exit(1)
    plt.close("all")

print("\nDone. All figures + CSVs are under:")
print("    ~/GCA/github_vignette_output/diarrhea_ionic_programs/")
