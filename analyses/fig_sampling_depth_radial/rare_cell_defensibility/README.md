# Figure 3i — follicle / rare-cell defensibility

Featured rare / depth-associated cell types and follicle niche capture
from composition tables.

**DEMO MODE: results from the bundled slice are for software checking,
not manuscript figures.**

## System requirements

- OS tested: macOS 15
- Python 3.12 with `pandas`, `numpy`, `scipy`, `statsmodels`
- `decoupler` only for the optional GSVA script
- Non-standard hardware: none

## Installation

From the repository root: `python -m pip install -r requirements.txt`
(under 5 minutes).

## Demo

First write demo CLR tables, then:

```bash
python analyses/fig_sampling_depth_radial/rare_cell_defensibility/src/compute_defensibility.py \
  --clr-long data/demo/expected/clr/clr_long.csv \
  --outdir data/demo/expected/follicle
```

| | |
|---|---|
| Input | `clr_long.csv` from the demo slice (must include `n_cells`) |
| Expected output | `niche_capture_rates.csv` and featured-type contrast tables |
| Runtime | under a minute |

The demo slice was built so 17 samples have ≥3 GC B LZ/DZ cells and 109
have zero. k=3 Youden on this slice is a software check, not the paper
claim.

Or run `python src/run_demo.py`.

## Instructions for use

```bash
python analyses/fig_sampling_depth_radial/rare_cell_defensibility/src/compute_defensibility.py \
  --clr-long /path/to/clr_long.csv --outdir /tmp/follicle
```

Niche primary rule: ≥1 GC B compartment (LZ or DZ) with ≥3 cells, and
≥1 support type among {fDC, Tfh, Tfr, FARM, FRC, mLTo} with ≥3 cells.
This is a composition proxy, not histology.
