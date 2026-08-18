# Figure 3i and Supplementary Figure 12 — follicle niche

Featured rare and depth-associated cell types, follicle niche capture
from composition (Figure 3i), and the GC B k-scan (Supplementary
Figure 12).

Input: `clr_long.csv` from
[`data/composition/`](../../../data/composition/) (full atlas; must
include `n_cells`). This analysis is HGCA-only; there is no TAURUS
follicle panel in the paper.

Script: `src/compute_defensibility.py`

Output: `niche_capture_rates.csv` and featured-type contrast tables.

```bash
python analyses/fig3_clr_contrasts/fig3i_sfig12_follicle/src/compute_defensibility.py \
  --clr-long /path/to/clr_long.csv --outdir /tmp/follicle
```

Primary composition call used in the figures: a sample is follicle+ if
**either** GC B LZ **or** GC B DZ has at least **3** cells
(`(n_LZ >= 3) | (n_DZ >= 3)`). This is a composition proxy, not
histology. The k-scan in Supplementary Figure 12 compares other cutoffs;
k=3 is the main-text rule.

GSVA on lymphoid pseudobulk needs the lineage objects and `decoupler`.
The bundled subset was built so 17 samples have at least three GC B
LZ/DZ cells and 109 have zero; the k=3 Youden threshold on that subset
is not the paper claim.
