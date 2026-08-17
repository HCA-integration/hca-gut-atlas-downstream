"""Paths and constants for follicle-associated sample-level L–R analysis."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUB = Path("/Users/kylekimler/Projects/GCA/publication2026/supp_follicle_epithelial")
FOLLICLE_DATA = PUB / "data"
AUDIT_CSV = FOLLICLE_DATA / "sample_classification_audit.csv"
FAE_MODULE_SAMPLE = FOLLICLE_DATA / "FAE_module_scores_sample.csv"
FAE_MODULE_EFFECTS = FOLLICLE_DATA / "FAE_module_effects_by_celltype.csv"
DA_FOCUSED = FOLLICLE_DATA / "da_focused_forest.csv"

ATLAS = Path("/Users/kylekimler/Projects/GCA/meta_datasets/integrated-objects/hgca_all_lineages_v1.h5ad")
LIANA_COMBINED = Path(
    "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/"
    "combined_lr_per_tissue_level_1.csv"
)
LINEAGE_CSV = REPO / "data" / "hgca_celltype_v1_lineage.csv"

OUT = Path(
    "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/"
    "follicle_associated_lr"
)
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

GROUP_KEY = "hgca_celltype_v1"
POWERED_SEGMENTS = ("ileum", "colon")
MIN_CT_CELLS = 10
MIN_SAMPLES_POS = 8
MIN_SAMPLES_NEG = 15
MIN_DONORS_POS = 5
MIN_DONORS_NEG = 8
MIN_DATASETS = 3
EXPR_PROP_SOFT = 0.10  # prevalence gate within sample×celltype
RANDOM_STATE = 0
FDR_ALPHA = 0.05

# Follicle niche ecosystem (sources/targets considered)
ECOSYSTEM = {
    # epithelial
    "Microfold Cells (M Cells)",
    "Enterocyte Progenitors",
    "Colonocyte Progenitors",
    "BEST4 Enterocytes",
    "BEST4 Colonocytes",
    "Goblet Cells",
    "Mature Goblet Cells",
    "Villus Tip Enterocytes",
    "Mid Villus Enterocytes",
    "Lower Villus Enterocytes",
    "Crypt Top Colonocytes",
    "Mid Crypt Colonocytes",
    "Lower Crypt Colonocytes",
    "Intestinal Stem Cells (ISC)",
    "Transiently Amplifying Cells (TA)",
    # stromal / FDC
    "Fibroblastic Reticular Cells (FRC)",
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
    "Marginal Reticular Cells (MRC)",
    "Follicular Dendritic Cells (fDC)",
    "Lamina propria Fibroblasts (S1)",
    "Crypt Bottom Fibroblasts (S2A)",
    "Crypt Top Fibroblasts (S2B)",
    "Submucosal Fibroblasts (S3)",
    # endothelial / lymphatic
    "Lymphatic Endothelial",
    "Medullary Sinus Endothelial",
    "Post Arteriole Capillary Endothelial (PAC)",
    "Pre Venule Capillary Endothelial (PVC)",
    "Arteriolar Endothelial",
    "Venular Endothelial",
    "Capillary Endothelial",
    # myeloid
    "Follicle Associated Resident Macrophages",
    "Homeostatic Macrophages",
    "M0 Macrophages",
    "Perivascular Resident Macrophages",
    "migDC",
    "cDC2",
    "Tolerogenic cDC2",
    # lymphoid
    "GC B Dark Zone (GC B DZ)",
    "GC B Light Zone (GC B LZ)",
    "Memory B",
    "CD4 Tfh",
    "CD4 Tfr",
    "CD4 Memory",
    "CD4 pTreg",
    "CD4 tTreg",
}

CLASSIFICATION_DERIVED = {
    "GC B Dark Zone (GC B DZ)",
    "GC B Light Zone (GC B LZ)",
}

EPITHELIAL = {
    "Microfold Cells (M Cells)",
    "Enterocyte Progenitors",
    "Colonocyte Progenitors",
    "BEST4 Enterocytes",
    "BEST4 Colonocytes",
    "Goblet Cells",
    "Mature Goblet Cells",
    "Villus Tip Enterocytes",
    "Mid Villus Enterocytes",
    "Lower Villus Enterocytes",
    "Crypt Top Colonocytes",
    "Mid Crypt Colonocytes",
    "Lower Crypt Colonocytes",
    "Intestinal Stem Cells (ISC)",
    "Transiently Amplifying Cells (TA)",
}

# Curated circuits always tested (database names)
CURATED = [
    # M-cell induction
    ("TNFSF11", "TNFRSF11A"),
    ("TNF", "TNFRSF1A"),
    ("TNF", "TNFRSF1B"),
    ("TNFSF11", "TNFRSF11B"),
    # Follicular organization
    ("CXCL13", "CXCR5"),
    ("CCL19", "CCR7"),
    ("CCL21", "CCR7"),
    ("TNFSF13B", "TNFRSF13C"),  # BAFF–BAFFR
    ("TNFSF13B", "TNFRSF17"),   # BAFF–BCMA
    ("TNFSF13", "TNFRSF13B"),   # APRIL–TACI
    ("LTB", "LTBR"),
    ("LTA", "LTBR"),
    ("ICAM1", "ITGAL_ITGB2"),
    ("VCAM1", "ITGA4_ITGB1"),
    ("VCAM1", "ITGA4_ITGB7"),
    # Lymphatic chemokine
    ("CCL25", "CCR9"),
    ("CCL19", "ACKR4"),
    ("CCL21", "ACKR4"),
    ("CCL2", "ACKR2"),
    ("CCL5", "ACKR2"),
    # Epithelial regulatory interface
    ("NECTIN2", "TIGIT"),
    ("NECTIN3", "TIGIT"),
    ("ALCAM", "CD6"),
    ("LGALS3", "LAG3"),
    ("CD24", "SIGLEC10"),
    # Macrophage / complement / clearance
    ("C1QA", "CR1"),
    ("C3", "CR2"),
    ("GAS6", "MERTK"),
    ("PROS1", "MERTK"),
    ("CXCL13", "CXCR5"),
]

# Explicit artifact / banned pairs (ligand, receptor) — exact or prefix rules
BANNED_EXACT = {
    ("APP", "CD74"),
    ("COPA", "CD74"),
    ("HLA-B", "CD3D"),
    ("CCL19", "ADRA2A"),
    ("CCL21", "ADRA2A"),
    ("DBP", "ACKR1"),
    ("FAM3D", "FPR1"),
    ("FAM3D", "FPR2"),
}
BANNED_LIGAND_PREFIX = (
    "RPL", "RPS", "MRPL", "MRPS", "HIST", "H2A", "H2B", "H3", "H4",
)
BANNED_SOURCE_OR_TARGET_SUBSTR = (
    "Neutrophil", "Eosinophil", "Basophil", "Mast Cells",  # mast kept? user said neutrophils/granulocytes — keep mast out of headline
)
# User: avoid neutrophils, granulocytes, FAM3D. Mast is not granulocyte in same sense — exclude Neutrophil/Eos/Basophil only
BANNED_CT_SUBSTR = ("Neutrophil", "Eosinophil", "Basophil")

WONG = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "black": "#000000",
}

MODULE_COLORS = {
    "follicular_stromal_organization": WONG["purple"],
    "bcell_recruitment_retention": WONG["blue"],
    "tfh_tfr_positioning": WONG["sky"],
    "mcell_induction": WONG["green"],
    "epithelial_ag_presentation_interface": WONG["orange"],
    "epithelial_regulatory_adhesion": WONG["vermillion"],
    "macrophage_antigen_handling": "#8C564B",
    "lymphatic_chemokine_gradient": WONG["black"],
}
