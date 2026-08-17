#!/usr/bin/env python
"""
One-slide-per-concept summary. Produces:
  concept_slides.csv   machine-readable (title / biology / result / next_step / validation)
  fig_concept_slides.png/pdf  rendered table for slide transformation
Pulls live numbers from believability_by_dataset.csv and patientmap/association_stats.csv.
"""
import os, textwrap, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import gca_plot_style as gps
gps.set_style()

BASE = "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA/concept_validation"
bel = pd.read_csv(f"{BASE}/believability_by_dataset.csv")
stat = pd.read_csv(f"{BASE}/patientmap/association_stats.csv")

def b(gene, pat):
    r = bel[(bel.gene==gene) & (bel.celltype_pat.str.contains(pat, regex=False))]
    if len(r)==0: return ""
    r=r.iloc[0]
    return f"{gene}: {int(r.n_datasets_detected)}/{int(r.n_datasets)} datasets, {int(r.n_donors_detected)}/{int(r.n_donors)} donors"

def assoc(concept):
    """Composition-adjusted, patient-random-intercept mixed model on the CLR map.
    Report contexts surviving BH<0.05 after adjusting for sender/receiver
    abundance; annotate whether the effect is per-cell ligand or receptor."""
    s = stat[stat.concept == concept].copy()
    s = s[(s.nA >= 4) & (s.nB >= 4)]
    sig = s[s.p_score_adj_BH < 0.05].sort_values("p_score_adj_BH")
    def driver(r):
        pl = r.p_lig_intensity if pd.notna(r.p_lig_intensity) else 1
        pr = r.p_rec_intensity if pd.notna(r.p_rec_intensity) else 1
        if pl < 0.05 and pr >= 0.05: return "per-cell ligand"
        if pr < 0.05 and pl >= 0.05: return "per-cell receptor"
        if pl < 0.05 and pr < 0.05: return "per-cell both"
        return "abundance-linked"
    def fmt(r):
        dirn = "up" if r.effect_rankbiserial > 0 else "down"
        return (f"{r.context} {r.contrast}: {dirn} "
                f"(adj BH={r.p_score_adj_BH:.1e}, {driver(r)})")
    if len(sig):
        return "; ".join(fmt(r) for _, r in sig.head(3).iterrows())
    tr = s[s.p_score_adj.notna()].sort_values("p_score_adj").head(1)
    if len(tr):
        r = tr.iloc[0]
        return "no context survives composition adjustment (best " + \
               f"{r.context} {r.contrast}, adj p={r.p_score_adj:.1e})"
    return ("not testable in TAURUS - receiver cells too sparse "
            "(neutrophils barely captured in dissociated scRNA)")

CONCEPTS = [
 dict(
   tag="Concept 1",
   title="A distributed enteroendocrine peptidergic command layer runs the ileo-colonic axis",
   biology=("Enterochromaffin cells (TPH1/serotonin) and L/N/S enteroendocrine cells "
            "(PYY, GCG, NTS, SCT) form a chemical command layer. Their receptors sit on "
            "stem/TA cells (HTR4), macrophages (HTR7), pericytes (NPY1R) and BEST4 "
            "enterocytes (guanylin-GUCY2C). It is one coordinated peptide/receptor grid, "
            "not isolated hormones."),
   validation=("TPH1 in EC: 6/6 datasets, 23/23 donors, ~99% frac. NTS/PYY/GCG peak in "
               "ileum; receptors broad but low. Duodenum/jejunum EEC panels are sparse "
               "(few normal proximal samples) - a metadata-breadth limit, not biology."),
   result="C1_serotonin_EC_to_ISC",
   outro=("EXCITING: the atlas resolves the full ligand->receptor grid in one map, and the "
          "EC-serotonin->stem-cell (HTR4) arm is higher in healthy than UC even after "
          "composition adjustment (adj BH<0.05, driven per-cell on the receptor side). "
          "NEXT: proximal (duo/jej) EEC sampling is thin - targeted proximal profiling "
          "needed before claiming a true proximal-distal peptide gradient."),
 ),
 dict(
   tag="Concept 2",
   title="An enteroendocrine proSAAS/BigLEN -> CD8 GPR171 neuro-immune checkpoint",
   biology=("EEC cells secrete PCSK1N (proSAAS -> BigLEN), whose receptor GPR171 is on "
            "mucosal CD8 T cells (IEL/TRM/effector-memory). This is a hormone-driven "
            "brake on tissue-resident cytotoxic T cells - a largely unexplored gut "
            "neuro-immune checkpoint."),
   validation=("PCSK1N in EEC: 9/9 datasets, 76/76 donors, ~91% frac. GPR171 on CD8: "
               "19/19 datasets, 180/181 donors (broad, low per-cell). Believable on both "
               "ligand and receptor side across the atlas."),
   result="C2_checkpoint_EEC_to_CD8",
   outro=("EXCITING: survives a patient-random-intercept mixed model AFTER adjusting for "
          "cell-type abundance - lower in inflamed tissue in UC-Pre, UC-Post and CD-Post "
          "(adj BH<0.05). The driver is LESS PCSK1N PER EEC CELL (p~1e-10), not simply fewer "
          "EEC - a genuine per-cell rewiring, not a composition artifact. NEXT: test whether "
          "restoring GPR171 tone dampens cytotoxic T cells - a novel druggable axis."),
 ),
 dict(
   tag="Concept 3",
   title="Mast-cell tryptase -> epithelial PAR2 is a homeostatic (not just allergic) pillar",
   biology=("Mast cells constitutively express tryptases (TPSAB1/TPSB2); the receptor "
            "F2RL1 (PAR2) is broad across goblet, enterocyte, BEST4 and Paneth cells. "
            "In the healthy atlas this is a standing homeostatic circuit tuning "
            "secretion and barrier, not only an allergy/mastocytosis axis."),
   validation=("TPSAB1 in mast: 13/13 datasets, 62/62 donors, 100% frac. F2RL1 on "
               "epithelium: 14/14 datasets, 148/150 donors. Extremely robust both sides."),
   result="C3_tryptase_Mast_to_Epi",
   outro=("EXCITING: mast->PAR2 tone shifts with treatment context (rises Pre->Post in CD, "
          "lower in non-remission UC) - concordant trends, not yet BH-significant after "
          "stratification. NEXT: PAR2 antagonists are in trials; the atlas nominates the "
          "epithelial targets and the treatment contexts where the circuit moves."),
 ),
 dict(
   tag="Concept 4",
   title="Enteric glia are the highest-quality coordinating hub of the gut wall",
   biology=("Glia broadcast neuronal-adhesion and trophic ligands (NRXN1, GDNF, NRG1) to "
            "smooth muscle, myofibroblasts, endothelium and each other. They behave as a "
            "connectivity hub wiring the muscularis/stroma, beyond classic neuron support."),
   validation=("NRXN1 in glia: 7/7 datasets, 24/24 donors, ~99% frac. Partner receptors "
               "(NLGN1) on SMC/myofibroblasts are sparser (4/6 datasets) - hub identity "
               "solid, some partner edges depend on rarer stromal capture."),
   result="C4_glia_hub",
   outro=("EXCITING: glia rank as the cleanest coordinating node across segments. "
          "NEXT: deeper muscularis/myenteric sampling would confirm glia->stroma edges "
          "that are currently metadata-limited."),
 ),
 dict(
   tag="Pillar / HEV",
   title="Gut lymphocyte efflux runs through a MADCAM1+ ACKR1+ venular sink, not classic HEV",
   biology=("Venular & pre-venule endothelium express the chemokine scavenger ACKR1 and "
            "the mucosal addressin MADCAM1; medullary-sinus/lymphatic endothelium make "
            "CCL21. Classic HEV sulfation machinery (CHST4/CHST2/B3GNT3/FUT7/GLCE, "
            "PNAd/L-selectin) is essentially ABSENT - so immune efflux to gut tissue is "
            "MADCAM1/alpha4beta7-driven, the vedolizumab axis, not PNAd HEVs."),
   validation=("ACKR1 in venular/med-sinus: 7/7 datasets, 41/42 donors. MADCAM1: 7/7 "
               "datasets, 39/42 donors. HEV sulfotransferases undetectable across all "
               "endothelial subsets -> confident negative."),
   result="pillar_MADCAM1_addressin",
   outro=("EXCITING: MADCAM1 addressin score is markedly higher in disease than in the few "
          "healthy samples (descriptive; stratified test underpowered by n=4 healthy) - the "
          "atlas independently surfaces the vedolizumab target axis. NEXT: profile more "
          "healthy controls to power the disease contrast and link ACKR1 sink tone to response."),
 ),
 dict(
   tag="Concept 6",
   title="A perivascular gut-wall wiring unit couples fibrosis and neovascularisation",
   biology=("Enteric glia and crypt fibroblasts feed pericytes/fibroblasts via PDGFA->PDGFRA, "
            "MFGE8->PDGFRB and WNT5A->MCAM, while arteriolar endothelium instructs pericytes "
            "through JAG1/DLL4->NOTCH3 and venular endothelium drives lymphatics via "
            "VEGFC->LYVE1. It is the strongest lineage channel in the atlas "
            "(stroma<->endothelium<->glia) and the cellular substrate of Crohn's stricturing "
            "and mucosal neovascularisation."),
   validation=("Receptor side is rock-solid: PDGFRA/PDGFRB on fibroblasts/pericytes (12/12 "
               "datasets), NOTCH3 on pericytes (8/8), VEGFC in venular endo (7/7). PDGFA "
               "ligand is sparser/variable across datasets (2/12) - the receptors anchor the "
               "module. Ranks: VEGFC->LYVE1 #101, MFGE8->PDGFRB #109, JAG1->NOTCH3 #420. "
               "Densest lineage channels atlas-wide (glia<->stroma<->endothelium)."),
   result="C6_perivascular_NOTCH",
   outro=("EXCITING: endothelial JAG1->NOTCH3 instruction of pericytes is significantly HIGHER "
          "in inflamed UC even after composition adjustment (adj BH=1e-3, per-cell receptor) - "
          "active-disease pericyte NOTCH activation, a pre-fibrotic switch. NEXT: contrast "
          "against stricturing Crohn's to nominate the first-igniting anti-fibrotic edge."),
 ),
 dict(
   tag="Concept 7",
   title="A stromal chemokine recruitment amplifier is buffered by the ACKR1 venular sink",
   biology=("Lamina-propria and reticular fibroblasts (and myeloid cells) broadcast CXCL1/2/8 "
            "onto CXCR1/2+ neutrophils - the push that drives the crypt-abscess influx of "
            "active IBD. The ACKR1+ venular/PVC endothelium scavenges those same chemokines - "
            "the pull that buffers it. Recruitment vs buffering is one push-pull system that "
            "sets how many immune cells efflux into tissue."),
   validation=("CXCL1/CXCL2 in S1/FRC fibroblasts and CXCL8 in macrophages are robust; CXCR2 "
               "sits on neutrophils (rare in dissociated data - a capture caveat). The sink "
               "half is rock-solid: CCL14->ACKR1 is the #1 axis atlas-wide, CCL2->ACKR1 #39."),
   result="C7_chemokine_recruit",
   outro=("EXCITING: the atlas resolves both halves of neutrophil trafficking - the fibroblast "
          "amplifier and the endothelial buffer - in one map. NEXT: test whether a weakened "
          "ACKR1 sink (not just a stronger amplifier) tips tissue toward inflamed neutrophilia."),
 ),
]

# fill live association strings
for c in CONCEPTS:
    c["result_str"] = assoc(c["result"])

rows=[]
for c in CONCEPTS:
    rows.append(dict(concept=c["tag"], title=c["title"], biology=c["biology"],
                     validation=c["validation"], clinical=c["result_str"],
                     outro=c["outro"]))
pd.DataFrame(rows).to_csv(f"{BASE}/concept_slides.csv", index=False)

# ---------- render table figure ----------
def wrap(t,w): return "\n".join(textwrap.wrap(t,w))
n=len(CONCEPTS)
fig,ax=plt.subplots(figsize=(13, 2.55*n))
ax.axis("off")
COL_L=["#0072B2","#D55E00","#009E73","#CC79A7","#7B3294","#8C564B","#111111"]
row_h=1.0/n
for i,c in enumerate(CONCEPTS):
    y1=1-(i)*row_h; y0=1-(i+1)*row_h
    ax.add_patch(plt.Rectangle((0,y0+0.004),1,row_h-0.008,transform=ax.transAxes,
                 facecolor="#FBFBFB",edgecolor="#CCC",lw=0.8))
    # left tag band
    ax.add_patch(plt.Rectangle((0,y0+0.004),0.012,row_h-0.008,transform=ax.transAxes,
                 facecolor=COL_L[i],edgecolor="none"))
    yc=y1-0.02
    ax.text(0.022, yc, c["tag"], transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", color=COL_L[i], va="top")
    ax.text(0.11, yc, wrap(c["title"],60), transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top")
    # biology box (side)
    ax.text(0.075, yc-0.30*row_h/ (row_h) *row_h, "", transform=ax.transAxes) # noop keep spacing
    ax.text(0.075, y0+0.62*row_h, "BIOLOGY", transform=ax.transAxes, fontsize=6.5,
            color="#666", fontweight="bold")
    ax.text(0.075, y0+0.58*row_h, wrap(c["biology"],72), transform=ax.transAxes,
            fontsize=7.4, va="top")
    # validation box (right column)
    ax.text(0.63, y0+0.62*row_h, "VALIDATION (atlas breadth)", transform=ax.transAxes,
            fontsize=6.5, color="#666", fontweight="bold")
    ax.text(0.63, y0+0.58*row_h, wrap(c["validation"],48), transform=ax.transAxes,
            fontsize=7.0, va="top")
    ax.text(0.63, y0+0.24*row_h, "PATIENT-MAP (CLR compositional, TAURUS; stratified)",
            transform=ax.transAxes, fontsize=6.5, color="#666", fontweight="bold")
    ax.text(0.63, y0+0.20*row_h, wrap(c["result_str"],48), transform=ax.transAxes,
            fontsize=7.0, va="top", color="#A1430F")
    # outro band bottom-left
    ax.text(0.075, y0+0.045*row_h, wrap(c["outro"],110), transform=ax.transAxes,
            fontsize=7.2, va="bottom", style="italic", color="#1a1a1a")
fig.suptitle("HGCA CCC synthesis - one slide per concept (title / biology / validation / next step)",
             fontsize=12, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0,0,1,0.985])
gps.save(fig, f"{BASE}/fig_concept_slides")
print("wrote", f"{BASE}/fig_concept_slides.png")
print("wrote", f"{BASE}/concept_slides.csv")
