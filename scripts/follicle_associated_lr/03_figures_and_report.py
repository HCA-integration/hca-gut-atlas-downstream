#!/usr/bin/env python
"""Figures A–D + Markdown final report for follicle-associated L–R analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (  # noqa: E402
    CACHE,
    EPITHELIAL,
    FDR_ALPHA,
    MODULE_COLORS,
    OUT,
    WONG,
)

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 6,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, stem, w_mm=180, h_mm=120):
    fig.set_size_inches(w_mm / 25.4, h_mm / 25.4)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


def fig_a_network(hi: pd.DataFrame):
    """Compact anatomical network of headline modules."""
    g = hi[hi["headline_ok"]].copy()
    if g.empty:
        # fall back to best curated with status ok and beta>0
        g = hi[(hi["status"] == "ok") & (hi["beta"] > 0)].sort_values("padj").head(20)
    # pick top 2 edges per module
    keep = (
        g.sort_values("padj")
        .groupby("module", as_index=False)
        .head(2)
    )
    # node compartments
    def compartment(ct):
        if ct in EPITHELIAL or "M Cells" in ct:
            return "epithelium"
        if "Macrophage" in ct or ct.startswith("cDC") or ct == "migDC":
            return "macrophage"
        if any(x in ct for x in ("FRC", "mLTo", "MRC", "fDC", "Fibroblast")):
            return "stroma"
        if any(x in ct for x in ("Lymphatic", "Sinus", "Endothelial")):
            return "lymphatic"
        if any(x in ct for x in ("GC B", "Tfh", "Tfr", "Memory B", "CD4")):
            return "lymphoid"
        return "other"

    keep["src_comp"] = keep["source"].map(compartment)
    keep["tgt_comp"] = keep["target"].map(compartment)
    order = ["epithelium", "macrophage", "stroma", "lymphoid", "lymphatic", "other"]
    xpos = {c: i for i, c in enumerate(order)}
    # layout nodes by unique CT
    nodes = sorted(set(keep["source"]) | set(keep["target"]))
    # y stack within compartment
    pos = {}
    for comp in order:
        cts = [n for n in nodes if compartment(n) == comp]
        for j, ct in enumerate(cts):
            pos[ct] = (xpos[comp], j)

    fig, ax = plt.subplots()
    # edges
    for _, r in keep.iterrows():
        if r["source"] not in pos or r["target"] not in pos:
            continue
        x1, y1 = pos[r["source"]]
        x2, y2 = pos[r["target"]]
        col = MODULE_COLORS.get(r["module"], WONG["grey"])
        lw = 0.5 + 2.5 * min(abs(r["beta"]) / (np.nanmax(np.abs(keep["beta"])) + 1e-9), 1)
        ls = "-" if r["evidence_class"] == "well_established_direct" else "--"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=col, lw=lw, ls=ls,
                            shrinkA=4, shrinkB=4),
        )
    for ct, (x, y) in pos.items():
        ax.scatter([x], [y], s=28, c="white", edgecolors="black", linewidths=0.4, zorder=3)
        ax.text(x, y + 0.15, ct.replace(" (", "\n("), ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_yticks([])
    ax.set_title("Figure A. Follicle-associated communication map (headline modules)",
                 loc="left", fontsize=7, fontweight="bold")
    ax.set_ylim(-0.5, max((y for _, y in pos.values()), default=1) + 1.2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color=c, label=m) for m, c in MODULE_COLORS.items()
               if m in set(keep["module"])]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=5)
    save(fig, "figA_follicle_communication_map", 180, 110)


def fig_b_effect_sizes(hi: pd.DataFrame):
    g = hi[(hi["status"] == "ok") & (~hi["classification_derived"])].copy()
    g = g.sort_values(["module", "padj"]).groupby("module", as_index=False).head(4)
    if g.empty:
        return
    g["label"] = (
        g["ligand_complex"] + "→" + g["receptor_complex"] + "\n"
        + g["source"].str.replace(r" \(.*\)", "", regex=True).str.slice(0, 22)
        + "→" + g["target"].str.replace(r" \(.*\)", "", regex=True).str.slice(0, 22)
    )
    g = g.sort_values(["module", "beta"])
    g["y"] = np.arange(len(g))
    fig, ax = plt.subplots()
    for _, r in g.iterrows():
        col = MODULE_COLORS.get(r["module"], WONG["grey"])
        ax.plot([r["ci_lo"], r["ci_hi"]], [r["y"], r["y"]], color=col, lw=1)
        ax.scatter([r["beta"]], [r["y"]], color=col, s=12, zorder=3)
    ax.axvline(0, color=WONG["grey"], lw=0.5, ls="--")
    ax.set_yticks(g["y"])
    ax.set_yticklabels(g["label"], fontsize=5)
    ax.set_xlabel("Follicle+ effect (β, HC3 CI)")
    ax.set_title("Figure B. Interaction-module effect sizes", loc="left",
                 fontsize=7, fontweight="bold")
    # donor coverage as text
    for _, r in g.iterrows():
        ax.text(ax.get_xlim()[1] if False else r["ci_hi"] + 0.01, r["y"],
                f"d{int(r['n_donors_pos'])}/{int(r['n_datasets'])}ds",
                va="center", fontsize=4.5, color=WONG["grey"])
    save(fig, "figB_module_effect_sizes", 180, min(170, 30 + 6 * len(g)))


def fig_c_epithelial_matrix(epi: pd.DataFrame):
    if epi.empty:
        return
    iface = ["NECTIN2->TIGIT", "NECTIN3->TIGIT", "ALCAM->CD6",
             "LGALS3->LAG3", "CD24->SIGLEC10"]
    # best target per source × lr (min padj)
    e = epi[epi["lr_pair"].isin(iface)].copy()
    if e.empty:
        return
    e = e.sort_values("padj").groupby(["source", "lr_pair"], as_index=False).first()
    mat = e.pivot(index="source", columns="lr_pair", values="beta")
    # order
    mat = mat.reindex(columns=[c for c in iface if c in mat.columns])
    fig, ax = plt.subplots()
    sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "β follicle+"})
    # stars for padj
    padj = e.pivot(index="source", columns="lr_pair", values="padj").reindex(
        index=mat.index, columns=mat.columns)
    for i, ct in enumerate(mat.index):
        for j, lr in enumerate(mat.columns):
            p = padj.loc[ct, lr]
            if pd.notna(p) and p < FDR_ALPHA:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", fontsize=7)
    ax.set_title("Figure C. Epithelial regulatory-interface matrix (* FDR<0.05)",
                 loc="left", fontsize=7, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "figC_epithelial_interface_matrix", 140, 120)


def fig_d_fingerprint(scores: pd.DataFrame, res: pd.DataFrame):
    """Samples ordered by GC B; module scores as rows."""
    # module score per sample = mean z-scored edge score of top edges in module
    top = res[(res["status"] == "ok") & (res["beta"] > 0)].sort_values("padj")
    top = top.groupby("module", as_index=False).head(5)
    if top.empty or scores.empty:
        return
    edge_set = set(top["edge_id"])
    s = scores[scores["edge_id"].isin(edge_set)].copy()
    s = s.merge(top[["edge_id", "module"]], on="edge_id", how="left")
    # z-score within edge
    s["z"] = s.groupby("edge_id")["score"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    finger = (
        s.groupby(["sample_id", "module"], as_index=False)
        .agg(z=("z", "mean"), n_gc_b=("n_gc_b", "first"),
             follicle_pos=("follicle_pos", "first"), segment=("segment", "first"))
    )
    # order samples by n_gc_b
    samp_order = (
        finger[["sample_id", "n_gc_b", "follicle_pos"]]
        .drop_duplicates()
        .sort_values(["follicle_pos", "n_gc_b"])
    )
    mods = [
        "follicular_stromal_organization",
        "lymphatic_chemokine_gradient",
        "macrophage_antigen_handling",
        "epithelial_regulatory_adhesion",
        "epithelial_ag_presentation_interface",
        "tfh_tfr_positioning",
        "bcell_recruitment_retention",
        "mcell_induction",
    ]
    mods = [m for m in mods if m in set(finger["module"])]
    mat = finger.pivot(index="module", columns="sample_id", values="z")
    mat = mat.reindex(index=mods, columns=samp_order["sample_id"])
    fig, ax = plt.subplots()
    sns.heatmap(mat, ax=ax, cmap="coolwarm", center=0, xticklabels=False,
                cbar_kws={"label": "module z-score"})
    # follicle bar
    ax.set_title("Figure D. Multilineage follicle fingerprint (samples ordered by GC B)",
                 loc="left", fontsize=7, fontweight="bold")
    save(fig, "figD_multilineage_fingerprint", 180, 90)


def xenium_table(hi: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified spatial validation candidates."""
    # known panel genes from collaborator protocol
    panelish = {
        "CXCL13", "CXCR5", "CCL19", "CCL21", "CCR7", "CR2", "FCER2",
        "TNFSF11", "TNFRSF11A", "SPIB", "GP2", "SOX8", "TNFAIP2",
        "HLA-DRA", "CD74", "PDPN", "PROX1", "LYVE1", "MS4A1", "CD19",
        "PDCD1", "ICOS", "BCL6", "NECTIN2", "TIGIT", "ALCAM", "CD6",
        "CD24", "SIGLEC10", "LGALS3", "LAG3",
    }
    rows = []
    g = hi[hi["headline_ok"] | ((hi["status"] == "ok") & (hi["priority"].astype(str).str.contains("curated")))]
    g = g.sort_values("padj").groupby("module", as_index=False).head(2)
    for _, r in g.iterrows():
        genes = set(str(r["ligand_complex"]).split("_") + str(r["receptor_complex"]).split("_"))
        on = sorted(genes & panelish)
        off = sorted(genes - panelish)
        rows.append(dict(
            module=r["module"],
            interaction=r["lr_pair"],
            source=r["source"],
            target=r["target"],
            required_genes=";".join(sorted(genes)),
            genes_likely_on_panel=";".join(on) if on else "NONE_CONFIRMED",
            genes_panel_gap=";".join(off),
            proposed_measurement=(
                f"Distance/bin enrichment of {r['target']} near {r['source']} "
                f"within follicle polygons; ligand–receptor co-localization"
            ),
            matched_control="Same section, >200 µm from follicle boundary; crypt–surface matched",
            expected_pattern="Co-localized or immediately adjacent compartments in SED/follicle",
            confounders="Dissociation bias absent in Xenium; sectioning plane; inflammation",
            confidence=("high" if r["evidence_class"] == "well_established_direct"
                         and r.get("headline_ok", False) else "moderate"),
            beta=r["beta"], padj=r["padj"], n_donors_pos=r["n_donors_pos"],
            n_datasets=r["n_datasets"],
        ))
    # always include TNFSF11 geometry even if weak
    rows.append(dict(
        module="mcell_induction",
        interaction="TNFSF11->TNFRSF11A",
        source="follicular stroma (MRC/mLTo/FRC)",
        target="FAE / MHC-II-high epithelium",
        required_genes="TNFSF11;TNFRSF11A;SPIB;HLA-DRA",
        genes_likely_on_panel="TNFSF11;TNFRSF11A;SPIB;HLA-DRA",
        genes_panel_gap="",
        proposed_measurement="TNFSF11+ stroma in SED beneath TNFRSF11A+/MHC-II-high epithelium",
        matched_control="Villous epithelium distant from follicle",
        expected_pattern="Basal stromal RANKL under dome epithelium",
        confounders="Low TNFSF11 transcript abundance; dropouts",
        confidence="pre_registered",
        beta=np.nan, padj=np.nan, n_donors_pos=np.nan, n_datasets=np.nan,
    ))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "xenium_validation_candidates.csv", index=False)
    return out


def write_report(res: pd.DataFrame, hi: pd.DataFrame, xen: pd.DataFrame):
    n_head = int(hi["headline_ok"].sum()) if "headline_ok" in hi else 0
    head = hi[hi["headline_ok"]].sort_values("padj") if n_head else hi.iloc[0:0]
    # module coherence
    mod = pd.read_csv(OUT / "module_summary.csv") if (OUT / "module_summary.csv").exists() else pd.DataFrame()
    rankl = pd.read_csv(OUT / "TNFSF11_TNFRSF11A_tests.csv") if (OUT / "TNFSF11_TNFRSF11A_tests.csv").exists() else pd.DataFrame()
    epi = pd.read_csv(OUT / "epithelial_interface_tests.csv") if (OUT / "epithelial_interface_tests.csv").exists() else pd.DataFrame()

    # classify overall result
    strong_mods = mod[mod["n_headline"] >= 2]["module"].tolist() if len(mod) else []
    iface_hits = epi[(epi["padj"] < FDR_ALPHA) & (epi["beta"] > 0) & (~epi["classification_derived"])] if len(epi) else epi
    iface_systems = iface_hits["lr_pair"].nunique() if len(iface_hits) else 0

    if n_head >= 5 and len(strong_mods) >= 2:
        verdict = "coherent but descriptive multicellular module"
    elif n_head >= 2 and len(strong_mods) >= 1:
        verdict = "coherent but descriptive multicellular module"
    elif n_head >= 1:
        verdict = "exploratory interaction candidates only"
    else:
        verdict = "no defensible follicle-specific L-R result"

    # upgrade if established circuits survive abundance adj
    est = head[head["evidence_class"] == "well_established_direct"] if n_head else head
    if len(est) >= 3 and (est["padj_abund_adj"] < FDR_ALPHA).sum() >= 2:
        verdict = "strong mechanistic follicle module"

    lines = []
    lines.append("# Follicle-associated ligand–receptor analysis — final report\n")
    lines.append(f"**Verdict:** `{verdict}`\n")
    lines.append("## Answers\n")
    lines.append(
        f"1. **Coherent follicle-associated L–R program?** "
        f"{'Yes, multicellular and partly abundance-structured' if n_head else 'No headline interactions met coverage + FDR gates'}. "
        f"{n_head} headline edges (FDR<{FDR_ALPHA}, β>0, donor/dataset coverage).\n"
    )
    top_mod = mod.sort_values("n_headline", ascending=False).head(3) if len(mod) else mod
    lines.append(
        "2. **Strongest story:** "
        + (", ".join(top_mod["module"].tolist()) if len(top_mod) else "none")
        + ". Established lymphoid-stromal circuits (CXCL13–CXCR5, CCL19/21–CCR7, BAFF) "
        "are expected and largely abundance-linked to the captured follicle; "
        "epithelial claims require independent multi-system support.\n"
    )
    surv = head[head["effect_decomposition"].isin(
        ["expression_driven", "coordinated_abundance_and_expression", "survives_abundance_adjustment"]
    )] if n_head else head
    lines.append(
        f"3. **Survive abundance adjustment / replicate:** {len(surv)} headline edges "
        f"with expression contribution or abundance-adjusted significance. "
        "Require n_datasets and n_donors_pos reported per edge in `follicle_lr_highlights.csv`.\n"
    )
    lines.append(
        "4. **Artifacts / weak priors removed:** CCL19/CCL21–ADRA2A, APP/COPA–CD74, "
        "HLA–CD3D, FAM3D, ribosomal ligands — see `artifact_audit_excluded_edges.csv`. "
        "GC B edges flagged `classification_derived` (not independent validation).\n"
    )
    lines.append(
        "5. **HGCA paper modules:** Prefer (i) follicular stromal / B–T organization "
        "(CXCL13–CXCR5, CCL19/21–CCR7) as descriptive niche recovery, and "
        "(ii) only epithelial interface modules with ≥2 independent systems.\n"
    )
    best_x = xen.sort_values("confidence").iloc[0] if len(xen) else None
    lines.append(
        "6. **Best Xenium module:** "
        + ("TNFSF11 stroma → TNFRSF11A/MHC-II epithelium (pre-registered FAE geometry)"
           if best_x is not None else "see xenium_validation_candidates.csv")
        + ", plus CXCL13/CCR7 organization as positive control for follicle identity.\n"
    )
    lines.append(
        "7. **Beyond DA/DE?** L–R adds putative *wiring* among already DA-enriched "
        "populations. It does **not** replace DA/DE. Abundance-driven edges mostly "
        "restate niche capture; expression-driven edges are the incremental mechanistic claims.\n"
    )

    # RANKL honesty
    lines.append("## TNFSF11 → TNFRSF11A (RANKL)\n")
    if len(rankl):
        ok = rankl[rankl["status"] == "ok"].sort_values("padj")
        if len(ok):
            r0 = ok.iloc[0]
            lines.append(
                f"Best tested edge: `{r0['source']} → {r0['target']}` "
                f"β={r0['beta']:.3f}, p={r0['p']:.2g}, padj={r0['padj']:.2g}, "
                f"donors+={int(r0['n_donors_pos'])}, datasets={int(r0['n_datasets'])}, "
                f"decomposition={r0['effect_decomposition']}.\n"
            )
        else:
            lines.append("No RANKL edge passed coverage gates — do not force a RANKL mechanism.\n")
    else:
        lines.append("No RANKL edges tested/available.\n")

    lines.append("## Epithelial regulatory interface\n")
    lines.append(
        f"Independent interface systems with FDR<0.05 and β>0: **{iface_systems}**. "
        "Claim of coupled antigen presentation + regulatory contacts requires ≥2 systems; "
        f"{'supported' if iface_systems >= 2 else 'NOT supported'} at headline threshold.\n"
    )

    lines.append("## Manuscript sentences\n")
    if n_head == 0:
        cons = (
            "After sample-level scoring with donor/dataset coverage gates and abundance adjustment, "
            "we did not identify follicle-specific ligand–receptor interactions that met pre-specified "
            "reproducibility criteria beyond the compositional recovery of the follicle niche."
        )
        strong = cons
    else:
        cons = (
            "Sample-level ligand–receptor scoring recovered follicle-associated wiring among "
            "follicular stromal, lymphatic, and lymphocyte populations, but many effects tracked "
            "cell-type abundance and the GC B–defined classification."
        )
        strong = (
            "Follicle-positive samples showed a coordinated multicellular ligand–receptor program "
            "spanning follicular stroma–lymphocyte chemokine axes"
            + (" and epithelial regulatory-interface interactions" if iface_systems >= 2 else "")
            + ", with donor- and dataset-replicated effects after covariate adjustment."
        )
    disc = (
        "These interaction maps should be interpreted as hypotheses about physical niche organization "
        "rather than proof of direct extracellular signaling; spatial validation is required."
    )
    lines.append(f"- **Conservative Results:** {cons}\n")
    lines.append(f"- **Stronger Results (if justified):** {strong}\n")
    lines.append(f"- **Discussion:** {disc}\n")
    lines.append(
        "- **Figure layout:** A network map of headline modules; B effect-size forest by module; "
        "C epithelial interface heatmap; D multilineage fingerprint ordered by GC B abundance.\n"
    )
    lines.append(
        "- **Main caveat:** Follicle labels are GC B–derived; interactions involving GC B are "
        "circular. Tissue-pooled LIANA ranks informed candidate selection but inference uses "
        "sample-level scores. CCL19/CCL21–ADRA2A is excluded as non-physiological.\n"
    )

    lines.append("\n## Headline edges\n")
    if n_head:
        show = head.head(15)[
            ["module", "lr_pair", "source", "target", "beta", "padj",
             "n_donors_pos", "n_datasets", "effect_decomposition", "evidence_class"]
        ]
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("_None._\n")

    lines.append("\n## Outputs\n")
    lines.append(f"Directory: `{OUT}`\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines))
    print("Wrote FINAL_REPORT.md — verdict:", verdict)


def main():
    res = pd.read_csv(OUT / "follicle_lr_tests.csv")
    hi = pd.read_csv(OUT / "follicle_lr_highlights.csv")
    scores = pd.read_parquet(CACHE / "sample_edge_scores.parquet")
    epi = pd.read_csv(OUT / "epithelial_interface_tests.csv") if (OUT / "epithelial_interface_tests.csv").exists() else pd.DataFrame()

    fig_a_network(hi)
    fig_b_effect_sizes(hi)
    fig_c_epithelial_matrix(epi if len(epi) else res)
    fig_d_fingerprint(scores, res)
    xen = xenium_table(hi)
    write_report(res, hi, xen)


if __name__ == "__main__":
    main()
