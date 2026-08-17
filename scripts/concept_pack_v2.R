#!/usr/bin/env Rscript
## Concept pack v2 — new highlight set on LIANA rank_aggregate tables.
##
## Themes:
##   1. Follicle / germinal-center wiring
##   2. Deep perivascular (PV resident mac ↔ deep endothelia)
##   3. BEST4 → non-epithelial (Enterocytes, Colonocytes, combined)
##   4. CCL19/CCL21 → ADRA2A on BEST4 (sender identity × segment)
##   5. IL27 → CD4 T cells (resource check; usually absent after expr_prop)
##
## Outputs under <OUT>/concept_pack_v2/ (separate from ensemble_synthesis/).

set.seed(1L)
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr, warn.conflicts = FALSE)
  library(ggplot2)
  library(ggrepel)
  library(scales)
  library(stringr, warn.conflicts = FALSE)
  library(patchwork)
})

INPUT_CSV <- Sys.getenv(
  "CCC_EDGE_CSV",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/combined_lr_per_tissue_level_1.csv")
BASE_OUT <- Sys.getenv(
  "CCC_OUTPUT_DIR",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate")
OUT_DIR <- file.path(BASE_OUT, "concept_pack_v2")
LINEAGE_CSV <- Sys.getenv("CCC_LINEAGE_LOOKUP_CSV",
                          "data/hgca_celltype_v1_lineage.csv")
DROP_STATES <- trimws(strsplit(Sys.getenv("CCC_DROP_STATES", "Epithelial"),
                               "[;,]")[[1]])
SEG_ORDER <- c("duodenum", "jejunum", "ileum", "colon")
RANK_Q <- 0.99
EPS <- 1e-300
TOP_LOLLI <- as.integer(Sys.getenv("CCC_TOP_LOLLI", "12"))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ---------- cell-type sets ------------------------------------------------
FOLLICLE <- c(
  "GC B Dark Zone (GC B DZ)", "GC B Light Zone (GC B LZ)",
  "Follicular Dendritic Cells (fDC)", "CD4 Tfh", "CD4 Tfr",
  "Follicle Associated Resident Macrophages",
  "Microfold Cells (M Cells)",
  "Fibroblastic Reticular Cells (FRC)",
  "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
  "Marginal Reticular Cells (MRC)"
)
DEEP_ENDO <- c(
  "Post Arteriole Capillary Endothelial (PAC)",
  "Pre Venule Capillary Endothelial (PVC)",
  "Arteriolar Endothelial"
)
PV_MAC <- "Perivascular Resident Macrophages"
BEST4_ENT <- "BEST4 Enterocytes"
BEST4_COL <- "BEST4 Colonocytes"
BEST4 <- c(BEST4_ENT, BEST4_COL)
CD4 <- c("CD4 Memory", "CD4 Naive", "CD4 Tfh", "CD4 Tfr",
         "CD4 Th17", "CD4 Tr1", "CD4 pTreg", "CD4 tTreg")

## Curated LR axes for the rank-highlight curve (cell-agnostic).
HL <- data.table(
  lr_pair = c(
    ## Follicle / GC
    "CXCL13->CXCR5", "C3->CR2", "FCER2->CR2",
    "TNFSF13B->TNFRSF13C", "CD40LG->CD40",
    "CCL19->CCR7", "CCL21->CCR7", "CXCL13->ACKR1",
    ## Deep perivascular
    "C1QA->CD93", "CXCL8->ACKR1", "PDGFC->FLT1",
    "VEGFC->LYVE1", "S100A9->CD36", "MMRN2->CLEC14A",
    "EDN1->ADGRL4", "CCL14->ACKR1",
    ## BEST4 → non-epithelial (global axis; cell context in lollipops)
    "CEACAM5->CD8A", "FAM3D->FPR1", "LGALS3->MCAM",
    "LGALS3->ENG", "EDN3->EDNRB", "PDGFA->PDGFRA", "CD24->SELP",
    ## CCL19/21 → ADRA2A (BEST4 receiver context in dedicated panel)
    "CCL19->ADRA2A", "CCL21->ADRA2A",
    ## IL27 complexes (may be absent)
    "EBI3_IL27->IL27RA_IL6ST", "IL12A_IL27->IL12RB2_IL27RA"
  ),
  concept = c(
    rep("Follicle / germinal center", 8),
    rep("Deep perivascular", 8),
    rep("BEST4 → non-epithelial", 7),
    rep("CCL19/21 → ADRA2A", 2),
    rep("IL27 → CD4", 2)
  ),
  label = c(
    "CXCL13→CXCR5", "C3→CR2", "FCER2→CR2",
    "BAFF→BAFFR", "CD40LG→CD40",
    "CCL19→CCR7", "CCL21→CCR7", "CXCL13→ACKR1",
    "C1QA→CD93", "CXCL8→ACKR1", "PDGFC→FLT1",
    "VEGFC→LYVE1", "S100A9→CD36", "MMRN2→CLEC14A",
    "EDN1→ADGRL4", "CCL14→ACKR1",
    "CEACAM5→CD8A", "FAM3D→FPR1", "LGALS3→MCAM",
    "LGALS3→ENG", "EDN3→EDNRB", "PDGFA→PDGFRA", "CD24→SELP",
    "CCL19→ADRA2A", "CCL21→ADRA2A",
    "IL27 (EBI3)", "IL27 (IL12A)"
  )
)

CONCEPT_COLS <- c(
  "Follicle / germinal center" = "#CC79A7",
  "Deep perivascular"          = "#8C564B",
  "BEST4 → non-epithelial"     = "#009E73",
  "CCL19/21 → ADRA2A"          = "#0072B2",
  "IL27 → CD4"                 = "#D55E00"
)

short_ct <- function(x) {
  x |>
    str_replace("Follicle Associated Resident Macrophages", "FA Res Mac") |>
    str_replace("Perivascular Resident Macrophages", "PV Res Mac") |>
    str_replace("Follicular Dendritic Cells \\(fDC\\)", "fDC") |>
    str_replace("GC B Dark Zone \\(GC B DZ\\)", "GC B DZ") |>
    str_replace("GC B Light Zone \\(GC B LZ\\)", "GC B LZ") |>
    str_replace("Fibroblastic Reticular Cells \\(FRC\\)", "FRC") |>
    str_replace("Mesenchymal Lymphoid Tissue Organizer Cells \\(mLTo Cells\\)", "mLTo") |>
    str_replace("Marginal Reticular Cells \\(MRC\\)", "MRC") |>
    str_replace("Microfold Cells \\(M Cells\\)", "M cells") |>
    str_replace("Post Arteriole Capillary Endothelial \\(PAC\\)", "PAC") |>
    str_replace("Pre Venule Capillary Endothelial \\(PVC\\)", "PVC") |>
    str_replace("Arteriolar Endothelial", "Arter Endo") |>
    str_replace("BEST4 Enterocytes", "BEST4 Ent") |>
    str_replace("BEST4 Colonocytes", "BEST4 Col") |>
    str_replace("Lymphatic Endothelial", "Lymph Endo") |>
    str_replace("Medullary Sinus Endothelial", "Med Sinus") |>
    str_replace("CD8 Effector Memory", "CD8 EM") |>
    str_replace("CD8 Circulating Effector Memory", "CD8 Circ EM")
}

theme_gca <- function(sz = 6L) {
  theme_classic(base_size = sz, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black"),
      plot.title = element_text(face = "bold", size = 7, hjust = 0),
      plot.subtitle = element_text(size = 5.5),
      axis.line = element_line(linewidth = 0.25),
      axis.ticks = element_line(linewidth = 0.25),
      axis.text = element_text(colour = "black", size = 6),
      axis.title = element_text(size = 6),
      panel.grid = element_blank(),
      legend.key.size = unit(3, "mm"),
      legend.text = element_text(size = 5.5),
      legend.title = element_text(size = 6, face = "bold"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 6)
    )
}

save_fig <- function(p, stem, w_mm, h_mm) {
  wi <- w_mm / 25.4; hi <- h_mm / 25.4
  pdf_dev <- if (isTRUE(capabilities("cairo"))) grDevices::cairo_pdf else pdf
  ggsave(paste0(stem, ".pdf"), p, width = wi, height = hi, device = pdf_dev)
  ggsave(paste0(stem, ".png"), p, width = wi, height = hi, dpi = 300)
  message("wrote ", stem)
}

## ---------- load ----------------------------------------------------------
message("Reading ", INPUT_CSV)
dt <- fread(INPUT_CSV)
need <- c("magnitude_rank", "specificity_rank")
miss <- setdiff(need, names(dt))
if (length(miss)) stop("Need rank_aggregate columns: ", paste(miss, collapse = ", "))
setnames(dt, "tissue_level_1", "segment")
dt[, segment := tolower(trimws(segment))]
dt <- dt[segment %in% SEG_ORDER &
           !is.na(magnitude_rank) & !is.na(specificity_rank)]
dt <- dt[!(source %in% DROP_STATES) & !(target %in% DROP_STATES)]
dt[, lr_pair := paste(ligand_complex, receptor_complex, sep = "->")]
dt[, ensemble_rank := sqrt(pmax(magnitude_rank, EPS) *
                             pmax(specificity_rank, EPS))]
dt[, ensemble_score := -log10(pmax(ensemble_rank, EPS))]
dt[, is_top := magnitude_rank <= quantile(magnitude_rank, 1 - RANK_Q),
   by = segment]
if (!"lr_means" %in% names(dt)) dt[, lr_means := NA_real_]

lk <- fread(LINEAGE_CSV)[, .(cell_state, plot_lineage)]
setkey(lk, cell_state)
dt[, src_lin := lk[.(source), plot_lineage]]
dt[, tgt_lin := lk[.(target), plot_lineage]]
dt[is.na(src_lin), src_lin := "Other"]
dt[is.na(tgt_lin), tgt_lin := "Other"]
EPI <- lk[plot_lineage == "Epithelial", cell_state]

## ---------- global axis ranks (for fig_rank_highlight) --------------------
ax <- dt[, .(
  ensemble_score = max(ensemble_score),
  magnitude_rank = min(magnitude_rank),
  specificity_rank = min(specificity_rank),
  lr_means = max(lr_means),
  is_top = as.integer(any(is_top))
), by = .(segment, lr_pair)]
axes_all <- ax[, .(
  ensemble_score_mean = mean(ensemble_score),
  mag_rank_mean = mean(magnitude_rank),
  spec_rank_mean = mean(specificity_rank),
  n_seg_top = sum(is_top),
  n_seg = .N
), by = lr_pair][order(-ensemble_score_mean)]
axes_all[, rank := .I]
N_axes <- nrow(axes_all)
top_cut <- axes_all[, quantile(ensemble_score_mean, RANK_Q)]

hl <- merge(HL, axes_all, by = "lr_pair", all.x = TRUE)
hl[, pct_top := 100 * rank / N_axes]
hl[, lab_full := ifelse(is.na(rank), paste0(label, "  (absent)"),
                        sprintf("%s  (#%d)", label, rank))]
fwrite(hl[order(concept, rank)], file.path(OUT_DIR, "highlight_axis_ranks.csv"))
cat("\n==== Concept pack v2 axis ranks ====\n")
print(hl[order(rank)], nrows = 50)

## ---------- fig_rank_highlight --------------------------------------------
hl_plot <- hl[!is.na(rank)]
p_rank <- ggplot(axes_all, aes(rank, ensemble_score_mean)) +
  geom_line(colour = "#BBBBBB", linewidth = 0.4) +
  geom_hline(yintercept = top_cut, linetype = "dashed",
             colour = "#999999", linewidth = 0.25) +
  geom_point(data = hl_plot, aes(rank, ensemble_score_mean, fill = concept),
             shape = 21, size = 1.9, stroke = 0.3, colour = "black") +
  geom_text_repel(
    data = hl_plot,
    aes(rank, ensemble_score_mean, label = lab_full, colour = concept),
    size = 1.75, max.overlaps = Inf, box.padding = 0.55,
    force = 4, force_pull = 0.2, segment.size = 0.2,
    min.segment.length = 0, seed = 1, show.legend = FALSE) +
  scale_x_log10(labels = comma) +
  scale_fill_manual(values = CONCEPT_COLS, name = "Concept") +
  scale_colour_manual(values = CONCEPT_COLS, guide = "none") +
  labs(
    title = "Concept pack v2 — where curated axes rank in the gut interactome",
    subtitle = sprintf(
      "LIANA rank_aggregate ensemble score (mean over 4 segments). N = %s L–R axes. Absent: %s",
      comma(N_axes),
      {
        abs_lab <- hl[is.na(rank), label]
        if (!length(abs_lab)) "none" else paste(abs_lab, collapse = "; ")
      }),
    x = "Rank among all L–R axes (log scale; 1 = strongest)",
    y = "Ensemble score (mean over segments)") +
  theme_gca()
save_fig(p_rank, file.path(OUT_DIR, "fig_rank_highlight"), 180, 100)

## ---------- lollipop helper -----------------------------------------------
lollipop <- function(d, title, subtitle, stem, colour_col = "segment",
                     colour_vals = NULL, w_mm = 160, h_mm = 110) {
  if (!nrow(d)) {
    message("SKIP empty lollipop: ", stem)
    return(invisible(NULL))
  }
  d <- copy(d)
  d[, hit := as.character(hit)]
  d[, hit := reorder(hit, ensemble_score)]
  x0 <- max(0, min(d$ensemble_score, na.rm = TRUE) * 0.92)
  if (is.null(colour_vals)) {
    p <- ggplot(d, aes(x = ensemble_score, y = hit, colour = .data[[colour_col]])) +
      geom_segment(aes(x = x0, xend = ensemble_score, yend = hit), linewidth = 0.35) +
      geom_point(aes(size = pmax(lr_means, 0.05)), shape = 16)
  } else {
    p <- ggplot(d, aes(x = ensemble_score, y = hit, colour = .data[[colour_col]])) +
      geom_segment(aes(x = x0, xend = ensemble_score, yend = hit), linewidth = 0.35) +
      geom_point(aes(size = pmax(lr_means, 0.05)), shape = 16) +
      scale_colour_manual(values = colour_vals, name = colour_col)
  }
  p <- p +
    scale_size_continuous(range = c(1.1, 3.6), name = "lr_means") +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.08))) +
    labs(title = title, subtitle = subtitle,
         x = "Ensemble score (−log10 rank)", y = NULL) +
    theme_gca() +
    theme(axis.text.y = element_text(size = 5.2))
  save_fig(p, stem, w_mm, h_mm)
}

SEG_HEX <- c(duodenum = "#E69F00", jejunum = "#56B4E9",
             ileum = "#009E73", colon = "#D55E00")

## ========================================================================
## 1. Follicle / GC lollipops
## ========================================================================
fol_edges <- dt[source %in% FOLLICLE & target %in% FOLLICLE]
fwrite(fol_edges[, .(segment, source, target, lr_pair, lr_means,
                     magnitude_rank, specificity_rank, ensemble_score, is_top)],
       file.path(OUT_DIR, "follicle_edges.csv"))

## Top cell-resolved follicle edges overall (by mean score across segs present)
fol_best <- fol_edges[, .(
  ensemble_score = max(ensemble_score),
  lr_means = max(lr_means),
  mag_rank = min(magnitude_rank),
  best_segment = segment[which.max(ensemble_score)],
  n_seg = uniqueN(segment)
), by = .(source, target, lr_pair)]
fol_top <- fol_best[order(-ensemble_score)][1:TOP_LOLLI]
fol_top[, hit := paste0(short_ct(source), " → ", short_ct(target),
                        "  |  ", lr_pair)]
fol_top[, segment := factor(best_segment, levels = SEG_ORDER)]
lollipop(
  fol_top, "Follicle ↔ follicle — top L–R × partners",
  "Germinal-center / GALT stroma wiring; best segment coloured",
  file.path(OUT_DIR, "fig_lollipop_follicle_top"),
  colour_col = "segment", colour_vals = SEG_HEX, w_mm = 170, h_mm = 100)

## Classic GC axes: best edge per (segment, lr) among follicle cells
classic_gc <- c("CXCL13->CXCR5", "C3->CR2", "FCER2->CR2",
                "TNFSF13B->TNFRSF13C", "CD40LG->CD40",
                "CCL19->CCR7", "CCL21->CCR7")
fol_classic <- fol_edges[lr_pair %in% classic_gc]
fol_classic_best <- fol_classic[, .SD[which.max(ensemble_score)],
                                by = .(segment, lr_pair)]
fol_classic_best[, hit := paste0(lr_pair, "  |  ",
                                 short_ct(source), " → ", short_ct(target))]
fol_classic_best[, segment := factor(segment, levels = SEG_ORDER)]
## facet by segment
if (nrow(fol_classic_best)) {
  d <- copy(fol_classic_best)
  d[, hit := reorder(hit, ensemble_score)]
  x0 <- max(0, min(d$ensemble_score) * 0.9)
  p <- ggplot(d, aes(ensemble_score, hit, colour = lr_pair)) +
    geom_segment(aes(x = x0, xend = ensemble_score, yend = hit), linewidth = 0.35) +
    geom_point(aes(size = pmax(lr_means, 0.05)), shape = 16) +
    facet_wrap(~ segment, scales = "free_y", ncol = 2) +
    scale_size_continuous(range = c(1.1, 3.4), name = "lr_means") +
    labs(title = "Classic germinal-center axes within follicle cell set",
         subtitle = "Best sender→receiver per segment for curated GC LR pairs",
         x = "Ensemble score", y = NULL, colour = "L–R") +
    theme_gca() +
    theme(axis.text.y = element_text(size = 4.8),
          legend.text = element_text(size = 5))
  save_fig(p, file.path(OUT_DIR, "fig_lollipop_follicle_classic_by_segment"),
           180, 140)
}

## ========================================================================
## 2. Deep perivascular lollipops
## ========================================================================
pv <- dt[
  (source == PV_MAC & target %in% DEEP_ENDO) |
    (target == PV_MAC & source %in% DEEP_ENDO)
]
fwrite(pv[, .(segment, source, target, lr_pair, lr_means,
              magnitude_rank, specificity_rank, ensemble_score, is_top)],
       file.path(OUT_DIR, "deep_perivascular_edges.csv"))

pv_best <- pv[, .(
  ensemble_score = max(ensemble_score),
  lr_means = max(lr_means),
  best_segment = segment[which.max(ensemble_score)],
  n_seg = uniqueN(segment)
), by = .(source, target, lr_pair)]
pv_top <- pv_best[order(-ensemble_score)][1:TOP_LOLLI]
pv_top[, hit := paste0(short_ct(source), " → ", short_ct(target),
                       "  |  ", lr_pair)]
pv_top[, segment := factor(best_segment, levels = SEG_ORDER)]
lollipop(
  pv_top, "Deep perivascular — PV resident mac ↔ deep endothelia",
  "PAC / PVC / arteriolar endothelia; best segment coloured",
  file.path(OUT_DIR, "fig_lollipop_deep_perivascular_top"),
  colour_col = "segment", colour_vals = SEG_HEX, w_mm = 170, h_mm = 100)

## by segment facet
pv_seg <- pv[, .SD[order(-ensemble_score)][1:min(8L, .N)], by = segment]
pv_seg[, hit := paste0(short_ct(source), " → ", short_ct(target),
                       "  |  ", lr_pair)]
pv_seg[, segment := factor(segment, levels = SEG_ORDER)]
if (nrow(pv_seg)) {
  d <- copy(pv_seg)
  d[, hit := reorder(hit, ensemble_score)]
  x0 <- max(0, min(d$ensemble_score) * 0.9)
  p <- ggplot(d, aes(ensemble_score, hit, colour = segment)) +
    geom_segment(aes(x = x0, xend = ensemble_score, yend = hit), linewidth = 0.35) +
    geom_point(aes(size = pmax(lr_means, 0.05)), shape = 16) +
    facet_wrap(~ segment, scales = "free_y", ncol = 2) +
    scale_colour_manual(values = SEG_HEX, guide = "none") +
    scale_size_continuous(range = c(1.1, 3.4), name = "lr_means") +
    labs(title = "Deep perivascular top edges by gut segment",
         subtitle = "PV Res Mac ↔ PAC / PVC / arteriolar endothelium",
         x = "Ensemble score", y = NULL) +
    theme_gca() +
    theme(axis.text.y = element_text(size = 4.8))
  save_fig(p, file.path(OUT_DIR, "fig_lollipop_deep_perivascular_by_segment"),
           180, 140)
}

## ========================================================================
## 3. BEST4 → non-epithelial
## ========================================================================
mk_best4_lolli <- function(sources, tag, stem) {
  b <- dt[source %in% sources & !(target %in% EPI)]
  fwrite(b[, .(segment, source, target, lr_pair, lr_means,
               magnitude_rank, specificity_rank, ensemble_score, is_top)],
         file.path(OUT_DIR, paste0("best4_nonEpi_", tag, "_edges.csv")))
  if (!nrow(b)) return(invisible(NULL))
  best <- b[, .(
    ensemble_score = max(ensemble_score),
    lr_means = max(lr_means),
    best_segment = segment[which.max(ensemble_score)],
    best_source = source[which.max(ensemble_score)],
    n_seg = uniqueN(segment),
    tgt_lin = tgt_lin[which.max(ensemble_score)]
  ), by = .(target, lr_pair)]
  top <- best[order(-ensemble_score)][1:min(TOP_LOLLI, .N)]
  top[, hit := paste0(short_ct(best_source), " → ", short_ct(target),
                      "  |  ", lr_pair)]
  top[, segment := factor(best_segment, levels = SEG_ORDER)]
  lollipop(
    top,
    sprintf("BEST4 (%s) → non-epithelial partners", tag),
    "Outgoing only; epithelial receivers excluded; best segment coloured",
    stem, colour_col = "segment", colour_vals = SEG_HEX, w_mm = 175, h_mm = 105)

  ## by segment
  seg <- b[, .SD[order(-ensemble_score)][1:min(8L, .N)], by = segment]
  seg[, hit := paste0(short_ct(source), " → ", short_ct(target),
                      "  |  ", lr_pair)]
  seg[, segment := factor(segment, levels = SEG_ORDER)]
  d <- copy(seg)
  d[, hit := reorder(hit, ensemble_score)]
  x0 <- max(0, min(d$ensemble_score) * 0.9)
  p <- ggplot(d, aes(ensemble_score, hit, colour = tgt_lin)) +
    geom_segment(aes(x = x0, xend = ensemble_score, yend = hit), linewidth = 0.35) +
    geom_point(aes(size = pmax(lr_means, 0.05)), shape = 16) +
    facet_wrap(~ segment, scales = "free_y", ncol = 2) +
    scale_size_continuous(range = c(1.1, 3.4), name = "lr_means") +
    labs(title = sprintf("BEST4 (%s) → non-epi by segment", tag),
         subtitle = "Colour = receiver lineage",
         x = "Ensemble score", y = NULL, colour = "Receiver lineage") +
    theme_gca() +
    theme(axis.text.y = element_text(size = 4.6))
  save_fig(p, paste0(stem, "_by_segment"), 180, 145)
}

mk_best4_lolli(BEST4_ENT, "Enterocytes",
               file.path(OUT_DIR, "fig_lollipop_BEST4_Enterocytes_nonEpi"))
mk_best4_lolli(BEST4_COL, "Colonocytes",
               file.path(OUT_DIR, "fig_lollipop_BEST4_Colonocytes_nonEpi"))
mk_best4_lolli(BEST4, "combined",
               file.path(OUT_DIR, "fig_lollipop_BEST4_combined_nonEpi"))

## ========================================================================
## 4. CCL19/CCL21 → ADRA2A on BEST4 — sender × segment
## ========================================================================
adra <- dt[ligand_complex %in% c("CCL19", "CCL21") &
             receptor_complex == "ADRA2A" &
             target %in% BEST4]
fwrite(adra[order(segment, magnitude_rank)],
       file.path(OUT_DIR, "CCL19_21_ADRA2A_BEST4_edges.csv"))

if (nrow(adra)) {
  ## summarize: for each segment × ligand × BEST4 subtype, top senders
  adra[, hit := paste0(ligand_complex, " from ", short_ct(source),
                       " → ", short_ct(target))]
  ## keep top senders per segment×ligand (union of both BEST4 targets)
  adra_top <- adra[, .SD[order(-ensemble_score)][1:min(10L, .N)],
                   by = .(segment, ligand_complex)]
  adra_top[, segment := factor(segment, levels = SEG_ORDER)]
  adra_top[, hit := reorder(hit, ensemble_score)]
  x0 <- max(0, min(adra_top$ensemble_score) * 0.9)
  p <- ggplot(adra_top, aes(ensemble_score, hit, colour = ligand_complex)) +
    geom_segment(aes(x = x0, xend = ensemble_score, yend = hit), linewidth = 0.35) +
    geom_point(aes(size = pmax(lr_means, 0.05), shape = target), stroke = 0.2) +
    facet_wrap(~ segment, scales = "free_y", ncol = 2) +
    scale_colour_manual(values = c(CCL19 = "#0072B2", CCL21 = "#E69F00"),
                        name = "Ligand") +
    scale_shape_manual(values = c("BEST4 Enterocytes" = 16,
                                  "BEST4 Colonocytes" = 17),
                       name = "BEST4 subtype",
                       labels = c("Enterocytes", "Colonocytes")) +
    scale_size_continuous(range = c(1.2, 3.6), name = "lr_means") +
    labs(
      title = "Who sends CCL19/CCL21 → ADRA2A on BEST4?",
      subtitle = "Top senders per segment × ligand; shape = BEST4 Enterocytes vs Colonocytes",
      x = "Ensemble score", y = NULL) +
    theme_gca() +
    theme(axis.text.y = element_text(size = 5))
  save_fig(p, file.path(OUT_DIR, "fig_lollipop_CCL19_21_ADRA2A_BEST4_by_segment"),
           180, 150)

  ## compact sender×segment heatmap (max score over BEST4 subtypes)
  heat <- adra[, .(score = max(ensemble_score),
                   lr_means = max(lr_means),
                   best_tgt = target[which.max(ensemble_score)]),
               by = .(segment, ligand_complex, source)]
  heat[, src_lab := short_ct(source)]
  heat[, segment := factor(segment, levels = SEG_ORDER)]
  ## keep senders that appear in top ranks somewhere
  keep_src <- heat[, .(m = max(score)), by = src_lab][order(-m)][1:min(12L, .N)]$src_lab
  heat <- heat[src_lab %in% keep_src]
  heat[, src_lab := factor(src_lab, levels = rev(keep_src))]
  fwrite(heat, file.path(OUT_DIR, "CCL19_21_ADRA2A_BEST4_sender_segment.csv"))
  p_h <- ggplot(heat, aes(segment, src_lab, fill = score)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.1f", score)), size = 1.7) +
    facet_wrap(~ ligand_complex, nrow = 1) +
    scale_fill_gradient(low = "#F2F2F2", high = "#0072B2", name = "Ensemble") +
    labs(title = "CCL19/CCL21 → ADRA2A on BEST4: sender strength by segment",
         subtitle = "Max ensemble score over BEST4 Enterocytes/Colonocytes receivers",
         x = "Gut segment", y = "Sender") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 5.5),
          axis.text.y = element_text(size = 5.5))
  save_fig(p_h, file.path(OUT_DIR, "fig_heatmap_CCL19_21_ADRA2A_BEST4_senders"),
           160, 90)
} else {
  message("No CCL19/CCL21→ADRA2A edges onto BEST4 found.")
}

## ========================================================================
## 5. IL27 → CD4
## ========================================================================
il_rx <- "IL27|EBI3"
il <- dt[grepl(il_rx, ligand_complex) | grepl("IL27", receptor_complex)]
il_cd4 <- il[target %in% CD4]
fwrite(il, file.path(OUT_DIR, "IL27_all_edges.csv"))
fwrite(il_cd4, file.path(OUT_DIR, "IL27_to_CD4_edges.csv"))

note <- c(
  "IL27 status in LIANA rank_aggregate run",
  "--------------------------------------",
  "Consensus resource contains complex interactions:",
  "  EBI3_IL27 -> IL27RA_IL6ST",
  "  IL12A_IL27 -> IL12RB2_IL27RA",
  sprintf("Edges matching IL27/EBI3 in combined table: %d", nrow(il)),
  sprintf("Of which target is a CD4 T subtype: %d", nrow(il_cd4)),
  "",
  "If zero, the complexes were filtered out during LIANA (typically expr_prop:",
  "both subunits of ligand and receptor complexes must pass 10% expression",
  "in the sender/receiver). See companion expression check script output",
  "IL27_expression_note.txt / concept_expression if generated.",
  ""
)
if (nrow(il_cd4)) {
  il_cd4[, hit := paste0(short_ct(source), " → ", short_ct(target),
                         "  |  ", lr_pair)]
  il_top <- il_cd4[, .SD[order(-ensemble_score)][1:min(TOP_LOLLI, .N)]]
  il_top[, segment := factor(segment, levels = SEG_ORDER)]
  lollipop(
    il_top, "IL27-complex → CD4 T cells",
    "Present in rank_aggregate table",
    file.path(OUT_DIR, "fig_lollipop_IL27_to_CD4"),
    colour_col = "segment", colour_vals = SEG_HEX)
  note <- c(note, "IL27→CD4 edges WERE recovered; see fig_lollipop_IL27_to_CD4.*")
} else {
  note <- c(note, "NO IL27→CD4 edges in this table — cannot place on rank curve or lollipop from LIANA output.")
}
writeLines(note, file.path(OUT_DIR, "IL27_LIANA_status.txt"))
message(paste(note, collapse = "\n"))

## ---------- concept edge export (cell-resolved support for highlighted axes)
hl_edges <- dt[lr_pair %in% HL$lr_pair]
hl_edges <- merge(hl_edges, HL[, .(lr_pair, concept, label)], by = "lr_pair",
                  all.x = TRUE)
fwrite(hl_edges[order(concept, magnitude_rank)],
       file.path(OUT_DIR, "highlight_edges_cell_resolved.csv"))

cat("\nDONE → ", OUT_DIR, "\n", sep = "")
