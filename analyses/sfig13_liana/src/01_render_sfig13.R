#!/usr/bin/env Rscript
## Supplementary Figure 13 — niche CCC panels + tissue_level_1 centrality

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(scales)
  library(stringr)
  library(ggrepel)
  library(patchwork)
})

.args <- commandArgs(trailingOnly = FALSE)
.file <- sub("^--file=", "", .args[grepl("^--file=", .args)])
.here <- if (length(.file)) dirname(normalizePath(.file)) else getwd()
ROOT <- Sys.getenv("SFIG13_ROOT", normalizePath(file.path(.here, "..")))
DATA <- file.path(ROOT, "data")
OUT  <- file.path(ROOT, "out")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

SEG_COLS <- c(
  duodenum = "#E9C61D",
  jejunum  = "#E96475",
  ileum    = "#208230",
  colon    = "#3A68AE"
)
SEG_ORDER <- names(SEG_COLS)
LINEAGE_COLS <- c(
  Epithelial   = "#009E73",
  Lymphoid     = "#0072B2",
  Myeloid      = "#D55E00",
  Stromal      = "#999999",
  Endothelial  = "#56B4E9",
  Glial        = "#CC79A7",
  Other        = "#E0E0E0"
)

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", size = base + 0.5, colour = "black"),
      plot.subtitle = element_text(size = base, colour = "black"),
      axis.title = element_text(size = base, colour = "black"),
      axis.text = element_text(size = base, colour = "black"),
      legend.title = element_text(size = base, colour = "black"),
      legend.text = element_text(size = base - 0.5, colour = "black"),
      legend.key.size = unit(2.5, "mm"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = base, colour = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(2, 4, 2, 2)
    )
}

short_ct <- function(x) {
  x |>
    str_replace_all("Perivascular Resident Macrophages", "PV Res Mac") |>
    str_replace_all("Follicle Associated Resident Macrophages", "FARM") |>
    str_replace_all("Homeostatic Macrophages", "Homeo Mac") |>
    str_replace_all("Cycling Macrophages", "Cycling Mac") |>
    str_replace_all("M0 Macrophages", "M0 Mac") |>
    str_replace_all("Post Arteriole Capillary Endothelial \\(PAC\\)", "PAC") |>
    str_replace_all("Pre Venule Capillary Endothelial \\(PVC\\)", "PVC") |>
    str_replace_all("Arteriolar Endothelial", "Arter Endo") |>
    str_replace_all("Venular Endothelial", "Venular Endo") |>
    str_replace_all("Capillary Endothelial", "Cap Endo") |>
    str_replace_all("Lymphatic Endothelial", "Lymph Endo") |>
    str_replace_all("Medullary Sinus Endothelial", "Med Sinus Endo") |>
    str_replace_all("Lamina propria Fibroblasts \\(S1\\)", "LP Fibro (S1)") |>
    str_replace_all("Submucosal Fibroblasts \\(S3\\)", "Submuc Fibro (S3)") |>
    str_replace_all("Crypt Top Fibroblasts \\(S2B\\)", "Crypt Top Fibro") |>
    str_replace_all("Fibroblastic Reticular Cells \\(FRC\\)", "FRC") |>
    str_replace_all("Follicular Dendritic Cells \\(fDC\\)", "fDC") |>
    str_replace_all("Mesenchymal Lymphoid Tissue Organizer Cells \\(mLTo Cells\\)", "mLTo") |>
    str_replace_all("Marginal Reticular Cells \\(MRC\\)", "MRC") |>
    str_replace_all("Transiently Amplifying Cells \\(TA\\)", "TA") |>
    str_replace_all("Intestinal Stem Cells \\(ISC\\)", "ISC") |>
    str_replace_all("Enterocyte Progenitors", "Ent Prog") |>
    str_replace_all("Lower Villus Enterocytes", "Low Villus Ent") |>
    str_replace_all("Mid Villus Enterocytes", "Mid Villus Ent") |>
    str_replace_all("Mature Goblet Cells", "Mature Goblet") |>
    str_replace_all("Goblet Cells", "Goblet") |>
    str_replace_all("CD8 Circulating Effector Memory", "CD8 circ EM") |>
    str_replace_all("CD8 Effector Memory", "CD8 EM") |>
    str_replace_all("CD8 Memory Exhausted", "CD8 Mem Exh") |>
    str_replace_all("Gamma Delta T Cells", "γδ T") |>
    str_replace_all("NKT Cells", "NKT") |>
    str_replace_all("NK Cells", "NK") |>
    str_replace_all("Classical Monocytes", "cMono") |>
    str_replace_all("Monocyte Derived Dendritic Cells", "moDC") |>
    str_replace_all("Tuft Progenitors", "Tuft Prog") |>
    str_replace_all("Secretory Progenitors", "Sec Prog") |>
    str_replace_all("BEST4 Enterocytes", "BEST4 Ent") |>
    str_replace_all("BEST4 Colonocytes", "BEST4 Col") |>
    str_replace_all("Adipocytes", "Adipocytes")
}

pretty_lr <- function(x) str_replace(x, "->", " → ")

save_panel <- function(p, stem, w_mm, h_mm) {
  wi <- w_mm / 25.4
  hi <- h_mm / 25.4
  ggsave(paste0(stem, ".pdf"), p, width = wi, height = hi, device = grDevices::cairo_pdf)
  ggsave(paste0(stem, ".svg"), p, width = wi, height = hi, device = svglite::svglite)
  ggsave(paste0(stem, ".png"), p, width = wi, height = hi, dpi = 300)
  message("Wrote ", stem)
}

## --------------------------------------------------------------------------
## a — compact centrality bump (tissue_level_1)
## --------------------------------------------------------------------------
nc <- read_csv(file.path(DATA, "ccc_node_centrality_by_segment.csv"), show_col_types = FALSE) |>
  mutate(
    segment = factor(ccc_segment_std, levels = SEG_ORDER),
    lineage = if_else(lineage %in% names(LINEAGE_COLS), lineage, "Other")
  ) |>
  filter(!is.na(segment), cell_state != "Epithelial")

## Within-segment percentile rank (1 = top hub)
nc <- nc |>
  group_by(segment) |>
  mutate(
    centrality_rank_pct = percent_rank(desc(total_strength)) * 100,
    share = total_strength / sum(total_strength, na.rm = TRUE)
  ) |>
  ungroup()

## Union of top-6 hubs per segment by share — compact, legible
top_states <- nc |>
  group_by(segment) |>
  slice_max(order_by = share, n = 6, with_ties = FALSE) |>
  ungroup() |>
  distinct(cell_state) |>
  pull(cell_state)

d_a <- nc |>
  filter(cell_state %in% top_states) |>
  mutate(lab = short_ct(cell_state))

## End labels only (first and last observed segment per cell state)
lab_df <- bind_rows(
  d_a |> group_by(cell_state) |> slice_min(as.integer(segment), n = 1, with_ties = FALSE),
  d_a |> group_by(cell_state) |> slice_max(as.integer(segment), n = 1, with_ties = FALSE)
) |>
  ungroup() |>
  distinct(cell_state, segment, .keep_all = TRUE)

p_a <- ggplot(d_a, aes(segment, centrality_rank_pct,
                       group = cell_state, colour = lineage)) +
  geom_line(linewidth = 0.45, alpha = 0.9) +
  geom_point(size = 1.35) +
  ggrepel::geom_text_repel(
    data = lab_df,
    aes(label = lab),
    size = 1.7,
    family = "Helvetica",
    colour = "black",
    max.overlaps = 40,
    box.padding = 0.12,
    point.padding = 0.05,
    min.segment.length = 0,
    segment.size = 0.2,
    segment.colour = "grey50",
    seed = 1
  ) +
  scale_y_reverse(expand = expansion(mult = c(0.02, 0.08))) +
  scale_colour_manual(values = LINEAGE_COLS, name = "Lineage") +
  labs(
    title = "Centrality rank shifts for top-6 hubs across gut segments",
    x = NULL,
    y = "Within-segment percentile rank (1 = top)"
  ) +
  theme_gca(6) +
  theme(legend.position = "right")

save_panel(p_a, file.path(OUT, "sfig13_a_centrality_bump_tissue_level_1"), 180, 85)

## --------------------------------------------------------------------------
## b — ACKR1 sink (PVC / venular)
## --------------------------------------------------------------------------
b <- read_csv(file.path(DATA, "panel_b_ackr1_sink_top.csv"), show_col_types = FALSE) |>
  mutate(
    segment = factor(tissue_level_1, levels = SEG_ORDER),
    y = paste0(short_ct(source), " → ", short_ct(target), " | ", pretty_lr(lr_pair)),
    y = reorder(y, ensemble_score)
  )

p_b <- ggplot(b, aes(ensemble_score, y, colour = segment)) +
  geom_segment(aes(x = min(ensemble_score) - 0.15, xend = ensemble_score,
                   yend = y), linewidth = 0.35) +
  geom_point(aes(size = lr_means)) +
  scale_colour_manual(values = SEG_COLS, drop = FALSE, name = "Best segment") +
  scale_size_continuous(range = c(1.2, 3.2), name = "lr_means") +
  labs(
    title = "Endothelial ACKR1 sink (PVC / venular)",
    x = "Ensemble score (−log10 magnitude rank)",
    y = NULL
  ) +
  theme_gca(6) +
  theme(legend.position = "right")

save_panel(p_b, file.path(OUT, "sfig13_b_ackr1_sink_pvc_venular"), 180, 95)

## Also a deep-PV mac↔endo companion (curated concept-pack edges)
## Ban unsupported C1q–CD93 (and other sticky hits aligned with panel d)
BAN_LR_DP <- c(
  "C1QA->CD93", "C1QB->CD93", "C1QC->CD93",
  "APP->CD74", "RPS19->C5AR1", "B2M->KLRD1", "VIM->CD44",
  ## intracellular / MHC sticky after C1q–CD93 removal
  "GNAI2->CAV1", "HLA-A->APLP2", "HLA-B->APLP2", "HLA-C->APLP2"
)
dp <- read_csv(file.path(DATA, "deep_perivascular_edges.csv"), show_col_types = FALSE) |>
  filter(is_top, !(lr_pair %in% BAN_LR_DP)) |>
  mutate(
    segment = factor(segment, levels = SEG_ORDER),
    y = paste0(short_ct(source), " → ", short_ct(target), " | ", pretty_lr(lr_pair))
  )
dp_top <- dp |>
  arrange(desc(ensemble_score)) |>
  distinct(source, target, lr_pair, .keep_all = TRUE) |>
  slice_head(n = 14) |>
  mutate(y = reorder(y, ensemble_score))

p_b2 <- ggplot(dp_top, aes(ensemble_score, y, colour = segment)) +
  geom_segment(aes(x = min(ensemble_score) - 0.1, xend = ensemble_score,
                   yend = y), linewidth = 0.35) +
  geom_point(aes(size = lr_means)) +
  scale_colour_manual(values = SEG_COLS, drop = FALSE, name = "Best segment") +
  scale_size_continuous(range = c(1.2, 3.2), name = "lr_means") +
  labs(
    title = "Deep perivascular wiring (PV Res Mac ↔ deep endothelia)",
    x = "Ensemble score (−log10 magnitude rank)",
    y = NULL
  ) +
  theme_gca(6) +
  theme(legend.position = "right")

save_panel(p_b2, file.path(OUT, "sfig13_b2_deep_perivascular_pv_mac"), 180, 95)

## --------------------------------------------------------------------------
## c — Lymphatic / sinus CCL21 → CCR7
## --------------------------------------------------------------------------
c <- read_csv(file.path(DATA, "panel_c_lymphatic_ccl21_ccr7_top.csv"), show_col_types = FALSE) |>
  mutate(
    segment = factor(tissue_level_1, levels = SEG_ORDER),
    y = paste0(short_ct(source), " → ", short_ct(target)),
    y = reorder(y, ensemble_score)
  )

p_c <- ggplot(c, aes(ensemble_score, y, colour = segment)) +
  geom_segment(aes(x = min(ensemble_score) - 0.15, xend = ensemble_score,
                   yend = y), linewidth = 0.35) +
  geom_point(aes(size = lr_means)) +
  scale_colour_manual(values = SEG_COLS, drop = FALSE, name = "Best segment") +
  scale_size_continuous(range = c(1.2, 3.2), name = "lr_means") +
  labs(
    title = "Lymphatic / medullary sinus → CCR7",
    x = "Ensemble score (−log10 magnitude rank)",
    y = NULL
  ) +
  theme_gca(6) +
  theme(legend.position = "right")

save_panel(p_c, file.path(OUT, "sfig13_c_lymphatic_ccl21_ccr7"), 140, 85)

## --------------------------------------------------------------------------
## d — Macrophage subtype niche contrast (PV vs FARM)
## --------------------------------------------------------------------------
d <- read_csv(file.path(DATA, "panel_d_mac_subtype_niche_top.csv"), show_col_types = FALSE) |>
  mutate(
    focus_lab = factor(short_ct(focus), levels = c("PV Res Mac", "FARM")),
    segment = factor(tissue_level_1, levels = SEG_ORDER),
    y_raw = paste0(
      if_else(direction == "outgoing", "→ ", "← "),
      short_ct(partner), " | ", pretty_lr(lr_pair)
    ),
    ## Facet-safe ordering: unique y per focus
    y = paste0(as.character(focus_lab), "___", y_raw)
  ) |>
  arrange(focus_lab, ensemble_score) |>
  mutate(y = factor(y, levels = unique(y)))

## Pretty strip labels via labeller on focus_lab; strip y back for display
y_labs <- setNames(
  sub("^.*?___", "", levels(d$y)),
  levels(d$y)
)

p_d <- ggplot(d, aes(ensemble_score, y, colour = segment)) +
  geom_segment(aes(x = min(ensemble_score) - 0.1, xend = ensemble_score,
                   yend = y), linewidth = 0.35) +
  geom_point(aes(size = lr_means, shape = direction)) +
  facet_wrap(~ focus_lab, scales = "free_y", ncol = 2) +
  scale_y_discrete(labels = y_labs) +
  scale_colour_manual(values = SEG_COLS, drop = FALSE, name = "Best segment") +
  scale_size_continuous(range = c(1.1, 3.0), name = "lr_means") +
  scale_shape_manual(values = c(outgoing = 16, incoming = 17), name = "Direction") +
  labs(
    title = "High-resolution macrophage niches",
    x = "Ensemble score (−log10 magnitude rank)",
    y = NULL
  ) +
  theme_gca(6) +
  theme(legend.position = "bottom", legend.box = "horizontal")

save_panel(p_d, file.path(OUT, "sfig13_d_macrophage_subtype_niches"), 180, 95)

## --------------------------------------------------------------------------
## e — Follicle exploratory L–R (nominal; not FDR)
## --------------------------------------------------------------------------
ex <- read_csv(file.path(DATA, "exploratory_candidates.csv"), show_col_types = FALSE) |>
  filter(lr_pair %in% c("CXCL13->CXCR5", "LGALS3->LAG3"), status == "ok") |>
  mutate(
    y = paste0(short_ct(source), " → ", short_ct(target), " | ", pretty_lr(lr_pair)),
    y = reorder(y, beta),
    sig = if_else(p < 0.001, "p < 0.001", if_else(p < 0.01, "p < 0.01", "p < 0.05"))
  )

## Keep top by |beta| within each LR (max 6 LGALS3, all CXCL13)
ex_plot <- bind_rows(
  ex |> filter(lr_pair == "CXCL13->CXCR5"),
  ex |> filter(lr_pair == "LGALS3->LAG3") |>
    arrange(desc(abs(beta))) |>
    slice_head(n = 6)
) |>
  mutate(y = reorder(y, beta))

p_e <- ggplot(ex_plot, aes(beta, y, colour = lr_pair)) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey60") +
  geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi),
                orientation = "y", width = 0.25, linewidth = 0.35) +
  geom_point(size = 1.8) +
  scale_colour_manual(
    values = c("CXCL13->CXCR5" = "#0072B2", "LGALS3->LAG3" = "#D55E00"),
    labels = c("CXCL13 → CXCR5", "LGALS3 → LAG3"),
    name = "L–R"
  ) +
  labs(
    title = "Follicle-associated L–R (exploratory)",
    x = "β (follicle+ vs follicle−)",
    y = NULL
  ) +
  theme_gca(6) +
  theme(legend.position = "right")

save_panel(p_e, file.path(OUT, "sfig13_e_follicle_exploratory_lr"), 160, 85)

## --------------------------------------------------------------------------
## Optional assembled preview (stacked; Illustrator will do final layout)
## --------------------------------------------------------------------------
assembled <- (p_a / p_b / p_c / p_d / p_e) +
  plot_layout(heights = c(1.05, 1.15, 0.95, 1.15, 1.0)) +
  plot_annotation(
    title = "Supplemental Figure 12. Niche-associated communication motifs (exploratory)",
    theme = theme(
      plot.title = element_text(face = "bold", size = 7, family = "Helvetica")
    )
  )

save_panel(assembled, file.path(OUT, "sfig13_assembled_preview"), 180, 280)

message("Done. Panels in ", OUT)
