#!/usr/bin/env Rscript
# Revised Fig 3b / 3c with CIs and null-standardized effects.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(file_arg)) normalizePath(file.path(dirname(file_arg), "..")) else getwd()
tables <- file.path(root, "tables")
figdir <- file.path(root, "figures")
dir.create(figdir, showWarnings = FALSE, recursive = TRUE)

PRETTY <- c(
  sampled_site_condition = "Sample condition",
  radial_tissue_term = "Radial layer",
  sample_preservation_method = "Preservation",
  sex_ontology_term = "Sex",
  age_range = "Age",
  dataset_id = "Study / batch",
  assay = "Assay",
  sample_collection_method = "Biopsy vs resection",
  sequenced_fragment = "Sequenced fragment",
  gene_annotation_version = "Gene annotation",
  tissue_level_1 = "Gut segment"
)

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      axis.line = element_line(linewidth = 0.4),
      axis.ticks = element_line(linewidth = 0.4),
      plot.title = element_text(size = base + 0.5, face = "plain"),
      legend.key.size = unit(2.5, "mm"),
      panel.grid = element_blank()
    )
}

save_fig <- function(p, stem, w_mm, h_mm) {
  w <- w_mm / 25.4; h <- h_mm / 25.4
  ggsave(file.path(figdir, paste0(stem, ".pdf")), p, width = w, height = h, device = cairo_pdf)
  ggsave(file.path(figdir, paste0(stem, ".svg")), p, width = w, height = h)
  ggsave(file.path(figdir, paste0(stem, ".png")), p, width = w, height = h, dpi = 300)
}

# ---- Fig 3b revised: all cell types as points; top-2 highlighted with p ----
comp <- read_csv(file.path(tables, "composition_celltype_estimates.csv"), show_col_types = FALSE)
# Move radial layer into specimen/technical block (Task 9 choice)
comp <- comp %>%
  mutate(
    block_rev = case_when(
      covariate %in% c("radial_tissue_term", "sample_collection_method", "dataset_id",
                       "assay", "sequenced_fragment", "gene_annotation_version") ~ "Specimen / technical",
      covariate == "tissue_level_1" ~ "Anatomy",
      TRUE ~ "Biological"
    ),
    cov_lab = factor(unname(PRETTY[covariate]), levels = unname(PRETTY[names(PRETTY)]))
  )

top2 <- comp %>%
  group_by(covariate) %>%
  slice_max(partial_r2, n = 2, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(label = sprintf("%s\nω²=%.2f; p=%.3f", celltype, omega2_trunc, empirical_p))

p_b <- ggplot(comp, aes(cov_lab, omega2_trunc)) +
  geom_jitter(aes(colour = null_z), width = 0.15, height = 0, size = 0.7, alpha = 0.55) +
  geom_point(data = top2, aes(cov_lab, omega2_trunc), shape = 21, size = 2.2,
             fill = "white", colour = "black", stroke = 0.4) +
  geom_text(data = top2, aes(cov_lab, omega2_trunc, label = sprintf("%.2f\np=%.3f", omega2_trunc, empirical_p)),
            size = 1.6, vjust = -0.6, family = "Helvetica") +
  facet_grid(~ block_rev, scales = "free_x", space = "free_x") +
  scale_colour_gradientn(
    colours = c("#FFFFFF", "#F0E442", "#E69F00", "#D55E00"),
    name = "Null-standardised\neffect (z)"
  ) +
  labs(
    title = "Composition ω² per cell type (permutation p for published top-2)",
    x = NULL, y = expression("One-way " * omega^2 * " (truncated)")
  ) +
  theme_gca(5.5) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, size = 5))

save_fig(p_b, "fig3b_revised", 180, 90)

# Heatmap-style top-2 with p annotations (closer to original panel)
hm <- top2 %>%
  mutate(
    cell_lab = celltype,
    cov_lab = factor(unname(PRETTY[covariate]), levels = unique(unname(PRETTY[covariate])))
  )
# assignment order like published
hm_rows <- c()
for (cv in unique(as.character(hm$covariate))) {
  cts <- hm %>% filter(covariate == cv) %>% pull(celltype)
  for (ct in cts) if (!(ct %in% hm_rows)) hm_rows <- c(hm_rows, ct)
}
mat <- comp %>%
  filter(celltype %in% hm_rows, covariate %in% unique(hm$covariate)) %>%
  mutate(
    celltype = factor(celltype, levels = rev(hm_rows)),
    cov_lab = factor(unname(PRETTY[covariate]), levels = unname(PRETTY[unique(hm$covariate)]))
  )

p_b2 <- ggplot(mat, aes(cov_lab, celltype, fill = omega2_trunc)) +
  geom_tile(colour = "white", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.2f\nq=%.2f", omega2_trunc, fdr_q)), size = 1.7, family = "Helvetica") +
  scale_fill_gradientn(
    colours = c("#FFFFFF", "#F0E442", "#E69F00", "#D55E00"),
    limits = c(0, 0.6), oob = scales::squish,
    name = expression(omega^2)
  ) +
  labs(
    title = "Fig 3b revised: top-2 cell types with FDR q from permutation nulls",
    x = NULL, y = NULL
  ) +
  theme_gca(5.5) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1))

save_fig(p_b2, "fig3b_revised_heatmap", 180, 70)

# ---- Fig 3c revised: PCR ranking with study bootstrap CIs ----
comp_pcr <- read_csv(file.path(tables, "composition_pcr_lineage.csv"), show_col_types = FALSE)
boot_c <- tryCatch(
  read_csv(file.path(tables, "composition_pcr_study_bootstrap.csv"), show_col_types = FALSE),
  error = function(e) NULL
)
expr_lin <- tryCatch(
  read_csv(file.path(tables, "expression_pcr_lineage.csv"), show_col_types = FALSE),
  error = function(e) NULL
)
mixed_c <- tryCatch(
  read_csv(file.path(tables, "composition_mixed_study_bootstrap.csv"), show_col_types = FALSE),
  error = function(e) NULL
)
mixed_e <- tryCatch(
  read_csv(file.path(tables, "expression_mixed_study_bootstrap.csv"), show_col_types = FALSE),
  error = function(e) NULL
)

# Pooled PCR means
pool_comp <- comp_pcr %>%
  group_by(covariate) %>%
  summarise(composition = mean(omega2_trunc, na.rm = TRUE), .groups = "drop")
if (!is.null(expr_lin)) {
  pool_expr <- expr_lin %>%
    group_by(covariate) %>%
    summarise(expression = mean(pcr_weighted, na.rm = TRUE), .groups = "drop")
} else {
  pool_expr <- tibble(covariate = character(), expression = double())
}
wide <- full_join(pool_comp, pool_expr, by = "covariate") %>%
  filter(is.finite(composition) | is.finite(expression)) %>%
  mutate(
    cov_lab = factor(unname(PRETTY[covariate]), levels = unname(PRETTY[names(PRETTY)]))
  )

# attach bootstrap CIs if available (composition PCR study boot)
if (!is.null(boot_c)) {
  bpool <- boot_c %>%
    group_by(covariate) %>%
    summarise(boot_lo = mean(boot_lo, na.rm = TRUE), boot_hi = mean(boot_hi, na.rm = TRUE), .groups = "drop")
  wide <- left_join(wide, bpool, by = "covariate")
} else {
  wide$boot_lo <- NA; wide$boot_hi <- NA
}

ord <- wide %>% arrange(expression) %>% pull(cov_lab)
wide <- wide %>% mutate(cov_lab = factor(cov_lab, levels = ord))
long <- wide %>%
  pivot_longer(c(composition, expression), names_to = "modality", values_to = "pcr")

p_c <- ggplot(wide, aes(y = cov_lab)) +
  geom_segment(aes(x = composition, xend = expression, yend = cov_lab), colour = "grey70", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = boot_lo, xmax = boot_hi, y = cov_lab), height = 0.2, colour = "#0072B2", linewidth = 0.3) +
  geom_point(data = long, aes(x = pcr, fill = modality, colour = modality), shape = 21, size = 1.6, stroke = 0.3) +
  scale_fill_manual(values = c(composition = "white", expression = "#D55E00")) +
  scale_colour_manual(values = c(composition = "#0072B2", expression = "black")) +
  labs(
    title = "Fig 3c revised: variance explained with study-bootstrap CIs (composition)",
    x = expression("Variance explained (weighted " * omega^2 * ")"), y = NULL
  ) +
  theme_gca(5.5) +
  theme(legend.position = "bottom")

save_fig(p_c, "fig3c_revised", 120, 90)

# Anatomy-to-study ratio with CIs from mixed or PCR bootstrap
ratio_rows <- list()
if (!is.null(mixed_c) && nrow(mixed_c)) {
  for (lin in unique(mixed_c$lineage)) {
    a <- mixed_c %>% filter(lineage == lin, covariate == "tissue_level_1")
    s <- mixed_c %>% filter(lineage == lin, covariate == "dataset_id")
    # dataset may be absent as fixed; fall back to PCR boot
  }
}
# Use composition/expression PCR lineage tables for ratio + bootstrap from composition_pcr_study_bootstrap
if (!is.null(boot_c)) {
  # need dataset_id and tissue_level_1 boots; compute ratio of bootstrap means with CI via bootstrap of ratio
  # Recompute from lineage PCR point estimates + CI propagation as ratio of boots when both present
  for (mod_name in c("composition", "expression")) {
    if (mod_name == "composition") {
      src <- comp_pcr %>% rename(pcr = omega2_trunc)
    } else if (!is.null(expr_lin)) {
      src <- expr_lin %>% rename(pcr = pcr_weighted)
    } else next
    lin_r <- src %>%
      filter(covariate %in% c("dataset_id", "tissue_level_1")) %>%
      select(lineage, covariate, pcr) %>%
      pivot_wider(names_from = covariate, values_from = pcr) %>%
      mutate(ratio = tissue_level_1 / dataset_id, modality = mod_name)
    ratio_rows[[length(ratio_rows) + 1]] <- lin_r
  }
}
ratios <- bind_rows(ratio_rows)
if (nrow(ratios)) {
  # attach composition bootstrap CI for ratio if we can join boot table
  if (!is.null(boot_c)) {
    # approximate CI: for each lineage, ratio of boot means not available jointly; show point only + pooled
  }
  pooled <- ratios %>%
    group_by(modality) %>%
    summarise(
      tissue_level_1 = mean(tissue_level_1, na.rm = TRUE),
      dataset_id = mean(dataset_id, na.rm = TRUE),
      ratio = tissue_level_1 / dataset_id,
      .groups = "drop"
    )
  p_r <- ggplot(ratios, aes(ratio, modality, fill = lineage)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey60", linewidth = 0.3) +
    geom_point(shape = 21, size = 2, colour = "black", stroke = 0.25,
               position = position_dodge(width = 0.35)) +
    geom_point(data = pooled, aes(ratio, modality), inherit.aes = FALSE,
               shape = 23, size = 2.4, fill = "black") +
    geom_text(data = pooled, aes(ratio, modality, label = sprintf("%.2f", ratio)),
              inherit.aes = FALSE, nudge_y = -0.22, size = 2, fontface = "bold") +
    scale_fill_manual(values = c(epithelial = "#009E73", lymphoid = "#0072B2",
                                 myeloid = "#E69F00", stroma = "#CC79A7")) +
    labs(
      title = "Anatomy-to-study ratio (revised; lineage points + pooled diamond)",
      x = expression(omega^2 * " gut segment / " * omega^2 * " study"), y = NULL
    ) +
    theme_gca(5.5) +
    theme(legend.position = "bottom")
  save_fig(p_r, "fig3c_revised_ratio", 90, 55)
  write_csv(ratios, file.path(tables, "task7_anatomy_study_ratio.csv"))
  write_csv(pooled, file.path(tables, "task7_anatomy_study_ratio_pooled.csv"))
}

# Mixed-model ranking panel if available
if (!is.null(mixed_c)) {
  mc <- mixed_c %>%
    mutate(cov_lab = factor(unname(PRETTY[covariate]), levels = unname(PRETTY[names(PRETTY)])))
  p_m <- ggplot(mc, aes(fixed_frac, cov_lab, colour = lineage)) +
    geom_errorbarh(aes(xmin = boot_lo, xmax = boot_hi), height = 0.2, position = position_dodge(0.6)) +
    geom_point(position = position_dodge(0.6), size = 1.3) +
    scale_colour_manual(values = c(epithelial = "#009E73", lymphoid = "#0072B2",
                                   myeloid = "#E69F00", stroma = "#CC79A7")) +
    labs(
      title = "Composition fixed-effect variance fraction (mixed model, study bootstrap CI)",
      x = "Fixed-effect variance fraction", y = NULL
    ) +
    theme_gca(5.5) +
    theme(legend.position = "bottom")
  save_fig(p_m, "fig3c_revised_mixed", 120, 90)
}

message("Wrote figures to ", figdir)
