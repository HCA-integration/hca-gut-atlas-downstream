#!/usr/bin/env Rscript
# Bar plots of follicle (GC B ≥5) capture across associated covariates.
# Wong / Nature colourblind-safe palettes; composite of the strongest hits.
#
# Inputs:
#   follicle_covariate_plot_rates.csv
#   follicle_covariate_fisher_by_segment.csv
#   follicle_covariate_screen.csv
#   follicle_capture_by_study.csv

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(tidyr); library(patchwork); library(svglite); library(scales)
  library(forcats)
})

HERE <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) "."
)
if (length(HERE) == 0 || HERE == "") HERE <- "."
DATA <- file.path(HERE, "..", "data")
OUT  <- file.path(HERE, "..", "out")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
MM <- 25.4
PT <- 2.845

# Distinct Wong bi-/multi-colour sets per panel (avoid biopsy sky/orange clash
# where possible; site uses green/purple already established).
COL_SITE <- c("Healthy" = "#009E73", "Disease-adjacent" = "#CC79A7")
COL_COLL <- c("Biopsy" = "#56B4E9", "Resection" = "#E69F00")
COL_FRAC <- c("Unfractionated" = "#0072B2", "Fractionated" = "#D55E00")
COL_SEG  <- c("ileum" = "#0072B2", "colon" = "#E69F00")
COL_RAD  <- c(
  "EPI" = "#56B4E9",
  "LP" = "#009E73",
  "EPI+LP" = "#E69F00",
  "Full thickness" = "#D55E00",
  "Muscle wall" = "#CC79A7"
)
# Age: sequential blue→vermillion through Wong-ish intermediates
COL_AGE <- c(
  "0-9" = "#0072B2", "10-19" = "#56B4E9", "20-29" = "#009E73",
  "30-39" = "#F0E442", "40-49" = "#E69F00", "50-59" = "#D55E00",
  "60-69" = "#CC79A7", "70-79" = "#000000", "80-89" = "#999999"
)

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 6.5, hjust = 0, face = "plain",
                               margin = margin(b = 1, t = 0)),
      axis.line = element_line(colour = "black", linewidth = 0.25),
      axis.ticks = element_line(colour = "black", linewidth = 0.25),
      axis.ticks.length = unit(0.55, "mm"),
      axis.text = element_text(colour = "black", size = 5.5),
      axis.title = element_text(colour = "black", size = 6),
      panel.grid = element_blank(),
      legend.position = "none",
      strip.background = element_blank(),
      strip.text = element_text(size = 6, colour = "black",
                               margin = margin(b = 0.5, t = 0)),
      panel.spacing = unit(1.2, "mm"),
      plot.margin = margin(1, 2, 1, 1, "pt")
    )
}

save_panel <- function(p, stem, w_mm, h_mm) {
  wi <- w_mm / MM; hi <- h_mm / MM
  ggsave(file.path(OUT, paste0(stem, ".pdf")), p, width = wi, height = hi,
         device = cairo_pdf, bg = "transparent")
  ggsave(file.path(OUT, paste0(stem, ".svg")), p, width = wi, height = hi,
         device = svglite, bg = "transparent")
  ggsave(file.path(OUT, paste0(stem, ".png")), p, width = wi, height = hi,
         dpi = 300, bg = "white")
  message("wrote ", stem, " (", w_mm, "×", h_mm, " mm)")
}

p_to_stars <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 1e-4 ~ "****",
    p < 1e-3 ~ "***",
    p < 1e-2 ~ "**",
    p < 5e-2 ~ "*",
    TRUE ~ "n.s."
  )
}

format_p <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 1e-4 ~ "P < 0.0001",
    p < 1e-3 ~ sprintf("P = %.1e", p),
    TRUE ~ sprintf("P = %.3g", p)
  )
}

rates_all <- read_csv(file.path(DATA, "follicle_covariate_plot_rates.csv"),
                      show_col_types = FALSE)
fish_seg <- read_csv(file.path(DATA, "follicle_covariate_fisher_by_segment.csv"),
                     show_col_types = FALSE)
screen <- read_csv(file.path(DATA, "follicle_covariate_screen.csv"),
                   show_col_types = FALSE)

ann_for <- function(cov) {
  fish_seg %>%
    filter(covariate == cov) %>%
    mutate(
      scope = segment,
      stars = p_to_stars(p_adj),
      lab = paste0(stars, " ", format_p(p_adj))
    )
}

screen_lab <- function(cov) {
  r <- screen %>% filter(covariate == !!cov)
  if (nrow(r) == 0) return("")
  sprintf("pooled FDR=%s, V=%.2f", format_p(r$p_adj[1]), r$cramers_v[1])
}

# Generic dodged/faceted bars for binary covariates by segment
make_binary_by_segment <- function(cov, fill_vals, title, level_order) {
  d <- rates_all %>%
    filter(covariate == cov, scope %in% c("ileum", "colon")) %>%
    mutate(
      scope = factor(scope, levels = c("ileum", "colon")),
      level_label = factor(level_label, levels = level_order),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * rate, n_pos, n)
    )
  ann <- ann_for(cov) %>%
    mutate(scope = factor(scope, levels = c("ileum", "colon")))
  ymax <- max(d$ci_hi, na.rm = TRUE) * 1.32

  ggplot(d, aes(level_label, rate, fill = level_label)) +
    geom_col(width = 0.72, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.14, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.12, size = 5 / PT, family = "Helvetica",
              lineheight = 0.85) +
    geom_text(
      data = ann, aes(x = 1.5, y = ymax * 0.96, label = lab),
      inherit.aes = FALSE, size = 5.2 / PT, family = "Helvetica",
      fontface = "bold", vjust = 1
    ) +
    facet_wrap(~ scope, nrow = 1) +
    scale_fill_manual(values = fill_vals, drop = FALSE) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = title, x = NULL, y = "GC B detection") +
    theme_gca() +
    theme(axis.text.x = element_text(size = 5, angle = 25, hjust = 1))
}

make_radial <- function() {
  d <- rates_all %>%
    filter(covariate == "radial_tissue_term", scope == "pooled") %>%
    mutate(
      level_label = factor(
        level_label,
        levels = c("EPI", "LP", "EPI+LP", "Full thickness", "Muscle wall")
      ),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * rate, n_pos, n)
    )
  ymax <- max(d$ci_hi, na.rm = TRUE) * 1.28
  ttl <- paste0(
    "Radial layer (", screen_lab("radial_tissue_term"), ")"
  )
  ggplot(d, aes(level_label, rate, fill = level_label)) +
    geom_col(width = 0.75, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.14, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.12, size = 5 / PT, family = "Helvetica",
              lineheight = 0.85) +
    scale_fill_manual(values = COL_RAD, drop = FALSE) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = ttl, x = NULL, y = "GC B detection") +
    theme_gca() +
    theme(axis.text.x = element_text(size = 5, angle = 30, hjust = 1))
}

make_age <- function() {
  age_lvls <- c("0-9", "10-19", "20-29", "30-39", "40-49",
                "50-59", "60-69", "70-79", "80-89")
  d <- rates_all %>%
    filter(covariate == "age_range", scope == "pooled") %>%
    mutate(
      level_label = factor(level_label, levels = age_lvls),
      lab = sprintf("%.0f%%\n(%d)", 100 * rate, n)
    ) %>%
    filter(!is.na(level_label))
  ymax <- max(d$ci_hi, na.rm = TRUE) * 1.22
  ttl <- paste0("Age (", screen_lab("age_range"), ")")
  ggplot(d, aes(level_label, rate, fill = level_label)) +
    geom_col(width = 0.8, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.12, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.1, size = 4.5 / PT, family = "Helvetica",
              lineheight = 0.85) +
    scale_fill_manual(values = COL_AGE, drop = FALSE) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = ttl, x = NULL, y = "GC B detection") +
    theme_gca() +
    theme(axis.text.x = element_text(size = 5, angle = 35, hjust = 1))
}

make_segment <- function() {
  d <- rates_all %>%
    filter(covariate == "tissue_level_1", scope == "pooled") %>%
    mutate(
      level_label = factor(tolower(level_label), levels = c("ileum", "colon")),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * rate, n_pos, n)
    )
  ymax <- max(d$ci_hi, na.rm = TRUE) * 1.28
  ttl <- paste0("Gut segment (", screen_lab("tissue_level_1"), ")")
  ggplot(d, aes(level_label, rate, fill = level_label)) +
    geom_col(width = 0.7, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.14, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.12, size = 5 / PT, family = "Helvetica",
              lineheight = 0.85) +
    scale_fill_manual(values = COL_SEG, drop = FALSE) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = ttl, x = NULL, y = "GC B detection") +
    theme_gca()
}

make_study <- function(top_n = 8) {
  d <- read_csv(file.path(DATA, "follicle_capture_by_study.csv"),
                show_col_types = FALSE) %>%
    arrange(desc(rate)) %>%
    slice_head(n = top_n) %>%
    mutate(
      level = fct_reorder(level, rate),
      lab = sprintf("%.0f%% (%d/%d)", 100 * rate, n_pos, n)
    )
  ttl <- paste0("Study (≥15 samples; ", screen_lab("dataset_id"), ")")
  ggplot(d, aes(rate, level)) +
    geom_col(fill = "#0072B2", colour = "black", linewidth = 0.2, width = 0.75) +
    geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.2, linewidth = 0.25) +
    geom_text(aes(label = lab), hjust = -0.05, size = 5 / PT, family = "Helvetica") +
    scale_x_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, max(d$ci_hi, na.rm = TRUE) * 1.35),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = ttl, x = "GC B detection", y = NULL) +
    theme_gca()
}

render <- function() {
  p_site <- make_binary_by_segment(
    "sampled_site_condition", COL_SITE,
    paste0("Site condition (", screen_lab("sampled_site_condition"), ")"),
    c("Healthy", "Disease-adjacent")
  )
  p_coll <- make_binary_by_segment(
    "sample_collection_method", COL_COLL,
    paste0("Collection (", screen_lab("sample_collection_method"), ")"),
    c("Biopsy", "Resection")
  )
  p_frac <- make_binary_by_segment(
    "chemical_fractionation", COL_FRAC,
    paste0("Chemical fractionation (", screen_lab("chemical_fractionation"), ")"),
    c("Unfractionated", "Fractionated")
  )
  p_rad <- make_radial()
  p_age <- make_age()
  p_seg <- make_segment()
  p_study <- make_study()

  save_panel(p_site, "follicle_covar_site_condition", 88, 46)
  save_panel(p_coll, "follicle_covar_collection", 88, 46)
  save_panel(p_frac, "follicle_covar_fractionation", 88, 46)
  save_panel(p_rad, "follicle_covar_radial", 95, 46)
  save_panel(p_age, "follicle_covar_age", 110, 46)
  save_panel(p_seg, "follicle_covar_segment", 55, 46)
  save_panel(p_study, "follicle_covar_study", 90, 55)

  # Best composite: strongest / biologically clearest panels
  # Row1: radial (strongest) | site | collection
  # Row2: fractionation | age | segment
  design <- "
  AAAABBCC
  DDDDEEFF
  "
  fig <- wrap_plots(
    A = p_rad, B = p_site, C = p_coll,
    D = p_frac, E = p_age, F = p_seg,
    design = design, heights = c(1, 1)
  ) +
    plot_annotation(
      title = "Covariates associated with follicle (GC B ≥5) capture",
      theme = theme(
        plot.title = element_text(
          family = "Helvetica", size = 7, hjust = 0, face = "plain",
          margin = margin(b = 2)
        )
      )
    )

  save_panel(fig, "follicle_covariate_bars_composite", 180, 95)

  # Alternate: add study as a third row for sharing
  fig2 <- (fig) / p_study + plot_layout(heights = c(2.1, 1))
  save_panel(fig2, "follicle_covariate_bars_composite_with_study", 180, 130)
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) render()
