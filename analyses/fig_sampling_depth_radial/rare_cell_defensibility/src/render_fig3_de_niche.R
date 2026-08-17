#!/usr/bin/env Rscript
# Compact Fig. 3 d+e (ileum + colon): biopsy vs resection.
#
# d — GC-module GSVA ridges; annotate GMM bimodality (ΔBIC, Ashman’s D)
# e — GC B detection rate at cell-count cutoffs; Fisher’s exact + BH-FDR
#
# Inputs (from compute / prior python):
#   niche_gsva_scores_long.csv
#   fig3_d_bimodality_stats.csv
#   fig3_e_capture_by_cutoff.csv
#   fig3_e_fisher_by_cutoff.csv

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(tidyr); library(patchwork); library(svglite); library(scales)
  library(ggridges)
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

COLL_COL <- c(Biopsy = "#56B4E9", Resection = "#E69F00")

# Nature guide: 5–7 pt at final size; prefer 6 pt. geom_text size ≈ pt / 2.845
PT <- 2.845

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 7, hjust = 0, face = "plain",
                               margin = margin(b = 0.5, t = 0)),
      axis.line = element_line(colour = "black", linewidth = 0.25),
      axis.ticks = element_line(colour = "black", linewidth = 0.25),
      axis.ticks.length = unit(0.6, "mm"),
      axis.text = element_text(colour = "black", size = 6),
      axis.title = element_text(colour = "black", size = 6,
                               margin = margin(t = 1, r = 1)),
      axis.title.x = element_text(margin = margin(t = 1)),
      panel.grid = element_blank(),
      legend.position = "none",
      strip.background = element_blank(),
      strip.text = element_text(size = 6.5, colour = "black", face = "plain",
                               margin = margin(b = 0.5, t = 0)),
      panel.spacing.x = unit(1.2, "mm"),
      plot.margin = margin(0.5, 1, 0.5, 0.5, "pt")
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
    is.na(p) ~ "n.s.",
    p < 1e-4 ~ "****",
    p < 1e-3 ~ "***",
    p < 1e-2 ~ "**",
    p < 5e-2 ~ "*",
    TRUE ~ "n.s."
  )
}

format_p <- function(p) {
  case_when(
    is.na(p) ~ "n.s.",
    p < 1e-4 ~ "P < 0.0001",
    p < 1e-3 ~ sprintf("P = %.1e", p),
    TRUE ~ sprintf("P = %.3g", p)
  )
}

load_gsva <- function() {
  read_csv(file.path(DATA, "niche_gsva_scores_long.csv"), show_col_types = FALSE) %>%
    filter(
      scope_gut, program == "GC_module",
      segment %in% c("ileum", "colon")
    ) %>%
    mutate(
      collection = case_when(
        tolower(sample_collection_method) == "biopsy" ~ "Biopsy",
        tolower(sample_collection_method) == "surgical resection" ~ "Resection",
        TRUE ~ NA_character_
      ),
      segment = factor(segment, levels = c("ileum", "colon")),
      collection = factor(collection, levels = c("Biopsy", "Resection"))
    ) %>%
    filter(!is.na(collection))
}

# d: ridges + per-segment bimodality (ΔBIC, Ashman D)
make_d <- function() {
  d <- load_gsva()
  bi <- read_csv(file.path(DATA, "fig3_d_bimodality_stats.csv"),
                 show_col_types = FALSE) %>%
    filter(strata == "segment", segment %in% c("ileum", "colon")) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      lab = sprintf("ΔBIC=%.0f  D=%.1f", delta_bic, ashman_d)
    )

  ggplot(d, aes(gsva, collection, fill = collection)) +
    geom_density_ridges(
      scale = 0.95, rel_min_height = 0.02, alpha = 0.92,
      colour = "black", linewidth = 0.2, bandwidth = 0.09
    ) +
    geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.25) +
    geom_text(
      data = bi, aes(x = 1.02, y = 2.15, label = lab),
      inherit.aes = FALSE, hjust = 1, vjust = 1,
      size = 5.5 / PT, family = "Helvetica", colour = "grey10"
    ) +
    facet_wrap(~ segment, nrow = 1) +
    scale_fill_manual(values = COLL_COL) +
    scale_x_continuous(limits = c(-1.0, 1.08), breaks = c(-1, 0, 1)) +
    scale_y_discrete(expand = expansion(mult = c(0.05, 0.18))) +
    labs(
      title = "Follicle capture is binary",
      x = "GC-module GSVA", y = NULL
    ) +
    theme_gca()
}

# e: one cutoff, Fisher bracket
make_e_cutoff <- function(cutoff = 5) {
  rates <- read_csv(file.path(DATA, "fig3_e_capture_by_cutoff.csv"),
                    show_col_types = FALSE) %>%
    filter(cutoff == !!cutoff) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      collection = factor(collection, levels = c("Biopsy", "Resection")),
      # single label above bar: percent + n (saves vertical space)
      lab = sprintf("%.0f%% (%d/%d)", 100 * rate, n_pos, n)
    )

  fish <- read_csv(file.path(DATA, "fig3_e_fisher_by_cutoff.csv"),
                   show_col_types = FALSE) %>%
    filter(cutoff == !!cutoff) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      stars = p_to_stars(p_adj),
      lab = paste0(stars, " ", format_p(p_adj))
    )

  ymax <- max(rates$ci_hi, na.rm = TRUE) * 1.22
  br <- fish %>% mutate(x1 = 1, x2 = 2, y = ymax * 0.86, y2 = ymax * 0.90)
  ann <- fish %>% mutate(x = 1.5, y = ymax * 0.985)

  title <- sprintf("Capture rate (≥%d GC B cells)", cutoff)

  ggplot(rates, aes(collection, rate, fill = collection)) +
    geom_col(width = 0.72, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.14, linewidth = 0.28) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.2, size = 5.5 / PT, family = "Helvetica") +
    geom_segment(
      data = br, aes(x = x1, xend = x2, y = y2, yend = y2),
      inherit.aes = FALSE, linewidth = 0.28
    ) +
    geom_segment(
      data = br, aes(x = x1, xend = x1, y = y, yend = y2),
      inherit.aes = FALSE, linewidth = 0.28
    ) +
    geom_segment(
      data = br, aes(x = x2, xend = x2, y = y, yend = y2),
      inherit.aes = FALSE, linewidth = 0.28
    ) +
    geom_text(
      data = ann, aes(x = x, y = y, label = lab),
      inherit.aes = FALSE, size = 5.5 / PT, family = "Helvetica",
      fontface = "bold", vjust = 0.1
    ) +
    facet_wrap(~ segment, nrow = 1) +
    scale_fill_manual(values = COLL_COL) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.01)),
      breaks = c(0, 0.25, 0.5)
    ) +
    labs(title = title, x = NULL, y = "GC B detection") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 6,
                                     margin = margin(t = 0)))
}

# sensitivity: all cutoffs in one compact panel
make_e_sensitivity <- function() {
  rates <- read_csv(file.path(DATA, "fig3_e_capture_by_cutoff.csv"),
                    show_col_types = FALSE) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      collection = factor(collection, levels = c("Biopsy", "Resection")),
      cutoff_lab = factor(
        paste0("≥", cutoff),
        levels = c("≥3", "≥5", "≥10")
      ),
      lab = sprintf("%.0f%%", 100 * rate)
    )
  fish <- read_csv(file.path(DATA, "fig3_e_fisher_by_cutoff.csv"),
                   show_col_types = FALSE) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      cutoff_lab = factor(paste0("≥", cutoff), levels = c("≥3", "≥5", "≥10")),
      stars = p_to_stars(p_adj)
    )

  ggplot(rates, aes(collection, rate, fill = collection)) +
    geom_col(width = 0.7, colour = "black", linewidth = 0.15,
             position = position_dodge(width = 0.8)) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.15, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.15, size = 5 / PT, family = "Helvetica") +
    geom_text(
      data = fish, aes(x = 1.5, y = 0.62, label = stars),
      inherit.aes = FALSE, size = 6 / PT, fontface = "bold", family = "Helvetica"
    ) +
    facet_grid(segment ~ cutoff_lab) +
    scale_fill_manual(values = COLL_COL) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, 0.72),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = "GC B detection vs cell-count cutoff (Fisher BH-FDR stars)",
      x = NULL, y = "Detection rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 5.5))
}

# Half-page (90 mm column): stack GSVA ridges over GC B detection bars.
make_halfpage_de <- function(p_d, p_e) {
  (p_d / p_e) + plot_layout(heights = c(1.05, 1))
}

render <- function() {
  p_d <- make_d()
  # primary e at ≥5 (middle stringency; ≥3/≥10 in sensitivity + text)
  p_e5 <- make_e_cutoff(5)
  p_e3 <- make_e_cutoff(3)
  p_e10 <- make_e_cutoff(10)
  p_sens <- make_e_sensitivity()

  # Compact Nature sizes: shorter than before, 6–7 pt text at final mm.
  # Slightly narrower width improves panel aspect (less flat plot area).
  w <- 78; h <- 40; W <- 160; H <- 42
  save_panel(p_d, "fig3_d_follicle_bimodal", w, h)
  save_panel(p_e5, "fig3_e_capture_by_collection", w, h)
  save_panel(p_e3, "fig3_e_capture_by_collection_k3", w, h)
  save_panel(p_e10, "fig3_e_capture_by_collection_k10", w, h)
  save_panel(
    (p_d | p_e5) + plot_layout(widths = c(1, 1)),
    "fig3_de_niche_capture", W, H
  )
  save_panel(
    (p_d | p_e3) + plot_layout(widths = c(1, 1)),
    "fig3_de_niche_capture_k3", W, H
  )
  save_panel(
    (p_d | p_e10) + plot_layout(widths = c(1, 1)),
    "fig3_de_niche_capture_k10", W, H
  )
  # Fig. 4e half-page column (90 mm): ileum/colon GSVA + GC B detection
  save_panel(
    make_halfpage_de(p_d, p_e5),
    "fig4_e_gc_niche_halfpage", 90, 78
  )
  save_panel(
    make_halfpage_de(p_d, p_e3),
    "fig4_e_gc_niche_halfpage_k3", 90, 78
  )
  save_panel(
    make_halfpage_de(p_d, p_e10),
    "fig4_e_gc_niche_halfpage_k10", 90, 78
  )
  save_panel(p_sens, "fig3_e_cutoff_sensitivity", 100, 50)
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) render()
