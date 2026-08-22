#!/usr/bin/env Rscript
# Follicle (GC B) capture: Healthy vs disease-adjacent, by ileum/colon.
#
# Wong bi-color (distinct from biopsy/resection sky-blue/orange):
#   Healthy           = bluish green  #009E73
#   Disease-adjacent  = reddish purple #CC79A7
#
# Inputs:
#   follicle_capture_by_site_condition.csv
#   follicle_fisher_site_within_segment.csv
#   follicle_fisher_segment_within_site.csv
#   follicle_capture_by_site_condition_collection.csv

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(tidyr); library(patchwork); library(svglite); library(scales)
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

SITE_COL <- c(
  "Healthy" = "#009E73",
  "Disease-adjacent" = "#CC79A7"
)

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
      panel.grid = element_blank(),
      legend.position = "none",
      strip.background = element_blank(),
      strip.text = element_text(size = 6.5, colour = "black", face = "plain",
                               margin = margin(b = 0.5, t = 0)),
      panel.spacing.x = unit(1.5, "mm"),
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

load_rates <- function(cutoff = 5) {
  read_csv(file.path(DATA, "follicle_capture_by_site_condition.csv"),
           show_col_types = FALSE) %>%
    filter(cutoff == !!cutoff) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      site = factor(site, levels = c("Healthy", "Disease-adjacent")),
      lab = sprintf("%.0f%% (%d/%d)", 100 * rate, n_pos, n)
    )
}

load_fish_site <- function(cutoff = 5) {
  read_csv(file.path(DATA, "follicle_fisher_site_within_segment.csv"),
           show_col_types = FALSE) %>%
    filter(cutoff == !!cutoff) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      stars = p_to_stars(p_adj),
      lab = paste0(stars, " ", format_p(p_adj))
    )
}

# Primary: within each segment, Healthy vs disease-adjacent
make_site_within_segment <- function(cutoff = 5) {
  rates <- load_rates(cutoff)
  fish <- load_fish_site(cutoff)

  ymax <- max(rates$ci_hi, na.rm = TRUE) * 1.28
  br <- fish %>% mutate(x1 = 1, x2 = 2, y = ymax * 0.86, y2 = ymax * 0.90)
  ann <- fish %>% mutate(x = 1.5, y = ymax * 0.985)

  ggplot(rates, aes(site, rate, fill = site)) +
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
    scale_fill_manual(values = SITE_COL, drop = FALSE) +
    scale_x_discrete(labels = c(
      "Healthy" = "Healthy",
      "Disease-adjacent" = "Disease-\nadjacent"
    )) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.01)),
      breaks = pretty_breaks(4)
    ) +
    labs(
      title = sprintf("Follicle capture by site condition (≥%d GC B cells)", cutoff),
      x = NULL, y = "GC B detection"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(size = 5.5, lineheight = 0.9))
}

# Secondary: within each site class, ileum vs colon
make_segment_within_site <- function(cutoff = 5) {
  rates <- load_rates(cutoff) %>%
    mutate(segment = factor(segment, levels = c("ileum", "colon")))
  fish <- read_csv(file.path(DATA, "follicle_fisher_segment_within_site.csv"),
                   show_col_types = FALSE) %>%
    filter(cutoff == !!cutoff) %>%
    mutate(
      site = factor(site, levels = c("Healthy", "Disease-adjacent")),
      stars = p_to_stars(p_adj),
      lab = paste0(stars, " ", format_p(p_adj))
    )

  ymax <- max(rates$ci_hi, na.rm = TRUE) * 1.28
  br <- fish %>% mutate(x1 = 1, x2 = 2, y = ymax * 0.86, y2 = ymax * 0.90)
  ann <- fish %>% mutate(x = 1.5, y = ymax * 0.985)

  # segment on x; fill still encodes site so palette stays consistent
  ggplot(rates, aes(segment, rate, fill = site)) +
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
    facet_wrap(~ site, nrow = 1) +
    scale_fill_manual(values = SITE_COL, drop = FALSE) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.01))
    ) +
    labs(
      title = sprintf("Ileum vs colon within site condition (≥%d GC B cells)", cutoff),
      x = NULL, y = "GC B detection"
    ) +
    theme_gca()
}

# Cutoff sensitivity (site within segment)
make_sensitivity <- function() {
  rates <- read_csv(file.path(DATA, "follicle_capture_by_site_condition.csv"),
                    show_col_types = FALSE) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      site = factor(site, levels = c("Healthy", "Disease-adjacent")),
      cutoff_lab = factor(paste0("≥", cutoff), levels = c("≥3", "≥5", "≥10")),
      lab = sprintf("%.0f%%", 100 * rate)
    )
  fish <- read_csv(file.path(DATA, "follicle_fisher_site_within_segment.csv"),
                   show_col_types = FALSE) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      cutoff_lab = factor(paste0("≥", cutoff), levels = c("≥3", "≥5", "≥10")),
      stars = p_to_stars(p_adj)
    )

  ggplot(rates, aes(site, rate, fill = site)) +
    geom_col(width = 0.7, colour = "black", linewidth = 0.15) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.14, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.15, size = 5 / PT, family = "Helvetica") +
    geom_text(
      data = fish, aes(x = 1.5, y = 0.72, label = stars),
      inherit.aes = FALSE, size = 6 / PT, fontface = "bold", family = "Helvetica"
    ) +
    facet_grid(segment ~ cutoff_lab) +
    scale_fill_manual(values = SITE_COL, drop = FALSE) +
    scale_x_discrete(labels = c(
      "Healthy" = "Healthy",
      "Disease-adjacent" = "Dis.-adj."
    )) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, 0.82),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = "GC B detection vs site condition across cutoffs (Fisher BH-FDR)",
      x = NULL, y = "Detection rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 5.5))
}

# Collection-stratified transparency panel (ileum has most adjacent samples)
make_collection_strata <- function(cutoff = 5) {
  rates <- read_csv(
    file.path(DATA, "follicle_capture_by_site_condition_collection.csv"),
    show_col_types = FALSE
  ) %>%
    filter(cutoff == !!cutoff) %>%
    mutate(
      segment = factor(segment, levels = c("ileum", "colon")),
      site = factor(site, levels = c("Healthy", "Disease-adjacent")),
      collection = factor(collection, levels = c("Biopsy", "Resection")),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * rate, n_pos, n)
    )

  ggplot(rates, aes(site, rate, fill = site)) +
    geom_col(width = 0.7, colour = "black", linewidth = 0.15) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.12, linewidth = 0.25) +
    geom_text(aes(y = pmax(ci_hi, rate), label = lab),
              vjust = -0.1, size = 4.8 / PT, family = "Helvetica",
              lineheight = 0.85) +
    facet_grid(collection ~ segment) +
    scale_fill_manual(values = SITE_COL, drop = FALSE) +
    scale_x_discrete(labels = c(
      "Healthy" = "Healthy",
      "Disease-adjacent" = "Dis.-adj."
    )) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.18))
    ) +
    labs(
      title = sprintf(
        "Site condition × collection (≥%d GC B; n shown — colon adjacent is sparse)",
        cutoff
      ),
      x = NULL, y = "GC B detection"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 5.5))
}

render <- function() {
  p_main <- make_site_within_segment(5)
  p_seg <- make_segment_within_site(5)
  p_sens <- make_sensitivity()
  p_coll <- make_collection_strata(5)

  save_panel(p_main, "follicle_capture_by_site_condition", 90, 48)
  save_panel(p_main, "follicle_capture_by_site_condition_k5", 90, 48)
  save_panel(make_site_within_segment(3), "follicle_capture_by_site_condition_k3", 90, 48)
  save_panel(make_site_within_segment(10), "follicle_capture_by_site_condition_k10", 90, 48)
  save_panel(p_seg, "follicle_capture_segment_within_site", 90, 48)
  save_panel(
    (p_main | p_seg) + plot_layout(widths = c(1, 1)),
    "follicle_capture_site_condition_composite", 180, 50
  )
  save_panel(p_sens, "follicle_capture_site_condition_sensitivity", 110, 55)
  save_panel(p_coll, "follicle_capture_site_by_collection", 120, 70)
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) render()
