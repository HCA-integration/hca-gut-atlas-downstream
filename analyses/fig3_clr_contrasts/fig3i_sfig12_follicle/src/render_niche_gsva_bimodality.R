#!/usr/bin/env Rscript
# Nature-style GSVA bimodality panels for gut and LN niche programs.
#
# Outputs:
#   panel_gsva_a_ridges_gut.*     — score distributions by program (gut)
#   panel_gsva_b_vs_detect_gut.*  — GSVA vs cell-count detection
#   panel_gsva_c_strata_gut.*     — ΔBIC heatmap segment × dataset
#   fig_gsva_bimodality_gut.*     — a|b / c composite
#   panel_gsva_a_ridges_ln.*      — LN scope (+ mesentery)
#   panel_gsva_b_vs_detect_ln.*
#   panel_gsva_c_strata_ln.*
#   fig_gsva_bimodality_ln.*

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

FEATURED <- c("GC_module", "GC_DZ", "GC_LZ", "Tfh", "Tfr", "FARM", "fDC", "Med_sinus")
NEG <- c("Goblet", "CD8_IEL")
PROG_LAB <- c(
  GC_module = "GC module", GC_DZ = "GC DZ", GC_LZ = "GC LZ",
  Tfh = "Tfh", Tfr = "Tfr", FARM = "FARM", fDC = "fDC",
  Med_sinus = "Med. sinus", Goblet = "Goblet", CD8_IEL = "CD8 IEL"
)

GUT_COL <- c(
  featured = "#0072B2",
  negative_control = "#999999"
)
LN_COL <- c(
  featured = "#D55E00",
  negative_control = "#7A6A5A"
)

theme_gca <- function(base = 5.5) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 6, hjust = 0, face = "plain",
                               margin = margin(b = 2)),
      axis.line = element_line(colour = "black", linewidth = 0.25),
      axis.ticks = element_line(colour = "black", linewidth = 0.25),
      axis.text = element_text(colour = "black", size = base),
      axis.title = element_text(colour = "black", size = base),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(size = 5),
      legend.text = element_text(size = 5),
      legend.key.size = unit(2.5, "mm"),
      strip.background = element_blank(),
      strip.text = element_text(size = 5.5, colour = "black"),
      plot.margin = margin(2, 2, 2, 2, "pt")
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
  message("wrote ", stem)
}

load_scores <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  d <- read_csv(file.path(DATA, "niche_gsva_scores_long.csv"), show_col_types = FALSE)
  if (scope == "gut") d <- d %>% filter(scope_gut)
  else d <- d %>% filter(scope_ln)
  d %>%
    mutate(
      program = factor(program, levels = c(FEATURED, NEG)),
      program_lab = factor(
        PROG_LAB[as.character(program)],
        levels = unname(PROG_LAB[c(FEATURED, NEG)])
      ),
      role = factor(role, levels = c("featured", "negative_control"))
    )
}

make_ridges <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  d <- load_scores(scope)
  cols <- if (scope == "gut") GUT_COL else LN_COL
  title <- if (scope == "gut") {
    "GSVA score distributions (gut wall)"
  } else {
    "GSVA score distributions (incl. mesentery mLN)"
  }
  # annotate bimodality flags
  bi <- read_csv(file.path(DATA, "niche_gsva_bimodality_global.csv"),
                 show_col_types = FALSE) %>%
    filter(strata == if (scope == "gut") "global_gut" else "global_ln") %>%
    select(program, likely_bimodal, delta_bic, ashman_d)

  d <- d %>% left_join(bi, by = "program")
  lab <- d %>%
    distinct(program_lab, likely_bimodal, delta_bic, ashman_d) %>%
    mutate(
      tag = ifelse(
        isTRUE(likely_bimodal) | (!is.na(likely_bimodal) & likely_bimodal),
        sprintf("bimodal  ΔBIC=%.0f  D=%.1f", delta_bic, ashman_d),
        sprintf("ΔBIC=%.0f  D=%.1f", delta_bic, ashman_d)
      )
    )

  ggplot(d, aes(gsva, program_lab, fill = role)) +
    geom_density_ridges(
      scale = 1.1, rel_min_height = 0.01, alpha = 0.85,
      colour = "black", linewidth = 0.2
    ) +
    geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.25) +
    geom_text(
      data = lab, aes(x = Inf, y = program_lab, label = tag),
      hjust = 1.05, vjust = -0.4, size = 1.5, family = "Helvetica",
      inherit.aes = FALSE, colour = "grey20"
    ) +
    scale_fill_manual(
      values = cols,
      labels = c(featured = "Niche program", negative_control = "Negative control"),
      name = NULL
    ) +
    labs(title = title, x = "GSVA enrichment score", y = NULL) +
    theme_gca() +
    theme(legend.position = "bottom")
}

make_vs_detect <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  d <- load_scores(scope) %>%
    filter(program %in% FEATURED, !is.na(detected_ge3)) %>%
    mutate(
      detected = factor(
        ifelse(detected_ge3, "Detected (≥3 cells)", "Not detected"),
        levels = c("Not detected", "Detected (≥3 cells)")
      )
    )
  cols <- if (scope == "gut") {
    c(`Not detected` = "#A6CEE3", `Detected (≥3 cells)` = "#1F78B4")
  } else {
    c(`Not detected` = "#FDBF6F", `Detected (≥3 cells)` = "#E31A1C")
  }
  title <- "GSVA vs composition detection"

  ggplot(d, aes(detected, gsva, fill = detected)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_violin(scale = "width", colour = "black", linewidth = 0.2, alpha = 0.9) +
    geom_boxplot(width = 0.18, outlier.shape = NA, fill = "white",
                 linewidth = 0.25, alpha = 0.9) +
    facet_wrap(~ program_lab, nrow = 2) +
    scale_fill_manual(values = cols, guide = "none") +
    labs(title = title, x = NULL, y = "GSVA score") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 4.5))
}

make_strata_heatmap <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  bi <- read_csv(file.path(DATA, "niche_gsva_bimodality_strata.csv"),
                 show_col_types = FALSE) %>%
    filter(strata == "segment×dataset", program %in% FEATURED)

  if (scope == "gut") {
    bi <- bi %>% filter(segment %in% c("duodenum", "jejunum", "ileum", "colon"))
  } else {
    bi <- bi %>%
      filter(segment %in% c("duodenum", "jejunum", "ileum", "colon",
                            "mesentery", "accessory"))
  }

  bi <- bi %>%
    mutate(
      program_lab = factor(
        PROG_LAB[as.character(program)],
        levels = unname(PROG_LAB[FEATURED])
      ),
      stratum = paste(segment, dataset_id, sep = " | "),
      delta_clip = pmax(pmin(delta_bic, 80), -20)
    )

  # keep strata with any program n, order by mean ΔBIC
  ord <- bi %>%
    group_by(stratum) %>%
    summarise(m = mean(delta_bic, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(m)) %>%
    pull(stratum)
  bi$stratum <- factor(bi$stratum, levels = rev(ord))

  fill_scale <- if (scope == "gut") {
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, name = "ΔBIC\n(1−2)",
      limits = c(-20, 80), oob = squish
    )
  } else {
    scale_fill_gradient2(
      low = "#4D9221", mid = "white", high = "#C51B7D",
      midpoint = 0, name = "ΔBIC\n(1−2)",
      limits = c(-20, 80), oob = squish
    )
  }

  ggplot(bi, aes(program_lab, stratum, fill = delta_clip)) +
    geom_tile(colour = "white", linewidth = 0.2) +
    geom_point(
      data = bi %>% filter(likely_bimodal),
      aes(program_lab, stratum),
      shape = 8, size = 0.9, colour = "black", inherit.aes = FALSE
    ) +
    fill_scale +
    labs(
      title = "Bimodality by segment × dataset (* = Ashman D>2 & ΔBIC>10)",
      x = NULL, y = NULL
    ) +
    theme_gca(base = 5) +
    theme(
      axis.text.y = element_text(size = 4),
      axis.text.x = element_text(angle = 35, hjust = 1, size = 5),
      legend.position = "right",
      legend.key.height = unit(5, "mm"),
      legend.key.width = unit(2.2, "mm")
    )
}

render_scope <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  p_a <- make_ridges(scope)
  p_b <- make_vs_detect(scope)
  p_c <- make_strata_heatmap(scope)
  save_panel(p_a, paste0("panel_gsva_a_ridges_", scope), 90, 85)
  save_panel(p_b, paste0("panel_gsva_b_vs_detect_", scope), 120, 80)
  save_panel(p_c, paste0("panel_gsva_c_strata_", scope), 120, 110)
  comp <- (p_a | p_b) / p_c + plot_layout(heights = c(1, 1.15))
  save_panel(comp, paste0("fig_gsva_bimodality_", scope), 180, 160)
}

.render <- function() {
  render_scope("gut")
  render_scope("ln")
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) {
  .render()
}
