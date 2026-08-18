#!/usr/bin/env Rscript
# Final two-panel niche-capture figure for scRNA-seq sampling story.
#
# 1) GC-module GSVA is bimodal (on/off follicle capture), stratified by
#    gut segment × biopsy vs resection; Goblet / CD8 IEL shown as contrast
#    (bimodal for different reasons).
# 2) Capture prevalence depends on radial sampling depth (composition call).
#
# Outputs:
#   fig_final_niche_capture.{pdf,svg,png}
#   panel_final_a_gc_bimodal_ridges.*
#   panel_final_b_capture_by_thickness.*

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

RADIAL_ORDER <- c("EPI", "EPI_LP", "LP", "EPI_LP_MUSC", "WM")
RADIAL_LAB <- c(
  EPI = "Epi", EPI_LP = "Epi+LP", LP = "LP",
  EPI_LP_MUSC = "Full thickness", WM = "WM"
)
RADIAL_COL <- c(
  EPI = "#009E73", EPI_LP = "#56B4E9", LP = "#0072B2",
  EPI_LP_MUSC = "#E69F00", WM = "#CC79A7"
)
SEG_ORDER <- c("duodenum", "jejunum", "ileum", "colon")
COLL_ORDER <- c("biopsy", "resection")

theme_gca <- function(base = 5.5) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 6.5, hjust = 0, face = "plain",
                               margin = margin(b = 3)),
      plot.subtitle = element_text(size = 5, hjust = 0, colour = "grey25",
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
      plot.margin = margin(3, 3, 2, 3, "pt")
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

load_gsva <- function() {
  read_csv(file.path(DATA, "niche_gsva_scores_long.csv"), show_col_types = FALSE) %>%
    filter(scope_gut) %>%
    mutate(
      collection = case_when(
        tolower(sample_collection_method) == "biopsy" ~ "biopsy",
        tolower(sample_collection_method) == "surgical resection" ~ "resection",
        TRUE ~ NA_character_
      ),
      segment = factor(segment, levels = SEG_ORDER),
      collection = factor(collection, levels = COLL_ORDER),
      radial_lab = factor(
        RADIAL_LAB[as.character(radial)],
        levels = unname(RADIAL_LAB[RADIAL_ORDER])
      )
    ) %>%
    filter(!is.na(collection), !is.na(segment))
}

# ── Panel 1: GC-module ridges (segment × collection) + contrast programs ──
make_bimodal_ridges <- function() {
  d <- load_gsva() %>%
    filter(program %in% c("GC_module", "Goblet", "CD8_IEL")) %>%
    mutate(
      program_lab = factor(
        program,
        levels = c("GC_module", "Goblet", "CD8_IEL"),
        labels = c(
          "GC module (follicle)",
          "Goblet (epithelial)",
          "CD8 IEL (intraepithelial)"
        )
      )
    )

  # Focus ileum+colon for main ridges (powered); keep all segments in facets
  # but drop tiny duodenum/jejunum empty facets gracefully via scales
  p_main <- ggplot(
    d %>% filter(program == "GC_module"),
    aes(gsva, collection, fill = collection)
  ) +
    geom_density_ridges(
      scale = 1.15, rel_min_height = 0.01, alpha = 0.9,
      colour = "black", linewidth = 0.2
    ) +
    geom_vline(xintercept = 0, colour = "grey45", linewidth = 0.3) +
    facet_wrap(~ segment, nrow = 1, drop = TRUE) +
    scale_fill_manual(
      values = c(biopsy = "#56B4E9", resection = "#E69F00"),
      name = NULL
    ) +
    labs(
      title = "a  Follicle capture is bimodal (GC-module GSVA)",
      subtitle = "High mode ≈ niche present; low mode ≈ missed. Stratified by segment and collection.",
      x = "GSVA enrichment score", y = NULL
    ) +
    theme_gca() +
    theme(legend.position = "bottom")

  # Contrast: same stratification, Goblet + CD8 IEL — show bimodality is not
  # unique, but driven by layer/protocol (annotated in subtitle)
  p_ctrl <- ggplot(
    d %>% filter(program %in% c("Goblet", "CD8_IEL"),
                 segment %in% c("ileum", "colon")),
    aes(gsva, collection, fill = program_lab)
  ) +
    geom_density_ridges(
      scale = 1.05, rel_min_height = 0.01, alpha = 0.85,
      colour = "black", linewidth = 0.15
    ) +
    geom_vline(xintercept = 0, colour = "grey45", linewidth = 0.25) +
    facet_grid(program_lab ~ segment) +
    scale_fill_manual(
      values = c(
        "Goblet (epithelial)" = "#999999",
        "CD8 IEL (intraepithelial)" = "#666666"
      ),
      guide = "none"
    ) +
    labs(
      title = "Controls also bimodal — for different reasons",
      subtitle = "Goblet: biopsy vs resection / segment. CD8 IEL: epithelial enrichment, not sparse niche.",
      x = "GSVA score", y = NULL
    ) +
    theme_gca(base = 5) +
    theme(
      strip.text = element_text(size = 4.5),
      axis.text.y = element_text(size = 4.5)
    )

  p_main / p_ctrl + plot_layout(heights = c(1.15, 1))
}

# GSVA high-mode rate by thickness (GMM assign or gsva>0) next to composition
make_capture_by_thickness <- function() {
  # Composition call (primary)
  comp <- read_csv(file.path(DATA, "gc_module_pooled_gut.csv"),
                   show_col_types = FALSE) %>%
    filter(metric == "primary") %>%
    mutate(
      radial = factor(context, levels = RADIAL_ORDER),
      x = factor(context_lab, levels = unname(RADIAL_LAB[RADIAL_ORDER])),
      call = "Composition (≥3 GC B cells)",
      rate = capture_rate,
      lo = ci_lo, hi = ci_hi,
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * capture_rate, n_pos, n_samples)
    )

  # GSVA high mode (score > 0) on same samples / layers
  gs <- load_gsva() %>%
    filter(program == "GC_module", radial %in% RADIAL_ORDER)
  # Wilson-ish via prop.test CI
  gsva_rate <- gs %>%
    group_by(radial) %>%
    summarise(
      n_samples = n(),
      n_pos = sum(gsva > 0, na.rm = TRUE),
      rate = n_pos / n_samples,
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      lo = binom.test(n_pos, n_samples)$conf.int[1],
      hi = binom.test(n_pos, n_samples)$conf.int[2]
    ) %>%
    ungroup() %>%
    mutate(
      x = factor(RADIAL_LAB[as.character(radial)],
                 levels = unname(RADIAL_LAB[RADIAL_ORDER])),
      call = "GSVA high mode (score > 0)",
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * rate, n_pos, n_samples)
    )

  d <- bind_rows(
    comp %>% select(x, radial, call, rate, lo, hi, lab, n_samples),
    gsva_rate %>% select(x, radial, call, rate, lo, hi, lab, n_samples)
  ) %>%
    mutate(
      call = factor(
        call,
        levels = c(
          "Composition (≥3 GC B cells)",
          "GSVA high mode (score > 0)"
        )
      ),
      fill_key = factor(as.character(radial), levels = RADIAL_ORDER)
    )

  ymax <- max(d$hi, na.rm = TRUE) * 1.28

  ggplot(d, aes(x, rate, fill = fill_key)) +
    geom_col(
      position = position_dodge(width = 0.78), width = 0.72,
      colour = "black", linewidth = 0.2, aes(group = call),
      alpha = 0.95
    ) +
    # dodge manually by call via facet instead — cleaner
    facet_wrap(~ call, nrow = 1) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, linewidth = 0.3) +
    geom_text(
      aes(y = pmax(hi, rate), label = lab),
      vjust = -0.12, size = 1.55, family = "Helvetica", lineheight = 0.9
    ) +
    scale_fill_manual(values = RADIAL_COL, guide = "none") +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = "b  Capture prevalence depends on sampling depth",
      subtitle = "Same GC-module call: cell-count detection and GSVA high-mode rate by radial layer.",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 5))
}

# Cleaner panel b: composition bars (main story) with GSVA overlay as points
make_capture_by_thickness_v2 <- function() {
  comp <- read_csv(file.path(DATA, "gc_module_pooled_gut.csv"),
                   show_col_types = FALSE) %>%
    filter(metric == "primary") %>%
    mutate(
      radial = factor(context, levels = RADIAL_ORDER),
      x = factor(context_lab, levels = unname(RADIAL_LAB[RADIAL_ORDER])),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * capture_rate, n_pos, n_samples)
    )

  gs <- load_gsva() %>%
    filter(program == "GC_module", radial %in% RADIAL_ORDER)
  gsva_rate <- gs %>%
    group_by(radial) %>%
    summarise(
      n_samples = n(),
      n_pos = sum(gsva > 0, na.rm = TRUE),
      rate = mean(gsva > 0, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      x = factor(RADIAL_LAB[as.character(radial)],
                 levels = unname(RADIAL_LAB[RADIAL_ORDER]))
    )

  ymax <- max(comp$ci_hi, gsva_rate$rate, na.rm = TRUE) * 1.25

  ggplot(comp, aes(x, capture_rate)) +
    geom_col(aes(fill = radial), width = 0.72, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.18, linewidth = 0.3) +
    geom_point(
      data = gsva_rate, aes(x, rate),
      shape = 23, size = 2.2, fill = "white", colour = "black", stroke = 0.4
    ) +
    geom_text(
      aes(y = pmax(ci_hi, capture_rate), label = lab),
      vjust = -0.15, size = 1.6, family = "Helvetica", lineheight = 0.9
    ) +
    scale_fill_manual(values = RADIAL_COL, guide = "none") +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = "b  Capture prevalence depends on sampling depth",
      subtitle = "Bars: GC B ≥3 cells (Wilson CI). Diamonds: GSVA high-mode rate (score > 0).",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 5.5))
}

# Concordance strip: GSVA vs y/n composition detection (supports panel a)
make_concordance <- function() {
  d <- load_gsva() %>%
    filter(program == "GC_module", !is.na(detected_ge3)) %>%
    mutate(
      detected = factor(
        ifelse(detected_ge3, "GC B detected", "GC B absent"),
        levels = c("GC B absent", "GC B detected")
      )
    )

  ggplot(d, aes(detected, gsva, fill = detected)) +
    geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.25) +
    geom_violin(scale = "width", colour = "black", linewidth = 0.2, alpha = 0.9) +
    geom_boxplot(width = 0.2, outlier.shape = NA, fill = "white", linewidth = 0.25) +
    facet_wrap(~ segment, nrow = 1) +
    scale_fill_manual(
      values = c(`GC B absent` = "#D0D0D0", `GC B detected` = "#0072B2"),
      guide = "none"
    ) +
    labs(
      title = "GSVA tracks per-sample composition call",
      x = NULL, y = "GC-module GSVA"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 25, hjust = 1, size = 4.5))
}

render_final <- function() {
  p_a <- make_bimodal_ridges()
  p_b <- make_capture_by_thickness_v2()
  p_c <- make_concordance()

  save_panel(p_a, "panel_final_a_gc_bimodal_ridges", 180, 110)
  save_panel(p_b, "panel_final_b_capture_by_thickness", 90, 60)
  save_panel(p_c, "panel_final_c_gsva_vs_detection", 140, 55)

  # User's two main panels: a (bimodal) + b (thickness)
  # Add concordance as thin strip under a
  p_a_only <- {
    d <- load_gsva() %>% filter(program == "GC_module")
    ggplot(d, aes(gsva, collection, fill = collection)) +
      geom_density_ridges(
        scale = 1.2, rel_min_height = 0.01, alpha = 0.9,
        colour = "black", linewidth = 0.2
      ) +
      geom_vline(xintercept = 0, colour = "grey45", linewidth = 0.3) +
      facet_wrap(~ segment, nrow = 1) +
      scale_fill_manual(
        values = c(biopsy = "#56B4E9", resection = "#E69F00"),
        name = NULL
      ) +
      labs(
        title = "a  Follicle capture is bimodal (GC-module GSVA)",
        subtitle = "High ≈ niche present in the library; low ≈ missed. By segment × biopsy vs resection.",
        x = "GSVA enrichment score", y = NULL
      ) +
      theme_gca() +
      theme(legend.position = "bottom")
  }

  final <- (p_a_only | (p_b / p_c)) +
    plot_layout(widths = c(1.25, 1))
  # Actually stack a over b as the two story panels; concordance under a
  final2 <- (p_a_only / p_c) | p_b + plot_layout(widths = c(1.35, 1))
  # patchwork syntax
  final2 <- ((p_a_only / p_c) + plot_layout(heights = c(1.3, 0.9))) | p_b
  save_panel(final2, "fig_final_niche_capture", 180, 120)

  # Strict two-panel version (main ask)
  two <- p_a_only / p_b + plot_layout(heights = c(1.35, 1))
  save_panel(two, "fig_final_niche_capture_two_panel", 180, 130)
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) {
  render_final()
}
