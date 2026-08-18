#!/usr/bin/env Rscript
# Main + ED panels for GC-module (gut) and LN-Tfh (lymph node / mesentery) stories.
# Gut palette: Wong blues/greens/orange. LN palette: warm vermillion/wine/ochre.
#
# Outputs under ../out/:
#   panel_gc_a_pooled_gut.*          — main subplot 1 (gut)
#   panel_gc_b_markers_gut.*         — main subplot 2 (gut)
#   panel_gc_l_study_gut.*           — ED study boxplot (gut)
#   fig_gc_module_main_gut.*         — a+b composite
#   panel_ln_a_pooled_ln.*           — LN version of subplot 1
#   panel_ln_b_markers_ln.*          — LN version of subplot 2
#   panel_ln_l_study_ln.*            — LN version of study boxplot
#   fig_ln_tfh_main_ln.*             — LN a+b composite

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
      plot.margin = margin(2, 2, 2, 2, "pt"),
      plot.background = element_blank(),
      panel.background = element_blank()
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

# Gut: Wong cool palette (existing radial story)
GUT_CTX <- c("EPI", "EPI_LP", "LP", "EPI_LP_MUSC", "WM")
GUT_LAB <- c(
  EPI = "Epi", EPI_LP = "Epi+LP", LP = "LP",
  EPI_LP_MUSC = "Full thickness", WM = "WM"
)
GUT_COL <- c(
  EPI = "#009E73", EPI_LP = "#56B4E9", LP = "#0072B2",
  EPI_LP_MUSC = "#E69F00", WM = "#CC79A7"
)

# LN: distinct warm / wine palette (mesentery highlighted)
LN_CTX <- c(
  "Epi", "Epi+LP", "LP", "Full thickness", "WM",
  "Accessory", "Mesentery (mLN)"
)
LN_COL <- c(
  "Epi" = "#7A9E7E",
  "Epi+LP" = "#C4A35A",
  "LP" = "#B86B4B",
  "Full thickness" = "#8B4A5E",
  "WM" = "#6B5B7A",
  "Accessory" = "#A67C52",
  "Mesentery (mLN)" = "#D55E00"  # vermillion positive-control accent
)

MARKER_ORDER <- c(
  "GC B LZ", "GC B DZ", "Tfh", "Tfr", "FARM",
  "fDC", "FRC", "mLTo", "MRC",
  "Med. sinus endo.", "Lymphatic endo."
)

# Heatmap fills — cool teal for gut, warm rose for LN
GUT_HEAT <- scale_fill_gradient(
  low = "#F7FBFF", high = "#08519C",
  labels = percent_format(accuracy = 1),
  name = "Detection", limits = c(0, 1)
)
LN_HEAT <- scale_fill_gradient(
  low = "#FFF5F0", high = "#A50F15",
  labels = percent_format(accuracy = 1),
  name = "Detection", limits = c(0, 1)
)

make_pooled <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  d <- read_csv(file.path(DATA, paste0("gc_module_pooled_", scope, ".csv")),
                show_col_types = FALSE) %>%
    filter(metric == "primary")

  if (scope == "gut") {
    d <- d %>%
      mutate(
        x = factor(context, levels = GUT_CTX, labels = unname(GUT_LAB[GUT_CTX])),
        fill_key = factor(context, levels = GUT_CTX)
      )
    cols <- GUT_COL
    title <- "GC-associated lymphoid module by radial layer"
    ylab <- "Detection rate"
  } else {
    d <- d %>%
      mutate(
        x = factor(context, levels = LN_CTX),
        fill_key = factor(context, levels = LN_CTX)
      )
    cols <- LN_COL
    title <- "LN Tfh program by tissue context"
    ylab <- "Detection rate (Tfh ≥3 cells)"
  }

  d <- d %>%
    mutate(lab = sprintf("%.0f%%\n(%d/%d)", 100 * capture_rate, n_pos, n_samples))

  ymax <- max(d$ci_hi, na.rm = TRUE) * 1.22

  ggplot(d, aes(x, capture_rate, fill = fill_key)) +
    geom_col(width = 0.72, colour = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.18,
                  linewidth = 0.3, colour = "black") +
    geom_text(aes(y = pmax(ci_hi, capture_rate), label = lab),
              vjust = -0.15, size = 1.55, family = "Helvetica", lineheight = 0.9) +
    scale_fill_manual(values = cols, guide = "none") +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = title, x = NULL, y = ylab) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 5))
}

make_markers <- function(scope = c("gut", "ln")) {
  scope <- match.arg(scope)
  d <- read_csv(file.path(DATA, paste0("gc_module_markers_", scope, ".csv")),
                show_col_types = FALSE)

  if (scope == "gut") {
    d <- d %>%
      mutate(
        x = factor(context, levels = GUT_CTX, labels = unname(GUT_LAB[GUT_CTX]))
      )
    heat <- GUT_HEAT
    title <- "Marker detection by radial layer"
  } else {
    d <- d %>%
      mutate(x = factor(context, levels = LN_CTX))
    heat <- LN_HEAT
    title <- "Marker detection by tissue context"
  }

  d <- d %>%
    mutate(marker = factor(marker, levels = rev(MARKER_ORDER))) %>%
    filter(!is.na(marker), !is.na(x))

  ggplot(d, aes(x, marker, fill = detection_rate)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    geom_text(
      aes(label = ifelse(n_samples >= 5, sprintf("%.0f", 100 * detection_rate), "")),
      size = 1.45, family = "Helvetica", colour = "black"
    ) +
    heat +
    labs(title = title, x = NULL, y = NULL) +
    theme_gca() +
    theme(
      axis.text.x = element_text(angle = 35, hjust = 1, size = 5),
      axis.text.y = element_text(size = 5),
      legend.position = "right",
      legend.key.height = unit(4, "mm"),
      legend.key.width = unit(2.2, "mm")
    )
}

make_study <- function(scope = c("gut", "ln"), min_n = 2) {
  scope <- match.arg(scope)

  if (scope == "gut") {
    d <- read_csv(
      file.path(DATA, "gc_module_study_segment_gut.csv"),
      show_col_types = FALSE
    ) %>%
      filter(
        n_samples >= min_n,
        segment %in% c("duodenum", "jejunum", "ileum", "colon"),
        context %in% GUT_CTX
      ) %>%
      mutate(
        segment = factor(segment, levels = c("duodenum", "jejunum", "ileum", "colon")),
        x = factor(context, levels = GUT_CTX, labels = unname(GUT_LAB[GUT_CTX])),
        fill_key = factor(context, levels = GUT_CTX)
      )
    cols <- GUT_COL
    title <- "GC-module detection across studies by segment and radial layer"
  } else {
    d <- read_csv(
      file.path(DATA, "gc_module_study_ln.csv"),
      show_col_types = FALSE
    ) %>%
      filter(n_samples >= min_n, context %in% LN_CTX) %>%
      mutate(
        x = factor(context, levels = LN_CTX),
        fill_key = factor(context, levels = LN_CTX),
        # facet: gut wall vs LN-oriented contexts
        panel = ifelse(
          context %in% c("Accessory", "Mesentery (mLN)"),
          "LN / accessory",
          "Gut wall"
        ),
        panel = factor(panel, levels = c("Gut wall", "LN / accessory"))
      )
    cols <- LN_COL
    title <- "LN Tfh program across studies (mesentery = mLN positive control)"
  }

  p <- ggplot(d, aes(x, capture_rate)) +
    geom_boxplot(
      aes(group = x),
      width = 0.55, outlier.shape = NA,
      fill = "grey92", colour = "black", linewidth = 0.3
    ) +
    geom_point(
      aes(size = n_samples, fill = fill_key),
      shape = 21, colour = "black", stroke = 0.28, alpha = 0.95,
      position = position_jitter(width = 0.12, height = 0, seed = 1)
    ) +
    scale_fill_manual(values = cols, guide = "none") +
    scale_size_continuous(
      name = "Samples", range = c(1.4, 3.2), breaks = c(3, 10, 30)
    ) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, 1),
      expand = expansion(mult = c(0.02, 0.04)),
      breaks = seq(0, 1, 0.25)
    ) +
    labs(title = title, x = NULL, y = "Detection rate") +
    theme_gca(base = 6) +
    theme(
      legend.position = "bottom",
      legend.margin = margin(1, 0, 0, 0),
      axis.text.x = element_text(size = 5, angle = 35, hjust = 1),
      strip.text = element_text(size = 6),
      plot.margin = margin(3, 4, 2, 3, "pt")
    )

  if (scope == "gut") {
    p <- p + facet_wrap(~ segment, nrow = 1)
  } else {
    p <- p + facet_wrap(~ panel, nrow = 1, scales = "free_x")
  }
  p
}

render_all <- function() {
  # Gut main pair
  p_a <- make_pooled("gut")
  p_b <- make_markers("gut")
  save_panel(p_a, "panel_gc_a_pooled_gut", 75, 58)
  save_panel(p_b, "panel_gc_b_markers_gut", 95, 70)
  save_panel(
    (p_a + labs(title = "a  GC-associated lymphoid module")) |
      (p_b + labs(title = "b  Marker detection")),
    "fig_gc_module_main_gut",
    180, 72
  )
  save_panel(make_study("gut"), "panel_gc_l_study_gut", 180, 70)

  # LN set (mesentery retained as positive control; warm palette)
  p_la <- make_pooled("ln")
  p_lb <- make_markers("ln")
  save_panel(p_la, "panel_ln_a_pooled_ln", 95, 58)
  save_panel(p_lb, "panel_ln_b_markers_ln", 120, 70)
  save_panel(
    (p_la + labs(title = "a  LN Tfh program")) |
      (p_lb + labs(title = "b  Marker detection")),
    "fig_ln_tfh_main_ln",
    180, 72
  )
  save_panel(make_study("ln"), "panel_ln_l_study_ln", 180, 70)
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) {
  render_all()
}
