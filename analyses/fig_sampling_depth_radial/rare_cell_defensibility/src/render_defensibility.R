#!/usr/bin/env Rscript
# Compact Nature-style rare-cell defensibility panels.
# Reads ../data/*.csv from compute_defensibility.py; writes ../out/.
#
# Follows ../../plot_specs.md (Helvetica 5–7 pt, Wong lineage colours,
# open L axes, vector PDF/SVG + 300 dpi PNG).

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(tidyr); library(patchwork); library(svglite); library(ragg)
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

FEATURED_ORDER <- c(
  "Tfr", "Med. sinus endo.", "FARM", "Tuft progenitors",
  "BEST4 enterocytes", "BEST4 colonocytes"
)
NEG_ORDER <- c(
  "S1 fibroblasts", "Goblet cells", "Villus-tip enterocytes",
  "CD8 IEL", "Homeostatic mac.", "Paneth cells"
)
ALL_ORDER <- c(FEATURED_ORDER, NEG_ORDER)

ROLE_COL <- c(featured = "#0072B2", negative_control = "#999999")
LINEAGE_HINT <- c(
  "Tfr" = "#0072B2", "Med. sinus endo." = "#CC79A7", "FARM" = "#E69F00",
  "Tuft progenitors" = "#009E73", "BEST4 enterocytes" = "#009E73",
  "BEST4 colonocytes" = "#009E73",
  "S1 fibroblasts" = "#CC79A7", "Goblet cells" = "#009E73",
  "Villus-tip enterocytes" = "#009E73", "CD8 IEL" = "#0072B2",
  "Homeostatic mac." = "#E69F00", "Paneth cells" = "#009E73"
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
      strip.text = element_text(size = 5, colour = "black"),
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

# ── a. LODO stability ──────────────────────────────────────────────────────
make_lodo <- function() {
  d <- read_csv(file.path(DATA, "lodo_full_thickness.csv"), show_col_types = FALSE) %>%
    filter(analysis == "lodo", role %in% c("featured", "negative_control")) %>%
    mutate(
      short_name = factor(short_name, levels = rev(ALL_ORDER)),
      role = factor(role, levels = c("featured", "negative_control"))
    )
  ref <- read_csv(file.path(DATA, "lodo_full_thickness.csv"), show_col_types = FALSE) %>%
    filter(analysis == "all") %>%
    select(short_name, delta_all = delta_CLR)

  ggplot(d, aes(delta_CLR, short_name)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_point(aes(colour = role), alpha = 0.35, size = 0.9,
               position = position_jitter(height = 0.15, seed = 1)) +
    geom_point(
      data = ref %>% mutate(short_name = factor(short_name, levels = rev(ALL_ORDER))),
      aes(x = delta_all, y = short_name),
      shape = 23, fill = "black", colour = "black", size = 1.4, stroke = 0.2
    ) +
    scale_colour_manual(
      values = ROLE_COL,
      labels = c(featured = "Featured", negative_control = "Negative control"),
      name = NULL
    ) +
    labs(
      title = "Leave-one-dataset-out full-thickness ΔCLR",
      x = "ΔCLR (full − rest)", y = NULL
    ) +
    theme_gca() +
    theme(legend.position = "bottom")
}

# ── b. Forest (within-study + pooled between-study) ─────────────────────────
make_forest <- function() {
  d <- read_csv(file.path(DATA, "forest_full_thickness.csv"), show_col_types = FALSE) %>%
    filter(role == "featured", !is.na(delta_CLR)) %>%
    mutate(
      short_name = factor(short_name, levels = rev(FEATURED_ORDER)),
      lab = ifelse(
        is_pooled,
        paste0("Pooled (", inference, ")"),
        paste0(dataset_id, " · within-study")
      )
    ) %>%
    group_by(short_name) %>%
    arrange(desc(is_pooled), dataset_id, .by_group = TRUE) %>%
    mutate(y = factor(lab, levels = rev(unique(lab)))) %>%
    ungroup()

  ggplot(d, aes(delta_CLR, y)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi), width = 0.25,
                  linewidth = 0.25, colour = "grey30", orientation = "y") +
    geom_point(aes(fill = inference, shape = is_pooled),
               colour = "black", stroke = 0.2, size = 1.6) +
    facet_wrap(~ short_name, scales = "free_y", ncol = 2) +
    scale_fill_manual(
      values = c(
        "within-study" = "#0072B2",
        "between-study" = "#E69F00",
        "mixed" = "#56B4E9"
      ),
      name = "Inference"
    ) +
    scale_shape_manual(values = c(`TRUE` = 23, `FALSE` = 21), guide = "none") +
    labs(
      title = "Dataset-level full-thickness effects",
      subtitle = "Diamonds = pooled; circles = within-study (only Elmentaite2020 has both arms)",
      x = "ΔCLR (full − rest)", y = NULL
    ) +
    theme_gca() +
    theme(
      plot.subtitle = element_text(size = 5, colour = "grey20"),
      axis.text.y = element_text(size = 4.5),
      legend.position = "bottom"
    )
}

# ── c. Donor vs sample aggregation ─────────────────────────────────────────
make_donor <- function() {
  d <- read_csv(file.path(DATA, "donor_vs_sample_contrasts.csv"), show_col_types = FALSE) %>%
    mutate(
      short_name = factor(short_name, levels = rev(ALL_ORDER)),
      aggregation = factor(aggregation, levels = c("sample", "donor"))
    )
  ggplot(d, aes(delta_CLR, short_name, colour = aggregation, shape = aggregation)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi),
                  width = 0.25, linewidth = 0.25, orientation = "y",
                  position = position_dodge(width = 0.55)) +
    geom_point(size = 1.5, position = position_dodge(width = 0.55)) +
    scale_colour_manual(
      values = c(sample = "#0072B2", donor = "#D55E00"),
      labels = c(sample = "Sample-level", donor = "Donor-aggregated"),
      name = NULL
    ) +
    scale_shape_manual(
      values = c(sample = 16, donor = 17),
      labels = c(sample = "Sample-level", donor = "Donor-aggregated"),
      name = NULL
    ) +
    labs(
      title = "Donor aggregation (equal donor weight)",
      x = "ΔCLR (full − rest)", y = NULL
    ) +
    theme_gca()
}

# ── d. Cell-count sensitivity ──────────────────────────────────────────────
make_sensitivity <- function() {
  d <- read_csv(file.path(DATA, "cellcount_sensitivity.csv"), show_col_types = FALSE) %>%
    filter(role == "featured", filter_family == "sample_depth") %>%
    mutate(
      short_name = factor(short_name, levels = FEATURED_ORDER),
      threshold = as.numeric(threshold)
    )
  ggplot(d, aes(threshold, delta_CLR, colour = short_name, group = short_name)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_line(linewidth = 0.4) +
    geom_point(size = 1.2) +
    scale_colour_manual(values = LINEAGE_HINT[FEATURED_ORDER], name = NULL) +
    scale_x_continuous(breaks = sort(unique(d$threshold))) +
    labs(
      title = "Cell-count sensitivity (min sample depth)",
      x = "Minimum sample total cells", y = "ΔCLR (full − rest)"
    ) +
    theme_gca() +
    theme(legend.position = "bottom", legend.key.width = unit(3, "mm"))
}

make_sensitivity_ct <- function() {
  d <- read_csv(file.path(DATA, "cellcount_sensitivity.csv"), show_col_types = FALSE) %>%
    filter(role == "featured", filter_family == "ct_min_cells") %>%
    mutate(
      short_name = factor(short_name, levels = FEATURED_ORDER),
      threshold = as.numeric(threshold)
    )
  ggplot(d, aes(threshold, delta_CLR, colour = short_name, group = short_name)) +
    geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_line(linewidth = 0.4) +
    geom_point(size = 1.2) +
    scale_colour_manual(values = LINEAGE_HINT[FEATURED_ORDER], name = NULL) +
    scale_x_continuous(breaks = sort(unique(d$threshold))) +
    labs(
      title = "Drop sparse positives (0 < n < threshold)",
      x = "Minimum cells of type", y = "ΔCLR (full − rest)"
    ) +
    theme_gca() +
    theme(legend.position = "bottom")
}

# ── e. Featured vs negative controls ───────────────────────────────────────
make_negctrl <- function() {
  d <- read_csv(file.path(DATA, "primary_contrasts_sample.csv"), show_col_types = FALSE) %>%
    mutate(
      short_name = factor(short_name, levels = rev(ALL_ORDER)),
      role = factor(role, levels = c("featured", "negative_control")),
      sig = ifelse(!is.na(p_adj) & p_adj < 0.05, "FDR < 0.05", "n.s.")
    )
  ggplot(d, aes(delta_CLR, short_name)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_errorbar(aes(xmin = ci_lo, xmax = ci_hi), width = 0.25,
                  linewidth = 0.25, colour = "grey35", orientation = "y") +
    geom_point(aes(fill = role, shape = sig), colour = "black",
               stroke = 0.2, size = 1.7) +
    scale_fill_manual(
      values = ROLE_COL,
      labels = c(featured = "Featured", negative_control = "Negative control"),
      name = NULL
    ) +
    scale_shape_manual(values = c("FDR < 0.05" = 21, "n.s." = 1), name = NULL) +
    labs(
      title = "Featured types vs mucosal negative controls",
      x = "ΔCLR (full − rest)", y = NULL
    ) +
    theme_gca()
}

# ── f. Niche capture probabilities ─────────────────────────────────────────
make_niche_depth <- function() {
  d <- read_csv(file.path(DATA, "niche_capture_rates.csv"), show_col_types = FALSE) %>%
    filter(strata == "depth", rule == "niche_primary", context != "unknown") %>%
    mutate(
      context = factor(
        context,
        levels = c("not_full_thickness", "full_thickness"),
        labels = c("Not full thickness", "Full thickness")
      ),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * capture_rate, n_pos, n_samples)
    )
  ggplot(d, aes(context, capture_rate, fill = context)) +
    geom_col(width = 0.65, colour = "black", linewidth = 0.2) +
    geom_text(aes(label = lab), vjust = -0.15, size = 1.7, family = "Helvetica") +
    scale_fill_manual(values = c("#56B4E9", "#0072B2"), guide = "none") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, max(d$capture_rate) * 1.25)) +
    labs(
      title = "Follicle/TLS niche capture (GC B + support)",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca()
}

make_niche_segment <- function() {
  d <- read_csv(file.path(DATA, "niche_capture_rates.csv"), show_col_types = FALSE) %>%
    filter(
      strata == "segment×depth",
      rule == "niche_primary",
      context %in% c("not_full_thickness", "full_thickness"),
      segment %in% c("duodenum", "jejunum", "ileum", "colon")
    ) %>%
    mutate(
      segment = factor(segment, levels = c("duodenum", "jejunum", "ileum", "colon")),
      context = factor(
        context,
        levels = c("not_full_thickness", "full_thickness"),
        labels = c("Not full thickness", "Full thickness")
      )
    )
  ggplot(d, aes(segment, capture_rate, fill = context)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.65,
             colour = "black", linewidth = 0.2) +
    scale_fill_manual(values = c("#56B4E9", "#0072B2"), name = NULL) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Niche capture by segment and depth",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

make_niche_dataset <- function() {
  d <- read_csv(file.path(DATA, "niche_capture_rates.csv"), show_col_types = FALSE) %>%
    filter(strata == "dataset", rule == "niche_primary", n_samples >= 5) %>%
    arrange(capture_rate) %>%
    mutate(dataset_id = factor(dataset_id, levels = dataset_id))
  ggplot(d, aes(capture_rate, dataset_id)) +
    geom_col(fill = "#0072B2", colour = "black", linewidth = 0.15, width = 0.75) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Niche capture by study (≥5 samples)",
      x = "Capture rate", y = NULL
    ) +
    theme_gca() +
    theme(axis.text.y = element_text(size = 4.5))
}

make_niche_markers <- function() {
  d <- read_csv(file.path(DATA, "niche_marker_detection_rates.csv"), show_col_types = FALSE) %>%
    filter(
      strata == "depth",
      context %in% c("not_full_thickness", "full_thickness")
    ) %>%
    mutate(
      context = factor(
        context,
        levels = c("not_full_thickness", "full_thickness"),
        labels = c("Not full thickness", "Full thickness")
      ),
      celltype = factor(celltype, levels = rev(sort(unique(celltype))))
    )
  ggplot(d, aes(detection_rate, celltype, fill = context)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7,
             colour = "black", linewidth = 0.15) +
    scale_fill_manual(values = c("#56B4E9", "#0072B2"), name = NULL) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Follicle-marker detection by depth",
      x = "Detection rate (≥3 cells)", y = NULL
    ) +
    theme_gca() +
    theme(axis.text.y = element_text(size = 4.5))
}

RADIAL_ORDER <- c("EPI", "EPI_LP", "LP", "EPI_LP_MUSC", "WM")
RADIAL_COL <- c(
  EPI = "#009E73",
  EPI_LP = "#56B4E9",
  LP = "#0072B2",
  EPI_LP_MUSC = "#E69F00",
  WM = "#CC79A7"
)

make_niche_radial <- function() {
  d <- read_csv(file.path(DATA, "niche_capture_rates.csv"), show_col_types = FALSE) %>%
    filter(strata == "radial_layer", rule == "niche_primary") %>%
    mutate(
      radial_layer = factor(radial_layer, levels = RADIAL_ORDER),
      lab = sprintf("%.0f%%\n(%d/%d)", 100 * capture_rate, n_pos, n_samples)
    ) %>%
    filter(!is.na(radial_layer))
  ggplot(d, aes(radial_layer, capture_rate, fill = radial_layer)) +
    geom_col(width = 0.7, colour = "black", linewidth = 0.2) +
    geom_text(aes(label = lab), vjust = -0.1, size = 1.6, family = "Helvetica") +
    scale_fill_manual(values = RADIAL_COL, guide = "none") +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      limits = c(0, max(d$capture_rate, na.rm = TRUE) * 1.28)
    ) +
    labs(
      title = "Niche capture by radial layer",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

make_niche_segment_radial <- function() {
  d <- read_csv(file.path(DATA, "niche_capture_rates.csv"), show_col_types = FALSE) %>%
    filter(
      strata == "segment×radial_layer",
      rule == "niche_primary",
      segment %in% c("duodenum", "jejunum", "ileum", "colon"),
      radial_layer %in% RADIAL_ORDER
    ) %>%
    mutate(
      segment = factor(segment, levels = c("duodenum", "jejunum", "ileum", "colon")),
      radial_layer = factor(radial_layer, levels = RADIAL_ORDER),
      lab = ifelse(n_samples > 0, sprintf("%d/%d", n_pos, n_samples), "")
    )
  ggplot(d, aes(radial_layer, capture_rate, fill = radial_layer)) +
    geom_col(width = 0.7, colour = "black", linewidth = 0.15) +
    geom_text(aes(label = lab, y = capture_rate), vjust = -0.2,
              size = 1.35, family = "Helvetica") +
    facet_wrap(~ segment, nrow = 1) +
    scale_fill_manual(values = RADIAL_COL, guide = "none") +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.22))
    ) +
    labs(
      title = "Niche capture by segment × radial layer",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 4.5))
}

make_niche_markers_radial <- function() {
  d <- read_csv(file.path(DATA, "niche_marker_detection_rates.csv"), show_col_types = FALSE) %>%
    filter(strata == "radial_layer", radial_layer %in% RADIAL_ORDER) %>%
    mutate(
      radial_layer = factor(radial_layer, levels = RADIAL_ORDER),
      celltype = factor(celltype, levels = rev(sort(unique(celltype))))
    )
  ggplot(d, aes(detection_rate, celltype, fill = radial_layer)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75,
             colour = "black", linewidth = 0.12) +
    scale_fill_manual(values = RADIAL_COL, name = NULL) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Follicle-marker detection by radial layer",
      x = "Detection rate (≥3 cells)", y = NULL
    ) +
    theme_gca() +
    theme(
      axis.text.y = element_text(size = 4.5),
      legend.position = "bottom"
    )
}

# Study-level capture by radial layer — boxplot + sized points, faceted by segment
make_niche_radial_study_ci <- function(min_n = 2) {
  layer_lab <- c(
    EPI = "Epi",
    EPI_LP = "Epi+LP",
    LP = "LP",
    EPI_LP_MUSC = "Full thickness",
    WM = "WM"
  )
  x_order <- c("EPI", "EPI_LP", "LP", "EPI_LP_MUSC", "WM")
  seg_order <- c("duodenum", "jejunum", "ileum", "colon")

  d <- read_csv(
    file.path(DATA, "niche_capture_by_dataset_segment_radial.csv"),
    show_col_types = FALSE
  ) %>%
    filter(
      n_samples >= min_n,
      radial_layer %in% x_order,
      segment %in% seg_order
    ) %>%
    mutate(
      radial_layer = factor(radial_layer, levels = x_order),
      segment = factor(segment, levels = seg_order),
      x_lab = factor(
        layer_lab[as.character(radial_layer)],
        levels = unname(layer_lab[x_order])
      )
    )

  ggplot(d, aes(x_lab, capture_rate)) +
    geom_boxplot(
      aes(group = x_lab),
      width = 0.55, outlier.shape = NA,
      fill = "grey92", colour = "black", linewidth = 0.3
    ) +
    geom_point(
      aes(size = n_samples, fill = radial_layer),
      shape = 21, colour = "black", stroke = 0.28, alpha = 0.95,
      position = position_jitter(width = 0.12, height = 0, seed = 1)
    ) +
    facet_wrap(~ segment, nrow = 1) +
    scale_fill_manual(values = RADIAL_COL, guide = "none") +
    scale_size_continuous(
      name = "Samples", range = c(1.4, 3.2), breaks = c(3, 10, 30)
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      limits = c(0, 1),
      expand = expansion(mult = c(0.02, 0.04)),
      breaks = seq(0, 1, 0.25)
    ) +
    labs(
      title = "Niche capture across studies by segment and radial layer",
      x = NULL, y = "Capture rate"
    ) +
    theme_gca(base = 6) +
    theme(
      legend.position = "bottom",
      legend.margin = margin(1, 0, 0, 0),
      axis.text.x = element_text(size = 5, angle = 35, hjust = 1),
      strip.text = element_text(size = 6),
      plot.margin = margin(3, 4, 2, 3, "pt")
    )
}

# ── Composite evidence panel ───────────────────────────────────────────────
make_composite <- function() {
  p_a <- make_lodo() + labs(title = "a  LODO stability")
  p_b <- make_forest() + labs(title = "b  Dataset forest")
  p_c <- make_donor() + labs(title = "c  Donor aggregation")
  p_d <- make_sensitivity() + labs(title = "d  Sample-depth sensitivity")
  p_e <- make_negctrl() + labs(title = "e  Negative controls")
  p_f <- make_niche_depth() + labs(title = "f  Niche capture")
  p_g <- make_niche_segment() + labs(title = "g  Capture by segment")
  p_h <- make_niche_markers() + labs(title = "h  Marker detection")

  top <- (p_a | p_e) + plot_layout(widths = c(1.1, 1))
  mid <- (p_c | p_d) + plot_layout(widths = c(1, 1))
  bot <- (p_f | p_g | p_h) + plot_layout(widths = c(0.8, 1.1, 1.3))
  # forest is dense — full width
  comp <- top / p_b / mid / bot + plot_layout(heights = c(1.0, 1.35, 0.95, 1.05))
  save_panel(comp, "fig_rare_cell_defensibility_composite", 180, 170)

  # also save individual panels for Illustrator
  save_panel(make_lodo(), "panel_a_lodo", 90, 70)
  save_panel(make_forest(), "panel_b_forest", 180, 95)
  save_panel(make_donor(), "panel_c_donor", 90, 70)
  save_panel(make_sensitivity(), "panel_d_sensitivity_depth", 90, 60)
  save_panel(make_sensitivity_ct(), "panel_d2_sensitivity_ct", 90, 60)
  save_panel(make_negctrl(), "panel_e_negctrl", 90, 70)
  save_panel(make_niche_depth(), "panel_f_niche_depth", 55, 55)
  save_panel(make_niche_segment(), "panel_g_niche_segment", 80, 55)
  save_panel(make_niche_dataset(), "panel_g2_niche_dataset", 90, 100)
  save_panel(make_niche_markers(), "panel_h_niche_markers", 90, 70)

  # radial-layer stratified niche panels
  save_panel(make_niche_radial(), "panel_i_niche_radial", 80, 55)
  save_panel(make_niche_segment_radial(), "panel_j_niche_segment_radial", 180, 55)
  save_panel(make_niche_markers_radial(), "panel_k_niche_markers_radial", 120, 75)

  rad_comp <- (
    make_niche_radial() + labs(title = "a  Capture by radial layer")
  ) / (
    make_niche_segment_radial() + labs(title = "b  Segment × radial layer")
  ) / (
    make_niche_markers_radial() + labs(title = "c  Marker detection by radial layer")
  ) + plot_layout(heights = c(0.9, 1.0, 1.2))
  save_panel(rad_comp, "fig_niche_capture_by_radial_layer", 180, 150)

  save_panel(
    make_niche_radial_study_ci(),
    "panel_l_niche_radial_study_ci",
    180, 70
  )
}

# Only auto-render when executed as a script, not when sourced
.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) {
  make_composite()
}
