#!/usr/bin/env Rscript
## Supplementary Figure 1a — dataset selection flow.
## Extracted from byTheNumbers/Plotting_samples_cells.R (July 2026 audit).
## Counts are study-level; 24 retained studies = 27 dataset_id values.

args <- commandArgs(trailingOnly = TRUE)
out_base <- if (length(args)) args[[1]] else {
  here <- if (length(grep("^--file=", commandArgs(FALSE)))) {
    dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
  } else getwd()
  file.path(dirname(here), "out", "sfig1a_dataset_selection_flow")
}
dir.create(dirname(out_base), recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(svglite)
})
PAL <- c(
  retained = "#0072B2",
  excluded = "#D55E00",
  text = "#000000",
  mid_grey = "#999999",
  light_grey = "#E0E0E0"
)

arrow_retained <- arrow(
  length = grid::unit(2.2, "mm"),
  type = "closed"
)
arrow_excluded <- arrow(
  length = grid::unit(1.7, "mm"),
  type = "closed"
)

# Candidate-inventory values are conservative sums from the 60-column working-
# group table (coverage: donors 53/60, samples 39/60, cells 49/60); they are
# reported-study totals rather than deduplicated individuals.
selection_stages <- data.frame(
  x = c(0.58, 2.18, 3.78, 5.38),
  studies = c(60, 28, 26, 24),
  stage = c(
    "Candidate inventory",
    "Dataset preparation",
    "Strict scRNA-seq compatibility",
    "Final integrated atlas"
  ),
  stage_note = c(
    "Formally tracked studies",
    "12 panGI core + 16 additions",
    "After modality review",
    "Current HGCA"
  ),
  totals = c(
    "≥857 donors\n≥1,186 samples\n≥6.2M cells",
    "712 donors\n770 samples\n≈2.5M cells",
    "469 donors\n1,026 samples\n2,428,530 cells",
    "265 donors\n502 samples\n944,502 cells"
  )
)

selection_losses <- data.frame(
  x = c(1.38, 2.98, 4.58),
  title = c(
    "32 not advanced",
    "2 snRNA-seq datasets put aside",
    "2 final integration losses"
  ),
  detail = c(
    "Boland2020 · quality control\nYokoi2022 · did not integrate\n30 other candidates",
    "Hickey2023\nDrokhylansky2020",
    "AgaceHelmsley1\nSmillie2019"
  )
)

p_selection <- ggplot() +
  # Proportional study-count ribbons: 60 -> 28 -> 26 -> 24.
  annotate(
    "polygon",
    x = c(0.70, 2.06, 2.06, 0.70),
    y = c(6.40 - 0.60, 6.40 - 0.28, 6.40 + 0.28, 6.40 + 0.60),
    fill = PAL["retained"], alpha = 0.16, color = NA
  ) +
  annotate(
    "polygon",
    x = c(2.30, 3.66, 3.66, 2.30),
    y = c(6.40 - 0.28, 6.40 - 0.26, 6.40 + 0.26, 6.40 + 0.28),
    fill = PAL["retained"], alpha = 0.16, color = NA
  ) +
  annotate(
    "polygon",
    x = c(3.90, 5.26, 5.26, 3.90),
    y = c(6.40 - 0.26, 6.40 - 0.24, 6.40 + 0.24, 6.40 + 0.26),
    fill = PAL["retained"], alpha = 0.16, color = NA
  ) +
  annotate(
    "segment", x = 0.70, xend = 5.26, y = 6.40, yend = 6.40,
    color = PAL["retained"], linewidth = 0.62
  ) +

  # Stage bubbles and labels.
  geom_point(
    data = selection_stages,
    aes(x = x, y = 6.40),
    shape = 21, size = 13, stroke = 0.75,
    fill = "white", color = PAL["retained"]
  ) +
  geom_text(
    data = selection_stages,
    aes(x = x, y = 6.40, label = studies),
    family = "Helvetica", size = 2.45, fontface = "bold"
  ) +
  geom_text(
    data = selection_stages,
    aes(x = x, y = 7.52, label = stage),
    family = "Helvetica", size = 1.82, fontface = "bold"
  ) +
  geom_text(
    data = selection_stages,
    aes(x = x, y = 7.18, label = stage_note),
    family = "Helvetica", size = 1.52, color = PAL["mid_grey"]
  ) +
  geom_text(
    data = selection_stages,
    aes(x = x, y = 5.30, label = totals),
    family = "Helvetica", size = 1.68, lineheight = 1.08
  ) +

  # Attrition branches placed exactly between their corresponding stages.
  geom_curve(
    data = selection_losses,
    aes(x = x, y = 6.12, xend = x, yend = 3.95),
    curvature = 0.08, color = PAL["excluded"], linewidth = 0.42,
    arrow = arrow_excluded
  ) +
  geom_point(
    data = selection_losses,
    aes(x = x, y = 3.76),
    shape = 21, size = 3.4, stroke = 0.5,
    fill = "white", color = PAL["excluded"]
  ) +
  geom_text(
    data = selection_losses,
    aes(x = x, y = 3.30, label = title),
    family = "Helvetica", size = 1.82, fontface = "bold"
  ) +
  geom_text(
    data = selection_losses,
    aes(x = x, y = 2.48, label = detail),
    family = "Helvetica", size = 1.55, lineheight = 1.04
  ) +

  # Minimal key.
  annotate("segment", x = 0.10, xend = 0.35, y = 1.32, yend = 1.32,
           color = PAL["retained"], linewidth = 0.82) +
  annotate("text", x = 0.41, y = 1.32, label = "Retained",
           family = "Helvetica", size = 1.48, hjust = 0) +
  annotate("segment", x = 1.02, xend = 1.27, y = 1.32, yend = 1.32,
           color = PAL["excluded"], linewidth = 0.62) +
  annotate("text", x = 1.33, y = 1.32, label = "Not retained",
           family = "Helvetica", size = 1.48, hjust = 0) +
  coord_cartesian(xlim = c(0, 5.90), ylim = c(1.10, 7.80), clip = "off") +
  labs(
    title = "Dataset selection for the Human Gut Cell Atlas",
    subtitle = "Four progressively stricter stages retained 24 of 60 candidate studies",
    caption = paste(
      strwrap(
        paste(
          "Candidate totals are conservative sums of reported values in the working-group table",
          "(donors available for 53/60 studies, samples for 39/60 and cells for 49/60); overlapping studies are not deduplicated.",
          "Later-stage values are successive metadata snapshots, so sample definitions and totals are not directly monotonic."
        ),
        width = 180
      ),
      collapse = "\n"
    )
  ) +
  theme_void(base_size = 6, base_family = "Helvetica") +
  theme(
    text = element_text(color = PAL["text"]),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(size = 7, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 6, hjust = 0),
    plot.caption = element_text(size = 5, hjust = 0, lineheight = 0.95),
    plot.margin = margin(3, 4, 3, 4, unit = "mm")
  )
save_selection_flow <- function(base) {
  ggsave(
    paste0(base, ".pdf"), p_selection,
    width = 180 / 25.4, height = 88 / 25.4,
    device = grDevices::cairo_pdf
  )
  ggsave(
    paste0(base, ".svg"), p_selection,
    width = 180 / 25.4, height = 88 / 25.4,
    device = svglite::svglite
  )
  ggsave(
    paste0(base, ".png"), p_selection,
    width = 180 / 25.4, height = 88 / 25.4,
    dpi = 300
  )
}
save_selection_flow(out_base)
message("Wrote ", out_base, ".{pdf,svg,png}")
