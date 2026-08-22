#!/usr/bin/env Rscript
# Fig. 1 flanking donor-age histograms + density curves from the integrated atlas.
# Left = ileum (light blue), right = colon (darker blue). No "1-5" age bin.
# Density is fit on known age-bin midpoints (unique donors); "unknown" is bars only.

suppressPackageStartupMessages({
  library(tidyverse)
  library(svglite)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Cannot determine script location")
script_path <- normalizePath(sub("^--file=", "", script_arg))
figure_dir <- normalizePath(file.path(dirname(script_path), ".."))
data_path <- file.path(figure_dir, "data", "donor_age_by_tissue.csv")
out_dir <- file.path(figure_dir, "out")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_path)) {
  stop("Missing ", data_path, ". Build counts from the integrated object first.")
}

# Colors sampled from the published Fig. 1 flanking plots.
tissue_colors <- c(
  ileum = "#9FC4E4",
  colon = "#08649C"
)

age_levels <- c(
  "0-9", "10-19", "20-29", "30-39", "40-49",
  "50-59", "60-69", "70-79", "80-89", "unknown"
)

# Midpoints for continuous density; unknown kept as a trailing discrete bar.
age_midpoints <- c(
  "0-9" = 5, "10-19" = 15, "20-29" = 25, "30-39" = 35, "40-49" = 45,
  "50-59" = 55, "60-69" = 65, "70-79" = 75, "80-89" = 85, "unknown" = 95
)
bin_width <- 10

counts <- readr::read_csv(data_path, show_col_types = FALSE) %>%
  mutate(
    tissue = factor(tissue, levels = c("ileum", "colon")),
    age_range = factor(age_range, levels = age_levels),
    age_mid = unname(age_midpoints[as.character(age_range)])
  ) %>%
  arrange(tissue, age_range)

if (any(counts$age_range == "1-5", na.rm = TRUE)) {
  stop("Unexpected residual 1-5 age bin in donor_age_by_tissue.csv")
}

make_age_plot <- function(tissue_name) {
  dat <- counts %>% filter(tissue == tissue_name)
  fill_col <- tissue_colors[[tissue_name]]
  # Slightly darker stroke for the density curve so it reads over the bars.
  line_col <- colorspace::darken(fill_col, amount = 0.25)

  density_dat <- dat %>%
    filter(age_range != "unknown", n_donors > 0) %>%
    tidyr::uncount(n_donors)

  n_known <- nrow(density_dat)

  p <- ggplot(dat, aes(x = age_mid, y = n_donors)) +
    geom_col(
      width = 8,
      fill = fill_col,
      color = NA,
      alpha = 0.85
    )

  if (n_known >= 2) {
    p <- p +
      geom_density(
        data = density_dat,
        aes(x = age_mid, y = after_stat(density) * n_known * bin_width),
        inherit.aes = FALSE,
        color = line_col,
        fill = NA,
        linewidth = 0.9,
        adjust = 1.1,
        bw = "nrd0"
      )
  }

  p +
    scale_y_continuous(
      name = "Number of donors",
      limits = c(0, 50),
      breaks = seq(0, 50, by = 10),
      expand = c(0, 0)
    ) +
    scale_x_continuous(
      name = "Age range",
      breaks = unname(age_midpoints[age_levels]),
      labels = age_levels,
      limits = c(0, 100),
      expand = c(0.02, 0)
    ) +
    labs(title = "Donor age distribution") +
    coord_cartesian(clip = "off") +
    theme_classic(base_family = "Helvetica", base_size = 6) +
    theme(
      text = element_text(color = "black"),
      plot.title = element_text(
        face = "bold", size = 7, hjust = 0.5,
        margin = margin(b = 4)
      ),
      axis.title = element_text(size = 6, color = "black"),
      axis.title.x = element_text(margin = margin(t = 3)),
      axis.title.y = element_text(margin = margin(r = 3)),
      axis.text = element_text(size = 5.5, color = "black"),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      axis.line = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      axis.ticks.length = unit(1.2, "mm"),
      panel.grid = element_blank(),
      plot.margin = margin(t = 4, r = 6, b = 4, l = 6),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
}

# Match the original flanking-panel footprint (~square single-column insets).
panel_width_in <- 2.55
panel_height_in <- 2.35

for (tissue_name in c("ileum", "colon")) {
  plot <- make_age_plot(tissue_name)
  base <- file.path(out_dir, paste0("fig1_donor_age_", tissue_name))
  ggsave(
    paste0(base, ".pdf"), plot,
    width = panel_width_in, height = panel_height_in,
    device = cairo_pdf, bg = "white"
  )
  ggsave(
    paste0(base, ".svg"), plot,
    width = panel_width_in, height = panel_height_in,
    device = svglite::svglite, bg = "white"
  )
  ggsave(
    paste0(base, ".png"), plot,
    width = panel_width_in, height = panel_height_in,
    dpi = 300, bg = "white"
  )
  message("Saved ", base, ".{pdf,svg,png}")
}

combined <- patchwork::wrap_plots(
  make_age_plot("ileum"),
  make_age_plot("colon"),
  nrow = 1
) +
  patchwork::plot_annotation(
    theme = theme(plot.background = element_rect(fill = "white", color = NA))
  )

combined_base <- file.path(out_dir, "fig1_donor_age_ileum_colon")
ggsave(
  paste0(combined_base, ".pdf"), combined,
  width = 5.3, height = 2.35, device = cairo_pdf, bg = "white"
)
ggsave(
  paste0(combined_base, ".svg"), combined,
  width = 5.3, height = 2.35, device = svglite::svglite, bg = "white"
)
ggsave(
  paste0(combined_base, ".png"), combined,
  width = 5.3, height = 2.35, dpi = 300, bg = "white"
)
message("Saved ", combined_base, ".{pdf,svg,png}")
