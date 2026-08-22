#!/usr/bin/env Rscript
# Supplementary Figure 8 — example CLR trajectories along the gut axis.
#
# Archive vignettes/Composition_along_gut_segments.ipynb had two plot types:
#   1. scatter + B-spline along duodenum → colon (pooled or per-covariate)
#   2. ileum vs colon boxplots, and naive-vs-metadata-adjusted overlays
# render_splines.R still has (2) for the Fig. 3 collection-method story.
# This script is the official S8 renderer and does type (1) only.
#
# Inputs: data/composition/clr_long.csv (within-lineage CLR, pseudocount 1).
#   Rscript analyses/fig3_clr_contrasts/src/render_sfig8_segment_examples.R
#   Rscript ... --clr-long data/composition/clr_long.csv --outdir /tmp/s8
#
# Env: SFIG8_CLR_LONG, SFIG8_OUTDIR.

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(svglite); library(ragg); library(splines)
})

HERE <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) "."
)
if (length(HERE) == 0 || HERE == "") HERE <- "."
figure_dir <- normalizePath(file.path(HERE, ".."))
repo_root <- normalizePath(file.path(figure_dir, "..", ".."))
MM <- 25.4

parse_cli <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(clr_long = "", outdir = "")
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--clr-long", "-i") && i < length(args)) {
      out$clr_long <- args[[i + 1L]]; i <- i + 2L
    } else if (key == "--outdir" && i < length(args)) {
      out$outdir <- args[[i + 1L]]; i <- i + 2L
    } else {
      stop("Unknown argument: ", key)
    }
  }
  out
}

cli <- parse_cli()
clr_path <- cli$clr_long
if (!nzchar(clr_path)) clr_path <- Sys.getenv("SFIG8_CLR_LONG", unset = "")
if (!nzchar(clr_path)) {
  candidates <- c(
    file.path(repo_root, "data", "composition", "clr_long.csv"),
    file.path(figure_dir, "data", "clr_long.csv"),
    file.path(repo_root, "data", "demo", "expected", "clr", "clr_long.csv")
  )
  hit <- candidates[file.exists(candidates)]
  if (!length(hit)) {
    stop("Missing clr_long.csv. Pass --clr-long or set SFIG8_CLR_LONG.")
  }
  clr_path <- hit[[1]]
}
OUT <- cli$outdir
if (!nzchar(OUT)) OUT <- Sys.getenv("SFIG8_OUTDIR", unset = "")
if (!nzchar(OUT)) OUT <- file.path(figure_dir, "out")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
message("S8 CLR table: ", clr_path)

TISSUE_ORDER <- c(duodenum = 1, jejunum = 2, ileum = 3, colon = 4)
TISSUE_ABBR <- c("1" = "Duo", "2" = "Jej", "3" = "Ile", "4" = "Col")

# Wong-ish hues reused from render_splines.R / the archive spotlight.
HUE_COLS <- list(
  sample_collection_method = c(
    "biopsy" = "#0072B2",
    "surgical resection" = "#D55E00"
  ),
  sample_preservation_method = c(
    "ambient temperature" = "#E69F00",
    "frozen at -80C" = "#0072B2"
  )
)
HUE_LABS <- list(
  sample_collection_method = c(
    "biopsy" = "Biopsy",
    "surgical resection" = "Resection"
  ),
  sample_preservation_method = c(
    "ambient temperature" = "Ambient",
    "frozen at -80C" = "Frozen (−80 °C)"
  )
)

EXAMPLES <- data.frame(
  celltype = c(
    "Perivascular Resident Macrophages",
    "Paneth Cells",
    "Mature Goblet Cells",
    "CD4 Tfr"
  ),
  short = c("PV resident mac.", "Paneth", "Mature goblet", "CD4 Tfr"),
  hue = c(
    "sample_collection_method",
    "sample_collection_method",
    "sample_preservation_method",
    "sample_collection_method"
  ),
  stem = c("sfig8_pv_mac", "sfig8_paneth", "sfig8_goblet_preservation", "sfig8_cd4_tfr"),
  stringsAsFactors = FALSE
)

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 7, hjust = 0, colour = "black"),
      axis.line = element_line(colour = "black", linewidth = 0.25),
      axis.ticks = element_line(colour = "black", linewidth = 0.25),
      axis.text = element_text(colour = "black", size = 5),
      axis.title = element_text(colour = "black", size = base),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = base, colour = "black"),
      legend.key.size = unit(3, "mm"),
      plot.background = element_blank(),
      panel.background = element_blank()
    )
}

save_panel <- function(p, stem, width_mm, height_mm) {
  wi <- width_mm / MM; hi <- height_mm / MM
  ggsave(file.path(OUT, paste0(stem, ".pdf")), p, width = wi, height = hi, device = cairo_pdf)
  ggsave(file.path(OUT, paste0(stem, ".svg")), p, width = wi, height = hi, device = svglite)
  ggsave(
    file.path(OUT, paste0(stem, ".png")), p, width = wi, height = hi, dpi = 300,
    device = ragg::agg_png
  )
  message("  wrote ", stem, ".{pdf,svg,png}")
}

fit_curve <- function(d, df_spline = 3, n_grid = 80) {
  d <- d[is.finite(d$tissue_order) & is.finite(d$clr), c("tissue_order", "clr")]
  names(d) <- c("x", "clr")
  nu <- length(unique(d$x))
  if (nrow(d) < 5 || nu < 2) return(NULL)
  grid <- data.frame(x = seq(min(d$x), max(d$x), length.out = n_grid))
  fit <- tryCatch({
    if (nu >= 4) lm(clr ~ ns(x, min(df_spline, nu - 1)), data = d)
    else lm(clr ~ x, data = d)
  }, error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  pr <- predict(fit, newdata = grid, se.fit = TRUE)
  data.frame(
    x = grid$x, fit = pr$fit,
    lo = pr$fit - 1.96 * pr$se.fit, hi = pr$fit + 1.96 * pr$se.fit
  )
}

long <- suppressMessages(read_csv(clr_path, show_col_types = FALSE)) %>%
  filter(tissue_level_1 %in% names(TISSUE_ORDER)) %>%
  mutate(tissue_order = unname(TISSUE_ORDER[tissue_level_1]))

missing <- setdiff(EXAMPLES$celltype, unique(long$celltype))
if (length(missing)) {
  warning("Cell types missing from CLR table: ", paste(missing, collapse = ", "))
}

for (i in seq_len(nrow(EXAMPLES))) {
  spec <- EXAMPLES[i, ]
  hue_col <- spec$hue
  d <- long %>%
    filter(celltype == spec$celltype, is.finite(clr), is.finite(tissue_order)) %>%
    filter(.data[[hue_col]] %in% names(HUE_COLS[[hue_col]]))
  if (!nrow(d)) {
    message("skip ", spec$stem, ": no rows")
    next
  }
  d[[hue_col]] <- factor(d[[hue_col]], levels = names(HUE_COLS[[hue_col]]))
  curves <- list()
  for (lev in levels(d[[hue_col]])) {
    f <- fit_curve(d[d[[hue_col]] == lev, ])
    if (!is.null(f)) {
      f$hue <- lev
      curves[[lev]] <- f
    }
  }
  curves <- if (length(curves)) bind_rows(curves) else NULL
  pal <- HUE_COLS[[hue_col]]
  labs <- HUE_LABS[[hue_col]]

  p <- ggplot() +
    geom_point(
      data = d,
      aes(tissue_order, clr, colour = .data[[hue_col]]),
      size = 0.55, alpha = 0.45, stroke = 0,
      position = position_jitter(width = 0.08, height = 0)
    )
  if (!is.null(curves)) {
    p <- p +
      geom_ribbon(
        data = curves,
        aes(x = x, ymin = lo, ymax = hi, fill = hue),
        alpha = 0.12
      ) +
      geom_line(
        data = curves,
        aes(x = x, y = fit, colour = hue),
        linewidth = 0.7
      )
  }
  p <- p +
    scale_colour_manual(values = pal, labels = labs, drop = FALSE) +
    scale_fill_manual(values = pal, guide = "none") +
    scale_x_continuous(breaks = 1:4, labels = unname(TISSUE_ABBR)) +
    labs(
      x = "Gut segment (proximal → distal)",
      y = "CLR composition",
      title = spec$short
    ) +
    theme_gca()

  save_panel(p, spec$stem, 88, 62)
}

message("done S8 example splines.")
