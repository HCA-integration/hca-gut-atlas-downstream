#!/usr/bin/env Rscript
# Fig. 3a — rotated anatomical CLR enrichment heatmaps.
# Cell types on the x-axis; gut segments + radial layers on the y-axis.
#
# Input is celltype_compositional_enrichment_long.csv from
# analyses/fig2_label_set/src/build_fig2_atlas_evidence.py
# (checked in at data/fig2/). That table stores unscaled category-mean
# CLR (mean_clr) plus a display-only row_z. This script paints row_z.
#
# Do not rebuild this heatmap from data/composition/clr_long.csv.
# Those CLRs are within-lineage. Fig. 3a uses one global composition
# per sample (all taxonomy v1 labels in one CLR, pseudocount 1).
#
#   Rscript analyses/fig3_clr_contrasts/src/render_fig3a_anatomical_clr_heatmap.R
#   Rscript .../render_fig3a_anatomical_clr_heatmap.R \
#     --enrichment /path/to/celltype_compositional_enrichment_long.csv \
#     --outdir /tmp/fig3a --datadir /tmp/fig3a
#
# Env: FIG3A_ENRICHMENT (or HGCA_ENRICHMENT_LONG), FIG3A_OUTDIR, FIG3A_DATADIR.
#
# Diverging fill: HCA blue (depletion) ↔ white ↔ vermillion (enrichment).

suppressPackageStartupMessages({
  library(tidyverse)
  library(svglite)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Cannot determine script location")
script_path <- normalizePath(sub("^--file=", "", script_arg))
figure_dir <- normalizePath(file.path(dirname(script_path), ".."))
repo_root <- normalizePath(file.path(figure_dir, "..", ".."))

parse_kv_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(enrichment = "", outdir = "", datadir = "")
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--enrichment", "-e") && i < length(args)) {
      out$enrichment <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--outdir" && i < length(args)) {
      out$outdir <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--datadir" && i < length(args)) {
      out$datadir <- args[[i + 1L]]
      i <- i + 2L
    } else {
      stop("Unknown argument: ", key)
    }
  }
  out
}

cli <- parse_kv_args()
env_enrichment <- Sys.getenv("FIG3A_ENRICHMENT", unset = "")
if (!nzchar(env_enrichment)) {
  env_enrichment <- Sys.getenv("HGCA_ENRICHMENT_LONG", unset = "")
}
enrichment_path <- cli$enrichment
if (!nzchar(enrichment_path)) enrichment_path <- env_enrichment
if (!nzchar(enrichment_path)) {
  candidates <- c(
    file.path(repo_root, "data", "fig2", "celltype_compositional_enrichment_long.csv"),
    file.path(figure_dir, "..", "fig2_label_set", "data", "celltype_compositional_enrichment_long.csv"),
    file.path(
      repo_root, "data", "demo", "expected", "fig2", "data",
      "celltype_compositional_enrichment_long.csv"
    )
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) {
    stop(
      "Missing enrichment table. Pass --enrichment, set FIG3A_ENRICHMENT, ",
      "or place celltype_compositional_enrichment_long.csv in data/fig2/."
    )
  }
  enrichment_path <- hit[[1]]
}
enrichment_path <- normalizePath(enrichment_path, mustWork = TRUE)

out_dir <- cli$outdir
if (!nzchar(out_dir)) out_dir <- Sys.getenv("FIG3A_OUTDIR", unset = "")
if (!nzchar(out_dir)) out_dir <- file.path(figure_dir, "out")
data_dir <- cli$datadir
if (!nzchar(data_dir)) data_dir <- Sys.getenv("FIG3A_DATADIR", unset = "")
if (!nzchar(data_dir)) data_dir <- file.path(figure_dir, "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
message("Fig. 3a enrichment: ", enrichment_path)

# Match Fig. 2 post-CAP leaf order within each lineage block.
lineage_leaf_orders <- list(
  Lymphoid = c(
    "NK Cells", "ILC3", "GC B Light Zone (GC B LZ)",
    "GC B Dark Zone (GC B DZ)", "Memory B", "Plasma IGG", "Plasma IGA",
    "Naive B", "MAIT Cells", "Gamma Delta T Cells", "NKT Cells",
    "CD8 Memory Exhausted", "CD8 Naive", "CD8 Effector Memory", "CD8 IEL",
    "CD8 Circulating Effector Memory", "CD8 TRM", "CD4 Tr1", "CD4 Tfh",
    "CD4 Naive", "CD4 Tfr", "CD4 tTreg", "CD4 pTreg", "CD4 Th17",
    "CD4 Memory"
  ),
  Stromal = c(
    "Adipocytes", "Smooth Muscle Cells (SMC)", "Glia",
    "Secretory Pericytes", "Angiogenic Pericytes", "Contractile Pericytes",
    "Venular Endothelial", "Medullary Sinus Endothelial",
    "Post Arteriole Capillary Endothelial (PAC)", "Arteriolar Endothelial",
    "Interstitial Cells of Cajal (ICC)",
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)",
    "Follicular Dendritic Cells (fDC)", "Marginal Reticular Cells (MRC)",
    "Myofibroblasts", "Submucosal Fibroblasts (S3)",
    "Muscularis Propria Fibroblasts", "Crypt Bottom Fibroblasts (S2A)",
    "Crypt Top Fibroblasts (S2B)", "Lamina propria Fibroblasts (S1)"
  ),
  Epithelial = c(
    "Microfold Cells (M Cells)", "EEC S", "EEC Progenitors", "EEC L",
    "EEC N", "EEC Enterochromaffin (EC)", "Brunners Gland Cells",
    "Foveolar Cells", "Paneth Cells", "Tuft Progenitors",
    "Mature Goblet Cells", "Intestinal Stem Cells (ISC)",
    "Secretory Progenitors", "BEST4 Enterocytes", "Enterocyte Progenitors",
    "Lower Villus Enterocytes", "Mid Villus Enterocytes",
    "Villus Tip Enterocytes", "Mid Crypt Colonocytes", "BEST4 Colonocytes",
    "Colonocyte Progenitors", "Lower Crypt Colonocytes",
    "Crypt Top Colonocytes"
  ),
  Myeloid = c(
    "Nonclassical Monocytes", "Classical Monocytes", "Cycling Macrophages",
    "Follicle Associated Resident Macrophages",
    "Perivascular Resident Macrophages", "M0 Macrophages",
    "Homeostatic Macrophages", "Monocyte Derived Dendritic Cells (MO DC)",
    "pDC", "cDC1", "migDC", "Tolerogenic cDC2", "Neutrophils",
    "Eosinophils", "Mast Cells"
  )
)

normalize_celltype_label <- function(x) {
  str_squish(str_replace_all(as.character(x), "[\\r\\n]+", " "))
}

anatomy_label <- function(level) {
  dplyr::recode(
    level,
    "WM" = "Whole mucosa",
    "EPI_LP_MUSC" = "Full thickness",
    "EPI_LP" = "EPI & LP",
    "EPI" = "EPI",
    "LP" = "LP",
    "duodenum" = "Duodenum",
    "jejunum" = "Jejunum",
    "ileum" = "Ileum",
    "colon" = "Colon",
    "mesentery" = "Mesentery",
    "accessory" = "Accessory",
    .default = level
  )
}

# Short display labels for long terminal names. Prefer parenthetical acronyms
# already used in the atlas nomenclature; otherwise use compact forms.
short_celltype_label <- function(x) {
  x <- normalize_celltype_label(x)
  dplyr::recode(
    x,
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)" = "mLTo",
    "Post Arteriole Capillary Endothelial (PAC)" = "PAC",
    "Follicle Associated Resident Macrophages" = "FA res. Mac",
    "Monocyte Derived Dendritic Cells (MO DC)" = "MO DC",
    "Interstitial Cells of Cajal (ICC)" = "ICC",
    "Perivascular Resident Macrophages" = "PV res. Mac",
    "Follicular Dendritic Cells (fDC)" = "fDC",
    "CD8 Circulating Effector Memory" = "CD8 circ. EM",
    "Lamina propria Fibroblasts (S1)" = "S1 Fibroblasts",
    "Marginal Reticular Cells (MRC)" = "MRC",
    "Muscularis Propria Fibroblasts" = "MP Fibroblasts",
    "Crypt Bottom Fibroblasts (S2A)" = "S2A Fibroblasts",
    "Medullary Sinus Endothelial" = "Med. sinus Endo",
    "Submucosal Fibroblasts (S3)" = "S3 Fibroblasts",
    "Crypt Top Fibroblasts (S2B)" = "S2B Fibroblasts",
    "Intestinal Stem Cells (ISC)" = "ISC",
    "GC B Light Zone (GC B LZ)" = "GC B LZ",
    "Smooth Muscle Cells (SMC)" = "SMC",
    "Microfold Cells (M Cells)" = "M Cells",
    "EEC Enterochromaffin (EC)" = "EEC EC",
    "GC B Dark Zone (GC B DZ)" = "GC B DZ",
    "Lower Villus Enterocytes" = "Lower villus Ent",
    "Mid Villus Enterocytes" = "Mid villus Ent",
    "Villus Tip Enterocytes" = "Villus tip Ent",
    "Lower Crypt Colonocytes" = "Lower crypt Col",
    "Mid Crypt Colonocytes" = "Mid crypt Col",
    "Crypt Top Colonocytes" = "Crypt top Col",
    "Colonocyte Progenitors" = "Colonocyte prog.",
    "Enterocyte Progenitors" = "Enterocyte prog.",
    "Secretory Progenitors" = "Secretory prog.",
    "Homeostatic Macrophages" = "Homeostatic Mac",
    "Cycling Macrophages" = "Cycling Mac",
    "M0 Macrophages" = "M0 Mac",
    "Nonclassical Monocytes" = "Nonclass. Mono",
    "Classical Monocytes" = "Classical Mono",
    "EEC Progenitors" = "EEC prog.",
    "Tuft Progenitors" = "Tuft prog.",
    "Arteriolar Endothelial" = "Arteriolar Endo",
    "Venular Endothelial" = "Venular Endo",
    "Angiogenic Pericytes" = "Angiogenic Peri",
    "Contractile Pericytes" = "Contractile Peri",
    "Secretory Pericytes" = "Secretory Peri",
    "Gamma Delta T Cells" = "GD T Cells",
    "CD8 Effector Memory" = "CD8 EM",
    "CD8 Memory Exhausted" = "CD8 Mem exh.",
    "Mature Goblet Cells" = "Mature Goblet",
    "Brunners Gland Cells" = "Brunner gland",
    "BEST4 Enterocytes" = "BEST4 Ent",
    "BEST4 Colonocytes" = "BEST4 Col",
    "Tolerogenic cDC2" = "Tol. cDC2",
    .default = x
  )
}

# Wong / Nature diverging scale for signed enrichment (colorblind-safe).
# Low = HCA blue, mid = white, high = vermillion.
clr_low <- "#0072B2"
clr_mid <- "#FFFFFF"
clr_high <- "#D55E00"
clr_limits <- c(-2.25, 2.25)

theme_gca_heatmap <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "Helvetica") +
    theme(
      text = element_text(family = "Helvetica", color = "black", size = base_size),
      plot.title = element_text(
        face = "bold", size = 7, color = "black", hjust = 0,
        margin = margin(b = 2)
      ),
      axis.line = element_blank(),
      axis.ticks = element_line(color = "black", linewidth = 0.25),
      axis.ticks.length = unit(0.8, "mm"),
      axis.text = element_text(color = "black", size = 7),
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
      axis.text.y = element_text(size = 7),
      axis.title = element_text(color = "black", size = 7),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "right",
      legend.title = element_text(size = 7, face = "bold"),
      legend.text = element_text(size = 6.5),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      strip.background = element_blank(),
      strip.text = element_text(size = 7, face = "bold", hjust = 0),
      plot.margin = margin(2, 4, 2, 4)
    )
}

celltype_lineage <- bind_rows(lapply(names(lineage_leaf_orders), function(lin) {
  tibble(
    hgca_celltype_v1 = normalize_celltype_label(lineage_leaf_orders[[lin]]),
    lineage = lin,
    celltype_order = seq_along(lineage_leaf_orders[[lin]])
  )
})) %>%
  mutate(
    lineage = factor(
      lineage, levels = c("Lymphoid", "Stromal", "Epithelial", "Myeloid")
    ),
    global_order = row_number()
  )

enrichment_all <- readr::read_csv(enrichment_path, show_col_types = FALSE) %>%
  mutate(
    hgca_celltype_v1 = normalize_celltype_label(hgca_celltype_v1),
    annotation_group = as.character(annotation_group)
  ) %>%
  filter(annotation_group %in% c("Tissue", "Radial layer"))

# Enrichment table also contains parent/internal taxonomy nodes; Fig. 3a uses
# the same terminal leaves as the Fig. 2 sidecars.
dropped_parents <- setdiff(
  unique(enrichment_all$hgca_celltype_v1),
  celltype_lineage$hgca_celltype_v1
)
if (length(dropped_parents) > 0) {
  message(
    "Excluding ", length(dropped_parents),
    " non-leaf enrichment labels from Fig. 3a: ",
    paste(dropped_parents, collapse = ", ")
  )
}

enrichment <- enrichment_all %>%
  inner_join(celltype_lineage, by = "hgca_celltype_v1")

missing_leaves <- setdiff(
  celltype_lineage$hgca_celltype_v1,
  unique(enrichment$hgca_celltype_v1)
)
if (length(missing_leaves) > 0) {
  stop(
    "Leaf cell types missing from enrichment table: ",
    paste(missing_leaves, collapse = ", ")
  )
}

plot_dat <- enrichment %>%
  mutate(
    anatomy = anatomy_label(annotation_level),
    celltype_label = short_celltype_label(hgca_celltype_v1),
    panel = recode(
      annotation_group,
      "Tissue" = "Segment",
      "Radial layer" = "Radial"
    ),
    panel = factor(panel, levels = c("Segment", "Radial"))
  ) %>%
  arrange(panel, level_order, global_order)

# Stable factor levels: anatomy within panel, cell types left→right by lineage.
anatomy_levels <- plot_dat %>%
  distinct(panel, anatomy, level_order) %>%
  arrange(panel, level_order) %>%
  pull(anatomy) %>%
  unique()

celltype_label_levels <- celltype_lineage %>%
  arrange(lineage, celltype_order) %>%
  mutate(celltype_label = short_celltype_label(hgca_celltype_v1)) %>%
  pull(celltype_label)

if (anyDuplicated(celltype_label_levels)) {
  stop(
    "Short cell-type labels are not unique: ",
    paste(
      unique(celltype_label_levels[duplicated(celltype_label_levels)]),
      collapse = ", "
    )
  )
}

plot_dat <- plot_dat %>%
  mutate(
    anatomy = factor(anatomy, levels = anatomy_levels),
    celltype_label = factor(celltype_label, levels = celltype_label_levels)
  )

readr::write_csv(
  plot_dat %>%
    select(
      panel, annotation_group, annotation_level, anatomy, level_order,
      lineage, hgca_celltype_v1, celltype_label, n_samples, mean_clr, row_z
    ),
  file.path(data_dir, "fig3a_anatomical_clr_long.csv")
)

lineage_bounds <- celltype_lineage %>%
  mutate(celltype_label = short_celltype_label(hgca_celltype_v1)) %>%
  group_by(lineage) %>%
  summarise(
    xmin = min(global_order) - 0.5,
    xmax = max(global_order) + 0.5,
    xmid = mean(global_order),
    .groups = "drop"
  )

n_types <- length(celltype_label_levels)
# Export near 2× Nature double-column width so labels stay legible at 6.5–7 pt.
width_in <- max(7.09, 0.11 * n_types + 1.6)
# Compact vertical footprint for Illustrator assembly.
height_in <- 2.55
# Per-lineage widths vary; keep height low so tiles stay wider than tall.
lineage_height_in <- 1.95

save_base <- function(plot, base, width, height) {
  ggsave(
    paste0(base, ".pdf"), plot,
    width = width, height = height, device = cairo_pdf, bg = "white"
  )
  ggsave(
    paste0(base, ".svg"), plot,
    width = width, height = height, device = svglite::svglite, bg = "white"
  )
  ggsave(
    paste0(base, ".png"), plot,
    width = width, height = height, dpi = 300, bg = "white"
  )
  message("Saved ", base, ".{pdf,svg,png}")
}

# Faceted stack keeps gut-segment / radial blocks tight (no patchwork gap).
make_faceted_heatmap <- function(
  dat,
  title = NULL,
  x_position = c("bottom", "top"),
  show_strip = TRUE,
  legend_barheight_mm = 16
) {
  x_position <- match.arg(x_position)
  x_hjust <- if (x_position == "top") 0 else 1
  p <- ggplot(dat, aes(x = celltype_label, y = anatomy, fill = row_z)) +
    geom_tile(color = "white", linewidth = 0.12, width = 0.96, height = 0.96) +
    geom_vline(
      data = lineage_bounds %>% filter(lineage != "Lymphoid"),
      aes(xintercept = xmin),
      inherit.aes = FALSE,
      color = "black", linewidth = 0.35
    ) +
    scale_fill_gradient2(
      low = clr_low, mid = clr_mid, high = clr_high,
      midpoint = 0, limits = clr_limits, oob = scales::squish,
      name = "CLR enrichment\n(row z-score)",
      breaks = c(-2, -1, 0, 1, 2),
      guide = guide_colorbar(
        barheight = grid::unit(legend_barheight_mm, "mm"),
        barwidth = grid::unit(2.4, "mm"),
        ticks.colour = "black",
        frame.colour = "black",
        frame.linewidth = 0.2
      )
    ) +
    scale_x_discrete(expand = c(0, 0), position = x_position) +
    scale_y_discrete(limits = rev, expand = c(0, 0)) +
    facet_grid(
      panel ~ .,
      scales = "free_y",
      space = "free_y",
      switch = if (show_strip) "y" else NULL
    ) +
    labs(title = title, x = NULL, y = NULL) +
    theme_gca_heatmap() +
    theme(
      axis.text.x = element_text(
        angle = 90, hjust = x_hjust, vjust = 0.5, size = 7
      ),
      panel.spacing.y = unit(1, "mm"),
      strip.placement = "outside",
      strip.text.y.left = element_text(
        angle = 90, size = 7, face = "bold", hjust = 0.5
      )
    )
  if (!show_strip) {
    # strip.text.y.left is more specific than strip.text; blank both.
    p <- p + theme(
      strip.text = element_blank(),
      strip.text.y = element_blank(),
      strip.text.y.left = element_blank(),
      strip.background = element_blank(),
      strip.placement = "inside"
    )
  }
  if (is.null(title) || identical(title, "")) {
    p <- p + theme(plot.title = element_blank())
  }
  p
}

save_base(
  make_faceted_heatmap(
    plot_dat,
    title = "Anatomical CLR enrichment",
    x_position = "bottom",
    show_strip = TRUE
  ),
  file.path(out_dir, "fig3a_anatomical_clr_heatmap_all_lineages"),
  width_in, height_in
)

make_lineage_faceted <- function(
  dat,
  title = NULL,
  x_position = c("bottom", "top"),
  show_strip = TRUE
) {
  x_position <- match.arg(x_position)
  x_hjust <- if (x_position == "top") 0 else 1
  p <- ggplot(dat, aes(x = celltype_label, y = anatomy, fill = row_z)) +
    geom_tile(color = "white", linewidth = 0.12, width = 0.96, height = 0.96) +
    scale_fill_gradient2(
      low = clr_low, mid = clr_mid, high = clr_high,
      midpoint = 0, limits = clr_limits, oob = scales::squish,
      name = "CLR enrichment\n(row z-score)",
      breaks = c(-2, -1, 0, 1, 2),
      guide = guide_colorbar(
        barheight = grid::unit(14, "mm"),
        barwidth = grid::unit(2.4, "mm"),
        ticks.colour = "black",
        frame.colour = "black",
        frame.linewidth = 0.2
      )
    ) +
    scale_x_discrete(expand = c(0, 0), position = x_position) +
    scale_y_discrete(limits = rev, expand = c(0, 0)) +
    facet_grid(
      panel ~ .,
      scales = "free_y",
      space = "free_y",
      switch = "y"
    ) +
    labs(title = title, x = NULL, y = NULL) +
    theme_gca_heatmap() +
    theme(
      axis.text.x = element_text(
        angle = 90, hjust = x_hjust, vjust = 0.5, size = 7
      ),
      panel.spacing.y = unit(1, "mm"),
      strip.placement = "outside",
      strip.text.y.left = element_text(
        angle = 90, size = 7, face = "bold", hjust = 0.5
      )
    )
  if (!show_strip) {
    p <- p + theme(
      strip.text = element_blank(),
      strip.text.y = element_blank(),
      strip.text.y.left = element_blank(),
      strip.background = element_blank(),
      strip.placement = "inside"
    )
  }
  if (is.null(title) || identical(title, "")) {
    p <- p + theme(plot.title = element_blank())
  }
  p
}

for (lin in levels(celltype_lineage$lineage)) {
  lin_labels <- celltype_lineage %>%
    filter(lineage == lin) %>%
    mutate(celltype_label = short_celltype_label(hgca_celltype_v1)) %>%
    pull(celltype_label)
  lin_dat <- plot_dat %>%
    filter(as.character(celltype_label) %in% lin_labels) %>%
    mutate(celltype_label = factor(celltype_label, levels = lin_labels))
  lin_width <- max(3.54, 0.16 * length(lin_labels) + 1.4)

  save_base(
    make_lineage_faceted(
      lin_dat,
      title = lin,
      x_position = "bottom",
      show_strip = TRUE
    ),
    file.path(out_dir, paste0("fig3a_anatomical_clr_heatmap_", tolower(lin))),
    width = lin_width,
    height = lineage_height_in
  )
}

# Variant: no panel titles; cell-type labels on the top axis.
save_base(
  make_faceted_heatmap(
    plot_dat,
    title = NULL,
    x_position = "top",
    show_strip = FALSE,
    legend_barheight_mm = 14
  ),
  file.path(
    out_dir, "fig3a_anatomical_clr_heatmap_all_lineages_no_title_labels_top"
  ),
  width_in, height_in
)

for (lin in levels(celltype_lineage$lineage)) {
  lin_labels <- celltype_lineage %>%
    filter(lineage == lin) %>%
    mutate(celltype_label = short_celltype_label(hgca_celltype_v1)) %>%
    pull(celltype_label)
  lin_dat <- plot_dat %>%
    filter(as.character(celltype_label) %in% lin_labels) %>%
    mutate(celltype_label = factor(celltype_label, levels = lin_labels))

  save_base(
    make_lineage_faceted(
      lin_dat,
      title = NULL,
      x_position = "top",
      show_strip = FALSE
    ),
    file.path(
      out_dir,
      paste0(
        "fig3a_anatomical_clr_heatmap_", tolower(lin), "_no_title_labels_top"
      )
    ),
    width = max(3.54, 0.16 * length(lin_labels) + 1.4),
    height = lineage_height_in
  )
}

message("Fig. 3a anatomical CLR heatmaps complete.")
