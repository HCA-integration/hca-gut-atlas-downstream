#!/usr/bin/env Rscript
# Gene-expression side of the sampling-depth story (see depth_expression_de.py):
#   r  variance-explained ranking: which cell types' pseudobulk GEX covaries most
#      with each depth category (radial layer, biopsy/resection, full thickness).
#   s  gene volcanoes (biopsy vs surgical resection) for the top depth-covarying
#      cell types + Glia.
#   t  gene volcanoes (full thickness vs rest) for the same selected set.
#
# plot_specs.md: Helvetica 5-7 pt, no gridlines, open axes, Wong palette,
# vector cairo PDF + SVG + 300 dpi PNG at exact final size.

suppressPackageStartupMessages({
  library(ggplot2); library(ggrepel); library(readr); library(dplyr)
  library(stringr); library(tidyr); library(svglite); library(ragg)
})

HERE <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 error = function(e) ".")
if (length(HERE) == 0 || HERE == "") HERE <- "."
DATA <- file.path(HERE, "..", "data")
DE   <- file.path(DATA, "depth_de")
OUT  <- file.path(HERE, "..", "out")
MM <- 25.4

LINEAGE_COL <- c(epithelial = "#009E73", lymphoid = "#0072B2",
                 myeloid = "#E69F00", stroma = "#CC79A7")
COV_LABEL <- c(radial_tissue_term = "Radial layer (5 levels)",
               sample_collection_method = "Biopsy vs resection",
               full_thickness = "Full thickness vs rest")
# DE volcano point colours: direction of the B - A contrast
UP_COL <- "#D55E00"; DN_COL <- "#0072B2"; NS_COL <- "#BFBFBF"

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
      legend.position = "bottom", legend.title = element_blank(),
      legend.text = element_text(size = base, colour = "black"),
      legend.key.size = unit(3, "mm"),
      strip.background = element_blank(),
      strip.text = element_text(size = 5, colour = "black"),
      panel.spacing = unit(1.5, "mm"),
      plot.background = element_blank(), panel.background = element_blank()
    )
}

save_panel <- function(p, stem, width_mm, height_mm) {
  wi <- width_mm / MM; hi <- height_mm / MM
  ggsave(file.path(OUT, paste0(stem, ".pdf")), p, width = wi, height = hi, device = cairo_pdf)
  ggsave(file.path(OUT, paste0(stem, ".svg")), p, width = wi, height = hi, device = svglite)
  ggsave(file.path(OUT, paste0(stem, ".png")), p, width = wi, height = hi, dpi = 300,
         device = ragg::agg_png)
  message("  wrote ", stem, ".{pdf,svg,png}")
}

# ---------------------------------------------------------------- r: ranking
ve <- suppressMessages(read_csv(file.path(DATA, "depth_gex_variance_explained.csv"),
                                show_col_types = FALSE)) %>%
  filter(!is.na(omega2)) %>%
  mutate(lineage = factor(lineage, levels = names(LINEAGE_COL)),
         covariate = factor(covariate,
                            levels = c("radial_tissue_term",
                                       "sample_collection_method", "full_thickness")))

N_SHOW <- 18
rank_top <- ve %>% group_by(covariate) %>%
  slice_max(omega2, n = N_SHOW, with_ties = FALSE) %>% ungroup()

# order cell types within each facet by omega2 (facet-independent ordering via tidytext-free trick)
rank_top <- rank_top %>%
  arrange(covariate, omega2) %>%
  mutate(row = row_number(),
         ct_lab = paste0(celltype, "___", as.integer(covariate)))

p_rank <- ggplot(rank_top, aes(x = omega2, y = reorder(ct_lab, row))) +
  geom_segment(aes(x = 0, xend = omega2, yend = reorder(ct_lab, row)),
               colour = "grey70", linewidth = 0.3) +
  geom_point(aes(colour = lineage), size = 1.4) +
  facet_wrap(~ covariate, scales = "free_y", ncol = 3,
             labeller = as_labeller(COV_LABEL)) +
  scale_colour_manual(values = LINEAGE_COL, drop = FALSE) +
  scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
  labs(x = "GEX variance explained (variance-weighted \u03c9\u00b2)", y = NULL,
       title = "r  Cell types whose pseudobulk expression covaries most with sampling depth") +
  theme_gca() +
  theme(axis.text.y = element_text(size = 4.3))

save_panel(p_rank, "panel_r_gex_variance_explained_by_depth", 180, 120)

# ---------------------------------------------------------------- s/t: gene volcanoes
sel <- suppressMessages(read_csv(file.path(DE, "selected_celltypes.csv"),
                                 show_col_types = FALSE))
sel_order <- sel %>% arrange(desc(max_depth_omega2)) %>% pull(celltype)

slugify <- function(x) gsub("_+", "_", gsub("[^0-9A-Za-z]+", "_", x)) |> (\(z) gsub("^_|_$", "", z))()

load_contrast <- function(contrast, celltypes) {
  rows <- list()
  for (ct in celltypes) {
    f <- file.path(DE, contrast, paste0(slugify(ct), "_de.csv"))
    if (file.exists(f)) {
      d <- suppressMessages(read_csv(f, show_col_types = FALSE))
      rows[[ct]] <- d
    }
  }
  if (!length(rows)) return(NULL)
  bind_rows(rows)
}

volcano_grid <- function(contrast, x_left, x_right, letter, stem) {
  d <- load_contrast(contrast, sel_order)
  if (is.null(d)) { message("  no DE for ", contrast); return(invisible()) }
  # per-cell-type adjustment mode -> facet label (flag study-confounded fallbacks)
  if (!"adjusted" %in% names(d)) d$adjusted <- TRUE
  mode_tbl <- d %>% distinct(celltype, adjusted) %>%
    mutate(ct_label = ifelse(adjusted, celltype,
                             paste0(celltype, "  \u2020 unadjusted")))
  lab_levels <- mode_tbl$ct_label[match(intersect(sel_order, mode_tbl$celltype),
                                        mode_tbl$celltype)]
  d <- d %>% left_join(mode_tbl, by = c("celltype", "adjusted")) %>%
    mutate(celltype = factor(ct_label, levels = lab_levels),
           dir = case_when(
             p_adj < 0.05 & log2fc_B_minus_A > 0 ~ "up",
             p_adj < 0.05 & log2fc_B_minus_A < 0 ~ "dn",
             TRUE ~ "ns"))
  # cap y for display and pick labels per facet (top by p_adj each side)
  lab <- d %>% filter(p_adj < 0.05) %>% group_by(celltype) %>%
    group_modify(~{
      up <- .x %>% filter(log2fc_B_minus_A > 0) %>% slice_min(p_adj, n = 6, with_ties = FALSE)
      dn <- .x %>% filter(log2fc_B_minus_A < 0) %>% slice_min(p_adj, n = 6, with_ties = FALSE)
      bind_rows(up, dn)
    }) %>% ungroup()

  ncol <- 4
  nfac <- length(levels(d$celltype))
  nrow <- ceiling(nfac / ncol)
  height_mm <- min(170, 18 + nrow * 34)

  p <- ggplot(d, aes(log2fc_B_minus_A, neglog10_p_adj)) +
    geom_hline(yintercept = -log10(0.05), colour = "grey60", linetype = "dashed",
               linewidth = 0.2) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.2) +
    geom_point(aes(colour = dir), size = 0.4, alpha = 0.55, stroke = 0) +
    geom_text_repel(data = lab, aes(label = gene_symbol), size = 1.4,
                    family = "Helvetica", colour = "black", segment.size = 0.12,
                    segment.colour = "grey60", min.segment.length = 0,
                    box.padding = 0.12, max.overlaps = Inf, seed = 1,
                    max.time = 1, max.iter = 12000) +
    facet_wrap(~ celltype, ncol = ncol, scales = "free",
               labeller = label_wrap_gen(width = 24)) +
    scale_colour_manual(values = c(up = UP_COL, dn = DN_COL, ns = NS_COL),
                        breaks = c("dn", "up"),
                        labels = c(paste0("higher in ", x_left),
                                   paste0("higher in ", x_right))) +
    labs(x = paste0("log\u2082 fold change (", x_right, " \u2212 ", x_left, ")"),
         y = "\u2212log\u2081\u2080 (FDR)",
         title = paste0(letter, "  Depth DE (", x_left, " vs ", x_right,
                        "): top depth-covarying cell types + Glia"),
         caption = paste0("dataset_id-adjusted where a study spans both arms; ",
                          "\u2020 unadjusted = contrast fully nested in study, ",
                          "effect confounded with study-of-origin")) +
    theme_gca() +
    theme(plot.caption = element_text(size = 4.5, colour = "grey30", hjust = 0)) +
    guides(colour = guide_legend(override.aes = list(size = 2, alpha = 1)))

  save_panel(p, stem, 180, height_mm)
}

volcano_grid("collection", "biopsy", "resection", "s",
             "panel_s_depth_de_collection_volcanoes")
volcano_grid("full_thickness", "rest", "full thickness", "t",
             "panel_t_depth_de_full_thickness_volcanoes")

message("done depth expression figures.")
