#!/usr/bin/env Rscript
# Nature-spec CLR composition volcanoes (ggplot2 + ggrepel) for the sampling
# depth / radial-layer story. Reads the recomputed Mann-Whitney tables from
# ../data/ (see recompute_clr_tables.py) and renders lineage-coloured volcanoes
# with proper label repulsion.
#
# Follows ../../plot_specs.md: Helvetica 5-7 pt, no gridlines, open L-shaped
# axes, colourblind-safe (Wong) palette, colour on markers only (labels black),
# vector cairo PDF + SVG + 300 dpi PNG at exact final size (90 or 180 mm).
#
# Two label modes per contrast:
#   default  - strongest significant hits per side
#   follicle - default PLUS every follicle/TLO niche cell type, even if it
#              does not pass FDR (so reviewers can see where the niche sits)

suppressPackageStartupMessages({
  library(ggplot2); library(ggrepel); library(readr)
  library(dplyr); library(stringr); library(tidyr)
  library(svglite); library(ragg); library(patchwork)
})

HERE <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 error = function(e) ".")
if (length(HERE) == 0 || HERE == "") HERE <- "."
DATA <- file.path(HERE, "..", "data")
OUT  <- file.path(HERE, "..", "out")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

MM <- 25.4

# Lineage colours (Wong colourblind-safe, 4 well-separated hues)
LINEAGE_COL <- c(
  epithelial = "#009E73",  # bluish green
  lymphoid   = "#0072B2",  # HCA blue
  myeloid    = "#E69F00",  # orange
  stroma     = "#CC79A7"   # reddish purple
)

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text            = element_text(colour = "black", family = "Helvetica"),
      plot.title      = element_text(size = 7, hjust = 0, face = "plain", colour = "black"),
      axis.line       = element_line(colour = "black", linewidth = 0.25),
      axis.ticks      = element_line(colour = "black", linewidth = 0.25),
      axis.text       = element_text(colour = "black", size = base),
      axis.title      = element_text(colour = "black", size = base),
      panel.grid      = element_blank(),
      legend.position = "bottom",
      legend.title    = element_blank(),
      legend.text     = element_text(size = base, colour = "black"),
      legend.key.size = unit(3, "mm"),
      strip.background = element_blank(),
      strip.text      = element_text(size = base, colour = "black", face = "plain"),
      plot.background = element_blank(),
      panel.background = element_blank()
    )
}

pick_labels <- function(df, n_side = 12, follicle = FALSE) {
  sig <- df %>% filter(p_adj < 0.05)
  up  <- sig %>% arrange(desc(delta_CLR_B_minus_A)) %>% slice_head(n = n_side)
  dn  <- sig %>% arrange(delta_CLR_B_minus_A) %>% slice_head(n = n_side)
  keep <- union(up$celltype, dn$celltype)
  if (follicle) keep <- union(keep, df$celltype[df$is_follicle_tlo])
  keep
}

make_volcano <- function(df, x_left, x_right, title, follicle = FALSE,
                         label_size = 1.7) {
  df <- df %>%
    mutate(neglog10_p_adj = -log10(pmax(p_adj, 1e-300)),
           lineage = factor(lineage, levels = names(LINEAGE_COL)))
  keep <- pick_labels(df, follicle = follicle)
  df$lab <- ifelse(df$celltype %in% keep, df$celltype, "")
  xmax <- max(abs(df$delta_CLR_B_minus_A), na.rm = TRUE) * 1.15
  ymax <- max(df$neglog10_p_adj, na.rm = TRUE) * 1.10

  ggplot(df, aes(delta_CLR_B_minus_A, neglog10_p_adj)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_hline(yintercept = -log10(0.05), colour = "grey60",
               linetype = "dashed", linewidth = 0.25) +
    geom_point(aes(fill = lineage), shape = 21, colour = "black",
               stroke = 0.15, size = 1.15, alpha = 0.9) +
    geom_text_repel(
      aes(label = lab), size = label_size, family = "Helvetica",
      colour = "black", segment.size = 0.15, segment.colour = "grey55",
      min.segment.length = 0, box.padding = 0.18, point.padding = 0.1,
      max.overlaps = Inf, seed = 1, force = 1.4, force_pull = 0.4,
      max.time = 2, max.iter = 20000
    ) +
    scale_fill_manual(values = LINEAGE_COL, drop = FALSE,
                      guide = guide_legend(override.aes = list(size = 2))) +
    annotate("text", x = -xmax * 0.98, y = ymax, hjust = 0, vjust = 1,
             label = paste0("Enriched: ", x_left), size = 1.75,
             fontface = "italic", colour = "black") +
    annotate("text", x = xmax * 0.98, y = ymax, hjust = 1, vjust = 1,
             label = paste0("Enriched: ", x_right), size = 1.75,
             fontface = "italic", colour = "black") +
    scale_x_continuous(limits = c(-xmax, xmax)) +
    scale_y_continuous(limits = c(0, ymax), expand = expansion(mult = c(0, 0.02))) +
    labs(x = bquote(Delta ~ "CLR composition  (" * .(x_right) ~ "\u2212" ~ .(x_left) * ")"),
         y = expression(-log[10] * " (FDR)"), title = title) +
    theme_gca()
}

save_panel <- function(p, stem, width_mm, height_mm) {
  wi <- width_mm / MM; hi <- height_mm / MM
  ggsave(file.path(OUT, paste0(stem, ".pdf")), p, width = wi, height = hi,
         device = cairo_pdf, bg = "white")
  ggsave(file.path(OUT, paste0(stem, ".svg")), p, width = wi, height = hi,
         device = svglite, bg = "white")
  ggsave(file.path(OUT, paste0(stem, ".png")), p, width = wi, height = hi,
         dpi = 300, device = ragg::agg_png, bg = "white")
  message("  wrote ", stem, ".{pdf,svg,png}")
}

rd <- function(...) suppressMessages(read_csv(file.path(DATA, ...), show_col_types = FALSE))

# ---------------------------------------------------------------------------
collection <- rd("clr_wilcoxon_collection.csv")
radial     <- rd("clr_wilcoxon_radial_epi_lp.csv")
fullthick  <- rd("clr_wilcoxon_full_thickness.csv")

# a: collection global (90 mm) + follicle variant
save_panel(make_volcano(collection, "biopsy", "resection",
           "a  Cell-type composition: biopsy vs surgical resection"),
           "panel_a_collection_volcano_global", 90, 100)
save_panel(make_volcano(collection, "biopsy", "resection",
           "a  Biopsy vs resection (follicle / TLO niche labelled)", follicle = TRUE),
           "panel_a_collection_volcano_global_follicle", 90, 100)

# b: radial EPI vs LP global (90 mm) + follicle variant
save_panel(make_volcano(radial, "epithelial layer", "lamina propria",
           "b  Composition across mucosal depth (EPI vs LP)"),
           "panel_b_radial_volcano_global", 90, 100)
save_panel(make_volcano(radial, "epithelial layer", "lamina propria",
           "b  Mucosal depth EPI vs LP (follicle / TLO niche labelled)", follicle = TRUE),
           "panel_b_radial_volcano_global_follicle", 90, 100)

# h: full-thickness (EPI_LP_MUSC) vs all other radial (90 mm) + follicle variant
save_panel(make_volcano(fullthick, "other layers", "full thickness",
           "h  Full-thickness (EPI_LP_MUSC) vs all other radial layers"),
           "panel_h_full_thickness_volcano_global", 90, 100)
save_panel(make_volcano(fullthick, "other layers", "full thickness",
           "h  Full thickness vs rest (follicle / TLO niche labelled)", follicle = TRUE),
           "panel_h_full_thickness_volcano_global_follicle", 90, 100)

# by-tissue small multiples (180 mm): collection + full-thickness
render_by_tissue <- function(kind, x_left, x_right, letter, follicle) {
  tissues <- c("colon", "ileum")
  frames <- lapply(tissues, function(t) {
    f <- file.path(DATA, "by_tissue", t, paste0("clr_wilcoxon_", kind, "_", t, ".csv"))
    if (!file.exists(f)) return(NULL)
    d <- suppressMessages(read_csv(f, show_col_types = FALSE)); d$tissue <- t; d
  })
  frames <- Filter(Negate(is.null), frames)
  if (!length(frames)) return(invisible())
  plots <- lapply(seq_along(frames), function(i) {
    t <- frames[[i]]$tissue[1]
    ttl <- paste0(if (i == 1) paste0(letter, "  ") else "",
                  str_to_title(t), ": ", x_left, " vs ", x_right)
    make_volcano(frames[[i]], x_left, x_right, ttl, follicle = follicle)
  })
  # combine side by side via patchwork if available, else cowplot-free grid
  if (requireNamespace("patchwork", quietly = TRUE)) {
    g <- patchwork::wrap_plots(plots, ncol = length(plots)) +
      patchwork::plot_layout(guides = "collect") &
      theme(legend.position = "bottom")
  } else {
    g <- plots[[1]]  # fallback: first tissue only
  }
  suffix <- if (follicle) "_follicle" else ""
  stem <- paste0("panel_", if (kind == "collection") "c" else "i",
                 "_", kind, "_volcano_by_tissue", suffix)
  save_panel(g, stem, 180, 105)
}

render_by_tissue("collection", "biopsy", "resection", "c", FALSE)
render_by_tissue("collection", "biopsy", "resection", "c", TRUE)
render_by_tissue("full_thickness", "other layers", "full thickness", "i", FALSE)
render_by_tissue("full_thickness", "other layers", "full thickness", "i", TRUE)

# ---------------------------------------------------------------------------
# Main-text Figure 3 composite: rendered once at final Nature dimensions.
#
# Layout at 180 x 170 mm (no post-hoc scaling):
#   a  cell-type / covariate sensitivity heat map             180 x 43 mm
#   b  PCR lollipops + directly plotted anatomy:batch ratio   180 x 37 mm
#   c  biopsy vs resection     | d full thickness vs rest      90 x 47 mm each
#   e  representative ileum/colon CLR box plots               180 x 37 mm
#
# Every text element is generated at 5-7 pt in the final composite. Nested
# plots are wrapped as single panels so patchwork adds exactly labels a-e.

pretty_cov <- c(
  sampled_site_condition = "Sample condition",
  radial_tissue_term = "Radial layer",
  sample_preservation_method = "Preservation",
  sex_ontology_term = "Sex",
  age_range = "Age",
  dataset_id = "Study / batch",
  assay = "Assay",
  sample_collection_method = "Collection",
  sequenced_fragment = "Sequenced fragment",
  gene_annotation_version = "Gene annotation",
  tissue_level_1 = "Gut region"
)

pretty_celltype <- function(x) {
  recode(
    x,
    "Submucosal Fibroblasts (S3)" = "S3 fibroblasts",
    "Lamina propria Fibroblasts (S1)" = "S1 fibroblasts",
    "Post Arteriole Capillary Endothelial (PAC)" = "PAC endothelial",
    "Crypt Top Fibroblasts (S2B)" = "S2B fibroblasts",
    "Crypt Bottom Fibroblasts (S2A)" = "S2A fibroblasts",
    "Perivascular Resident Macrophages" = "Perivascular macrophages",
    "Follicle Associated Resident Macrophages" = "Follicle-associated macrophages",
    "Lymphatic Endothelial" = "Lymphatic endothelial",
    "Arteriolar Endothelial" = "Arteriolar endothelial",
    "Mesenchymal Lymphoid Tissue Organizer Cells (mLTo Cells)" = "mLTo cells",
    "Follicular Dendritic Cells (fDC)" = "fDC",
    "GC B Light Zone (GC B LZ)" = "GC B LZ",
    "GC B Dark Zone (GC B DZ)" = "GC B DZ",
    "CD4 Tfh" = "Tfh",
    "CD4 Tfr" = "Tfr",
    "Villus Tip Enterocytes" = "Villus-tip enterocytes",
    .default = x
  )
}

theme_composite <- function(base = 5.5) {
  theme_gca(base) +
    theme(
      plot.title = element_text(size = 6, face = "plain", margin = margin(b = 1.5)),
      axis.text = element_text(size = 5),
      axis.title = element_text(size = 5.5),
      strip.text = element_text(size = 5, face = "plain"),
      legend.text = element_text(size = 5),
      legend.key.size = unit(2.5, "mm"),
      plot.margin = margin(2, 2, 2, 2, "pt")
    )
}

make_panel_a <- function() {
  h <- suppressMessages(read_csv(
    file.path(DATA, "celltype_metadata_sensitivity_top2.csv"),
    show_col_types = FALSE
  ))
  names(h)[1] <- "celltype"
  row_order <- rev(h$celltype)
  cov_order <- names(h)[-1]
  h <- h %>%
    pivot_longer(-celltype, names_to = "covariate", values_to = "sensitivity") %>%
    mutate(
      celltype = factor(celltype, levels = row_order),
      covariate = factor(covariate, levels = cov_order),
      block = factor(
        ifelse(covariate %in% c(
          "sampled site condition", "radial tissue term",
          "sample preservation method", "sex ontology term", "age range"
        ), "Biological", "Technical"),
        levels = c("Biological", "Technical")
      )
    )

  ggplot(h, aes(covariate, celltype, fill = sensitivity)) +
    geom_tile(colour = "white", linewidth = 0.15) +
    geom_text(aes(label = sprintf("%.2f", sensitivity)),
              family = "Helvetica", size = 1.75, colour = "black") +
    facet_grid(~ block, scales = "free_x", space = "free_x") +
    scale_fill_gradientn(
      colours = c("#FFFFFF", "#F0E442", "#E69F00", "#D55E00"),
      values = c(0, 0.32, 0.68, 1), limits = c(0, 0.60),
      oob = scales::squish, name = "Sensitivity\n(0-1)"
    ) +
    scale_x_discrete(labels = function(x) str_wrap(str_to_sentence(x), 14)) +
    labs(
      title = "Cell-type composition is sensitive to biological and technical context",
      x = NULL, y = NULL
    ) +
    theme_composite() +
    theme(
      axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, size = 5),
      axis.text.y = element_text(size = 5),
      strip.text = element_text(size = 5.5),
      panel.spacing.x = unit(1.5, "mm"),
      legend.position = "right"
    )
}

make_panel_b_parts <- function(compact = FALSE, show_lollipop_legend = FALSE) {
  pcr <- suppressMessages(read_csv(
    file.path(DATA, "composition_vs_expression_pcr_long.csv"),
    show_col_types = FALSE
  ))
  wide <- pcr %>%
    group_by(block, covariate, modality) %>%
    summarise(pcr = mean(pcr, na.rm = TRUE), .groups = "drop") %>%
    mutate(pcr = ifelse(is.nan(pcr), NA_real_, pcr)) %>%
    pivot_wider(names_from = modality, values_from = pcr) %>%
    filter(is.finite(composition), is.finite(expression)) %>%
    arrange(expression) %>%
    mutate(
      covariate_label = factor(
        unname(pretty_cov[covariate]),
        levels = unname(pretty_cov[covariate])
      )
    )
  long_pcr <- wide %>%
    select(covariate_label, composition, expression) %>%
    pivot_longer(c(composition, expression), names_to = "modality", values_to = "pcr") %>%
    mutate(modality = factor(modality, levels = c("composition", "expression")))

  point_size <- if (compact) 1.35 else 1.55
  lollipop <- ggplot(wide, aes(y = covariate_label)) +
    geom_segment(aes(x = composition, xend = expression,
                     yend = covariate_label),
                 colour = "grey70", linewidth = if (compact) 0.35 else 0.45) +
    geom_point(
      data = long_pcr,
      aes(x = pcr, fill = modality, colour = modality),
      shape = 21, size = point_size, stroke = 0.32
    ) +
    scale_fill_manual(
      values = c(composition = "white", expression = "#D55E00"),
      labels = c(composition = "Composition (CLR)",
                 expression = "Expression (pseudobulk)")
    ) +
    scale_colour_manual(
      values = c(composition = "#0072B2", expression = "black"),
      labels = c(composition = "Composition (CLR)",
                 expression = "Expression (pseudobulk)")
    ) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.06))) +
    labs(
      title = "Variance explained by each covariate",
      x = expression("Variance explained (weighted " * omega^2 * ")"),
      y = NULL
    ) +
    theme_composite() +
    theme(
      legend.position = if (show_lollipop_legend) "bottom" else "none",
      legend.box = "horizontal",
      legend.margin = margin(0, 0, 0, 0),
      plot.margin = margin(1, 1, 1, 1, "pt")
    )

  # Descriptive anatomy-to-study effect-size ratio. This is not an established
  # inferential metric: show all four lineage ratios and the ratio of lineage
  # means, rather than a fold-change annotation that hides heterogeneity.
  ratio_lineage <- pcr %>%
    filter(covariate %in% c("dataset_id", "tissue_level_1")) %>%
    select(lineage, modality, covariate, pcr) %>%
    pivot_wider(names_from = covariate, values_from = pcr) %>%
    mutate(
      ratio = tissue_level_1 / dataset_id,
      modality = factor(modality, levels = c("composition", "expression")),
      lineage = factor(lineage, levels = names(LINEAGE_COL))
    )
  pooled_components <- pcr %>%
    filter(covariate %in% c("dataset_id", "tissue_level_1")) %>%
    group_by(modality, covariate) %>%
    summarise(pcr = mean(pcr, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = covariate, values_from = pcr) %>%
    mutate(
      ratio = tissue_level_1 / dataset_id,
      modality = factor(modality, levels = c("composition", "expression"))
    )

  ratio_plot <- ggplot(
    ratio_lineage,
    aes(ratio, modality, fill = lineage)
  ) +
    geom_vline(xintercept = 1, linetype = "dashed",
               colour = "grey65", linewidth = 0.25) +
    geom_point(shape = 21, size = if (compact) 1.4 else 1.65,
               colour = "black", stroke = 0.25) +
    geom_point(
      data = pooled_components,
      aes(ratio, modality), inherit.aes = FALSE,
      shape = 23, size = if (compact) 1.85 else 2.1,
      fill = "black", colour = "black"
    ) +
    geom_text(
      data = pooled_components,
      aes(ratio, modality, label = sprintf("%.2f", ratio)),
      inherit.aes = FALSE, hjust = 0.5,
      position = position_nudge(y = -0.22), family = "Helvetica",
      fontface = "bold", size = if (compact) 1.55 else 1.75, colour = "black"
    ) +
    scale_fill_manual(values = LINEAGE_COL) +
    scale_y_discrete(labels = c(
      composition = "Composition\n(CLR)",
      expression = "Expression\n(pseudobulk)"
    )) +
    scale_x_continuous(
      limits = c(0, 1.02), breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = "Anatomy-to-study variance ratio",
      x = expression(omega^2 * " gut region / " * omega^2 * " study"),
      y = NULL
    ) +
    theme_composite() +
    theme(
      legend.position = "none",
      plot.margin = margin(1, 1, 1, 1, "pt")
    )

  list(lollipop = lollipop, ratio = ratio_plot)
}

make_panel_b <- function(compact = FALSE) {
  parts <- make_panel_b_parts(compact = compact, show_lollipop_legend = FALSE)
  wrap_plots(parts$lollipop, parts$ratio, widths = c(1.55, 1))
}

# Half-page (one column): lollipop fills the column; anatomy:study ratio under it.
make_panel_b_halfpage <- function() {
  parts <- make_panel_b_parts(compact = TRUE, show_lollipop_legend = TRUE)
  wrap_plots(parts$lollipop, parts$ratio, ncol = 1, heights = c(2.35, 1))
}

make_compact_volcano <- function(df, left, right, title,
                                 n_side = 6, extra_labels = character()) {
  d <- df %>%
    mutate(
      neglog10_p_adj = -log10(pmax(p_adj, 1e-300)),
      lineage = factor(lineage, levels = names(LINEAGE_COL))
    )
  sig <- d %>% filter(p_adj < 0.05)
  keep <- union(
    sig %>% slice_min(delta_CLR_B_minus_A, n = n_side,
                      with_ties = FALSE) %>% pull(celltype),
    sig %>% slice_max(delta_CLR_B_minus_A, n = n_side,
                      with_ties = FALSE) %>% pull(celltype)
  )
  keep <- union(keep, extra_labels)
  d <- d %>% mutate(label = ifelse(celltype %in% keep,
                                   pretty_celltype(celltype), ""))
  xmax <- max(abs(d$delta_CLR_B_minus_A), na.rm = TRUE) * 1.12
  ymax <- max(d$neglog10_p_adj, na.rm = TRUE) * 1.06

  ggplot(d, aes(delta_CLR_B_minus_A, neglog10_p_adj)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_hline(yintercept = -log10(0.05), colour = "grey60",
               linetype = "dashed", linewidth = 0.25) +
    geom_point(aes(fill = lineage), shape = 21, colour = "black",
               stroke = 0.12, size = 0.9) +
    geom_text_repel(
      aes(label = label), family = "Helvetica", size = 1.75,
      colour = "black", segment.size = 0.12, segment.colour = "grey55",
      min.segment.length = 0, box.padding = 0.12, point.padding = 0.06,
      max.overlaps = Inf, seed = 4, max.time = 4, max.iter = 30000,
      ylim = c(0, ymax * 0.91)
    ) +
    scale_fill_manual(values = LINEAGE_COL, drop = FALSE) +
    scale_x_continuous(limits = c(-xmax, xmax),
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(limits = c(0, ymax),
                       expand = expansion(mult = c(0, 0.02))) +
    annotate("text", x = -xmax, y = ymax, label = left,
             hjust = 0, vjust = 1, family = "Helvetica",
             fontface = "italic", size = 1.75) +
    annotate("text", x = xmax, y = ymax, label = right,
             hjust = 1, vjust = 1, family = "Helvetica",
             fontface = "italic", size = 1.75) +
    labs(
      title = title,
      x = expression(Delta * " CLR composition"),
      y = expression(-log[10] * " (FDR)")
    ) +
    theme_composite() +
    theme(legend.position = "none")
}

segment_support_label <- function(d) {
  abbrev <- c(
    epithelial = "Epi", lymphoid = "Lym",
    myeloid = "Mye", stroma = "Str"
  )
  d %>%
    distinct(lineage, n_A, n_B) %>%
    mutate(
      lineage = factor(lineage, levels = names(abbrev)),
      text = paste0(abbrev[as.character(lineage)], " ", n_A, "/", n_B)
    ) %>%
    arrange(lineage) %>%
    pull(text) %>%
    paste(collapse = " · ")
}

make_descriptive_segment <- function(d, title, n_side = 5) {
  shown <- bind_rows(
    d %>% slice_min(delta_CLR_B_minus_A, n = n_side, with_ties = FALSE),
    d %>% slice_max(delta_CLR_B_minus_A, n = n_side, with_ties = FALSE)
  ) %>%
    distinct(celltype, .keep_all = TRUE) %>%
    arrange(delta_CLR_B_minus_A) %>%
    mutate(
      label = factor(
        pretty_celltype(celltype),
        levels = pretty_celltype(celltype)
      ),
      lineage = factor(lineage, levels = names(LINEAGE_COL))
    )
  xmax <- max(abs(shown$delta_CLR_B_minus_A), na.rm = TRUE) * 1.12

  ggplot(shown, aes(delta_CLR_B_minus_A, label)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.25) +
    geom_segment(
      aes(x = 0, xend = delta_CLR_B_minus_A, yend = label),
      colour = "grey70", linewidth = 0.35
    ) +
    geom_point(
      aes(fill = lineage), shape = 21, colour = "black",
      stroke = 0.15, size = 1.35
    ) +
    scale_fill_manual(values = LINEAGE_COL, drop = FALSE) +
    scale_x_continuous(
      limits = c(-xmax, xmax),
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    labs(
      title = title,
      subtitle = paste0(
        "Descriptive only; rest/full n: ", segment_support_label(d)
      ),
      x = expression(Delta * " CLR composition"),
      y = NULL
    ) +
    theme_composite() +
    theme(
      legend.position = "none",
      plot.subtitle = element_text(size = 4.5, colour = "black"),
      axis.text.y = element_text(size = 4.5)
    )
}

read_segment_depth <- function(segment) {
  rd(
    "by_tissue", segment,
    paste0("clr_wilcoxon_full_thickness_", segment, ".csv")
  )
}

make_inferential_segment <- function(d, title) {
  make_compact_volcano(
    d, "Other layers", "Full thickness", title, n_side = 1,
    extra_labels = c(
      "Submucosal Fibroblasts (S3)",
      "Lymphatic Endothelial",
      "Perivascular Resident Macrophages",
      "Adipocytes",
      "Lamina propria Fibroblasts (S1)"
    )
  ) +
    labs(subtitle = paste0(
      "Rest/full n: ", segment_support_label(d)
    )) +
    theme(plot.subtitle = element_text(size = 4.5, colour = "black"))
}

make_panel_e <- function() {
  selected <- c(
    "Submucosal Fibroblasts (S3)",
    "Lamina propria Fibroblasts (S1)",
    "Lymphatic Endothelial",
    "Perivascular Resident Macrophages",
    "Adipocytes"
  )
  # Pack Ileum/Colon close (continuous x) so facets aren't sparse
  tissue_gap <- 0.95   # distance between tissue group centers
  pair_half  <- 0.18   # half-distance between biopsy/resection within a tissue
  box_w      <- 0.30
  jit_w      <- 0.04
  d <- suppressMessages(read_csv(file.path(DATA, "clr_long.csv"),
                                 show_col_types = FALSE)) %>%
    filter(
      celltype %in% selected,
      tissue_level_1 %in% c("ileum", "colon"),
      sample_collection_method %in% c("biopsy", "surgical resection")
    ) %>%
    mutate(
      celltype = factor(celltype, levels = selected),
      tissue = factor(str_to_title(tissue_level_1),
                      levels = c("Ileum", "Colon")),
      sample_collection_method = factor(
        sample_collection_method,
        levels = c("biopsy", "surgical resection")
      ),
      x_tissue = 1 + (as.numeric(tissue) - 1) * tissue_gap,
      x_center = x_tissue +
        ifelse(sample_collection_method == "biopsy", -pair_half, pair_half)
    )

  p_to_stars <- function(p) {
    ifelse(is.na(p), "n.s.",
      ifelse(p < 1e-4, "****",
      ifelse(p < 1e-3, "***",
      ifelse(p < 1e-2, "**",
      ifelse(p < 0.05, "*", "n.s.")))))
  }

  br_rows <- list()
  for (ct in levels(d$celltype)) {
    for (ti in levels(d$tissue)) {
      sub <- d[d$celltype == ct & d$tissue == ti, ]
      a <- sub$within_lineage_percentage[sub$sample_collection_method == "biopsy"]
      b <- sub$within_lineage_percentage[
        sub$sample_collection_method == "surgical resection"
      ]
      a <- a[is.finite(a)]; b <- b[is.finite(b)]
      if (length(a) < 3 || length(b) < 3) next
      pv <- tryCatch(
        stats::wilcox.test(a, b, exact = FALSE)$p.value,
        error = function(e) NA_real_
      )
      ymax <- max(c(a, b), na.rm = TRUE)
      x_t <- unique(sub$x_tissue)[1]
      br_rows[[length(br_rows) + 1L]] <- data.frame(
        celltype = ct, tissue = ti,
        x1 = x_t - pair_half, x2 = x_t + pair_half,
        y = ymax, p_value = pv, stringsAsFactors = FALSE
      )
    }
  }
  br <- bind_rows(br_rows) %>%
    mutate(
      p_adj = p.adjust(p_value, method = "BH"),
      label = p_to_stars(p_adj),
      celltype = factor(celltype, levels = levels(d$celltype)),
      y_bar = y + pmax(0.04 * abs(y), 0.8),
      y_lab = y + pmax(0.09 * abs(y), 1.8)
    )

  set.seed(1)
  d_plot <- d %>% mutate(x_jit = x_center + runif(n(), -jit_w, jit_w))
  x_breaks <- c(1, 1 + tissue_gap)
  x_lim <- c(1 - pair_half - box_w / 2 - 0.08,
             1 + tissue_gap + pair_half + box_w / 2 + 0.08)

  ggplot(d_plot, aes(y = within_lineage_percentage)) +
    geom_boxplot(
      aes(x = x_center, fill = sample_collection_method,
          group = interaction(tissue, sample_collection_method)),
      outlier.shape = NA, linewidth = 0.25, width = box_w,
      position = "identity", colour = "black"
    ) +
    geom_point(
      aes(x = x_jit),
      colour = "black", size = 0.28, alpha = 0.9, stroke = 0
    ) +
    geom_segment(
      data = br, aes(x = x1, xend = x2, y = y_bar, yend = y_bar),
      inherit.aes = FALSE, linewidth = 0.25
    ) +
    geom_segment(
      data = br,
      aes(x = x1, xend = x1, y = y_bar - 0.35 * (y_lab - y_bar), yend = y_bar),
      inherit.aes = FALSE, linewidth = 0.25
    ) +
    geom_segment(
      data = br,
      aes(x = x2, xend = x2, y = y_bar - 0.35 * (y_lab - y_bar), yend = y_bar),
      inherit.aes = FALSE, linewidth = 0.25
    ) +
    geom_text(
      data = br, aes(x = (x1 + x2) / 2, y = y_lab, label = label),
      inherit.aes = FALSE, size = 1.45, family = "Helvetica", vjust = 0
    ) +
    facet_wrap(
      ~ celltype, nrow = 1, scales = "free_y",
      labeller = labeller(celltype = function(x) str_wrap(x, 16))
    ) +
    scale_fill_manual(
      values = c("biopsy" = "#0072B2", "surgical resection" = "#D55E00"),
      labels = c("Biopsy", "Surgical resection")
    ) +
    scale_x_continuous(
      breaks = x_breaks, labels = levels(d$tissue), limits = x_lim,
      expand = c(0, 0)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.18))) +
    labs(
      title = "Sampling method changes the cellular view in both ileum and colon",
      x = NULL, y = "Within lineage (%)"
    ) +
    theme_composite() +
    theme(
      plot.title = element_text(size = 5.5, margin = margin(b = 0.5)),
      axis.text.x = element_text(size = 5),
      axis.title.y = element_text(size = 5),
      strip.text = element_text(size = 4.5, margin = margin(b = 0.5, t = 0)),
      panel.spacing.x = unit(0.6, "mm"),
      legend.position = "bottom",
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(-3, 0, 0, 0),
      plot.margin = margin(1, 1, 0, 1, "pt")
    )
}

make_best4_segment_dotplot <- function(compact = TRUE) {
  full_marker_order <- c(
    "CFTR", "FOLH1", "ONECUT2", "CACNA2D1", "SMIM24", "CPA2", "ADGRG4", "LYZ",
    "ANXA13", "CLDN15", "CCL25", "ALDOB", "ALDH1A1", "SI", "SULT1E1", "FAM3B",
    "DPEP1", "MALRD1", "CKB", "CEACAM5", "VSIG2",
    "C10orf99", "CA2", "HMGCS2", "FABP5", "MUC12"
  )
  compact_markers <- c(
    "CFTR", "FOLH1", "CPA2",
    "ANXA13", "CLDN15", "CCL25",
    "DPEP1", "MALRD1", "VSIG2",
    "C10orf99", "CA2", "HMGCS2"
  )
  shown_markers <- if (compact) compact_markers else full_marker_order
  d <- suppressMessages(read_csv(
    file.path(DATA, "best4_segment_marker_dotplot.csv"),
    show_col_types = FALSE
  )) %>%
    filter(
      celltype == "All BEST4 cells",
      marker %in% shown_markers
    ) %>%
    mutate(
      tissue = factor(
        str_to_title(tissue_level_1),
        levels = c("Duodenum", "Jejunum", "Ileum", "Colon")
      ),
      marker = factor(
        marker,
        levels = shown_markers
      )
    )
  # Compact: single-line y labels to cut vertical footprint
  tissue_labels <- d %>%
    distinct(tissue, n_cells, n_samples) %>%
    mutate(
      label = if (compact) {
        paste0(tissue, " (", n_cells, "/", n_samples, ")")
      } else {
        paste0(tissue, "\n", n_cells, " cells/", n_samples, " samples")
      }
    )
  tissue_lab <- setNames(
    tissue_labels$label, as.character(tissue_labels$tissue)
  )

  ggplot(
    d,
    aes(marker, tissue, size = pct_cells_detected, fill = mean_log1p_10k)
  ) +
    geom_point(shape = 21, colour = "black", stroke = 0.18) +
    scale_x_discrete(expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_discrete(
      labels = tissue_lab,
      # crush row gaps so the half-page panel stays short
      expand = expansion(mult = 0, add = if (compact) 0.35 else 0.55)
    ) +
    scale_fill_gradientn(
      colours = c("#FFFFFF", "#F0E442", "#E69F00", "#D55E00"),
      name = if (compact) "Mean log1p" else "Mean\nlog1p(10k)"
    ) +
    scale_size_continuous(
      # slightly larger dots when compact (half-column) so genes read denser
      range = if (compact) c(0.7, 2.6) else c(0.35, 2.5),
      limits = c(0, 100),
      breaks = c(25, 50, 75, 100),
      name = if (compact) "% cells" else "Cells\nexpressing (%)"
    ) +
    labs(
      title = "BEST4 segment-enriched marker programs",
      x = NULL, y = NULL
    ) +
    theme_composite() +
    theme(
      plot.title = element_text(
        size = if (compact) 5.5 else 6, margin = margin(b = 0.5)
      ),
      axis.text.x = element_text(
        size = if (compact) 4.5 else 4.8, face = "italic",
        angle = if (compact) 35 else 90,
        hjust = 1,
        vjust = 1,
        margin = margin(t = 0)
      ),
      axis.text.y = element_text(size = if (compact) 4.4 else 4.8),
      # compact: bottom legend keeps height short; genes stay dense on x
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(-1, 0, 0, 0),
      legend.key.size = unit(if (compact) 1.8 else 2.5, "mm"),
      legend.text = element_text(size = if (compact) 4.3 else 5),
      legend.title = element_text(size = if (compact) 4.3 else 5),
      legend.spacing.x = unit(1.5, "mm"),
      panel.spacing = unit(0, "mm"),
      # Force a short plotting region so the 4 tissue rows sit close
      aspect.ratio = if (compact) 0.22 else NULL,
      plot.margin = margin(
        if (compact) 0.5 else 2,
        if (compact) 1 else 2,
        if (compact) 0 else 2,
        if (compact) 1 else 2,
        "pt"
      )
    )
}

make_panel_f <- function(compact = FALSE) {
  # Representative cell types were chosen before inspecting individual genes:
  # high radial-expression omega2, an identifiable within-study depth contrast,
  # and coverage of epithelial, follicular immune and neuroglial compartments.
  de_specs <- tribble(
    ~file_stem, ~celltype_label,
    "Enterocyte_Progenitors", "Enterocyte\nprogenitors",
    "CD4_Tfh", "Tfh",
    "Glia", "Glia"
  )
  de_frames <- lapply(seq_len(nrow(de_specs)), function(i) {
    f <- file.path(
      DATA, "depth_de", "full_thickness",
      paste0(de_specs$file_stem[i], "_de.csv")
    )
    d <- suppressMessages(read_csv(f, show_col_types = FALSE))
    d %>%
      filter(adjusted) %>%
      mutate(
        celltype_label = de_specs$celltype_label[i],
        y_plot = pmin(neglog10_p_adj, 20),
        direction = case_when(
          p_adj < 0.05 & log2fc_B_minus_A > 1 ~ "Full thickness",
          p_adj < 0.05 & log2fc_B_minus_A < -1 ~ "Other layers",
          TRUE ~ "Not significant"
        )
      )
  })
  de <- bind_rows(de_frames) %>%
    mutate(
      celltype_label = factor(
        celltype_label,
        levels = de_specs$celltype_label
      )
    )

  de_plot <- ggplot(de, aes(log2fc_B_minus_A, y_plot, colour = direction)) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed",
               colour = "grey65", linewidth = 0.2) +
    geom_vline(xintercept = 0, colour = "grey65", linewidth = 0.2) +
    geom_point(size = 0.18, alpha = 0.45, stroke = 0) +
    facet_wrap(~ celltype_label, nrow = 1, scales = "free_x") +
    scale_colour_manual(
      values = c(
        "Other layers" = "#0072B2",
        "Full thickness" = "#D55E00",
        "Not significant" = "#BDBDBD"
      ),
      breaks = c("Other layers", "Full thickness")
    ) +
    labs(
      title = "Representative study-adjusted depth DE",
      x = expression(log[2] * " fold change (full thickness - other)"),
      y = expression(-log[10] * " (FDR), capped at 20")
    ) +
    theme_composite() +
    theme(
      legend.position = "bottom",
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(-3, 0, 0, 0),
      strip.text = element_text(size = 5)
    ) +
    guides(colour = guide_legend(override.aes = list(size = 1.5, alpha = 1)))

  marker_order <- c(
    "APOE", "FRZB", "RLBP1", "CRYM",
    "TF", "BCAN",
    "ENTPD2", "SFRP5",
    "GFRA3",
    "RELN", "MBP", "PLLP",
    "SLC5A7", "APOD", "KCNS3"
  )
  marker_group_order <- c(
    "Intra-ganglionic",
    "Myenteric plexus / type I",
    "Submucosal plexus / type I",
    "Extra-ganglionic",
    "Mucosal / type III",
    "Muscularis propria / type IV"
  )
  marker_raw <- suppressMessages(read_csv(
    file.path(DATA, "glia_depth_marker_pseudobulk.csv"),
    show_col_types = FALSE
  ))
  collection_summary <- marker_raw %>%
    filter(sample_collection_method %in% c("biopsy", "surgical resection")) %>%
    mutate(
      axis = "Collection",
      group_raw = sample_collection_method,
      x_order = ifelse(group_raw == "biopsy", 1, 2)
    ) %>%
    group_by(marker_group, marker, axis, group_raw, x_order) %>%
    summarise(
      mean_log2cpm = mean(log2cpm),
      pct_samples = mean(detected_cpm_gt_1) * 100,
      n_samples = n_distinct(sample_id), .groups = "drop"
    )
  radial_summary <- marker_raw %>%
    filter(radial_tissue_term %in% c("EPI_LP", "LP", "WM", "EPI_LP_MUSC")) %>%
    mutate(
      axis = "Radial layer",
      group_raw = radial_tissue_term,
      x_order = recode(
        group_raw, "EPI_LP" = 1, "LP" = 2, "WM" = 3, "EPI_LP_MUSC" = 4
      )
    ) %>%
    group_by(marker_group, marker, axis, group_raw, x_order) %>%
    summarise(
      mean_log2cpm = mean(log2cpm),
      pct_samples = mean(detected_cpm_gt_1) * 100,
      n_samples = n_distinct(sample_id), .groups = "drop"
    )
  marker_summary <- bind_rows(collection_summary, radial_summary) %>%
    mutate(
      marker = factor(marker, levels = rev(marker_order)),
      marker_group = factor(marker_group, levels = marker_group_order),
      axis = factor(axis, levels = c("Collection", "Radial layer")),
      group_label = case_when(
        group_raw == "biopsy" ~ paste0("Biopsy\nn=", n_samples),
        group_raw == "surgical resection" ~ paste0("Resection\nn=", n_samples),
        group_raw == "EPI_LP" ~ paste0("EPI+LP\nn=", n_samples),
        group_raw == "LP" ~ paste0("LP\nn=", n_samples),
        group_raw == "WM" ~ paste0("WM\nn=", n_samples),
        group_raw == "EPI_LP_MUSC" ~ paste0("Full\nn=", n_samples)
      )
    )
  levels(marker_summary$marker_group) <- c(
    "Intra-ganglionic",
    "Myenteric I",
    "Submucosal I",
    "Extra-ganglionic",
    "Mucosal III",
    "Muscularis IV"
  )
  group_levels <- marker_summary %>%
    arrange(axis, x_order) %>%
    distinct(group_label) %>%
    pull(group_label)
  marker_summary$group_label <- factor(
    marker_summary$group_label, levels = group_levels
  )

  if (compact) {
    compact_markers <- c(
      "RLBP1", "ENTPD2", "SFRP5", "GFRA3", "RELN", "APOD"
    )
    compact_labels <- c(
      RLBP1 = "RLBP1 (intra)",
      ENTPD2 = "ENTPD2 (submucosal I)",
      SFRP5 = "SFRP5 (submucosal I)",
      GFRA3 = "GFRA3 (extra)",
      RELN = "RELN (mucosal III)",
      APOD = "APOD (muscularis IV)"
    )
    marker_plot_data <- marker_summary %>%
      filter(as.character(marker) %in% compact_markers) %>%
      mutate(
        marker_display = factor(
          unname(compact_labels[as.character(marker)]),
          levels = rev(unname(compact_labels[compact_markers]))
        )
      )
    glia_plot <- ggplot(
      marker_plot_data,
      aes(group_label, marker_display, size = pct_samples, fill = mean_log2cpm)
    ) +
      geom_point(shape = 21, colour = "black", stroke = 0.18) +
      facet_grid(~ axis, scales = "free_x", space = "free_x") +
      labs(
        title = "Glial subtype markers across sampling depth",
        x = NULL, y = NULL
      ) +
      theme_composite() +
      theme(
        axis.text.x = element_text(size = 4.7),
        axis.text.y = element_text(size = 4.7, face = "italic"),
        strip.text.x = element_text(size = 5),
        panel.spacing.x = unit(1.2, "mm"),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(-3, 0, 0, 0)
      )
  } else {
    glia_plot <- ggplot(
      marker_summary,
      aes(group_label, marker, size = pct_samples, fill = mean_log2cpm)
    ) +
      geom_point(shape = 21, colour = "black", stroke = 0.18) +
      facet_grid(
        marker_group ~ axis, scales = "free", space = "free",
        switch = "y"
      ) +
      labs(
        title = "Glial subtype markers across sampling depth",
        x = NULL, y = NULL
      ) +
      theme_composite() +
      theme(
        axis.text.x = element_text(size = 4.7),
        axis.text.y = element_text(size = 4.7, face = "italic"),
        strip.placement = "outside",
        strip.text.x = element_text(size = 5),
        strip.text.y.left = element_text(size = 4.6, angle = 0),
        panel.spacing = unit(0.8, "mm"),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.margin = margin(0, 0, 0, 0),
        legend.box.margin = margin(-3, 0, 0, 0)
      )
  }
  glia_plot <- glia_plot +
    scale_fill_gradientn(
      colours = c("#FFFFFF", "#F0E442", "#E69F00", "#D55E00"),
      name = "Mean\nlog2 CPM"
    ) +
    scale_size_continuous(
      range = c(0.35, 2.15), limits = c(0, 100),
      breaks = c(25, 50, 75, 100), name = "Samples\n>1 CPM (%)"
    )

  wrap_plots(de_plot, glia_plot, widths = c(0.92, 1.35))
}

make_figure3_composite <- function() {
  panel_a <- make_panel_a()
  panel_b <- wrap_elements(full = make_panel_b())
  panel_c <- make_descriptive_segment(
    read_segment_depth("duodenum"), "Duodenum"
  )
  panel_d <- make_descriptive_segment(
    read_segment_depth("jejunum"), "Jejunum"
  )
  panel_e <- make_inferential_segment(
    read_segment_depth("ileum"), "Ileum: full thickness versus other layers"
  )
  panel_f <- make_inferential_segment(
    read_segment_depth("colon"), "Colon: full thickness versus other layers"
  )
  panel_g <- make_panel_e()
  save_panel(panel_g, "fig3_g_collection_boxplots_ileum_colon", 105, 36)
  panel_h <- make_best4_segment_dotplot(compact = TRUE)

  design <- "
  AA
  BB
  CD
  EF
  GG
  HH
  "
  fig <- wrap_plots(
    A = panel_a, B = panel_b, C = panel_c, D = panel_d,
    E = panel_e, F = panel_f, G = panel_g, H = panel_h,
    design = design, heights = c(31, 34, 25, 34, 16, 24)
  ) +
    plot_annotation(tag_levels = "a") &
    theme(
      plot.tag = element_text(
        family = "Helvetica", face = "bold", size = 7, colour = "black"
      ),
      plot.tag.position = c(0, 1),
      plot.margin = margin(1.5, 1.5, 1.5, 1.5, "pt")
    )

  save_panel(fig, "fig3_sampling_depth_composite", 180, 170)
  save_panel(
    wrap_plots(panel_c, panel_d, panel_e, panel_f, ncol = 2),
    "fig3_c_f_full_thickness_by_segment", 180, 105
  )
  # Half-column (90 mm): compact gene set, short height for half-page layout
  save_panel(
    make_best4_segment_dotplot(compact = TRUE),
    "fig3_h_best4_segment_dotplot", 90, 28
  )
  # Full-width archive with all markers
  save_panel(
    make_best4_segment_dotplot(compact = FALSE),
    "fig3_h_best4_segment_dotplot_full", 180, 75
  )

  # Half-page composition ω² PCR: 3b left (lollipop) + anatomy:study ratio stacked
  save_panel(
    make_panel_b_halfpage(),
    "fig3_b_composition_omega2_halfpage", 90, 78
  )
  # Same content as a short full-width strip (lollipop | ratio)
  save_panel(
    make_panel_b(compact = TRUE),
    "fig3_b_composition_omega2_row", 180, 42
  )
  # Standalone 3b-left lollipop for Illustrator mixes
  save_panel(
    make_panel_b_parts(compact = TRUE, show_lollipop_legend = TRUE)$lollipop,
    "fig3_b_pcr_lollipop_half", 90, 58
  )
}

make_figure3_composite()

# report outputs without parsing compressed PDF bytes
message("\nFinal exports (90 or 180 mm wide; composite = 180 x 170 mm):")
for (f in sort(list.files(OUT, pattern = "\\.pdf$", full.names = TRUE))) {
  message(sprintf("  %-60s %8.1f kB", basename(f), file.info(f)$size / 1024))
}
