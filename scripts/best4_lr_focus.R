#!/usr/bin/env Rscript
## BEST4 Enterocytes + BEST4 Colonocytes -> single "BEST4 cells" label.
## Aggregates an existing LIANA per-tissue table to highlight the strongest
## ingoing and outgoing BEST4 communication hits across the four major gut
## segments (duodenum, jejunum, ileum, colon). No re-running of LIANA.
##
## SCORING:
##   Prefers LIANA rank_aggregate consensus columns when present:
##     magnitude_rank, specificity_rank, ensemble_score = -log10(sqrt(mag*spec))
##   Also keeps lr_means (CellPhoneDB), and method scores when available
##   (spec_weight / NATMI, scaled_weight / Connectome). Homemade NATMI from
##   ligand_means is used only if means are present in the table.
##
## Style spec: ~/Projects/GCA/publication2026/plot_specs.md
##   - <= 180 mm wide / <= 170 mm tall, Helvetica 5-7 pt, no grid, white bg,
##     L-shape axes, Wong palette, cairo PDF + SVG + 300 dpi PNG.
##
## Inputs (env vars):
##   CCC_EDGE_CSV   path to LIANA edges CSV (tissue_level_1 long form)
##   CCC_OUTPUT_DIR base output dir; results land in <dir>/best4_lr_focus/
##   CCC_LINEAGE_LOOKUP_CSV path to cell_state -> plot_lineage CSV
##   CCC_DROP_STATES        ";"-list of cell_state to drop (default "Epithelial")
##   CCC_BEST4_LABELS       ";"-list of labels to merge (default
##                          "BEST4 Enterocytes;BEST4 Colonocytes")
##   CCC_EXPR_PROP_MIN      min ligand/receptor expressed fraction (default 0.10)
##   CCC_TOP_LR_PER_SEG     top-N LR pairs per segment for union (default 12)
##   CCC_TOP_PARTNERS       top-N partner cell states per segment (default 12)
##   CCC_BUBBLE_TOP_LR      LR pairs to keep in bubble plot   (default 15)
##   CCC_BUBBLE_TOP_PART    partners to keep in bubble plot   (default 18)

set.seed(1L)

suppressPackageStartupMessages({
  library(dplyr,   warn.conflicts = FALSE)
  library(tidyr,   warn.conflicts = FALSE)
  library(readr)
  library(tibble)
  library(stringr, warn.conflicts = FALSE)
  library(ggplot2)
  library(scales)
  library(ggrepel)
  library(patchwork)
  library(data.table, warn.conflicts = FALSE)
})

## ---------- params --------------------------------------------------------

INPUT_CSV     <- Sys.getenv(
  "CCC_EDGE_CSV",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/combined_lr_per_tissue_level_1.csv")
BASE_OUT      <- Sys.getenv(
  "CCC_OUTPUT_DIR",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate")
OUT_DIR       <- file.path(BASE_OUT, "best4_lr_focus")
LINEAGE_CSV   <- Sys.getenv("CCC_LINEAGE_LOOKUP_CSV",
                            "data/hgca_celltype_v1_lineage.csv")
DROP_STATES   <- {
  d <- Sys.getenv("CCC_DROP_STATES", "Epithelial")
  trimws(strsplit(d, "[;,]")[[1]])
}
BEST4_LABELS  <- {
  d <- Sys.getenv("CCC_BEST4_LABELS",
                  "BEST4 Enterocytes;BEST4 Colonocytes")
  trimws(strsplit(d, "[;,]")[[1]])
}
BEST4_MERGED  <- "BEST4 cells"
SEGMENT_ORDER <- c("duodenum", "jejunum", "ileum", "colon")
EXPR_PROP_MIN      <- as.numeric(Sys.getenv("CCC_EXPR_PROP_MIN", "0.10"))
TOP_LR_PER_SEG     <- as.integer(Sys.getenv("CCC_TOP_LR_PER_SEG", "12"))
TOP_PARTNERS       <- as.integer(Sys.getenv("CCC_TOP_PARTNERS",   "12"))
BUBBLE_TOP_LR      <- as.integer(Sys.getenv("CCC_BUBBLE_TOP_LR",  "15"))
BUBBLE_TOP_PART    <- as.integer(Sys.getenv("CCC_BUBBLE_TOP_PART","18"))
EPS <- 1e-9

## Scorings rendered as full plot/CSV sets. Each: col = column in `long`,
## label = axis/legend text, fname = filename suffix.
## Filtered at runtime to columns that exist after scoring.
SCORINGS <- list(
  list(key = "ensemble",   col = "combined",   label = "LIANA ensemble (−log10 rank)"),
  list(key = "magnitude",  col = "lr_means",   label = "lr_means (CellPhoneDB)"),
  list(key = "mag_rank",   col = "mag_score",  label = "−log10(magnitude_rank)"),
  list(key = "spec_rank",  col = "spec_score", label = "−log10(specificity_rank)"),
  list(key = "natmi_spec", col = "natmi_spec", label = "NATMI specificity"),
  list(key = "connectome", col = "connectome_z", label = "Connectome weight")
)

LINEAGE_ORDER <- c("Epithelial", "Lymphoid", "Myeloid", "Stromal",
                   "Endothelial", "Glial", "Other")
LINEAGE_HEX <- c(
  Epithelial  = "#009E73",
  Lymphoid    = "#0072B2",
  Myeloid     = "#D55E00",
  Stromal     = "#999999",
  Endothelial = "#56B4E9",
  Glial       = "#CC79A7",
  Other       = "#E0E0E0"
)

MAX_FIG_WIDTH_MM  <- 180
MAX_FIG_HEIGHT_MM <- 170
GG_PT_PER_SIZE    <- 2.845
MIN_GG_TEXT_SIZE  <- 5 / GG_PT_PER_SIZE

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ---------- theme + saver -------------------------------------------------

theme_gca <- function(fs = "Helvetica", sz = 6L) {
  theme_classic(base_size = sz, base_family = fs) +
    theme(
      text             = element_text(family = fs, colour = "black"),
      plot.title       = element_text(family = fs, face = "bold",
                                      size = 7, colour = "black", hjust = 0),
      plot.subtitle    = element_text(family = fs, colour = "black", size = 6),
      axis.line        = element_line(colour = "black", linewidth = 0.25),
      axis.ticks       = element_line(colour = "black", linewidth = 0.25),
      axis.text        = element_text(colour = "black", size = 6),
      axis.title       = element_text(colour = "black", size = 6),
      panel.grid       = element_blank(),
      panel.background = element_blank(),
      plot.background  = element_rect(fill = "white", colour = NA),
      legend.key       = element_blank(),
      legend.text      = element_text(colour = "black", size = 6),
      legend.title     = element_text(colour = "black", size = 6, face = "bold"),
      strip.background = element_blank(),
      strip.text       = element_text(colour = "black", face = "bold", size = 6)
    )
}

save_pair <- function(p, stem, w_mm, h_mm) {
  w_mm <- min(w_mm, MAX_FIG_WIDTH_MM)
  h_mm <- min(h_mm, MAX_FIG_HEIGHT_MM)
  wi <- w_mm / 25.4
  hi <- h_mm / 25.4
  pdf_dev <- if (isTRUE(capabilities("cairo"))) grDevices::cairo_pdf else pdf
  ggsave(paste0(stem, ".pdf"), p, width = wi, height = hi, device = pdf_dev)
  if (requireNamespace("svglite", quietly = TRUE)) {
    ggsave(paste0(stem, ".svg"), p, width = wi, height = hi,
           device = svglite::svglite)
  } else {
    ggsave(paste0(stem, ".svg"), p, width = wi, height = hi, device = "svg")
  }
  ggsave(paste0(stem, ".png"), p, width = wi, height = hi, dpi = 300)
}

## ---------- data load + reshape ------------------------------------------

load_edges <- function() {
  if (!file.exists(INPUT_CSV))
    stop("Edge CSV not found: ", INPUT_CSV)
  message("Reading: ", INPUT_CSV)
  dt <- data.table::fread(INPUT_CSV)
  seg_col <- intersect(c("tissue_level_1", "segment", "tissue", "tissue_subset"),
                       names(dt))[1]
  if (is.na(seg_col)) stop("No segment column found in ", INPUT_CSV)
  setnames(dt, seg_col, "segment")
  needed <- c("source", "target", "ligand_complex", "receptor_complex")
  miss <- setdiff(needed, names(dt))
  if (length(miss)) stop("Missing required columns: ",
                         paste(miss, collapse = ", "))
  for (cc in c("ligand_means", "receptor_means", "lr_means",
               "ligand_props", "receptor_props",
               "magnitude_rank", "specificity_rank",
               "spec_weight", "scaled_weight", "cellphone_pvals")) {
    if (!cc %in% names(dt)) dt[, (cc) := NA_real_]
  }
  dt[, segment := tolower(trimws(as.character(segment)))]
  dt[, source := as.character(source)]
  dt[, target := as.character(target)]
  dt[, ligand_complex := as.character(ligand_complex)]
  dt[, receptor_complex := as.character(receptor_complex)]
  for (cc in c("ligand_means", "receptor_means", "lr_means",
               "ligand_props", "receptor_props",
               "magnitude_rank", "specificity_rank",
               "spec_weight", "scaled_weight", "cellphone_pvals"))
    dt[, (cc) := as.numeric(get(cc))]
  if (dt[, any(!is.na(lr_means))]) {
    dt <- dt[!is.na(lr_means) & lr_means > 0]
  } else if (dt[, any(!is.na(magnitude_rank))]) {
    dt <- dt[!is.na(magnitude_rank)]
  }
  dt <- dt[segment %in% SEGMENT_ORDER]
  if (length(DROP_STATES))
    dt <- dt[!(source %in% DROP_STATES) & !(target %in% DROP_STATES)]
  ## merge BEST4 sub-labels
  dt[source %in% BEST4_LABELS, source := BEST4_MERGED]
  dt[target %in% BEST4_LABELS, target := BEST4_MERGED]
  ## Collapse colliding (segment, source, target, ligand, receptor) rows from
  ## the merge: keep strongest magnitude / best (lowest) ranks.
  dt <- dt[, .(
    lr_means         = suppressWarnings(max(lr_means, na.rm = TRUE)),
    ligand_means     = suppressWarnings(max(ligand_means, na.rm = TRUE)),
    receptor_means   = suppressWarnings(max(receptor_means, na.rm = TRUE)),
    ligand_props     = suppressWarnings(max(ligand_props, na.rm = TRUE)),
    receptor_props   = suppressWarnings(max(receptor_props, na.rm = TRUE)),
    magnitude_rank   = suppressWarnings(min(magnitude_rank, na.rm = TRUE)),
    specificity_rank = suppressWarnings(min(specificity_rank, na.rm = TRUE)),
    spec_weight      = suppressWarnings(max(spec_weight, na.rm = TRUE)),
    scaled_weight    = suppressWarnings(max(scaled_weight, na.rm = TRUE)),
    cellphone_pvals  = suppressWarnings(min(cellphone_pvals, na.rm = TRUE))
  ), by = .(segment, source, target, ligand_complex, receptor_complex)]
  for (cc in c("ligand_props", "receptor_props", "lr_means", "ligand_means",
               "receptor_means", "magnitude_rank", "specificity_rank",
               "spec_weight", "scaled_weight", "cellphone_pvals"))
    dt[is.infinite(get(cc)), (cc) := NA_real_]
  dt[, lr_pair := paste(ligand_complex, receptor_complex, sep = "->")]
  dt[, segment := factor(segment, levels = SEGMENT_ORDER)]
  as_tibble(dt)
}

## NATMI specificity + Connectome z, computed on the FULL sender/receiver
## landscape (all cell types), then attached back to every edge.
compute_specificity <- function(edges) {
  z_or0 <- function(x) {
    if (length(x) < 2L) return(rep(0, length(x)))
    s <- sd(x)
    if (is.na(s) || s == 0) return(rep(0, length(x)))
    (x - mean(x)) / s
  }
  lig_expr <- edges |>
    group_by(segment, ligand_complex, source) |>
    summarise(ligand_means = max(ligand_means, na.rm = TRUE), .groups = "drop") |>
    group_by(segment, ligand_complex) |>
    mutate(spec_ligand = ligand_means / (sum(ligand_means) + EPS),
           ligand_z    = z_or0(ligand_means),
           n_senders   = n()) |>
    ungroup() |>
    select(segment, ligand_complex, source, spec_ligand, ligand_z, n_senders)
  rec_expr <- edges |>
    group_by(segment, receptor_complex, target) |>
    summarise(receptor_means = max(receptor_means, na.rm = TRUE), .groups = "drop") |>
    group_by(segment, receptor_complex) |>
    mutate(spec_receptor = receptor_means / (sum(receptor_means) + EPS),
           receptor_z     = z_or0(receptor_means),
           n_receivers    = n()) |>
    ungroup() |>
    select(segment, receptor_complex, target, spec_receptor, receptor_z, n_receivers)
  edges |>
    left_join(lig_expr, by = c("segment", "ligand_complex", "source")) |>
    left_join(rec_expr, by = c("segment", "receptor_complex", "target")) |>
    mutate(
      spec_ligand   = tidyr::replace_na(spec_ligand, 0),
      spec_receptor = tidyr::replace_na(spec_receptor, 0),
      ligand_z      = tidyr::replace_na(ligand_z, 0),
      receptor_z    = tidyr::replace_na(receptor_z, 0),
      natmi_spec    = spec_ligand * spec_receptor,
      connectome_z  = (ligand_z + receptor_z) / 2
    )
}

load_lineage <- function() {
  if (!file.exists(LINEAGE_CSV))
    stop("Lineage CSV not found: ", LINEAGE_CSV)
  lk <- read_csv(LINEAGE_CSV, show_col_types = FALSE) |>
    select(cell_state, plot_lineage) |>
    distinct()
  bind_rows(lk, tibble(cell_state = BEST4_MERGED, plot_lineage = "Epithelial")) |>
    distinct(cell_state, .keep_all = TRUE)
}

attach_lineage <- function(df, lk, partner_col = "partner") {
  df |>
    left_join(rename(lk, !!partner_col := cell_state,
                     partner_lineage = plot_lineage),
              by = partner_col) |>
    mutate(partner_lineage = ifelse(is.na(partner_lineage) |
                                      !partner_lineage %in% LINEAGE_ORDER,
                                    "Other", partner_lineage),
           partner_lineage = factor(partner_lineage, levels = LINEAGE_ORDER))
}

split_directions <- function(edges, lk) {
  keep <- intersect(
    c("segment", "lr_pair", "ligand_complex", "receptor_complex",
      "lr_means", "ligand_means", "receptor_means",
      "ligand_props", "receptor_props",
      "magnitude_rank", "specificity_rank",
      "spec_ligand", "spec_receptor", "ligand_z", "receptor_z",
      "natmi_spec", "connectome_z", "spec_weight", "scaled_weight",
      "cellphone_pvals"),
    names(edges))
  out <- edges |>
    filter(source == BEST4_MERGED) |>
    mutate(partner = target, direction = "outgoing") |>
    select(all_of(c(keep, "partner", "direction"))) |>
    attach_lineage(lk)
  inn <- edges |>
    filter(target == BEST4_MERGED) |>
    mutate(partner = source, direction = "incoming") |>
    select(all_of(c(keep, "partner", "direction"))) |>
    attach_lineage(lk)
  bind_rows(out, inn) |>
    filter(partner != BEST4_MERGED) |>
    mutate(direction = factor(direction, levels = c("outgoing", "incoming")))
}

## ---------- score-generic aggregations ------------------------------------

agg_lr_pair <- function(long, score_col) {
  long |>
    mutate(.val = .data[[score_col]]) |>
    group_by(direction, segment, lr_pair, ligand_complex, receptor_complex) |>
    summarise(
      val_max = max(.val, na.rm = TRUE),
      best_partner = partner[which.max(.val)],
      best_partner_lineage = partner_lineage[which.max(.val)],
      n_partners   = n_distinct(partner),
      .groups = "drop"
    )
}

agg_partner <- function(long, score_col) {
  long |>
    mutate(.val = .data[[score_col]]) |>
    group_by(direction, segment, partner, partner_lineage) |>
    summarise(
      val_sum    = sum(.val, na.rm = TRUE),
      val_max    = max(.val, na.rm = TRUE),
      n_lr_pairs = n_distinct(lr_pair),
      .groups = "drop"
    )
}

## ---------- plot: LR-pair heatmap -----------------------------------------

plot_lr_heatmap <- function(lr_agg, direction_lbl, stem, score_label,
                            top_n = TOP_LR_PER_SEG) {
  d <- filter(lr_agg, direction == direction_lbl)
  if (!nrow(d)) return(invisible(NULL))
  top_set <- d |>
    group_by(segment) |>
    slice_max(val_max, n = top_n, with_ties = FALSE) |>
    pull(lr_pair) |>
    unique()
  d <- filter(d, lr_pair %in% top_set)
  full <- expand_grid(segment = factor(SEGMENT_ORDER, levels = SEGMENT_ORDER),
                      lr_pair = top_set) |>
    left_join(d, by = c("segment", "lr_pair")) |>
    mutate(val_max = ifelse(is.na(val_max), 0, val_max))
  ord <- d |>
    group_by(lr_pair) |>
    summarise(peak = max(val_max),
              peak_seg = segment[which.max(val_max)],
              .groups = "drop") |>
    arrange(peak_seg, desc(peak))
  full <- mutate(full,
                 lr_pair = factor(lr_pair, levels = ord$lr_pair),
                 segment = factor(segment, levels = SEGMENT_ORDER))
  p <- ggplot(full, aes(segment, lr_pair, fill = val_max)) +
    geom_tile(colour = "white", linewidth = 0.2) +
    scale_fill_gradientn(
      colours = c("#FFFFFF", "#FFE5C7", "#F1A340", "#A1430F"),
      name = score_label, na.value = "white") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(
      title = paste0("BEST4 cells - ", direction_lbl,
                     " LR pairs (top ", top_n, " per segment, union)"),
      subtitle = paste0("ranked by ", score_label,
                        "; ligand_complex -> receptor_complex; partner = ",
                        ifelse(direction_lbl == "outgoing", "target", "source")),
      x = NULL, y = NULL
    ) +
    theme_gca() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1, size = 6),
      axis.text.y = element_text(family = "Helvetica", size = 5, colour = "black"),
      axis.line   = element_blank(),
      axis.ticks  = element_blank(),
      legend.position = "right"
    )
  h_mm <- min(MAX_FIG_HEIGHT_MM, 6 + length(top_set) * 3.0)
  save_pair(p, stem, 110, h_mm)
}

## ---------- plot: top partner cell types per segment ----------------------

plot_top_partners <- function(part_agg, direction_lbl, stem, score_label,
                              top_n = TOP_PARTNERS) {
  d <- filter(part_agg, direction == direction_lbl)
  if (!nrow(d)) return(invisible(NULL))
  d_top <- d |>
    group_by(segment) |>
    slice_max(val_sum, n = top_n, with_ties = FALSE) |>
    ungroup() |>
    arrange(segment, val_sum) |>
    mutate(seg_id = paste(segment, partner),
           seg_id = factor(seg_id, levels = unique(seg_id)))
  p <- ggplot(d_top, aes(val_sum, seg_id, fill = partner_lineage)) +
    geom_col() +
    geom_text(aes(label = partner), x = 0, hjust = 0,
              colour = "black", family = "Helvetica", size = MIN_GG_TEXT_SIZE) +
    facet_wrap(~ segment, scales = "free", nrow = 1) +
    scale_fill_manual(values = LINEAGE_HEX, drop = FALSE,
                      name = "partner lineage") +
    scale_y_discrete(labels = NULL, breaks = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.04))) +
    labs(
      title = paste0("Top ", top_n, " partner cell states per segment - ",
                     direction_lbl, " from BEST4 cells"),
      subtitle = paste0("x = sum of ", score_label,
                        " across LR pairs (BEST4 <-> partner)"),
      x = paste0("summed ", score_label), y = NULL
    ) +
    theme_gca() +
    theme(panel.spacing = grid::unit(2, "mm"),
          legend.position = "bottom")
  save_pair(p, stem, MAX_FIG_WIDTH_MM, 95)
}

## ---------- plot: LR pair x partner bubble --------------------------------

plot_lr_partner_bubble <- function(long, lr_agg, part_agg,
                                   direction_lbl, stem, score_col, score_label,
                                   top_lr = BUBBLE_TOP_LR,
                                   top_part = BUBBLE_TOP_PART) {
  d <- filter(long, direction == direction_lbl)
  if (!nrow(d)) return(invisible(NULL))
  lr_top <- lr_agg |>
    filter(direction == direction_lbl) |>
    group_by(segment) |>
    slice_max(val_max, n = top_lr, with_ties = FALSE) |>
    pull(lr_pair) |>
    unique()
  part_top_per_seg <- part_agg |>
    filter(direction == direction_lbl) |>
    group_by(segment) |>
    slice_max(val_sum, n = top_part, with_ties = FALSE) |>
    select(segment, partner) |>
    ungroup()
  d <- d |>
    mutate(.val = .data[[score_col]]) |>
    filter(lr_pair %in% lr_top) |>
    semi_join(part_top_per_seg, by = c("segment", "partner"))
  if (!nrow(d)) return(invisible(NULL))
  lr_ord <- d |>
    group_by(lr_pair) |>
    summarise(v = max(.val), .groups = "drop") |>
    arrange(desc(v)) |>
    pull(lr_pair)
  part_ord <- part_top_per_seg |>
    semi_join(d, by = c("segment", "partner")) |>
    distinct() |>
    left_join(part_agg |>
                filter(direction == direction_lbl) |>
                select(segment, partner, partner_lineage, val_sum),
              by = c("segment", "partner")) |>
    arrange(segment, partner_lineage, desc(val_sum)) |>
    mutate(seg_id = paste(segment, partner, sep = "||")) |>
    pull(seg_id)
  d <- d |>
    mutate(seg_id = paste(segment, partner, sep = "||"),
           seg_id = factor(seg_id, levels = part_ord),
           lr_pair = factor(lr_pair, levels = rev(lr_ord)))
  p <- ggplot(d, aes(seg_id, lr_pair, size = .val, colour = partner_lineage)) +
    geom_point(alpha = 0.9, stroke = 0) +
    facet_wrap(~ segment, nrow = 1, scales = "free_x") +
    scale_x_discrete(labels = function(x) sub(".*\\|\\|", "", x)) +
    scale_colour_manual(values = LINEAGE_HEX, drop = FALSE,
                        name = "partner lineage") +
    scale_size_continuous(range = c(0.6, 3.5), name = score_label,
                          guide = guide_legend(override.aes = list(colour = "black"))) +
    labs(
      title = paste0("BEST4 cells - ", direction_lbl,
                     ": top LR pairs x top partners per segment"),
      subtitle = paste0("ranked by ", score_label, "; LR pairs = union of top ",
                        top_lr, " per segment; partners = top ", top_part,
                        " per segment"),
      x = NULL, y = NULL
    ) +
    theme_gca() +
    theme(
      axis.text.x = element_text(angle = 60, hjust = 1, size = 5),
      axis.text.y = element_text(size = 5),
      axis.line   = element_blank(),
      axis.ticks  = element_line(linewidth = 0.2),
      panel.spacing.x = grid::unit(3, "mm"),
      legend.position = "bottom",
      legend.box = "horizontal"
    )
  save_pair(p, stem, MAX_FIG_WIDTH_MM, MAX_FIG_HEIGHT_MM)
}

## ---------- plot: top-N membership across segments ------------------------

plot_membership <- function(lr_agg, direction_lbl, stem, score_label,
                            top_n = TOP_LR_PER_SEG) {
  d <- filter(lr_agg, direction == direction_lbl)
  if (!nrow(d)) return(invisible(NULL))
  top_set <- d |>
    group_by(segment) |>
    slice_max(val_max, n = top_n, with_ties = FALSE) |>
    pull(lr_pair) |>
    unique()
  per_seg_top <- d |>
    group_by(segment) |>
    slice_max(val_max, n = top_n, with_ties = FALSE) |>
    transmute(segment, lr_pair, in_top = TRUE)
  full <- expand_grid(segment = factor(SEGMENT_ORDER, levels = SEGMENT_ORDER),
                      lr_pair = top_set) |>
    left_join(per_seg_top, by = c("segment", "lr_pair")) |>
    mutate(in_top = !is.na(in_top))
  count_summary <- full |>
    group_by(lr_pair) |>
    summarise(n_seg_top = sum(in_top), .groups = "drop") |>
    arrange(desc(n_seg_top), lr_pair)
  full <- full |>
    left_join(count_summary, by = "lr_pair") |>
    mutate(lr_pair = factor(lr_pair, levels = rev(count_summary$lr_pair)))
  p <- ggplot(full, aes(segment, lr_pair)) +
    geom_tile(aes(fill = in_top), colour = "white", linewidth = 0.25) +
    scale_fill_manual(values = c(`TRUE` = "#A1430F", `FALSE` = "#F2F2F2"),
                      labels = c(`TRUE` = "in top", `FALSE` = "not in top"),
                      name = paste0("top ", top_n, " in segment")) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(
      title = paste0("BEST4 cells - ", direction_lbl,
                     ": LR-pair top-", top_n, " membership by segment"),
      subtitle = paste0("ranked by ", score_label,
                        "; universal vs segment-specific BEST4 communication"),
      x = NULL, y = NULL
    ) +
    theme_gca() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1, size = 6),
      axis.text.y = element_text(size = 5),
      axis.line   = element_blank(),
      axis.ticks  = element_blank(),
      legend.position = "right"
    )
  h_mm <- min(MAX_FIG_HEIGHT_MM, 6 + length(top_set) * 3.0)
  save_pair(p, stem, 110, h_mm)
}

## ---------- plot: specificity vs magnitude scatter ------------------------
## Illustrates why housekeeping genes top the magnitude list: they sit at
## high magnitude but low specificity (bottom-right). BEST4-specific biology
## sits high on the y-axis.

plot_spec_vs_magnitude <- function(long, direction_lbl, stem, top_lab = 16L) {
  d <- filter(long, direction == direction_lbl)
  if (!nrow(d)) return(invisible(NULL))
  ## one row per lr_pair: the partner/segment where it is most specific
  per_pair <- d |>
    group_by(lr_pair, ligand_complex, receptor_complex) |>
    slice_max(natmi_spec, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(lr_pair, natmi_spec, partner, partner_lineage, segment)
  mag <- d |>
    group_by(lr_pair) |>
    summarise(lr_means_max = max(lr_means), .groups = "drop")
  per_pair <- left_join(per_pair, mag, by = "lr_pair")
  lab_spec <- per_pair |> slice_max(natmi_spec, n = top_lab) |> pull(lr_pair)
  lab_mag  <- per_pair |> slice_max(lr_means_max, n = top_lab) |> pull(lr_pair)
  per_pair <- per_pair |>
    mutate(
      cls = case_when(
        lr_pair %in% lab_spec & lr_pair %in% lab_mag ~ "specific & high magnitude",
        lr_pair %in% lab_spec ~ "specific (informative)",
        lr_pair %in% lab_mag  ~ "high magnitude only (housekeeping-like)",
        TRUE ~ "other"
      ),
      cls = factor(cls, levels = c("specific (informative)",
                                   "specific & high magnitude",
                                   "high magnitude only (housekeeping-like)",
                                   "other")),
      lab = ifelse(lr_pair %in% union(lab_spec, lab_mag),
                   as.character(lr_pair), NA_character_)
    )
  pal <- c("specific (informative)" = "#009E73",
           "specific & high magnitude" = "#0072B2",
           "high magnitude only (housekeeping-like)" = "#D55E00",
           "other" = "#CCCCCC")
  p <- ggplot(per_pair, aes(lr_means_max, natmi_spec)) +
    geom_point(aes(colour = cls), size = 1.2, stroke = 0, alpha = 0.9) +
    ggrepel::geom_text_repel(
      aes(label = lab, colour = cls),
      size = MIN_GG_TEXT_SIZE, family = "Helvetica",
      min.segment.length = 0, max.overlaps = 30, segment.alpha = 0.4,
      box.padding = 0.25, show.legend = FALSE) +
    scale_colour_manual(values = pal, drop = FALSE, name = NULL) +
    labs(
      title = paste0("BEST4 cells - ", direction_lbl,
                     ": specificity vs magnitude"),
      subtitle = "Housekeeping/MHC genes: high magnitude, low specificity (lower-right). Specific biology rises on y.",
      x = "magnitude (max lr_means across segments)",
      y = "NATMI specificity (max across segments/partners)"
    ) +
    theme_gca() +
    theme(legend.position = "bottom")
  save_pair(p, stem, 130, 130)
}

## ---------- main ----------------------------------------------------------

render_scoring <- function(long, sc) {
  lr_agg   <- agg_lr_pair(long, sc$col)
  part_agg <- agg_partner(long, sc$col)
  for (dir_lbl in c("outgoing", "incoming")) {
    base <- file.path(OUT_DIR, paste0("fig_best4_", dir_lbl, "_", sc$key))
    plot_lr_heatmap(lr_agg, dir_lbl, paste0(base, "_lr_heatmap"), sc$label)
    plot_top_partners(part_agg, dir_lbl, paste0(base, "_top_partners"), sc$label)
    plot_lr_partner_bubble(long, lr_agg, part_agg, dir_lbl,
                           paste0(base, "_lr_x_partner_bubble"),
                           sc$col, sc$label)
    plot_membership(lr_agg, dir_lbl, paste0(base, "_membership"), sc$label)
  }
  list(lr_agg = lr_agg, part_agg = part_agg)
}

main <- function() {
  edges <- load_edges()
  lk    <- load_lineage()

  message("Edges (4 segments, post-merge): ",
          format(nrow(edges), big.mark = ","))
  if (!any(c(edges$source, edges$target) %in% BEST4_MERGED))
    stop("No BEST4 cells found in input after label merge.")

  ## Homemade NATMI/Connectome only if ligand/receptor means exist
  has_means <- all(c("ligand_means", "receptor_means") %in% names(edges)) &&
    any(!is.na(edges$ligand_means)) && any(!is.na(edges$receptor_means))
  if (has_means) {
    edges <- compute_specificity(edges)
  } else {
    message("No ligand/receptor means; using rank_aggregate method scores")
    edges <- edges |>
      mutate(
        natmi_spec = dplyr::coalesce(spec_weight, NA_real_),
        connectome_z = dplyr::coalesce(scaled_weight, NA_real_),
        spec_ligand = NA_real_, spec_receptor = NA_real_,
        ligand_z = NA_real_, receptor_z = NA_real_
      )
  }

  ## Expression-proportion filter only when props are present
  n_pre <- nrow(edges)
  if (any(!is.na(edges$ligand_props)) || any(!is.na(edges$receptor_props))) {
    edges_f <- edges |>
      filter(is.na(ligand_props)   | ligand_props   >= EXPR_PROP_MIN,
             is.na(receptor_props) | receptor_props >= EXPR_PROP_MIN)
    message("expr_prop >= ", EXPR_PROP_MIN, " filter: kept ",
            format(nrow(edges_f), big.mark = ","), " / ",
            format(n_pre, big.mark = ","), " edges")
  } else {
    edges_f <- edges
    message("No prop columns; keeping LIANA expr_prop-filtered edges (",
            format(n_pre, big.mark = ","), ")")
  }

  long <- split_directions(edges_f, lk)
  message("BEST4 outgoing rows: ",
          format(sum(long$direction == "outgoing"), big.mark = ","))
  message("BEST4 incoming rows: ",
          format(sum(long$direction == "incoming"), big.mark = ","))

  ## True LIANA ensemble score when ranks present; else geometric percentile proxy
  if (all(c("magnitude_rank", "specificity_rank") %in% names(long)) &&
      any(!is.na(long$magnitude_rank))) {
    long <- long |>
      mutate(
        mag_score = -log10(pmax(magnitude_rank, EPS)),
        spec_score = -log10(pmax(specificity_rank, EPS)),
        combined = -log10(pmax(sqrt(pmax(magnitude_rank, EPS) *
                                      pmax(specificity_rank, EPS)), EPS))
      )
  } else {
    long <- long |>
      group_by(direction, segment) |>
      mutate(
        mag_pct  = dplyr::percent_rank(lr_means),
        spec_pct = dplyr::percent_rank(natmi_spec),
        combined = sqrt(pmax(mag_pct, 0) * pmax(spec_pct, 0)),
        mag_score = mag_pct,
        spec_score = spec_pct
      ) |>
      ungroup()
  }

  ## ---- CSV: full scored long table -----
  write_csv(long, file.path(OUT_DIR, "best4_lr_long_scored.csv"))

  scorings <- Filter(function(sc) sc$col %in% names(long) &&
                       any(!is.na(long[[sc$col]])), SCORINGS)

  ## ---- CSV: per-score global + per-segment top hits -----
  for (sc in scorings) {
    keep_cols <- intersect(
      c("direction", "segment", "lr_pair", "partner", "partner_lineage",
        "lr_means", "natmi_spec", "connectome_z", "combined",
        "magnitude_rank", "specificity_rank", "mag_score", "spec_score",
        "spec_ligand", "spec_receptor", "ligand_props", "receptor_props"),
      names(long))
    gt <- long |>
      mutate(.val = .data[[sc$col]]) |>
      group_by(direction) |>
      slice_max(.val, n = 50, with_ties = FALSE) |>
      ungroup() |>
      arrange(direction, desc(.val)) |>
      select(all_of(keep_cols))
    write_csv(gt, file.path(OUT_DIR,
              paste0("best4_global_top50_", sc$key, "_per_direction.csv")))

    pst <- long |>
      mutate(.val = .data[[sc$col]]) |>
      group_by(direction, segment) |>
      slice_max(.val, n = 30, with_ties = FALSE) |>
      ungroup() |>
      arrange(direction, segment, desc(.val)) |>
      select(all_of(keep_cols))
    write_csv(pst, file.path(OUT_DIR,
              paste0("best4_top30_per_segment_", sc$key, "_per_direction.csv")))
  }

  ## ---- plots: one full set per scoring -----
  aggs <- list()
  SCORINGS <<- scorings
  for (sc in SCORINGS) aggs[[sc$key]] <- render_scoring(long, sc)

  ## ---- diagnostic scatter: specificity vs magnitude -----
  for (dir_lbl in c("outgoing", "incoming"))
    plot_spec_vs_magnitude(long, dir_lbl,
      file.path(OUT_DIR, paste0("fig_best4_", dir_lbl,
                                "_specificity_vs_magnitude")))

  ## ---- console summary -----
  show_top <- function(score_col, label) {
    cat("\n==== Top 10 BEST4 OUTGOING per segment by ", label, " ====\n", sep = "")
    long |> filter(direction == "outgoing") |>
      mutate(.val = .data[[score_col]]) |>
      group_by(segment) |>
      slice_max(.val, n = 10, with_ties = FALSE) |>
      select(segment, lr_pair, partner, partner_lineage,
             score = .val, lr_means) |>
      print(n = Inf)
    cat("\n==== Top 10 BEST4 INCOMING per segment by ", label, " ====\n", sep = "")
    long |> filter(direction == "incoming") |>
      mutate(.val = .data[[score_col]]) |>
      group_by(segment) |>
      slice_max(.val, n = 10, with_ties = FALSE) |>
      select(segment, lr_pair, partner, partner_lineage,
             score = .val, lr_means) |>
      print(n = Inf)
  }
  show_top("combined", "LIANA ensemble (−log10 sqrt(mag_rank × spec_rank))")

  message("\nWrote outputs under: ", OUT_DIR)
}

main()
