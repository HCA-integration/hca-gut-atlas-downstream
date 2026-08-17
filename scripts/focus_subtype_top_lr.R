#!/usr/bin/env Rscript
## Per-subtype top L–R + partner plots for:
##   BEST4, Macrophages, and all Endothelial subtypes.
##
## For each subtype, show the strongest interactions with:
##   - ligand→receptor pair
##   - partner cell type (receiver if outgoing, sender if incoming)
##   - direction (sent / received)
## Stratified by tissue_level_1 where the subtype has enough cells.
##
## Same expression + cell-count gates as focus_three_groups_liana.R.
## Style: ~/Projects/GCA/publication2026/plot_specs.md

set.seed(1L)
suppressPackageStartupMessages({
  library(dplyr, warn.conflicts = FALSE)
  library(tidyr, warn.conflicts = FALSE)
  library(readr)
  library(tibble)
  library(stringr, warn.conflicts = FALSE)
  library(ggplot2)
  library(scales)
  library(patchwork)
  library(data.table, warn.conflicts = FALSE)
})

INPUT_CSV <- Sys.getenv(
  "CCC_EDGE_CSV",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/combined_lr_per_tissue_level_1.csv")
BASE_OUT <- Sys.getenv(
  "CCC_OUTPUT_DIR",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate")
OUT_DIR <- file.path(BASE_OUT, "focus_three_groups", "subtype_top_lr")
LINEAGE_CSV <- Sys.getenv("CCC_LINEAGE_LOOKUP_CSV",
                          "data/hgca_celltype_v1_lineage.csv")
COUNTS_CSV <- Sys.getenv(
  "CCC_COUNTS_FULL_CSV",
  file.path(BASE_OUT, "focus_three_groups",
            "cell_counts_by_segment_subtype_full.csv"))
DROP_STATES <- trimws(strsplit(Sys.getenv("CCC_DROP_STATES", "Epithelial"), "[;,]")[[1]])
SEGMENT_ORDER <- c("duodenum", "jejunum", "ileum", "colon")
EXPR_PROP_MIN <- as.numeric(Sys.getenv("CCC_EXPR_PROP_MIN", "0.10"))
MEAN_MIN      <- as.numeric(Sys.getenv("CCC_MEAN_MIN", "0.10"))
MIN_CT_CELLS  <- as.integer(Sys.getenv("CCC_MIN_CT_CELLS", "25"))
TOP_N         <- as.integer(Sys.getenv("CCC_TOP_N_PER_DIR", "10"))
EPS <- 1e-9

lk <- read_csv(LINEAGE_CSV, show_col_types = FALSE)
BEST4 <- lk$cell_state[str_detect(lk$cell_state, "^BEST4")]
MAC   <- lk$cell_state[str_detect(lk$cell_state, "Macrophage")]
ENDO  <- lk$cell_state[lk$plot_lineage == "Endothelial"]
FOCUS <- unique(c(BEST4, MAC, ENDO))
GROUP_OF <- c(
  setNames(rep("BEST4", length(BEST4)), BEST4),
  setNames(rep("Macrophage", length(MAC)), MAC),
  setNames(rep("Endothelial", length(ENDO)), ENDO)
)

## Exploratory subtype panels are allowed wider than the 180 mm manuscript
## cap so segment facets and score axes stay legible.
MAX_FIG_WIDTH_MM <- 360
MAX_FIG_HEIGHT_MM <- 200
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

theme_gca <- function(fs = "Helvetica", sz = 6L) {
  theme_classic(base_size = sz, base_family = fs) +
    theme(
      text = element_text(family = fs, colour = "black"),
      plot.title = element_text(face = "bold", size = 7, hjust = 0),
      plot.subtitle = element_text(size = 6),
      axis.line = element_line(colour = "black", linewidth = 0.25),
      axis.ticks = element_line(colour = "black", linewidth = 0.25),
      axis.text = element_text(colour = "black", size = 6),
      axis.title = element_text(size = 6),
      panel.grid = element_blank(),
      panel.background = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      legend.key = element_blank(),
      legend.text = element_text(size = 6),
      legend.title = element_text(size = 6, face = "bold"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 6)
    )
}

save_pair <- function(p, stem, w_mm, h_mm) {
  w_mm <- min(w_mm, MAX_FIG_WIDTH_MM); h_mm <- min(h_mm, MAX_FIG_HEIGHT_MM)
  wi <- w_mm / 25.4; hi <- h_mm / 25.4
  pdf_dev <- if (isTRUE(capabilities("cairo"))) grDevices::cairo_pdf else pdf
  ggsave(paste0(stem, ".pdf"), p, width = wi, height = hi, device = pdf_dev)
  if (requireNamespace("svglite", quietly = TRUE))
    ggsave(paste0(stem, ".svg"), p, width = wi, height = hi, device = svglite::svglite)
  ggsave(paste0(stem, ".png"), p, width = wi, height = hi, dpi = 300)
}

short_ct <- function(x) {
  x |>
    str_replace("Follicle Associated Resident Macrophages", "FA Res Mac") |>
    str_replace("Perivascular Resident Macrophages", "PV Res Mac") |>
    str_replace("Homeostatic Macrophages", "Homeo Mac") |>
    str_replace("Cycling Macrophages", "Cyc Mac") |>
    str_replace("M0 Macrophages", "M0 Mac") |>
    str_replace("BEST4 Enterocytes", "BEST4 Ent") |>
    str_replace("BEST4 Colonocytes", "BEST4 Col") |>
    str_replace("Lymphatic Endothelial", "Lymph Endo") |>
    str_replace("Medullary Sinus Endothelial", "Med Sinus Endo") |>
    str_replace("Post Arteriole Capillary Endothelial \\(PAC\\)", "PAC Endo") |>
    str_replace("Pre Venule Capillary Endothelial \\(PVC\\)", "PVC Endo") |>
    str_replace("Arteriolar Endothelial", "Arter Endo") |>
    str_replace("Capillary Endothelial", "Cap Endo") |>
    str_replace("Venular Endothelial", "Ven Endo") |>
    str_replace("Transiently Amplifying Cells \\(TA\\)", "TA") |>
    str_replace("Intestinal Stem Cells \\(ISC\\)", "ISC") |>
    str_replace("Smooth Muscle Cells \\(SMC\\)", "SMC") |>
    str_replace("Fibroblastic Reticular Cells \\(FRC\\)", "FRC") |>
    str_replace("CD8 Circulating Effector Memory", "CD8 Circ EM") |>
    str_replace("CD8 Effector Memory", "CD8 EM") |>
    str_replace("Lamina propria Fibroblasts \\(S1\\)", "S1 Fib") |>
    str_replace("Crypt Bottom Fibroblasts \\(S2A\\)", "S2A Fib") |>
    str_replace("Crypt Top Fibroblasts \\(S2B\\)", "S2B Fib") |>
    str_replace("Submucosal Fibroblasts \\(S3\\)", "S3 Fib")
}

safe_name <- function(x) {
  x |> str_replace_all("[^A-Za-z0-9]+", "_") |> str_replace_all("_+", "_") |>
    str_replace("^_|_$", "")
}

## ---------- cell counts (atlas) -------------------------------------------
## Prefer precomputed full counts; otherwise require the Python export.
if (!file.exists(COUNTS_CSV)) {
  stop("Missing ", COUNTS_CSV,
       "\nExport cell counts for BEST4 + Macrophage + Endothelial first.")
}
counts <- read_csv(COUNTS_CSV, show_col_types = FALSE) |>
  mutate(segment = factor(segment, levels = SEGMENT_ORDER),
         usable = n_cells >= MIN_CT_CELLS)
usable_map <- counts |> filter(usable) |>
  transmute(segment = as.character(segment), cell_state, n_cells)

cat("\n=== Usable subtype × segment (n≥", MIN_CT_CELLS, ") ===\n", sep = "")
print(as.data.frame(usable_map |> arrange(cell_state, segment)))

## ---------- load + filter LIANA -------------------------------------------
message("Reading ", INPUT_CSV)
dt <- fread(INPUT_CSV)
need <- c("magnitude_rank", "specificity_rank")
miss <- setdiff(need, names(dt))
if (length(miss))
  stop("Need LIANA rank_aggregate columns: ", paste(miss, collapse = ", "))
setnames(dt, "tissue_level_1", "segment")
dt[, segment := tolower(trimws(segment))]
dt <- dt[segment %in% SEGMENT_ORDER &
           !is.na(magnitude_rank) & !is.na(specificity_rank)]
dt <- dt[!(source %in% DROP_STATES) & !(target %in% DROP_STATES)]
if (!"ligand_props" %in% names(dt)) dt[, ligand_props := NA_real_]
if (!"receptor_props" %in% names(dt)) dt[, receptor_props := NA_real_]
if (!"ligand_means" %in% names(dt)) dt[, ligand_means := NA_real_]
if (!"receptor_means" %in% names(dt)) dt[, receptor_means := NA_real_]
if (!"lr_means" %in% names(dt)) dt[, lr_means := NA_real_]
n_pre <- nrow(dt)
has_expr <- dt[, any(!is.na(ligand_props) & !is.na(receptor_props) &
                       !is.na(ligand_means) & !is.na(receptor_means))]
if (has_expr) {
  dt <- dt[ligand_props >= EXPR_PROP_MIN & receptor_props >= EXPR_PROP_MIN &
             ligand_means >= MEAN_MIN & receptor_means >= MEAN_MIN]
  message(sprintf("Expression filter: kept %s / %s",
                  format(nrow(dt), big.mark = ","), format(n_pre, big.mark = ",")))
} else {
  message("No means/props columns; keeping LIANA expr_prop-filtered edges (",
          format(n_pre, big.mark = ","), ")")
}
dt[, lr_pair := paste(ligand_complex, receptor_complex, sep = "→")]

## True LIANA ensemble ranks → plot score
dt[, ensemble_rank := sqrt(pmax(magnitude_rank, EPS) *
                             pmax(specificity_rank, EPS))]
dt[, combined := -log10(pmax(ensemble_rank, EPS))]
if ("spec_weight" %in% names(dt)) {
  dt[, natmi_spec := spec_weight]
} else {
  dt[, natmi_spec := NA_real_]
}

## ---------- build per-subtype long table ----------------------------------
rows <- list()
for (ct in FOCUS) {
  segs <- usable_map$segment[usable_map$cell_state == ct]
  if (!length(segs)) next
  for (s in segs) {
    out <- dt[segment == s & source == ct & target != ct,
              .(segment, focus = ct, direction = "sent (outgoing)",
                partner = target, lr_pair, ligand = ligand_complex,
                receptor = receptor_complex, lr_means, ligand_props,
                receptor_props, combined, natmi_spec)]
    inn <- dt[segment == s & target == ct & source != ct,
              .(segment, focus = ct, direction = "received (incoming)",
                partner = source, lr_pair, ligand = ligand_complex,
                receptor = receptor_complex, lr_means, ligand_props,
                receptor_props, combined, natmi_spec)]
    if (nrow(out)) rows[[length(rows) + 1L]] <- out
    if (nrow(inn)) rows[[length(rows) + 1L]] <- inn
  }
}
long <- rbindlist(rows, fill = TRUE)
if (!nrow(long)) stop("No edges left after gates.")
long[, group := GROUP_OF[focus]]
long[, focus_lab := short_ct(focus)]
long[, partner_lab := short_ct(partner)]
long[, segment := factor(segment, levels = SEGMENT_ORDER)]
long[, direction := factor(direction,
                           levels = c("sent (outgoing)", "received (incoming)"))]
fwrite(long, file.path(OUT_DIR, "all_subtype_edges_long.csv"))

## Top N per (focus, segment, direction)
top <- long[, .SD[order(-combined)][1:min(TOP_N, .N)],
            by = .(focus, segment, direction)]
fwrite(top, file.path(OUT_DIR, "top_lr_per_subtype_segment.csv"))

## Also a segment-collapsed top (max across segments) for a compact view
top_glob <- long[, .(
  combined = max(combined),
  lr_means = max(lr_means),
  best_segment = segment[which.max(combined)],
  n_seg = uniqueN(segment),
  partner = partner[which.max(combined)],
  partner_lab = partner_lab[which.max(combined)],
  ligand = ligand[which.max(combined)],
  receptor = receptor[which.max(combined)]
), by = .(focus, focus_lab, group, direction, lr_pair)]
top_glob <- top_glob[, .SD[order(-combined)][1:min(TOP_N, .N)],
                     by = .(focus, direction)]
fwrite(top_glob, file.path(OUT_DIR, "top_lr_per_subtype_global.csv"))

DIR_HEX <- c("sent (outgoing)" = "#0072B2", "received (incoming)" = "#D55E00")

## Explicit labels so ligand ownership is unambiguous:
##   sent:     "sends L → R on <partner>"
##   received: "receives L via R from <partner>"
make_hit_label <- function(direction, ligand, receptor, partner_lab) {
  ifelse(
    as.character(direction) == "sent (outgoing)",
    paste0("sends ", ligand, " → ", receptor, " on ", partner_lab),
    paste0("receives ", ligand, " via ", receptor, " from ", partner_lab)
  )
}

## ---------- plot: per subtype, facets = segment, y = LR | partner ---------
plot_subtype <- function(ct) {
  d <- top[focus == ct]
  if (!nrow(d)) return(invisible(NULL))
  d[, hit := make_hit_label(direction, ligand, receptor, partner_lab)]
  ## order hits within each segment×direction by score
  d[, hit := reorder(hit, combined)]
  segs <- levels(droplevels(d$segment))
  n_hits <- d[, .(n = uniqueN(hit)), by = .(segment, direction)][, max(n)]
  ## Wide panels: ~60 mm per segment column + room for y labels / legend
  h_mm <- min(MAX_FIG_HEIGHT_MM, 36 + 4.0 * n_hits * 2)
  w_mm <- min(MAX_FIG_WIDTH_MM, 80 + 60 * length(segs))
  ## Zoom x to the data range for -log10 ensemble ranks
  x0 <- max(0, min(d$combined, na.rm = TRUE) * 0.95)
  x1 <- max(d$combined, na.rm = TRUE) * 1.02
  x_breaks <- pretty(c(x0, x1), n = 4)

  p <- ggplot(d, aes(x = combined, y = hit, colour = direction)) +
    geom_segment(aes(x = x0, xend = combined, yend = hit), linewidth = 0.35) +
    geom_point(aes(size = pmax(lr_means, 0.01)), shape = 16) +
    scale_colour_manual(values = DIR_HEX, name = "Direction") +
    scale_size_continuous(range = c(1.2, 3.8), name = "lr_means") +
    scale_x_continuous(limits = c(x0, x1), name = "Ensemble score (−log10 rank)",
                       breaks = x_breaks,
                       expand = expansion(mult = c(0.01, 0.06))) +
    facet_grid(direction ~ segment, scales = "free_y", space = "free_y") +
    labs(
      title = paste0(short_ct(ct), " — top L–R × partner"),
      subtitle = sprintf(
        paste0("%s · top %d per direction × segment · LIANA rank_aggregate · n≥%d\n",
               "sent = focus expresses ligand; received = focus expresses receptor"),
        GROUP_OF[[ct]], TOP_N, MIN_CT_CELLS),
      y = NULL) +
    theme_gca() +
    theme(axis.text.x = element_text(size = 6),
          axis.text.y = element_text(size = 5.5),
          strip.text.y = element_text(size = 5.5),
          panel.spacing.x = unit(5, "mm"),
          plot.margin = margin(4, 10, 4, 4))
  stem <- file.path(OUT_DIR, paste0("fig_", safe_name(ct), "_top_lr"))
  save_pair(p, stem, w_mm, h_mm)
  message("wrote ", stem)
}

for (ct in FOCUS) {
  if (ct %in% unique(top$focus)) plot_subtype(ct)
}

## ---------- overview: one page per group (global top, direction split) ----
plot_group_overview <- function(g) {
  d <- top_glob[group == g]
  if (!nrow(d)) return(invisible(NULL))
  d[, hit := make_hit_label(direction, ligand, receptor, partner_lab)]
  d[, focus_lab := factor(focus_lab, levels = unique(focus_lab))]
  ## rank within focus×direction
  d[, hit := reorder(interaction(focus_lab, hit, drop = TRUE), combined)]
  n_f <- uniqueN(d$focus)
  h_mm <- min(MAX_FIG_HEIGHT_MM, 32 + 6.0 * n_f * TOP_N / 2)
  w_mm <- 280

  x0g <- max(0, min(d$combined, na.rm = TRUE) * 0.9)
  p <- ggplot(d, aes(x = combined, y = reorder(hit, combined), colour = direction)) +
    geom_segment(aes(x = x0g, xend = combined, yend = reorder(hit, combined)),
                 linewidth = 0.3) +
    geom_point(aes(shape = best_segment), size = 1.8) +
    scale_colour_manual(values = DIR_HEX, name = "Direction") +
    scale_shape_manual(values = c(duodenum = 16, jejunum = 17,
                                  ileum = 15, colon = 18),
                       name = "Best segment") +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.08))) +
    facet_wrap(~ focus_lab, scales = "free_y", ncol = 1) +
    labs(
      title = paste0(g, " subtypes — top L–R × partner (best segment)"),
      subtitle = sprintf(
        paste0("Top %d sent + top %d received per subtype · LIANA rank_aggregate · cell-count-filtered\n",
               "sent = focus expresses ligand; received = focus expresses receptor"),
        TOP_N, TOP_N),
      x = "Ensemble score (max over segments)",
      y = NULL) +
    theme_gca() +
    theme(axis.text.x = element_text(size = 6),
          axis.text.y = element_text(size = 5),
          plot.margin = margin(4, 8, 4, 4))
  stem <- file.path(OUT_DIR, paste0("fig_", g, "_overview_top_lr"))
  save_pair(p, stem, w_mm, h_mm)
  message("wrote ", stem)
}

for (g in c("BEST4", "Macrophage", "Endothelial")) plot_group_overview(g)

## ---------- compact summary table (CSV already) + printable strip ---------
## One strip plot: subtype on y, top hit label, direction colour — global
summ <- top_glob[, .SD[order(-combined)][1:min(5L, .N)], by = .(focus, direction)]
summ[, label := make_hit_label(direction, ligand, receptor, partner_lab)]
fwrite(summ, file.path(OUT_DIR, "top5_summary_per_subtype_direction.csv"))

cat("\nDONE → ", OUT_DIR, "\n", sep = "")
cat("Subtypes plotted: ", paste(unique(top$focus), collapse = "; "), "\n", sep = "")
