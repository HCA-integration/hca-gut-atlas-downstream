#!/usr/bin/env Rscript
## Focused HGCA LIANA for three cell-type groups:
##   BEST4  (BEST4 Enterocytes, BEST4 Colonocytes)
##   Macrophage (all *Macrophage* subtypes)
##   Lymphatic (Lymphatic Endothelial, Medullary Sinus Endothelial)
##
## Outputs (under <OUT>/focus_three_groups/):
##   1. Classic LIANA in/outgoing dotplots per group, stratified by tissue_level_1
##      (only segments where the focal group has enough cells).
##   2. Between-group top communications across tissue_level_1
##      (group→group flux + subtype-resolved LR trajectories).
##
## Scoring: LIANA rank_aggregate consensus ranks (lower = better):
##   ensemble_rank  = sqrt(magnitude_rank * specificity_rank)
##   combined       = -log10(ensemble_rank)   (higher = stronger; for plots)
## Expression filter: applied only if ligand/receptor means+props are present.
## rank_aggregate tables usually omit them; LIANA already used expr_prop at run.
## Cell-count gate (from atlas obs of normal cells):
##   a subtype is usable in a segment only if n_cells >= MIN_CT_CELLS
##   a group×segment panel is drawn only if sum(usable subtypes) >= MIN_GROUP_CELLS
##
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
  library(ggrepel)
  library(patchwork)
  library(data.table, warn.conflicts = FALSE)
})

## ---------- params --------------------------------------------------------
INPUT_CSV <- Sys.getenv(
  "CCC_EDGE_CSV",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate/combined_lr_per_tissue_level_1.csv")
BASE_OUT <- Sys.getenv(
  "CCC_OUTPUT_DIR",
  "/Users/kylekimler/Projects/GCA/github_vignette_output/LIANA_rank_aggregate")
OUT_DIR <- file.path(BASE_OUT, "focus_three_groups")
COUNTS_CSV <- Sys.getenv(
  "CCC_COUNTS_CSV",
  file.path(OUT_DIR, "cell_counts_by_segment_subtype.csv"))
DROP_STATES <- trimws(strsplit(Sys.getenv("CCC_DROP_STATES", "Epithelial"), "[;,]")[[1]])
SEGMENT_ORDER <- c("duodenum", "jejunum", "ileum", "colon")
EXPR_PROP_MIN <- as.numeric(Sys.getenv("CCC_EXPR_PROP_MIN", "0.10"))
MEAN_MIN      <- as.numeric(Sys.getenv("CCC_MEAN_MIN", "0.10"))
MIN_CT_CELLS  <- as.integer(Sys.getenv("CCC_MIN_CT_CELLS", "25"))
MIN_GROUP_CELLS <- as.integer(Sys.getenv("CCC_MIN_GROUP_CELLS", "50"))
TOP_LR <- as.integer(Sys.getenv("CCC_TOP_LR", "12"))
TOP_BETWEEN_LR <- as.integer(Sys.getenv("CCC_TOP_BETWEEN_LR", "15"))
EPS <- 1e-9

GROUP_DEFS <- list(
  BEST4 = c("BEST4 Enterocytes", "BEST4 Colonocytes"),
  Macrophage = c(
    "Homeostatic Macrophages", "M0 Macrophages", "Cycling Macrophages",
    "Perivascular Resident Macrophages",
    "Follicle Associated Resident Macrophages"),
  Lymphatic = c("Lymphatic Endothelial", "Medullary Sinus Endothelial")
)
GROUP_ORDER <- c("BEST4", "Macrophage", "Lymphatic")
GROUP_HEX <- c(BEST4 = "#009E73", Macrophage = "#D55E00", Lymphatic = "#56B4E9")

MAX_FIG_WIDTH_MM  <- 180
MAX_FIG_HEIGHT_MM <- 170
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

## ---------- theme ---------------------------------------------------------
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
    str_replace("Medullary Sinus Endothelial", "Med Sinus Endo")
}

## ---------- support table -------------------------------------------------
if (!file.exists(COUNTS_CSV))
  stop("Cell-count CSV missing: ", COUNTS_CSV,
       "\nRun the atlas cell-count export first.")
counts <- read_csv(COUNTS_CSV, show_col_types = FALSE) |>
  mutate(segment = factor(segment, levels = SEGMENT_ORDER),
         usable = n_cells >= MIN_CT_CELLS)
fwrite(as.data.table(counts), file.path(OUT_DIR, "support_gate.csv"))

supported_cts <- function(group, segment) {
  counts |>
    filter(group == !!group, segment == !!segment, usable) |>
    pull(cell_state)
}
supported_segments <- function(group) {
  counts |>
    filter(group == !!group, usable) |>
    group_by(segment) |>
    summarise(n = sum(n_cells), .groups = "drop") |>
    filter(n >= MIN_GROUP_CELLS) |>
    pull(segment) |>
    as.character()
}

cat("\n=== Segment support (MIN_CT=", MIN_CT_CELLS,
    ", MIN_GROUP=", MIN_GROUP_CELLS, ") ===\n", sep = "")
for (g in GROUP_ORDER) {
  segs <- supported_segments(g)
  cat(g, ": ", paste(segs, collapse = ", "), "\n", sep = "")
  for (s in SEGMENT_ORDER) {
    cts <- supported_cts(g, s)
    if (!length(cts)) next
    cat("  ", s, " subtypes: ", paste(short_ct(cts), collapse = ", "), "\n", sep = "")
  }
}

## ---------- load + filter edges -------------------------------------------
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
has_expr <- all(c("ligand_props", "receptor_props", "ligand_means",
                  "receptor_means") %in% names(dt)) &&
  dt[, any(!is.na(ligand_props) & !is.na(receptor_props) &
             !is.na(ligand_means) & !is.na(receptor_means))]
if (has_expr) {
  dt <- dt[ligand_props >= EXPR_PROP_MIN & receptor_props >= EXPR_PROP_MIN &
             ligand_means >= MEAN_MIN & receptor_means >= MEAN_MIN]
  message(sprintf(
    "Expression filter (prop>=%.2f, mean>=%.2f): kept %s / %s edges",
    EXPR_PROP_MIN, MEAN_MIN, format(nrow(dt), big.mark = ","),
    format(n_pre, big.mark = ",")))
} else {
  message("No ligand/receptor means+props in table; keeping LIANA expr_prop-filtered edges (",
          format(n_pre, big.mark = ","), ")")
}

dt[, lr_pair := paste(ligand_complex, receptor_complex, sep = "->")]
dt[, src_group := fcase(
  source %in% GROUP_DEFS$BEST4, "BEST4",
  source %in% GROUP_DEFS$Macrophage, "Macrophage",
  source %in% GROUP_DEFS$Lymphatic, "Lymphatic",
  default = NA_character_)]
dt[, tgt_group := fcase(
  target %in% GROUP_DEFS$BEST4, "BEST4",
  target %in% GROUP_DEFS$Macrophage, "Macrophage",
  target %in% GROUP_DEFS$Lymphatic, "Lymphatic",
  default = NA_character_)]

## True LIANA ensemble ranks (lower better) -> score for plotting (higher better)
dt[, ensemble_rank := sqrt(pmax(magnitude_rank, EPS) *
                             pmax(specificity_rank, EPS))]
dt[, combined := -log10(pmax(ensemble_rank, EPS))]
if ("spec_weight" %in% names(dt)) {
  dt[, natmi_spec := spec_weight]
} else {
  dt[, natmi_spec := NA_real_]
}

## Drop edges involving unsupported subtypes in that segment
ok_key <- counts |> filter(usable) |>
  transmute(segment = as.character(segment), cell_state, ok = TRUE) |>
  as.data.table()
dt <- merge(dt, ok_key[, .(segment, source = cell_state, src_ok = ok)],
            by = c("segment", "source"), all.x = TRUE)
dt <- merge(dt, ok_key[, .(segment, target = cell_state, tgt_ok = ok)],
            by = c("segment", "target"), all.x = TRUE)
## For partners outside the 3 groups, keep them (ok = NA → TRUE for non-focus)
all_focus <- unique(unlist(GROUP_DEFS))
dt[is.na(src_ok) & !(source %in% all_focus), src_ok := TRUE]
dt[is.na(tgt_ok) & !(target %in% all_focus), tgt_ok := TRUE]
dt[is.na(src_ok), src_ok := FALSE]
dt[is.na(tgt_ok), tgt_ok := FALSE]
dt <- dt[src_ok == TRUE & tgt_ok == TRUE]

fwrite(dt[!is.na(src_group) | !is.na(tgt_group),
          .(segment, source, target, src_group, tgt_group, lr_pair,
            ligand_complex, receptor_complex, lr_means, ligand_means,
            receptor_means, ligand_props, receptor_props,
            magnitude_rank, specificity_rank, ensemble_rank, natmi_spec, combined)],
       file.path(OUT_DIR, "filtered_edges_focus_related.csv"))

## ---------- classic LIANA dotplot (per group, in/out) ---------------------
## Classic style: y = ligand -> receptor, x = source | target (partner subtype),
## colour = lr_means, size = combined (or prop). Facet by segment.

plot_classic_liana <- function(edges_dir, group, direction, stem) {
  if (!nrow(edges_dir)) return(invisible(NULL))
  segs <- intersect(SEGMENT_ORDER, unique(edges_dir$segment))
  if (!length(segs)) return(invisible(NULL))

  ## top LR per segment (union)
  top_lr <- edges_dir[, .(score = max(combined)), by = .(segment, lr_pair)][
    order(-score), head(.SD, TOP_LR), by = segment]$lr_pair |> unique()
  d <- edges_dir[lr_pair %in% top_lr]
  ## keep top partners overall to avoid hairball
  top_part <- d[, .(score = max(combined)), by = partner][order(-score)][
    1:min(18L, .N)]$partner
  d <- d[partner %in% top_part]
  ## best edge per (segment, lr, partner) if duplicates
  d <- d[, .SD[which.max(combined)], by = .(segment, lr_pair, partner, focus_ct)]

  d[, segment := factor(segment, levels = segs)]
  d[, partner_lab := short_ct(partner)]
  d[, focus_lab := short_ct(focus_ct)]
  d[, edge_lab := paste(focus_lab, partner_lab, sep = " → ")]
  if (direction == "incoming")
    d[, edge_lab := paste(partner_lab, focus_lab, sep = " → ")]

  ## order LR by mean combined
  lr_ord <- d[, .(m = mean(combined)), by = lr_pair][order(m)]$lr_pair
  d[, lr_pair := factor(lr_pair, levels = lr_ord)]
  edge_ord <- d[, .(m = mean(combined)), by = edge_lab][order(-m)]$edge_lab
  d[, edge_lab := factor(edge_lab, levels = edge_ord)]

  n_y <- length(levels(d$lr_pair))
  n_x <- length(levels(d$edge_lab))
  h_mm <- min(170, 28 + 4.2 * n_y)
  w_mm <- min(180, 40 + 14 * length(segs) + 2.2 * min(n_x, 10))

  if (d[, any(!is.na(ligand_props) & !is.na(receptor_props))]) {
    d[, point_size := ligand_props * receptor_props]
    size_name <- "prop×prop"
  } else {
    d[, point_size := combined]
    size_name <- "ensemble"
  }
  p <- ggplot(d, aes(x = edge_lab, y = lr_pair)) +
    geom_point(aes(size = point_size, colour = combined),
               shape = 16) +
    scale_colour_gradient(low = "#F2F2F2", high = "#A1430F",
                          name = "ensemble\nscore") +
    scale_size_continuous(range = c(0.8, 4.2), name = size_name) +
    facet_wrap(~ segment, nrow = 1, scales = "free_x") +
    labs(
      title = sprintf("%s %s communication (LIANA rank_aggregate)",
                      group, direction),
      subtitle = sprintf(
        "Top %d LR / segment by ensemble score; n_cells≥%d",
        TOP_LR, MIN_CT_CELLS),
      x = if (direction == "outgoing") "Focus subtype → partner" else "Partner → focus subtype",
      y = "Ligand → receptor") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
          axis.text.y = element_text(size = 5))
  save_pair(p, stem, w_mm, h_mm)
  message("wrote ", stem)
}

for (g in GROUP_ORDER) {
  segs <- supported_segments(g)
  if (!length(segs)) {
    message("SKIP group ", g, " (no supported segments)")
    next
  }
  ## OUTGOING: focus subtypes as source → any partner (classic LIANA)
  out_list <- lapply(segs, function(s) {
    cts <- supported_cts(g, s)
    if (!length(cts)) return(NULL)
    dt[segment == s & source %in% cts & !(target %in% cts)]
  })
  out <- rbindlist(out_list[!vapply(out_list, is.null, logical(1))], fill = TRUE)
  if (nrow(out)) {
    out[, `:=`(focus_ct = source, partner = target, direction = "outgoing")]
    fwrite(out[, .(segment, focus_ct, partner, lr_pair, lr_means,
                   magnitude_rank, specificity_rank, ensemble_rank, combined)],
           file.path(OUT_DIR, paste0(g, "_outgoing_edges.csv")))
    plot_classic_liana(out, g, "outgoing",
                       file.path(OUT_DIR, paste0("fig_", g, "_outgoing_liana_dot")))
  }

  ## INCOMING: any partner → focus subtypes
  inn_list <- lapply(segs, function(s) {
    cts <- supported_cts(g, s)
    if (!length(cts)) return(NULL)
    dt[segment == s & target %in% cts & !(source %in% cts)]
  })
  inn <- rbindlist(inn_list[!vapply(inn_list, is.null, logical(1))], fill = TRUE)
  if (nrow(inn)) {
    inn[, `:=`(focus_ct = target, partner = source, direction = "incoming")]
    fwrite(inn[, .(segment, focus_ct, partner, lr_pair, lr_means,
                   magnitude_rank, specificity_rank, ensemble_rank, combined)],
           file.path(OUT_DIR, paste0(g, "_incoming_edges.csv")))
    plot_classic_liana(inn, g, "incoming",
                       file.path(OUT_DIR, paste0("fig_", g, "_incoming_liana_dot")))
  }
}

## ---------- between-group communications ----------------------------------
## Edges where both ends are in the 3 groups and groups differ.
btw_list <- list()
for (s in SEGMENT_ORDER) {
  for (sg in GROUP_ORDER) {
    for (tg in setdiff(GROUP_ORDER, sg)) {
      scts <- supported_cts(sg, s)
      tcts <- supported_cts(tg, s)
      if (!length(scts) || !length(tcts)) next
      if (!(s %in% supported_segments(sg) && s %in% supported_segments(tg))) next
      sub <- dt[segment == s & source %in% scts & target %in% tcts]
      if (nrow(sub)) btw_list[[length(btw_list) + 1L]] <- sub
    }
  }
}
btw <- rbindlist(btw_list, fill = TRUE)
message("Between-group edges after gates: ", nrow(btw))
fwrite(btw[, .(segment, source, target, src_group, tgt_group, lr_pair,
               lr_means, magnitude_rank, specificity_rank, ensemble_rank,
               combined, natmi_spec)],
       file.path(OUT_DIR, "between_group_edges.csv"))

## A) Group→group flux heatmap (sum of combined) by segment
flux <- btw[, .(flux = sum(combined), n_edges = .N,
                n_lr = uniqueN(lr_pair)),
            by = .(segment, src_group, tgt_group)]
flux[, channel := paste(src_group, tgt_group, sep = " → ")]
flux[, segment := factor(segment, levels = SEGMENT_ORDER)]
fwrite(flux, file.path(OUT_DIR, "between_group_flux.csv"))

chan_ord <- flux[, .(m = mean(flux)), by = channel][order(-m)]$channel
flux[, channel := factor(channel, levels = rev(chan_ord))]

p_flux <- ggplot(flux, aes(segment, channel, fill = flux)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.0f", flux)), size = 1.8, colour = "black") +
  scale_fill_gradient(low = "#F2F2F2", high = "#0072B2", name = "Sum ensemble") +
  labs(title = "Between-group communication flux across tissue_level_1",
       subtitle = "Sum of LIANA rank_aggregate ensemble scores; cell-count-filtered",
       x = "Gut segment", y = "Source group → target group") +
  theme_gca()
save_pair(p_flux, file.path(OUT_DIR, "fig_between_group_flux"), 120, 70)

## B) Top LR axes between groups — how they change across segments
## Aggregate to (segment, src_group, tgt_group, lr_pair) taking max combined
ax <- btw[, .(combined = max(combined), lr_means = max(lr_means),
              mag_rank = min(magnitude_rank),
              best_src = source[which.max(combined)],
              best_tgt = target[which.max(combined)]),
          by = .(segment, src_group, tgt_group, lr_pair)]
## pick top LR overall by mean combined across segments where present
top_ax <- ax[, .(m = mean(combined), n_seg = uniqueN(segment)), by = lr_pair][
  n_seg >= 2][order(-m)][1:TOP_BETWEEN_LR]
ax_top <- ax[lr_pair %in% top_ax$lr_pair]
ax_top[, channel := paste(src_group, tgt_group, sep = "→")]
ax_top[, segment := factor(segment, levels = SEGMENT_ORDER)]
lr_ord <- top_ax[order(m)]$lr_pair
ax_top[, lr_pair := factor(lr_pair, levels = lr_ord)]
fwrite(ax_top, file.path(OUT_DIR, "between_group_top_lr_by_segment.csv"))

p_lr <- ggplot(ax_top, aes(segment, lr_pair)) +
  geom_point(aes(size = combined, colour = channel), alpha = 0.9) +
  scale_colour_manual(
    values = c("BEST4→Macrophage" = "#009E73", "Macrophage→BEST4" = "#66C2A5",
               "BEST4→Lymphatic" = "#56B4E9", "Lymphatic→BEST4" = "#0072B2",
               "Macrophage→Lymphatic" = "#D55E00", "Lymphatic→Macrophage" = "#E69F00"),
    name = "Channel") +
  scale_size_continuous(range = c(1.2, 5), name = "Ensemble") +
  labs(title = "Top between-group L–R axes across tissue_level_1",
       subtitle = sprintf("Top %d LR by LIANA ensemble score; ≥2 supported segments",
                          TOP_BETWEEN_LR),
       x = "Gut segment", y = "Ligand → receptor") +
  theme_gca() +
  theme(axis.text.y = element_text(size = 5))
save_pair(p_lr, file.path(OUT_DIR, "fig_between_group_top_lr"), 140, 95)

## C) Subtype × subtype heatmap of top edges (mean over segments + per segment)
## For each segment, max combined per source-target subtype pair
st <- btw[, .(score = max(combined), n_lr = uniqueN(lr_pair)),
          by = .(segment, source, target, src_group, tgt_group)]
st[, src_lab := short_ct(source)]
st[, tgt_lab := short_ct(target)]
st[, segment := factor(segment, levels = SEGMENT_ORDER)]
fwrite(st, file.path(OUT_DIR, "between_subtype_scores.csv"))

## keep subtypes that appear
src_levels <- st[, .(m = mean(score)), by = src_lab][order(-m)]$src_lab
tgt_levels <- st[, .(m = mean(score)), by = tgt_lab][order(-m)]$tgt_lab
st[, src_lab := factor(src_lab, levels = src_levels)]
st[, tgt_lab := factor(tgt_lab, levels = tgt_levels)]

p_st <- ggplot(st, aes(tgt_lab, src_lab, fill = score)) +
  geom_tile(colour = "white", linewidth = 0.25) +
  scale_fill_gradient(low = "#F2F2F2", high = "#A1430F", name = "Max ensemble") +
  facet_wrap(~ segment, nrow = 1) +
  labs(title = "Subtype ↔ subtype communication among BEST4 / Macrophage / Lymphatic",
       subtitle = "Max LIANA ensemble score per pair; blank = unsupported / absent",
       x = "Target subtype", y = "Source subtype") +
  theme_gca() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.text.y = element_text(size = 5))
save_pair(p_st, file.path(OUT_DIR, "fig_between_subtype_heatmap"), 180, 85)

## D) Per-channel LR heatmap (facet by channel) — change across segments
ax_top2 <- copy(ax_top)
ax_top2[, channel := factor(channel)]
p_ch <- ggplot(ax_top2, aes(segment, lr_pair, fill = combined)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  scale_fill_gradient(low = "#F2F2F2", high = "#0072B2", name = "Ensemble") +
  facet_wrap(~ channel, scales = "free_y", ncol = 3) +
  labs(title = "Between-group L–R strength by channel × segment",
       x = "Gut segment", y = NULL) +
  theme_gca() +
  theme(axis.text.y = element_text(size = 5),
        axis.text.x = element_text(angle = 30, hjust = 1, size = 5))
save_pair(p_ch, file.path(OUT_DIR, "fig_between_channel_lr_heatmap"), 180, 120)

## ---------- support summary figure ----------------------------------------
sup <- counts |>
  mutate(lab = short_ct(cell_state),
         status = ifelse(usable, "used", "excluded (n < min)"))
p_sup <- ggplot(sup, aes(segment, lab, fill = n_cells)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = n_cells, colour = status), size = 1.7) +
  scale_fill_gradient(low = "#F2F2F2", high = "#0072B2", name = "n cells",
                      trans = "log1p") +
  scale_colour_manual(values = c("used" = "black",
                                 "excluded (n < min)" = "#999999"),
                      name = NULL) +
  facet_wrap(~ group, scales = "free_y", ncol = 1) +
  labs(title = "Cell-count support gate for three focus groups",
       subtitle = sprintf("Subtype used only if n ≥ %d; group panel if sum ≥ %d (normal cells)",
                          MIN_CT_CELLS, MIN_GROUP_CELLS),
       x = "Gut segment", y = NULL) +
  theme_gca() +
  theme(axis.text.y = element_text(size = 5))
save_pair(p_sup, file.path(OUT_DIR, "fig_support_gate"), 100, 110)

cat("\nDONE → ", OUT_DIR, "\n", sep = "")
