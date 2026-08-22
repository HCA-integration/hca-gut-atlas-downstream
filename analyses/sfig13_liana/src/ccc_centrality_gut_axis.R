#!/usr/bin/env Rscript
## HGCA CCC centrality + tidygraph/ggraph visualization along the gut axis.
##
## Style spec: ~/Projects/GCA/publication2026/plot_specs.md
##   - 180 mm width / <= 170 mm height, Helvetica 5-7 pt, no grid,
##     L-shape axes, white background, Wong palette, cairo PDF + SVG + 300 dpi PNG.
##
## Lineage policy (CRITICAL):
##   We do NOT infer lineage by string heuristics. We use a real cell-state ->
##   lineage lookup CSV (default: data/hgca_celltype_v1_lineage.csv, derived from
##   AnnData hgca_celltype_level1/2/3). If any cell_state present in the LIANA
##   edges is missing from the lookup, the script writes a complete editable
##   template covering ALL unique states and stops with instructions.
##
## Plotting policy: igraph is allowed for centrality math only. ALL network
## visualization goes through tidygraph + ggraph.
##
## Outputs are written under <CCC_OUTPUT_DIR>/revised/.

set.seed(1L)

suppressPackageStartupMessages({
  library(dplyr,        warn.conflicts = FALSE)
  library(tidyr,        warn.conflicts = FALSE)
  library(readr)
  library(tibble)
  library(stringr,      warn.conflicts = FALSE)
  library(ggplot2)
  library(scales)
  library(igraph,       warn.conflicts = FALSE)
  library(tidygraph,    warn.conflicts = FALSE)
  library(ggraph,       warn.conflicts = FALSE)
  library(graphlayouts)
  library(patchwork)
  library(grid)
})
if (!requireNamespace("ggrepel",  quietly = TRUE)) stop("install.packages('ggrepel')")
if (!requireNamespace("svglite",  quietly = TRUE)) stop("install.packages('svglite')")
library(ggrepel)
DT_OK <- requireNamespace("data.table", quietly = TRUE)

## ---------- params + constants --------------------------------------------

OUTPUT_BASE <- Sys.getenv("CCC_OUTPUT_DIR", "results/ccc_centrality_gut_axis")
OUTPUT_DIR  <- file.path(OUTPUT_BASE, "revised")
PARAM_BF <- as.numeric(Sys.getenv("CCC_BACKBONE_TOP_FRAC", "0.02"))
PARAM_BK <- as.integer(Sys.getenv("CCC_BACKBONE_MAX_KEEP", "150"))
PARAM_BTW <- as.integer(Sys.getenv("CCC_BTW_MAX_VERTICES", "180"))
PARAM_MAX_EDGE_ROWS <- as.numeric(Sys.getenv("CCC_MAX_EDGE_ROWS", "0"))
PARAM_RAW_SEGMENTS <- as.integer(Sys.getenv("CCC_RAW_SEGMENTS", "0"))
## Optional ";"- or ","-separated list of segments to exclude (post-normalize).
## Example: CCC_EXCLUDE_SEGMENTS="jejunum"  or  "mesentery;accessory"
EXCLUDE_SEGMENTS <- {
  d <- Sys.getenv("CCC_EXCLUDE_SEGMENTS", "")
  v <- if (nzchar(d)) trimws(strsplit(d, "[;,]")[[1]]) else character(0)
  v[nzchar(v)]
}
DEFAULT_LINEAGE_CSV <- "data/hgca_celltype_v1_lineage.csv"

LINEAGE_ORDER <- c("Epithelial", "Lymphoid", "Myeloid", "Stromal",
                   "Endothelial", "Glial", "Other")
LINEAGE_HEX <- c(
  Epithelial  = "#009E73",   # bluish green (Wong)
  Lymphoid    = "#0072B2",   # HCA blue (Wong)
  Myeloid     = "#D55E00",   # vermillion (Wong)
  Stromal     = "#999999",   # mid grey (Wong)
  Endothelial = "#56B4E9",   # sky blue (Wong)
  Glial       = "#CC79A7",   # reddish purple (Wong)
  Other       = "#E0E0E0"    # light grey (Wong "absent")
)
SEG_FULL <- c("duodenum", "jejunum", "ileum", "caecum",
              "ascending colon", "transverse colon", "descending colon",
              "sigmoid colon", "rectum")
SEG_SHORT <- c("duodenum", "jejunum", "ileum", "colon")

## column aliases for input CSVs
MAP_SEGMENT <- c("segment", "gut_segment", "tissue_segment",
                 "tissue_level_1", "tissue_level_2",
                 "highres_tissue_ontology", "tissue_subset",
                 "sample_region", "organ_part",
                 "tissue", "tissue_ontology_term")
MAP_SOURCE  <- c("source", "source_cell", "sender", "sender_cell")
MAP_TARGET  <- c("target", "target_cell", "receiver", "receiver_cell")
MAP_WEIGHT  <- c("lr_means", "lrscore", "magnitude_rank", "lr_logfc",
                 "expr_prod", "scaled_weight", "lr_mean", "magscore",
                 "aggregate_rank", "weight", "ccc_weight", "score", "mean_rank")

## focus states for sender x receiver labelling (always shown if present)
FOCUS_INTERPRETABLE <- c(
  "Paneth Cells", "BEST4 Enterocytes", "BEST4 Colonocytes",
  "M0 Macrophages", "Homeostatic Macrophages", "cDC2",
  "Capillary Endothelial",
  "Lamina propria Fibroblasts (S1)",
  "Submucosal Fibroblasts (S3)",
  "Marginal Reticular Cells (MRC)"
)

## flag any cell_state name that exactly matches a coarse lineage label
COARSE_NAMES <- c("Epithelial", "Lymphoid", "Myeloid", "Stromal",
                  "Endothelial", "Glial", "Stromal Cells")

## Cell-state labels to drop entirely from input (low-quality / placeholder
## clusters). Override with CCC_DROP_STATES="Foo;Bar".
DROP_STATES <- {
  d <- Sys.getenv("CCC_DROP_STATES", "Epithelial")
  trimws(strsplit(d, "[;,]")[[1]])
}

## Curated focus panel: epithelial / sensory / secretory / glial cell states
## the team wants tracked together. Override with CCC_FOCUS_PANEL=";"-list.
FOCUS_PANEL <- {
  d <- Sys.getenv(
    "CCC_FOCUS_PANEL",
    paste("BEST4 Enterocytes", "BEST4 Colonocytes",
          "EEC Enterochromaffin (EC)", "EEC L", "EEC N", "EEC S",
          "EEC Progenitors", "Enteroendocrine Cells (EEC)",
          "Glia",
          "Tuft Cells", "Tuft Progenitors",
          "Goblet Cells", "Mature Goblet Cells",
          "Paneth Cells",
          sep = ";")
  )
  trimws(strsplit(d, "[;]")[[1]])
}

## plot_specs.md §1: max page sizes
MAX_FIG_WIDTH_MM  <- 180
MAX_FIG_HEIGHT_MM <- 170
GG_PT_PER_SIZE <- 2.845
MIN_GG_TEXT_SIZE <- 5 / GG_PT_PER_SIZE  ## ~ 5 pt floor

## ---------- theme + savers ------------------------------------------------

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
  ggsave(paste0(stem, ".svg"), p, width = wi, height = hi, device = svglite::svglite)
  ggsave(paste0(stem, ".png"), p, width = wi, height = hi, dpi = 300)
}


## ---------- helpers -------------------------------------------------------

pick_col <- function(df, aliases) {
  h <- aliases[aliases %in% names(df)][1]
  if (is.na(h)) NA_character_ else h
}

normalize_segment <- function(x) {
  s <- ifelse(is.na(x) | trimws(as.character(x)) %in% c("", "NA"),
              NA_character_, trimws(as.character(x)))
  out <- case_when(
    str_detect(s, regex("duoden",     ignore_case = TRUE)) ~ "duodenum",
    str_detect(s, regex("jejun",      ignore_case = TRUE)) ~ "jejunum",
    str_detect(s, regex("(^|[^a-z])ile(um|al)?", ignore_case = TRUE)) &
      !str_detect(s, regex("colon", ignore_case = TRUE)) ~ "ileum",
    str_detect(s, regex("cecum|caecum", ignore_case = TRUE)) ~ "caecum",
    str_detect(s, regex("ascending",  ignore_case = TRUE)) ~ "ascending colon",
    str_detect(s, regex("transverse", ignore_case = TRUE)) ~ "transverse colon",
    str_detect(s, regex("descending", ignore_case = TRUE)) ~ "descending colon",
    str_detect(s, regex("sigmoid",    ignore_case = TRUE)) ~ "sigmoid colon",
    str_detect(s, regex("\\brect",    ignore_case = TRUE)) ~ "rectum",
    str_detect(s, regex("colon",      ignore_case = TRUE)) ~ "colon",
    TRUE ~ NA_character_
  )
  coalesce(out, ifelse(!is.na(s), tolower(s), NA_character_))
}

order_segments <- function(present) {
  u <- unique(present[!is.na(present)])
  if (!length(u)) return(character(0))
  inf_h <- intersect(SEG_FULL, u)
  if (length(inf_h) >= 2L && setequal(sort(inf_h), sort(u))) {
    return(SEG_FULL[SEG_FULL %in% u])
  }
  if (length(intersect(u, SEG_SHORT)) == length(u)) {
    return(SEG_SHORT[SEG_SHORT %in% u])
  }
  parent_rank <- function(x) {
    s <- tolower(x)
    case_when(
      grepl("duoden",       s) ~ 1L,
      grepl("jejun",        s) ~ 2L,
      grepl("(^|[^a-z])ile(um|al)", s) & !grepl("colon", s) ~ 3L,
      grepl("small intest|small_intest", s) ~ 4L,
      grepl("cecum|caecum", s) ~ 5L,
      grepl("ascending",    s) ~ 6L,
      grepl("transverse",   s) ~ 7L,
      grepl("descending",   s) ~ 8L,
      grepl("sigmoid",      s) ~ 9L,
      grepl("\\brect",      s) ~ 10L,
      grepl("colon",        s) ~ 7L,
      TRUE                     ~ 99L
    )
  }
  sub_rank <- function(x) {
    s <- tolower(x)
    case_when(
      grepl("epithel", s) ~ 2L,
      grepl("mucos",   s) ~ 3L,
      TRUE                ~ 1L
    )
  }
  u[order(parent_rank(u), sub_rank(u), u)]
}

clean_state_name <- function(x) {
  s <- gsub("[\r\n\t]+", " ", as.character(x))
  s <- gsub("\\s+", " ", s)
  trimws(s)
}

## ---------- I/O -----------------------------------------------------------

read_ccc_edges <- function(path, max_rows = PARAM_MAX_EDGE_ROWS) {
  if (DT_OK) {
    df <- as_tibble(data.table::fread(path, showProgress = FALSE))
  } else {
    df <- suppressWarnings(read_csv(path, show_col_types = FALSE,
                                    progress = FALSE))
  }
  if (max_rows > 0 && nrow(df) > max_rows) {
    nm <- tolower(names(df))
    weight_col <- nm[nm %in% MAP_WEIGHT][1]
    if (!is.na(weight_col)) {
      ord <- order(suppressWarnings(-as.numeric(df[[weight_col]])))
      df <- df[head(ord, as.integer(max_rows)), , drop = FALSE]
    } else {
      df <- df[seq_len(as.integer(max_rows)), , drop = FALSE]
    }
    message("read_ccc_edges: capped to top ",
            format(as.integer(max_rows), big.mark = ","),
            " rows by weight column '", weight_col, "'")
  }
  df
}

standardize_ccc_columns <- function(df_raw) {
  df <- rename_with(df_raw, tolower)
  seg <- pick_col(df, MAP_SEGMENT)
  sr  <- pick_col(df, MAP_SOURCE)
  tg  <- pick_col(df, MAP_TARGET)
  wt  <- pick_col(df, MAP_WEIGHT)
  if (any(is.na(c(sr, tg, wt)))) {
    stop("Missing source/target/weight column. source=", sr,
         " target=", tg, " weight=", wt, call. = FALSE)
  }
  seg_vec <- if (is.na(seg)) {
    rep("ileum", nrow(df))
  } else if (PARAM_RAW_SEGMENTS == 1L) {
    s <- trimws(as.character(df[[seg]]))
    ifelse(nzchar(s) & s != "NA", s, NA_character_)
  } else {
    normalize_segment(df[[seg]])
  }
  out <- tibble(
    ccc_segment_std = seg_vec,
    source = clean_state_name(df[[sr]]),
    target = clean_state_name(df[[tg]]),
    ccc_weight_std = suppressWarnings(as.numeric(df[[wt]]))
  )
  out <- out |>
    filter(!is.na(ccc_segment_std), nzchar(source), nzchar(target)) |>
    mutate(ccc_weight_std = tidyr::replace_na(
      pmax(ccc_weight_std, 0, na.rm = TRUE), 0))

  ## Drop low-quality / placeholder states (e.g. coarse "Epithelial" cluster).
  drop <- intersect(DROP_STATES,
                    unique(c(out$source, out$target)))
  if (length(drop)) {
    n_pre <- nrow(out)
    out <- filter(out, !(source %in% drop), !(target %in% drop))
    message("Dropped cell_state(s) [", paste(drop, collapse = ", "),
            "]: removed ", format(n_pre - nrow(out), big.mark = ","),
            " of ", format(n_pre, big.mark = ","), " edges")
  }
  ## Drop user-excluded segments (e.g. CCC_EXCLUDE_SEGMENTS="jejunum").
  if (length(EXCLUDE_SEGMENTS)) {
    excl_present <- intersect(EXCLUDE_SEGMENTS, unique(out$ccc_segment_std))
    if (length(excl_present)) {
      n_pre <- nrow(out)
      out <- filter(out, !(ccc_segment_std %in% excl_present))
      message("Excluded segment(s) [", paste(excl_present, collapse = ", "),
              "]: removed ", format(n_pre - nrow(out), big.mark = ","),
              " of ", format(n_pre, big.mark = ","), " edges")
    }
    miss <- setdiff(EXCLUDE_SEGMENTS, excl_present)
    if (length(miss)) {
      message("CCC_EXCLUDE_SEGMENTS values not found in input (ignored): ",
              paste(miss, collapse = ", "))
    }
  }
  out
}

## Strict lineage resolver. NO heuristic fallback. If any state is unmapped,
## write a *complete* template covering every unique state (with whatever
## was already known prefilled) and stop.
resolve_lineages <- function(states_chr, lk_csv, od) {
  ux <- distinct(tibble(cell_state = clean_state_name(states_chr)))
  ux <- filter(ux, nzchar(cell_state))
  if (is.na(lk_csv) || !nzchar(lk_csv) || !file.exists(lk_csv)) {
    tmpl_path <- file.path(od, "ccc_cell_state_lineage_lookup_template.csv")
    dir.create(od, recursive = TRUE, showWarnings = FALSE)
    write_csv(arrange(mutate(ux, plot_lineage = ""), cell_state), tmpl_path)
    stop("CCC_LINEAGE_LOOKUP_CSV not found at '", lk_csv,
         "'.\n  Wrote template (", nrow(ux), " rows): ", tmpl_path,
         "\n  Fill `plot_lineage` (one of: ",
         paste(LINEAGE_ORDER, collapse = ", "),
         ") and rerun with CCC_LINEAGE_LOOKUP_CSV=<path>",
         call. = FALSE)
  }
  lk_raw <- suppressWarnings(read_csv(lk_csv, show_col_types = FALSE)) |>
    rename_with(tolower)
  lk_state_col <- intersect(c("cell_state", "hgca_celltype_v1", "celltype",
                              "cell_type"), names(lk_raw))[1]
  lk_lin_col <- intersect(c("plot_lineage", "lineage", "hgca_lineage"),
                          names(lk_raw))[1]
  if (is.na(lk_state_col) || is.na(lk_lin_col)) {
    stop("Lookup CSV must have a cell_state column and a plot_lineage/lineage ",
         "column. Got: ", paste(names(lk_raw), collapse = ", "), call. = FALSE)
  }
  lk <- transmute(lk_raw,
                  cell_state = clean_state_name(.data[[lk_state_col]]),
                  plot_lineage = trimws(as.character(.data[[lk_lin_col]]))) |>
    filter(nzchar(cell_state), nzchar(plot_lineage)) |>
    mutate(plot_lineage = ifelse(plot_lineage %in% LINEAGE_ORDER,
                                 plot_lineage, "Other")) |>
    distinct(cell_state, .keep_all = TRUE)

  joined <- left_join(ux, lk, by = "cell_state")
  miss <- filter(joined, is.na(plot_lineage))
  if (nrow(miss) > 0L) {
    tmpl <- arrange(transmute(joined,
                              cell_state,
                              plot_lineage = coalesce(plot_lineage, "")),
                    cell_state)
    tmpl_path <- file.path(od, "ccc_cell_state_lineage_lookup_template.csv")
    dir.create(od, recursive = TRUE, showWarnings = FALSE)
    write_csv(tmpl, tmpl_path)
    stop(nrow(miss), " of ", nrow(ux), " cell_states are missing from the ",
         "lookup CSV.\n  Wrote complete template at: ", tmpl_path,
         "\n  Fill the empty plot_lineage entries (allowed: ",
         paste(LINEAGE_ORDER, collapse = ", "),
         ") and rerun with CCC_LINEAGE_LOOKUP_CSV=<that-path>",
         call. = FALSE)
  }
  ## flag coarse / generic state names that match a lineage label
  coarse <- intersect(joined$cell_state, COARSE_NAMES)
  if (length(coarse) > 0L) {
    message("WARN: cell_state names match coarse lineage labels (likely ",
            "summary clusters): ", paste(coarse, collapse = ", "),
            "\n  Consider relabeling them in the source AnnData.")
  }
  joined |>
    mutate(plot_lineage = factor(plot_lineage, levels = LINEAGE_ORDER)) |>
    select(cell_state, plot_lineage)
}

## ---------- graph + centrality (igraph used for math only) ----------------

make_segment_graphs <- function(edge_tbl) {
  edge_tbl |>
    group_by(ccc_segment_std, source, target) |>
    summarise(weight = sum(ccc_weight_std, na.rm = TRUE), .groups = "drop") |>
    split(~ccc_segment_std)
}

backbone_edges <- function(eg, top_frac = PARAM_BF, max_keep = PARAM_BK) {
  if (!nrow(eg)) return(eg)
  o <- eg[order(-eg$weight), , drop = FALSE]
  k <- min(max(1L, ceiling(top_frac * nrow(o))), max_keep)
  head(o, k)
}

compute_segment_centrality <- function(seg_nm, eg, lineage_tbl) {
  if (!nrow(eg)) return(tibble())
  vs <- sort(unique(c(eg$source, eg$target)))
  nod <- left_join(tibble(cell_state = vs), lineage_tbl, by = "cell_state") |>
    mutate(
      plot_lineage = ifelse(is.na(plot_lineage) |
                              !plot_lineage %in% LINEAGE_ORDER,
                            "Other", as.character(plot_lineage)),
      lineage = factor(plot_lineage, levels = LINEAGE_ORDER)
    )
  el <- transmute(eg, from = source, to = target, weight)
  ig <- igraph::graph_from_data_frame(
    el, directed = TRUE,
    vertices = transmute(nod, name = cell_state, lineage)
  )
  igraph::E(ig)$weight <- pmax(igraph::E(ig)$weight, 1e-12)
  n <- igraph::vcount(ig)

  in_str  <- igraph::strength(ig, mode = "in",  weights = igraph::E(ig)$weight)
  out_str <- igraph::strength(ig, mode = "out", weights = igraph::E(ig)$weight)
  pr <- tryCatch(igraph::page_rank(ig, weights = igraph::E(ig)$weight)$vector,
                 error = function(e) rep(NA_real_, n))
  if (length(pr) != n) pr <- rep(NA_real_, n)
  hubs <- autor <- rep(NA_real_, n)
  hs <- tryCatch(suppressWarnings(igraph::hits_scores(
    ig, weights = igraph::E(ig)$weight)),
    error = function(e) NULL)
  if (!is.null(hs) && length(hs$hub) == n && length(hs$authority) == n) {
    hubs  <- as.numeric(hs$hub)
    autor <- as.numeric(hs$authority)
  }
  btw <- rep(NA_real_, n)
  if (n <= PARAM_BTW) {
    inv <- 1 / igraph::E(ig)$weight
    bt <- tryCatch(igraph::betweenness(ig, weights = inv),
                   error = function(e) rep(NA_real_, n))
    if (length(bt) == n) btw <- as.numeric(bt)
  }
  l2 <- transmute(lineage_tbl, to = cell_state,
                  tlin = as.character(plot_lineage))
  pc <- el |>
    left_join(l2, by = "to") |>
    mutate(tlin = ifelse(is.na(tlin) | !tlin %in% LINEAGE_ORDER,
                         "Other", tlin)) |>
    group_by(from, tlin) |>
    summarise(sw = sum(weight), .groups = "drop") |>
    group_by(from) |>
    summarise(participation_out = {
      tot <- sum(sw)
      if (!is.finite(tot) || tot <= 0) NA_real_
      else as.numeric(1 - sum((sw / tot)^2))
    }, .groups = "drop")

  tibble(
    ccc_segment_std = seg_nm,
    cell_state      = igraph::V(ig)$name,
    lineage         = factor(igraph::V(ig)$lineage, levels = LINEAGE_ORDER),
    in_strength     = as.numeric(in_str),
    out_strength    = as.numeric(out_str),
    total_strength  = as.numeric(in_str + out_str),
    pagerank        = as.numeric(pr),
    hub_score       = as.numeric(hubs),
    authority_score = as.numeric(autor),
    betweenness     = as.numeric(btw)
  ) |>
    left_join(pc, by = c("cell_state" = "from"))
}

## Add normalized centrality variants (within-segment)
augment_centrality <- function(nc) {
  saf_share <- function(x) {
    s <- sum(x, na.rm = TRUE)
    if (!is.finite(s) || s <= 0) rep(NA_real_, length(x))
    else x / s
  }
  saf_pct <- function(x) {
    if (sum(!is.na(x)) < 2L) return(rep(NA_real_, length(x)))
    rk <- rank(x, ties.method = "average", na.last = "keep")
    rk / sum(!is.na(x))
  }
  nc |>
    group_by(ccc_segment_std) |>
    mutate(
      total_strength_share      = saf_share(total_strength),
      total_strength_percentile = saf_pct(total_strength),
      in_strength_share         = saf_share(in_strength),
      out_strength_share        = saf_share(out_strength),
      pagerank_percentile       = saf_pct(pagerank),
      sender_receiver_balance   = out_strength_share - in_strength_share,
      centrality_rank_total     = rank(-total_strength, ties.method = "min"),
      centrality_rank_pct       = rank(-total_strength_percentile,
                                       ties.method = "min", na.last = "keep")
    ) |>
    ungroup()
}

## Layout from aggregate undirected graph; nodes are anchored by *name*.
make_aggregate_layout <- function(edge_full) {
  ag <- edge_full |>
    group_by(source, target) |>
    summarise(w = sum(ccc_weight_std, na.rm = TRUE), .groups = "drop") |>
    filter(w > 0)
  vn <- sort(unique(c(ag$source, ag$target)))
  ig <- igraph::graph_from_data_frame(
    transmute(ag, from = source, to = target, w),
    directed = FALSE,
    vertices = tibble(name = vn)
  )
  ig_u <- igraph::as_undirected(ig, mode = "collapse",
                                edge.attr.comb = list(w = "sum"))
  igraph::E(ig_u)$w <- pmax(igraph::E(ig_u)$w, 1e-9)
  xy <- tryCatch(layout_with_stress(ig_u, weights = igraph::E(ig_u)$w),
                 error = function(e) igraph::layout_with_fr(ig_u,
                                          weights = igraph::E(ig_u)$w))
  cr <- igraph::norm_coords(xy, ymin = -1, ymax = 1, xmin = -1, xmax = 1)
  tibble(name = igraph::V(ig_u)$name, lx = cr[, 1], ly = cr[, 2])
}

## ---------- diagnostics ---------------------------------------------------

build_segment_diagnostics <- function(edge_tbl, seg_graphs, backbone_list,
                                      nc_aug, seg_ord) {
  full_edges <- bind_rows(seg_graphs, .id = "ccc_segment_std")
  back <- bind_rows(backbone_list, .id = "ccc_segment_std")
  diag_full <- full_edges |>
    group_by(ccc_segment_std) |>
    summarise(
      n_nodes              = n_distinct(c(source, target)),
      n_edges_full         = n(),
      sum_edge_weight_full = sum(weight, na.rm = TRUE),
      .groups = "drop"
    )
  diag_back <- back |>
    group_by(ccc_segment_std) |>
    summarise(
      n_edges_backbone        = n(),
      sum_edge_weight_backbone = sum(weight, na.rm = TRUE),
      .groups = "drop"
    )
  diag_nc <- nc_aug |>
    group_by(ccc_segment_std) |>
    summarise(
      n_nodes_centrality        = n(),
      sum_node_total_strength   = sum(total_strength, na.rm = TRUE),
      mean_node_total_strength  = mean(total_strength, na.rm = TRUE),
      n_lr_pairs_pre_aggregation = sum(!is.na(total_strength)),
      .groups = "drop"
    )
  diag_full |>
    left_join(diag_back, by = "ccc_segment_std") |>
    left_join(diag_nc, by = "ccc_segment_std") |>
    mutate(ccc_segment_std = factor(ccc_segment_std, levels = seg_ord)) |>
    arrange(ccc_segment_std)
}

plot_segment_diagnostics <- function(seg_diag, stem) {
  d <- seg_diag |>
    mutate(segment = ccc_segment_std)
  mk <- function(yvar, ylab) {
    ggplot(d, aes(x = segment, y = .data[[yvar]])) +
      geom_col(fill = "#0072B2", width = 0.65) +
      labs(x = NULL, y = ylab) +
      theme_gca() +
      theme(axis.text.x = element_text(angle = 32, hjust = 1))
  }
  p1 <- mk("n_nodes",                 "Nodes (cell states)")
  p2 <- mk("n_edges_full",            "Edges (full graph)")
  p3 <- mk("n_edges_backbone",        "Edges (backbone)")
  p4 <- mk("sum_edge_weight_full",    "Sum edge weight")
  p5 <- mk("sum_node_total_strength", "Sum node total strength")
  p <- (p1 | p2 | p3) / (p4 | p5 | plot_spacer()) +
    plot_annotation(title = "Segment coverage diagnostics",
                    theme = theme_gca())
  save_pair(p, stem, MAX_FIG_WIDTH_MM, 110)
}


## ---------- ggraph-based fixed-layout network -----------------------------
##
## We build one tbl_graph per segment, share the layout via a manual layout
## table, render with ggraph geoms, then assemble panels with patchwork.
## This is the idiomatic tidygraph + ggraph approach.

ggraph_panel <- function(seg_nm, bk, xy, nc_seg, vol_tbl, lab_states_chr) {
  layout_tbl <- xy |>
    left_join(nc_seg |> select(name = cell_state, total_strength_share,
                               lineage),
              by = "name") |>
    mutate(
      lineage = factor(ifelse(is.na(lineage), "Other",
                              as.character(lineage)),
                       levels = LINEAGE_ORDER),
      total_strength_share = ifelse(is.na(total_strength_share), 0,
                                    total_strength_share),
      do_label = name %in% lab_states_chr,
      label_text = ifelse(do_label, str_trunc(name, 24), "")
    )
  ## edges with within-segment scaled weights
  ed <- bk |>
    transmute(from = source, to = target, weight) |>
    mutate(w_scaled = if (n_distinct(weight) > 1L)
      (weight - min(weight, na.rm = TRUE)) /
        (max(weight, na.rm = TRUE) - min(weight, na.rm = TRUE) + 1e-12)
      else 1)
  g <- tidygraph::tbl_graph(nodes = layout_tbl, edges = ed,
                            directed = TRUE, node_key = "name")
  ggraph(g, layout = "manual", x = lx, y = ly) +
    geom_edge_link(aes(width = w_scaled, alpha = w_scaled),
                   colour = "#999999", lineend = "round",
                   show.legend = FALSE) +
    scale_edge_width(range = c(0.05, 0.85)) +
    scale_edge_alpha(range = c(0.10, 0.50)) +
    geom_node_point(aes(fill = lineage, size = total_strength_share),
                    shape = 21, stroke = 0.12, colour = "#00000033") +
    scale_fill_manual(values = LINEAGE_HEX, drop = FALSE,
                      name = "Lineage") +
    scale_size(range = c(0.5, 5.5), guide = "none") +
    coord_fixed(clip = "off") +
    labs(title = seg_nm, x = NULL, y = NULL) +
    theme_gca() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.line = element_blank(),
          plot.title = element_text(face = "bold", size = 7,
                                    hjust = 0.5, colour = "black"),
          plot.margin = margin(2, 4, 2, 4))
}

add_node_labels <- function(p, layout_tbl) {
  p +
    ggrepel::geom_text_repel(
      data = filter(layout_tbl, do_label),
      aes(x = lx, y = ly, label = label_text),
      size = MIN_GG_TEXT_SIZE, family = "Helvetica",
      colour = "black",
      segment.size = 0.11, max.overlaps = 60,
      min.segment.length = 0,
      inherit.aes = FALSE
    )
}

plot_fixed_layout_network <- function(seg_ord, backbone_list, xy, nc_aug,
                                      vol_tbl, stem_labeled, stem_unlabeled,
                                      lab_top_k = 5L, lab_volatile = 12L) {
  ## label set: top-k hubs in each segment (by share) + globally top volatile
  lbl_top <- nc_aug |>
    filter(!is.na(centrality_rank_total)) |>
    group_by(ccc_segment_std) |>
    slice_min(order_by = centrality_rank_total, n = lab_top_k,
              with_ties = FALSE) |>
    ungroup() |>
    distinct(cell_state) |>
    pull(cell_state)
  lbl_vol <- if (!is.null(vol_tbl) && nrow(vol_tbl)) {
    slice_max(vol_tbl, order_by = volatility,
              n = min(lab_volatile, nrow(vol_tbl)),
              with_ties = FALSE) |>
      pull(cell_state)
  } else character()
  lab_states <- unique(c(lbl_top, lbl_vol))

  ## build per-segment ggraph panels
  panels_unlab <- lapply(seg_ord, function(s) {
    ggraph_panel(
      s,
      backbone_list[[s]] %||% tibble(source = character(), target = character(),
                                      weight = numeric()),
      xy,
      filter(nc_aug, ccc_segment_std == s),
      vol_tbl, character(0)
    )
  })
  panels_lab <- lapply(seg_ord, function(s) {
    bk <- backbone_list[[s]] %||% tibble(source = character(),
                                         target = character(),
                                         weight = numeric())
    nc_seg <- filter(nc_aug, ccc_segment_std == s)
    layout_tbl <- xy |>
      left_join(nc_seg |> select(name = cell_state, lineage,
                                 total_strength_share),
                by = "name") |>
      mutate(
        lineage = factor(ifelse(is.na(lineage), "Other",
                                as.character(lineage)), levels = LINEAGE_ORDER),
        total_strength_share = ifelse(is.na(total_strength_share), 0,
                                      total_strength_share),
        do_label = name %in% lab_states,
        label_text = ifelse(do_label, str_trunc(name, 22), "")
      )
    add_node_labels(ggraph_panel(s, bk, xy, nc_seg, vol_tbl, lab_states),
                    layout_tbl)
  })

  ncol_facet <- if (length(seg_ord) <= 4L) 2L else 4L
  combine_panels <- function(plots) {
    ## Hide legends on every panel; show one global legend assembled by patchwork
    plots <- lapply(plots, function(p) p + theme(legend.position = "none"))
    plots[[length(plots)]] <- plots[[length(plots)]] +
      theme(legend.position = "right") +
      guides(fill = guide_legend(override.aes = list(size = 3)))
    wrap_plots(plots, ncol = ncol_facet) +
      plot_annotation(
        title = "Sparse CCC backbone (shared stress coordinates, ggraph)",
        theme = theme_gca()
      )
  }
  save_pair(combine_panels(panels_unlab),  stem_unlabeled,
            MAX_FIG_WIDTH_MM, MAX_FIG_HEIGHT_MM)
  save_pair(combine_panels(panels_lab),    stem_labeled,
            MAX_FIG_WIDTH_MM, MAX_FIG_HEIGHT_MM)
}

`%||%` <- function(a, b) if (is.null(a)) b else a


## ---------- sender x receiver, bump, heatmaps, dot-line handoff -----------

plot_sender_receiver <- function(nc_aug, seg_ord, vol_tbl, stem_labeled,
                                 stem_unlabeled,
                                 focus_extra = FOCUS_INTERPRETABLE) {
  d <- nc_aug |>
    filter(ccc_segment_std %in% seg_ord) |>
    mutate(ccc_segment_std = factor(ccc_segment_std, levels = seg_ord)) |>
    group_by(ccc_segment_std) |>
    mutate(
      ox_z = if (n_distinct(out_strength_share) > 1L)
        as.numeric(scale(out_strength_share)) else 0,
      iy_z = if (n_distinct(in_strength_share) > 1L)
        as.numeric(scale(in_strength_share)) else 0
    ) |>
    ungroup()
  if (length(seg_ord) >= 2L) {
    arrows <- d |>
      arrange(cell_state, ccc_segment_std) |>
      group_by(cell_state) |>
      mutate(xe = lead(ox_z), ye = lead(iy_z)) |>
      filter(!is.na(xe)) |>
      ungroup()
  } else {
    arrows <- d[0, ] |>
      mutate(xe = numeric(0), ye = numeric(0))
  }
  shapes <- rep(c(21L, 22L, 23L, 24L, 25L, 3L, 4L, 8L),
                length.out = max(1L, nlevels(d$ccc_segment_std)))

  base_p <- ggplot(d, aes(x = ox_z, y = iy_z)) +
    facet_wrap(~lineage, ncol = 3) +
    geom_hline(yintercept = 0, colour = "#DDDDDD", linewidth = 0.25) +
    geom_vline(xintercept = 0, colour = "#DDDDDD", linewidth = 0.25)
  if (nrow(arrows)) {
    base_p <- base_p +
      geom_segment(
        data = arrows,
        aes(x = ox_z, xend = xe, y = iy_z, yend = ye,
            colour = lineage, group = cell_state),
        linewidth = 0.25, alpha = 0.45,
        arrow = arrow(length = unit(0.09, "cm"), type = "closed"),
        inherit.aes = FALSE
      )
  }
  base_p <- base_p +
    geom_point(aes(shape = ccc_segment_std, fill = lineage),
               stroke = 0.2, size = 1.95) +
    scale_fill_manual(values = LINEAGE_HEX, guide = "none") +
    scale_colour_manual(values = LINEAGE_HEX, guide = "none") +
    scale_shape_manual(values = shapes,
                       guide = guide_legend(title = "Segment", ncol = 1)) +
    labs(title = "Sender x receiver trajectory",
         subtitle = "Within-segment z-scaled out-share / in-share; arrows along proximal-distal axis",
         x = "Out-strength share z", y = "In-strength share z") +
    theme_gca()

  ## label set: focus + top-volatility + top-share (max-by-state)
  vol_top <- if (!is.null(vol_tbl) && nrow(vol_tbl))
    slice_max(vol_tbl, order_by = volatility, n = 10L,
              with_ties = FALSE) |> pull(cell_state)
  else character()
  share_top <- nc_aug |>
    group_by(cell_state) |>
    summarise(peak = max(total_strength_share, na.rm = TRUE), .groups = "drop") |>
    slice_max(order_by = peak, n = 10L, with_ties = FALSE) |>
    pull(cell_state)
  lab_states <- unique(c(focus_extra, vol_top, share_top))
  lab_states <- intersect(lab_states, unique(d$cell_state))
  lab_d <- d |>
    filter(cell_state %in% lab_states) |>
    group_by(cell_state) |>
    slice_max(order_by = total_strength_share, n = 1L,
              with_ties = FALSE) |>
    ungroup()

  labelled <- base_p +
    ggrepel::geom_text_repel(
      data = lab_d,
      aes(x = ox_z, y = iy_z, label = str_trunc(cell_state, 18)),
      size = MIN_GG_TEXT_SIZE, family = "Helvetica", colour = "black",
      segment.size = 0.12, max.overlaps = 30,
      inherit.aes = TRUE
    )
  save_pair(base_p,   stem_unlabeled, MAX_FIG_WIDTH_MM, 135)
  save_pair(labelled, stem_labeled,   MAX_FIG_WIDTH_MM, 135)
}

plot_bump_chart <- function(nc_aug, seg_ord, vol_tbl,
                            stem_all_segments, stem_min3) {
  build_one <- function(seg_pres_min, label) {
    pres <- nc_aug |>
      filter(!is.na(total_strength)) |>
      group_by(cell_state) |>
      summarise(n_seg = n_distinct(ccc_segment_std), .groups = "drop")
    keep <- pres$cell_state[pres$n_seg >= seg_pres_min]
    vol_top <- if (!is.null(vol_tbl) && nrow(vol_tbl))
      slice_max(filter(vol_tbl, cell_state %in% keep),
                order_by = volatility, n = 15L, with_ties = FALSE) |>
        pull(cell_state)
    else character()
    hub_top <- nc_aug |>
      filter(cell_state %in% keep) |>
      group_by(cell_state) |>
      summarise(peak = max(total_strength_share, na.rm = TRUE),
                .groups = "drop") |>
      slice_max(order_by = peak, n = 10L, with_ties = FALSE) |>
      pull(cell_state)
    focus <- unique(c(vol_top, hub_top))
    d <- nc_aug |>
      filter(cell_state %in% keep) |>
      mutate(segment = factor(ccc_segment_std, levels = seg_ord),
             lineage = factor(as.character(lineage), levels = LINEAGE_ORDER))
    end_pts <- d |>
      filter(cell_state %in% focus) |>
      mutate(seg_idx = as.integer(segment)) |>
      group_by(cell_state) |>
      filter(seg_idx == min(seg_idx) | seg_idx == max(seg_idx)) |>
      ungroup() |>
      select(-seg_idx)
    p <- ggplot(filter(d, cell_state %in% focus),
                aes(segment, centrality_rank_pct,
                    group = cell_state, colour = lineage)) +
      geom_line(linewidth = 0.35, alpha = 0.85) +
      geom_point(size = 1.6, stroke = 0.15) +
      ggrepel::geom_text_repel(
        data = end_pts,
        aes(label = str_trunc(cell_state, 22)),
        size = MIN_GG_TEXT_SIZE, family = "Helvetica",
        colour = "black", segment.size = 0.11, max.overlaps = 40,
        inherit.aes = TRUE
      ) +
      scale_y_reverse() +
      scale_colour_manual(values = LINEAGE_HEX,
                          guide = guide_legend(title = "Lineage")) +
      labs(title = paste0("Centrality rank shifts (percentile)  -  ", label),
           subtitle = "Top 15 volatile + top 10 hubs (within filter)",
           x = NULL, y = "Within-segment percentile rank (1 = top)") +
      theme_gca() +
      theme(axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1))
    p
  }
  p_all <- build_one(length(seg_ord),                 "all segments")
  p_3p  <- build_one(min(3L, length(seg_ord)),        ">= 3 segments")
  save_pair(p_all, stem_all_segments, MAX_FIG_WIDTH_MM, 110)
  save_pair(p_3p,  stem_min3,         MAX_FIG_WIDTH_MM, 120)
}

## Bump chart focused on the union of top-N hubs *within each segment*,
## so segments with their own top hubs (not just duodenum's top) are all
## represented.
plot_bump_chart_per_segment_top <- function(nc_aug, seg_ord, stem,
                                            n_per_segment = 6L) {
  ## Union of top-N states (by total_strength_share) in EACH segment
  picks <- nc_aug |>
    filter(!is.na(total_strength_share)) |>
    group_by(ccc_segment_std) |>
    slice_max(order_by = total_strength_share,
              n = n_per_segment, with_ties = FALSE) |>
    ungroup() |>
    distinct(cell_state) |>
    pull(cell_state)
  if (!length(picks)) {
    warning("plot_bump_chart_per_segment_top: no states selected")
    return(invisible(NULL))
  }
  d <- nc_aug |>
    filter(cell_state %in% picks) |>
    mutate(segment = factor(ccc_segment_std, levels = seg_ord),
           lineage = factor(as.character(lineage), levels = LINEAGE_ORDER))
  end_pts <- d |>
    mutate(seg_idx = as.integer(segment)) |>
    group_by(cell_state) |>
    filter(seg_idx == min(seg_idx) | seg_idx == max(seg_idx)) |>
    ungroup() |>
    select(-seg_idx)
  p <- ggplot(d, aes(segment, centrality_rank_pct,
                     group = cell_state, colour = lineage)) +
    geom_line(linewidth = 0.35, alpha = 0.85) +
    geom_point(size = 1.6, stroke = 0.15) +
    ggrepel::geom_text_repel(
      data = end_pts,
      aes(label = str_trunc(cell_state, 22)),
      size = MIN_GG_TEXT_SIZE, family = "Helvetica",
      colour = "black", segment.size = 0.11, max.overlaps = 60,
      inherit.aes = TRUE
    ) +
    scale_y_reverse() +
    scale_colour_manual(values = LINEAGE_HEX,
                        guide = guide_legend(title = "Lineage")) +
    labs(title = paste0("Centrality rank shifts (percentile) — ",
                        "per-segment top ", n_per_segment),
         subtitle = "Union of top hubs per segment by within-segment share",
         x = NULL, y = "Within-segment percentile rank (1 = top)") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1))
  save_pair(p, stem, MAX_FIG_WIDTH_MM, 120)
}

## Curated focus-panel bump + dot-line: track a hand-picked set of states
## (BEST4 enterocytes/colonocytes, EEC types, Glia, Tuft, Goblet, Paneth) along
## the proximal-distal axis. Two views: rank shift + absolute share.
plot_focus_panel <- function(nc_aug, seg_ord, focus_chr, stem) {
  d <- nc_aug |>
    filter(cell_state %in% focus_chr) |>
    mutate(segment = factor(ccc_segment_std, levels = seg_ord),
           lineage = factor(as.character(lineage), levels = LINEAGE_ORDER))
  if (!nrow(d)) {
    warning("plot_focus_panel: none of the focus states found")
    return(invisible(NULL))
  }
  missing <- setdiff(focus_chr, unique(d$cell_state))
  if (length(missing)) {
    message("plot_focus_panel: ", length(missing),
            " focus state(s) not in input: ",
            paste(missing, collapse = ", "))
  }
  ## ordering of the curated cells: by group, then by lineage, then alpha
  group_of <- function(cs) {
    case_when(
      grepl("^BEST4", cs)                                 ~ "BEST4",
      grepl("^EEC|Enteroendocrine", cs)                   ~ "EEC",
      grepl("^Glia$|^Glial", cs)                          ~ "Glia",
      grepl("Tuft", cs)                                   ~ "Tuft",
      grepl("Goblet", cs)                                 ~ "Goblet",
      grepl("Paneth", cs)                                 ~ "Paneth",
      TRUE                                                ~ "Other"
    )
  }
  d <- mutate(d, panel_group = group_of(cell_state))
  panel_levels <- c("BEST4", "EEC", "Tuft", "Goblet", "Paneth", "Glia", "Other")
  d$panel_group <- factor(d$panel_group,
                          levels = intersect(panel_levels,
                                             unique(d$panel_group)))

  ## Panel A: rank-shift bump
  end_pts <- d |>
    mutate(seg_idx = as.integer(segment)) |>
    group_by(cell_state) |>
    filter(seg_idx == min(seg_idx) | seg_idx == max(seg_idx)) |>
    ungroup() |>
    select(-seg_idx)
  ## Stable color per cell_state within group, but readable in greys/Wong.
  state_levels <- d |>
    arrange(panel_group, cell_state) |>
    distinct(cell_state) |>
    pull(cell_state)
  d$cell_state <- factor(d$cell_state, levels = state_levels)
  end_pts$cell_state <- factor(end_pts$cell_state, levels = state_levels)
  p_rank <- ggplot(d, aes(segment, centrality_rank_pct,
                          group = cell_state, colour = panel_group)) +
    geom_line(linewidth = 0.35, alpha = 0.85) +
    geom_point(size = 1.7, stroke = 0.15) +
    ggrepel::geom_text_repel(
      data = end_pts,
      aes(label = str_trunc(as.character(cell_state), 22)),
      size = MIN_GG_TEXT_SIZE, family = "Helvetica",
      colour = "black", segment.size = 0.11, max.overlaps = 60,
      inherit.aes = TRUE
    ) +
    scale_y_reverse() +
    scale_colour_brewer(palette = "Dark2",
                        guide = guide_legend(title = "Group", ncol = 1)) +
    labs(title = "Curated focus panel — centrality rank shifts",
         subtitle = "BEST4 enterocytes/colonocytes, EECs, Glia, Tuft, Goblet, Paneth",
         x = NULL, y = "Within-segment percentile rank (1 = top)") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1))

  ## Panel B: absolute within-segment share
  p_share <- ggplot(d, aes(segment, total_strength_share,
                           group = cell_state, colour = panel_group)) +
    geom_line(linewidth = 0.35, alpha = 0.85) +
    geom_point(size = 1.7, stroke = 0.15) +
    ggrepel::geom_text_repel(
      data = end_pts,
      aes(label = str_trunc(as.character(cell_state), 22)),
      size = MIN_GG_TEXT_SIZE, family = "Helvetica",
      colour = "black", segment.size = 0.11, max.overlaps = 60,
      inherit.aes = TRUE
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    scale_colour_brewer(palette = "Dark2", guide = "none") +
    labs(title = "Curated focus panel — share of within-segment total strength",
         subtitle = "Same states as above, absolute share",
         x = NULL, y = "Share of segment total strength") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1))

  combined <- p_rank / p_share +
    plot_layout(heights = c(1, 1), guides = "collect") &
    theme(legend.position = "right")
  save_pair(combined, stem, MAX_FIG_WIDTH_MM, MAX_FIG_HEIGHT_MM)
}

plot_centrality_heatmaps <- function(nc_aug, seg_ord, stem_relative, stem_raw) {
  ## ROWS = states present in >= 3 segments
  pres <- nc_aug |>
    filter(!is.na(total_strength)) |>
    group_by(cell_state) |>
    summarise(n_seg = n_distinct(ccc_segment_std), .groups = "drop")
  keep <- pres$cell_state[pres$n_seg >= min(3L, length(seg_ord))]
  d <- nc_aug |>
    filter(cell_state %in% keep) |>
    mutate(ccc_segment_std = factor(ccc_segment_std, levels = seg_ord),
           lineage = factor(as.character(lineage), levels = LINEAGE_ORDER))
  ## relative: percentile, z-scored within state
  build_mat <- function(value_col) {
    w <- d |>
      mutate(seg_ch = as.character(ccc_segment_std)) |>
      select(cell_state, seg_ch, value = all_of(value_col)) |>
      pivot_wider(names_from = seg_ch, values_from = value,
                  values_fill = NA_real_)
    cols <- intersect(seg_ord, names(w))
    m <- as.matrix(w[, cols, drop = FALSE])
    rownames(m) <- w$cell_state
    m[!is.finite(m)] <- NA_real_
    m
  }
  z_within_state <- function(m) {
    if (ncol(m) < 2L) {
      out <- m
      out[!is.na(out)] <- 0
      return(out)
    }
    out <- t(apply(m, 1, function(r) {
      rr <- as.numeric(r)
      if (sum(!is.na(rr)) < 2 || stats::sd(rr, na.rm = TRUE) == 0) {
        rep(0, length(rr))
      } else as.numeric(scale(rr))
    }))
    rownames(out) <- rownames(m)
    colnames(out) <- colnames(m)
    out
  }

  m_pct <- build_mat("total_strength_percentile")
  m_raw <- build_mat("total_strength")
  z_pct <- z_within_state(m_pct)
  z_raw <- z_within_state(m_raw)

  lin_lookup <- distinct(d, cell_state, lineage)
  order_rows <- function(zmat) {
    pick_argmax <- function(r) {
      rr <- as.numeric(r); rr[!is.finite(rr)] <- NA_real_
      if (all(is.na(rr))) NA_integer_ else which.max(rr)
    }
    am <- apply(zmat, 1, pick_argmax)
    max_seg <- ifelse(is.na(am), NA_character_, colnames(zmat)[am])
    vol <- apply(zmat, 1, function(r) stats::sd(r, na.rm = TRUE))
    tibble(cell_state = rownames(zmat),
           max_seg = max_seg, vol = vol) |>
      left_join(lin_lookup, by = "cell_state") |>
      mutate(lineage = factor(coalesce(as.character(lineage), "Other"),
                              levels = LINEAGE_ORDER)) |>
      arrange(lineage, max_seg, desc(vol)) |>
      pull(cell_state)
  }

  draw <- function(zmat, raw_mat, title_main, subtitle, stem) {
    rk <- order_rows(zmat)
    zmat <- zmat[rk, , drop = FALSE]
    seg_levels <- intersect(seg_ord, colnames(zmat))
    ## Build long df from the matrix (column order = seg_levels)
    df <- as_tibble(zmat[, seg_levels, drop = FALSE], rownames = "cell_state") |>
      pivot_longer(-cell_state, names_to = "segment", values_to = "z") |>
      mutate(segment = factor(segment, levels = seg_levels),
             cell_state = factor(cell_state, levels = rk)) |>
      left_join(lin_lookup, by = "cell_state")
    miss_df <- df |> mutate(missing = is.na(z))
    p <- ggplot(df, aes(segment, cell_state, fill = z)) +
      geom_tile(data = filter(miss_df, missing),
                aes(segment, cell_state),
                fill = "#E0E0E0", inherit.aes = FALSE) +
      geom_tile(colour = "white", linewidth = 0.05) +
      scale_x_discrete(limits = seg_levels, expand = c(0, 0)) +
      scale_y_discrete(limits = rk, expand = c(0, 0)) +
      scale_fill_gradient2(low = "#0072B2", mid = "#F8F8F8", high = "#D55E00",
                           midpoint = 0, na.value = "#E0E0E0",
                           name = "z-score") +
      labs(title = title_main, subtitle = subtitle,
           x = NULL, y = NULL) +
      theme_gca() +
      theme(axis.text.y = element_text(size = 5),
            axis.text.x = element_text(angle = 32, hjust = 1, size = 5.5))
    hmm <- max(110, ceiling(nrow(zmat) * 1.6) + 30)
    save_pair(p, stem, MAX_FIG_WIDTH_MM, hmm)
  }

  draw(z_pct, m_pct,
       "Centrality heatmap (relative)",
       "Rows: states present in >= 3 segments; value = within-state z of total-strength percentile; grey = absent",
       stem_relative)
  draw(z_raw, m_raw,
       "Centrality heatmap (raw, diagnostic)",
       "Value = within-state z of raw total strength; sensitive to absolute LR-pair count - use for QC only",
       stem_raw)
}

plot_lineage_handoff_dot <- function(nc_aug, seg_ord, stem,
                                     n_per_lineage = 6L) {
  d <- nc_aug |>
    filter(!is.na(total_strength_share)) |>
    mutate(segment = factor(ccc_segment_std, levels = seg_ord),
           lineage = factor(as.character(lineage), levels = LINEAGE_ORDER))
  ## select states: top by max share OR by volatility per lineage
  vol_by <- d |>
    group_by(cell_state, lineage) |>
    summarise(vol = stats::sd(total_strength_share, na.rm = TRUE),
              peak = max(total_strength_share, na.rm = TRUE),
              .groups = "drop")
  picks <- vol_by |>
    group_by(lineage) |>
    arrange(desc(peak), desc(vol), .by_group = TRUE) |>
    slice_head(n = n_per_lineage) |>
    ungroup() |>
    select(cell_state, lineage)
  if (!nrow(picks)) return(invisible(NULL))
  fl <- semi_join(d, picks, by = c("cell_state", "lineage"))
  ## Order rows per lineage by peak share (descending peak at top)
  row_order <- fl |>
    group_by(lineage, cell_state) |>
    summarise(peak = max(total_strength_share, na.rm = TRUE), .groups = "drop") |>
    arrange(lineage, desc(peak)) |>
    mutate(row_id = paste(lineage, cell_state, sep = " | "))
  fl <- fl |>
    mutate(row_id = paste(lineage, cell_state, sep = " | "),
           row_id = factor(row_id, levels = row_order$row_id),
           y_label = factor(cell_state, levels = unique(row_order$cell_state)))
  p <- ggplot(fl, aes(segment, row_id)) +
    facet_wrap(~lineage, scales = "free_y", ncol = 2) +
    scale_y_discrete(labels = function(x) sub(".+? \\| ", "", x)) +
    geom_point(aes(size = total_strength_share,
                   fill = sender_receiver_balance),
               shape = 21, stroke = 0.15, colour = "#33333344") +
    scale_size(range = c(0.5, 4),
               name = "Share of total\nstrength (segment)",
               labels = scales::percent_format(accuracy = 0.1)) +
    scale_fill_gradient2(low = "#0072B2", mid = "#F8F8F8",
                         high = "#D55E00", midpoint = 0,
                         name = "Sender shift\n(out_share - in_share)") +
    labs(title = "Per-lineage hub handoff (dot-line)",
         subtitle = "Size = share of within-segment total strength; fill = sender vs receiver balance",
         x = NULL, y = NULL) +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 32, hjust = 1, vjust = 1),
          axis.text.y = element_text(size = 5.5))
  save_pair(p, stem, MAX_FIG_WIDTH_MM, 150)
}


## ---------- main driver ---------------------------------------------------

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  edge_csv <- Sys.getenv("CCC_EDGE_CSV", "")
  if (!nzchar(edge_csv) && length(args) >= 1L) edge_csv <- args[[1]]
  if (!nzchar(edge_csv) || !file.exists(edge_csv)) {
    stop("CCC_EDGE_CSV not set or file missing: '", edge_csv, "'", call. = FALSE)
  }
  message("Reading edges from: ", edge_csv)
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

  raw <- read_ccc_edges(edge_csv)
  message("rows read: ", format(nrow(raw), big.mark = ","))
  edges <- standardize_ccc_columns(raw)
  rm(raw); gc()
  message("standardized rows: ", format(nrow(edges), big.mark = ","),
          " | unique segments: ",
          paste(sort(unique(edges$ccc_segment_std)), collapse = ", "))

  states_chr <- unique(c(edges$source, edges$target))
  lk_csv <- Sys.getenv("CCC_LINEAGE_LOOKUP_CSV", "")
  if (!nzchar(lk_csv)) {
    cand <- file.path(getwd(), DEFAULT_LINEAGE_CSV)
    if (file.exists(cand)) lk_csv <- cand
  }
  lineage_tbl <- resolve_lineages(states_chr, lk_csv, OUTPUT_DIR)

  seg_graphs <- make_segment_graphs(edges)
  seg_ord <- order_segments(names(seg_graphs))
  seg_graphs <- seg_graphs[seg_ord]
  if (length(seg_ord) < 2L) {
    message("WARN: only one segment ('", seg_ord, "') in input -- ",
            "across-segment plots will be single-panel.")
  }

  backbone_list <- lapply(seg_ord, function(s) {
    backbone_edges(seg_graphs[[s]], PARAM_BF, PARAM_BK)
  })
  names(backbone_list) <- seg_ord

  nc_list <- lapply(seg_ord, function(s) {
    compute_segment_centrality(s, seg_graphs[[s]], lineage_tbl)
  })
  nc <- bind_rows(nc_list)
  nc_aug <- augment_centrality(nc)

  ## volatility: SD of within-segment z-scaled total_strength_share
  if (length(seg_ord) >= 2L) {
    z_seg <- nc_aug |>
      group_by(ccc_segment_std) |>
      mutate(ts_share_z = if (n_distinct(total_strength_share) > 1L)
        as.numeric(scale(total_strength_share)) else 0) |>
      ungroup()
    vol_tbl <- z_seg |>
      group_by(cell_state) |>
      summarise(volatility = stats::sd(ts_share_z, na.rm = TRUE),
                .groups = "drop") |>
      mutate(volatility = ifelse(is.na(volatility), 0, volatility))
  } else {
    vol_tbl <- distinct(nc_aug, cell_state) |> mutate(volatility = 0)
  }
  max_seg <- nc_aug |>
    group_by(cell_state) |>
    slice_max(order_by = total_strength_share, n = 1L,
              with_ties = FALSE) |>
    ungroup() |>
    transmute(cell_state, max_segment = ccc_segment_std)
  nc_export <- left_join(nc_aug, vol_tbl, by = "cell_state") |>
    left_join(max_seg, by = "cell_state")

  write_csv(nc_export,
            file.path(OUTPUT_DIR, "ccc_node_centrality_by_segment.csv"))
  write_csv(bind_rows(backbone_list, .id = "ccc_segment_std"),
            file.path(OUTPUT_DIR, "ccc_edge_backbone_by_segment.csv"))

  ## diagnostics
  seg_diag <- build_segment_diagnostics(edges, seg_graphs, backbone_list,
                                         nc_aug, seg_ord)
  write_csv(seg_diag, file.path(OUTPUT_DIR, "ccc_segment_diagnostics.csv"))
  plot_segment_diagnostics(seg_diag,
    file.path(OUTPUT_DIR, "fig_ccc_segment_diagnostics"))

  ## layout + ggraph network plots (labeled + unlabeled)
  message("computing aggregate layout for ",
          n_distinct(c(edges$source, edges$target)), " nodes")
  xy <- make_aggregate_layout(edges)
  plot_fixed_layout_network(
    seg_ord, backbone_list, xy, nc_aug, vol_tbl,
    stem_labeled   = file.path(OUTPUT_DIR, "fig_ccc_fixed_layout_network_labeled"),
    stem_unlabeled = file.path(OUTPUT_DIR, "fig_ccc_fixed_layout_network_unlabeled")
  )

  ## sender x receiver
  plot_sender_receiver(
    nc_aug, seg_ord, vol_tbl,
    stem_labeled   = file.path(OUTPUT_DIR, "fig_ccc_sender_receiver_labeled"),
    stem_unlabeled = file.path(OUTPUT_DIR, "fig_ccc_sender_receiver_unlabeled")
  )

  ## bump charts
  plot_bump_chart(
    nc_aug, seg_ord, vol_tbl,
    stem_all_segments = file.path(OUTPUT_DIR, "fig_ccc_bump_all_segments"),
    stem_min3         = file.path(OUTPUT_DIR, "fig_ccc_bump_min3_segments")
  )
  ## per-segment-top bump: union of top-N hubs in each segment (so each
  ## segment is represented, not just the proximal-anchored ones)
  plot_bump_chart_per_segment_top(
    nc_aug, seg_ord,
    stem = file.path(OUTPUT_DIR, "fig_ccc_bump_per_segment_top"),
    n_per_segment = 6L
  )

  ## curated focus panel (BEST4, EECs, Glia, Tuft, Goblet, Paneth)
  plot_focus_panel(nc_aug, seg_ord, FOCUS_PANEL,
    stem = file.path(OUTPUT_DIR, "fig_ccc_focus_panel"))

  ## heatmaps
  plot_centrality_heatmaps(nc_aug, seg_ord,
    stem_relative = file.path(OUTPUT_DIR, "fig_ccc_heatmap_relative"),
    stem_raw      = file.path(OUTPUT_DIR, "fig_ccc_heatmap_raw_diagnostic")
  )

  ## dot-line lineage handoff
  plot_lineage_handoff_dot(nc_aug, seg_ord,
    stem = file.path(OUTPUT_DIR, "fig_ccc_lineage_hub_handoff_dotline"))

  ## ---------- summary ----------------------------------------------------
  emit <- function(label, df, value_col, fmt = "%.3g", scale = 1,
                   suffix = "") {
    message(label)
    if (!nrow(df)) { message("  (none)"); return(invisible()) }
    vals <- df[[value_col]] * scale
    for (i in seq_len(nrow(df))) {
      message(sprintf("  %2d. %s  (%s=", i, df$cell_state[i], value_col),
              appendLF = FALSE)
      message(sprintf(fmt, vals[i]), suffix, ")")
    }
  }

  hubs_raw <- nc_aug |>
    group_by(cell_state) |>
    summarise(peak_total_strength = max(total_strength, na.rm = TRUE),
              .groups = "drop") |>
    slice_max(order_by = peak_total_strength, n = 10L, with_ties = FALSE)
  hubs_share <- nc_aug |>
    group_by(cell_state) |>
    summarise(peak_share = max(total_strength_share, na.rm = TRUE),
              .groups = "drop") |>
    slice_max(order_by = peak_share, n = 10L, with_ties = FALSE)
  vol_top <- vol_tbl |>
    slice_max(order_by = volatility, n = 10L, with_ties = FALSE)
  sender_top <- nc_aug |>
    group_by(cell_state) |>
    summarise(max_balance = max(sender_receiver_balance, na.rm = TRUE),
              .groups = "drop") |>
    slice_max(order_by = max_balance, n = 10L, with_ties = FALSE)
  receiver_top <- nc_aug |>
    group_by(cell_state) |>
    summarise(min_balance = min(sender_receiver_balance, na.rm = TRUE),
              .groups = "drop") |>
    slice_min(order_by = min_balance, n = 10L, with_ties = FALSE)

  message("\n=== CCC centrality summary ===")
  message("output dir: ", normalizePath(OUTPUT_DIR, mustWork = FALSE))
  message("segments (", length(seg_ord), "): ",
          paste(seg_ord, collapse = " | "))
  message("nodes: ", n_distinct(nc_aug$cell_state))
  message("backbone edges total: ",
          sum(vapply(backbone_list, nrow, integer(1))))
  emit("\nTop 10 hubs by raw peak total_strength:",
       hubs_raw, "peak_total_strength", "%.3g")
  emit("\nTop 10 hubs by peak total_strength_share (within segment):",
       hubs_share, "peak_share", "%.2f", scale = 100, suffix = "%")
  emit("\nTop 10 most volatile cell states (z-scaled share SD):",
       vol_top, "volatility", "%.3g")
  emit("\nTop 10 'sender-shifted' (max out-in balance):",
       sender_top, "max_balance", "%+.3g")
  emit("\nTop 10 'receiver-shifted' (min out-in balance):",
       receiver_top, "min_balance", "%+.3g")

  ## strongest segment-specific hubs (top 5 per segment by share)
  message("\nStrongest segment-specific hubs (top 5 by within-segment share):")
  for (s in seg_ord) {
    top5 <- nc_aug |>
      filter(ccc_segment_std == s) |>
      slice_max(order_by = total_strength_share, n = 5L, with_ties = FALSE)
    if (!nrow(top5)) next
    message("  [", s, "]")
    for (i in seq_len(nrow(top5))) {
      message(sprintf("    %d. %s  (share=%.2f%%, balance=%+.2f)",
                      i, top5$cell_state[i],
                      100 * top5$total_strength_share[i],
                      top5$sender_receiver_balance[i]))
    }
  }
}

if (!interactive()) {
  main()
}
