#!/usr/bin/env Rscript
# Nature-style compositional correlation panels (corrplot, circles only).
#
# - Circles encode Spearman r (colour + size)
# - Non-significant (BH-FDR ≥ 0.05) left blank so stars are unnecessary
# - Generous margins so colour legend is readable
# - Small titles; sample support in subtitle
# - Underpowered segments omitted (see analysis_support_summary.csv)

suppressPackageStartupMessages({
  library(corrplot)
  library(readr)
  library(dplyr)
  library(svglite)
  library(ggplot2)
})

HERE <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) "."
)
if (length(HERE) == 0 || HERE == "") HERE <- "."
DATA <- file.path(HERE, "..", "data")
OUT  <- file.path(HERE, "..", "out")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
MM <- 25.4

COL <- colorRampPalette(c(
  "#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7",
  "#FDDBC7", "#EF8A62", "#B2182B"
))(200)

read_mat <- function(path) {
  as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
}

shorten <- function(x) {
  sets <- read_csv(file.path(DATA, "celltype_sets.csv"), show_col_types = FALSE)
  mp <- setNames(sets$short_name, sets$celltype)
  ifelse(x %in% names(mp), unname(mp[x]), x)
}

support_line <- function(scope = "overall_epi_immune", segment = "all") {
  s <- read_csv(file.path(DATA, "analysis_support_summary.csv"),
                show_col_types = FALSE)
  if (scope %in% c("segment_epi_immune", "chemfrac_epi_immune", "lp_epi_immune")) {
    row <- s %>% filter(scope == !!scope, segment == !!segment)
  } else {
    row <- s %>% filter(scope == "overall_epi_immune")
  }
  if (nrow(row) == 0) return("")
  note <- as.character(row$note[1])
  warn <- if (grepl("underpowered", note)) " [UNDERPOWERED]" else ""
  sprintf(
    "n = %d epi+immune samples; detect = n_cells>=3 in >=20 samples; circles = FDR<0.05%s",
    row$n_samples[1], warn
  )
}

powered_segments <- function() {
  s <- read_csv(file.path(DATA, "analysis_support_summary.csv"),
                show_col_types = FALSE)
  s %>%
    filter(scope == "segment_epi_immune", note == "powered") %>%
    arrange(match(segment, c("ileum", "colon", "jejunum", "duodenum"))) %>%
    pull(segment)
}

shown_chemfrac <- function(pattern = NULL) {
  s <- read_csv(file.path(DATA, "analysis_support_summary.csv"),
                show_col_types = FALSE)
  ord <- c(
    "biopsy_unfractionated", "biopsy_unfractionated_ileum",
    "biopsy_unfractionated_colon",
    "biopsy_fractionated",
    "resection_unfractionated", "resection_fractionated"
  )
  out <- s %>%
    filter(
      scope == "chemfrac_epi_immune",
      note == "powered" | grepl("^shown_underpowered", note)
    ) %>%
    arrange(match(segment, ord)) %>%
    pull(segment)
  if (!is.null(pattern)) out <- out[grepl(pattern, out)]
  out
}

shown_lp <- function(pattern = NULL) {
  s <- read_csv(file.path(DATA, "analysis_support_summary.csv"),
                show_col_types = FALSE)
  ord <- c("biopsy_all", "biopsy_ileum", "biopsy_colon", "all_ileum")
  out <- s %>%
    filter(
      scope == "lp_epi_immune",
      note == "powered" | grepl("^shown_underpowered", note)
    ) %>%
    arrange(match(segment, ord)) %>%
    pull(segment)
  if (!is.null(pattern)) out <- out[grepl(pattern, out)]
  out
}

# Append detection support to axis labels (n samples with >=3 cells)
label_with_support <- function(names, segment = NULL) {
  sets <- read_csv(file.path(DATA, "celltype_sets.csv"), show_col_types = FALSE)
  prev <- read_csv(file.path(DATA, "celltype_prevalence_by_segment.csv"),
                   show_col_types = FALSE)
  short <- shorten(names)
  # map short/full -> detect counts
  if (is.null(segment)) {
    # overall: sum ge3 across segments among epi+immune? use pooled meta prevalence
    # approximate from sum of n_detect_ge3 / sum n_samples weighted — use max segment
    # Prefer overall epi+immune file if present; else sum detects
    agg <- prev %>%
      group_by(celltype, short_name) %>%
      summarise(n_detect_ge3 = sum(n_detect_ge3), .groups = "drop")
  } else {
    agg <- prev %>% filter(segment == !!segment)
  }
  det_by_full <- setNames(agg$n_detect_ge3, agg$celltype)
  det_by_short <- setNames(agg$n_detect_ge3, agg$short_name)
  vapply(seq_along(names), function(i) {
    nm <- names[i]
    sh <- short[i]
    n <- if (nm %in% names(det_by_full)) det_by_full[[nm]]
    else if (sh %in% names(det_by_short)) det_by_short[[sh]]
    else NA_integer_
    if (is.finite(n)) paste0(sh, " (", n, ")") else sh
  }, character(1))
}

# Blank non-significant entries so only supported significant associations show
mask_nonsig <- function(r, p, alpha = 0.05) {
  r2 <- r
  r2[!(is.finite(p) & p < alpha)] <- NA
  # keep diagonal if present
  if (nrow(r2) == ncol(r2) && all(rownames(r2) == colnames(r2))) {
    diag(r2) <- diag(r)
  }
  r2
}

save_devices <- function(stem, w_mm, h_mm, draw) {
  wi <- w_mm / MM; hi <- h_mm / MM
  pdf(file.path(OUT, paste0(stem, ".pdf")), width = wi, height = hi,
      useDingbats = FALSE, family = "Helvetica")
  draw(); dev.off()
  png(file.path(OUT, paste0(stem, ".png")), width = wi, height = hi,
      units = "in", res = 300, family = "Helvetica", bg = "white")
  draw(); dev.off()
  svglite(file.path(OUT, paste0(stem, ".svg")), width = wi, height = hi,
          bg = "transparent", system_fonts = list(sans = "Helvetica"))
  draw(); dev.off()
  message("wrote ", stem, " (", w_mm, "×", h_mm, " mm)")
}

reorder_by_hclust <- function(r) {
  if (nrow(r) < 2 || ncol(r) < 2) return(r)
  rc <- r
  rc[!is.finite(rc)] <- 0
  # square matrices: same order rows/cols
  if (nrow(r) == ncol(r) && all(rownames(r) == colnames(r))) {
    o <- hclust(dist(rc), method = "ward.D2")$order
    return(r[o, o, drop = FALSE])
  }
  r
}

draw_circle_corr <- function(r, p, title, subtitle,
                             tl_cex = 0.42, cl_cex = 0.5,
                             do_hclust = FALSE) {
  # Align p to r
  p <- p[rownames(r), colnames(r), drop = FALSE]
  if (do_hclust) {
    r <- reorder_by_hclust(r)
    p <- p[rownames(r), colnames(r), drop = FALSE]
  }

  # Blank non-significant / untested; corrplot leaves NA cells empty
  r_sig <- mask_nonsig(r, p)

  op <- par(
    family = "Helvetica",
    mar = c(3.0, 2.0, 3.6, 2.4),
    oma = c(0.2, 0.2, 0.2, 0.6),
    cex.main = 0.55
  )
  on.exit(par(op), add = TRUE)

  if (!any(is.finite(r_sig))) {
    # tested pairs exist but none pass FDR, or matrix empty after mask
    plot.new()
    mtext(title, side = 3, line = -1.2, cex = 0.48, font = 1, adj = 0)
    mtext(subtitle, side = 3, line = -2.0, cex = 0.38, col = "grey25", adj = 0)
    mtext("No BH-FDR < 0.05 pairs in this stratum", side = 3, line = -3.2,
          cex = 0.42, col = "grey40", adj = 0)
    return(invisible(NULL))
  }

  corrplot(
    as.matrix(r_sig),
    method = "circle",
    type = "full",
    order = "original",
    col = COL,
    bg = "white",
    tl.col = "black",
    tl.cex = tl_cex,
    tl.srt = 55,
    cl.cex = cl_cex,
    cl.ratio = 0.16,
    cl.align.text = "l",
    cl.pos = "r",
    addgrid.col = NA,
    outline = TRUE,
    col.lim = c(-1, 1),
    is.corr = TRUE,
    na.label = " ",
    na.label.col = "white",
    diag = FALSE,
    mar = c(0, 0, 1.0, 0)
  )
  mtext(title, side = 3, line = 2.35, cex = 0.48, font = 1, adj = 0)
  mtext(subtitle, side = 3, line = 0.9, cex = 0.38, col = "grey25", adj = 0)
}

save_square_corr <- function(r_path, p_path, stem, w_mm, h_mm, title, subtitle,
                             cluster = TRUE, tl_cex = 0.42,
                             segment = NULL, annotate_support = TRUE) {
  r <- read_mat(r_path)
  p <- read_mat(p_path)
  if (!any(is.finite(r))) {
    message("skip ", stem, " (no tested pairs)")
    return(invisible(NULL))
  }
  # keep rows/cols with any tested (finite r) entry
  keep_r <- rowSums(is.finite(r)) > 0
  keep_c <- colSums(is.finite(r)) > 0
  # Prefer showing all rare-epi rows so blank rows communicate low support
  keep_r <- rep(TRUE, nrow(r))
  if (!any(keep_c)) keep_c <- rep(TRUE, ncol(r))
  r <- r[keep_r, keep_c, drop = FALSE]
  p <- p[keep_r, keep_c, drop = FALSE]

  if (cluster && nrow(r) >= 2 && any(is.finite(r))) {
    rc <- r; rc[!is.finite(rc)] <- 0
    ord_r <- hclust(dist(rc), method = "ward.D2")$order
    r <- r[ord_r, , drop = FALSE]; p <- p[ord_r, , drop = FALSE]
  }
  if (cluster && ncol(r) >= 2 && any(is.finite(r))) {
    rc <- r; rc[!is.finite(rc)] <- 0
    ord_c <- hclust(dist(t(rc)), method = "ward.D2")$order
    r <- r[, ord_c, drop = FALSE]; p <- p[, ord_c, drop = FALSE]
  }

  rn <- rownames(r); cn <- colnames(r)
  if (annotate_support) {
    rownames(r) <- label_with_support(rn, segment)
    colnames(r) <- label_with_support(cn, segment)
  } else {
    rownames(r) <- shorten(rn); colnames(r) <- shorten(cn)
  }
  rownames(p) <- rownames(r); colnames(p) <- colnames(r)

  save_devices(stem, w_mm, h_mm, function() {
    draw_circle_corr(r, p, title, subtitle, tl_cex = tl_cex, do_hclust = FALSE)
  })
}

save_overall_top <- function() {
  r <- read_mat(file.path(DATA, "corr_matrix_overall.csv"))
  p <- read_mat(file.path(DATA, "corr_pmatrix_overall.csv"))
  # mean |r| among finite tested pairs
  abs_r <- abs(r)
  abs_r[!is.finite(r) | !is.finite(p)] <- NA
  mean_abs <- rowMeans(abs_r, na.rm = TRUE)
  keep <- names(sort(mean_abs, decreasing = TRUE))
  keep <- keep[is.finite(mean_abs[keep])]
  keep <- keep[seq_len(min(40, length(keep)))]
  r <- r[keep, keep, drop = FALSE]
  p <- p[keep, keep, drop = FALSE]
  rownames(r) <- shorten(rownames(r)); colnames(r) <- shorten(colnames(r))
  rownames(p) <- shorten(rownames(p)); colnames(p) <- shorten(colnames(p))

  sub <- support_line("overall_epi_immune")
  save_devices("panel_a_corr_overall_top40", 180, 165, function() {
    draw_circle_corr(
      r, p,
      title = "a  Compositional correlations (joint CLR, Spearman)",
      subtitle = paste0(sub, "; top 40 types by mean |r|"),
      tl_cex = 0.4, do_hclust = TRUE
    )
  })
}

save_het_bars <- function() {
  d <- read_csv(file.path(DATA, "corr_heterogeneity_by_covariate.csv"),
                show_col_types = FALSE) %>%
    arrange(mean_fisher_z_var) %>%
    mutate(covariate = factor(covariate, levels = covariate))

  p <- ggplot(d, aes(mean_fisher_z_var, covariate)) +
    geom_col(fill = "#0072B2", colour = "black", width = 0.7, linewidth = 0.25) +
    geom_text(
      aes(label = sprintf("|Δr| = %.2f", mean_abs_delta_r)),
      hjust = -0.05, size = 1.9, family = "Helvetica"
    ) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.28)),
      name = expression(Mean ~ italic(z) * "-variance of epi" %*% "immune " * italic(r))
    ) +
    labs(
      title = "Where epi × immune correlations change",
      subtitle = "Support-filtered pairs only; study = technical ceiling",
      y = NULL
    ) +
    theme_classic(base_size = 6, base_family = "Helvetica") +
    theme(
      plot.title = element_text(size = 7, hjust = 0, face = "plain",
                               margin = margin(b = 1)),
      plot.subtitle = element_text(size = 5, colour = "grey30",
                                  margin = margin(b = 2)),
      axis.text = element_text(size = 6, colour = "black"),
      axis.title = element_text(size = 6),
      axis.line = element_line(linewidth = 0.25),
      plot.margin = margin(3, 8, 3, 3)
    )

  ggsave(file.path(OUT, "panel_het_covariate_rank.pdf"), p,
         width = 90 / MM, height = 55 / MM, device = cairo_pdf, bg = "transparent")
  ggsave(file.path(OUT, "panel_het_covariate_rank.png"), p,
         width = 90 / MM, height = 55 / MM, dpi = 300, bg = "white")
  ggsave(file.path(OUT, "panel_het_covariate_rank.svg"), p,
         width = 90 / MM, height = 55 / MM, device = svglite, bg = "transparent")
  message("wrote panel_het_covariate_rank")
}

save_top_pairs <- function() {
  if (!file.exists(file.path(DATA, "corr_pair_heterogeneity_summary.csv"))) return()
  sum <- read_csv(file.path(DATA, "corr_pair_heterogeneity_summary.csv"),
                  show_col_types = FALSE) %>%
    filter(covariate == "segment", n_levels >= 2, min_detect >= 10) %>%
    arrange(desc(abs_delta_r)) %>%
    head(15) %>%
    mutate(pair = paste(shorten(rare_epithelial), "x", shorten(immune)))

  if (nrow(sum) == 0) return()

  long <- read_csv(file.path(DATA, "corr_pair_heterogeneity_long.csv"),
                   show_col_types = FALSE) %>%
    filter(covariate == "segment", tested) %>%
    mutate(
      pair = paste(shorten(rare_epithelial), "x", shorten(immune)),
      level = factor(level, levels = c("ileum", "colon", "jejunum", "duodenum"))
    ) %>%
    filter(pair %in% sum$pair) %>%
    mutate(pair = factor(pair, levels = rev(sum$pair)))

  p <- ggplot(long, aes(level, pair, fill = spearman_r)) +
    geom_tile(colour = "black", linewidth = 0.25) +
    geom_text(
      aes(label = sprintf("%.2f\n(%d)", spearman_r, n_detect_epi)),
      size = 1.5, family = "Helvetica", lineheight = 0.85
    ) +
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, limits = c(-1, 1), name = "r"
    ) +
    labs(
      title = "Top segment-variable epi x immune pairs",
      subtitle = "Cell values: Spearman r (n samples detecting the epithelial type)",
      x = NULL, y = NULL
    ) +
    theme_classic(base_size = 6, base_family = "Helvetica") +
    theme(
      plot.title = element_text(size = 7, hjust = 0, face = "plain"),
      plot.subtitle = element_text(size = 5, colour = "grey30"),
      axis.text.x = element_text(angle = 35, hjust = 1, size = 6, colour = "black"),
      axis.text.y = element_text(size = 5.5, colour = "black"),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(2, "mm"),
      legend.text = element_text(size = 5),
      legend.title = element_text(size = 5),
      axis.line = element_line(linewidth = 0.25)
    )

  ggsave(file.path(OUT, "panel_top_pairs_by_segment.pdf"), p,
         width = 110 / MM, height = 85 / MM, device = cairo_pdf, bg = "transparent")
  ggsave(file.path(OUT, "panel_top_pairs_by_segment.png"), p,
         width = 110 / MM, height = 85 / MM, dpi = 300, bg = "white")
  ggsave(file.path(OUT, "panel_top_pairs_by_segment.svg"), p,
         width = 110 / MM, height = 85 / MM, device = svglite, bg = "transparent")
  message("wrote panel_top_pairs_by_segment")
}

save_prevalence_heatmap <- function() {
  sets <- read_csv(file.path(DATA, "celltype_sets.csv"), show_col_types = FALSE)
  rare <- sets$celltype[sets$set == "rare_epithelial"]
  prev <- read_csv(file.path(DATA, "celltype_prevalence_by_segment.csv"),
                   show_col_types = FALSE) %>%
    filter(celltype %in% rare) %>%
    mutate(
      segment = factor(segment, levels = c("duodenum", "jejunum", "ileum", "colon")),
      short_name = factor(short_name, levels = rev(unique(short_name)))
    )

  p <- ggplot(prev, aes(segment, short_name, fill = prevalence_ge1)) +
    geom_tile(colour = "black", linewidth = 0.25) +
    geom_text(
      aes(label = sprintf("%.0f%%\n%d/%d", 100 * prevalence_ge1,
                          n_detect_ge1, n_samples)),
      size = 1.45, family = "Helvetica", lineheight = 0.85
    ) +
    scale_fill_gradient(
      low = "#F7FBFF", high = "#08306B",
      labels = scales::percent_format(accuracy = 1),
      name = "Detect ≥1"
    ) +
    labs(
      title = "Rare epithelial detection support by segment",
      subtitle = "Percent (n with >=3 cells / n samples); pairs tested only if detect >= 20",
      x = NULL, y = NULL
    ) +
    theme_classic(base_size = 6, base_family = "Helvetica") +
    theme(
      plot.title = element_text(size = 7, hjust = 0, face = "plain"),
      plot.subtitle = element_text(size = 5, colour = "grey30"),
      axis.text = element_text(size = 6, colour = "black"),
      axis.text.x = element_text(angle = 35, hjust = 1),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(2, "mm"),
      axis.line = element_line(linewidth = 0.25)
    )

  ggsave(file.path(OUT, "panel_prevalence_rare_epi.pdf"), p,
         width = 90 / MM, height = 70 / MM, device = cairo_pdf, bg = "transparent")
  ggsave(file.path(OUT, "panel_prevalence_rare_epi.png"), p,
         width = 90 / MM, height = 70 / MM, dpi = 300, bg = "white")
  ggsave(file.path(OUT, "panel_prevalence_rare_epi.svg"), p,
         width = 90 / MM, height = 70 / MM, device = svglite, bg = "transparent")
  message("wrote panel_prevalence_rare_epi")
}

render_all <- function() {
  save_overall_top()

  save_square_corr(
    file.path(DATA, "corr_rect_epi_immune_overall_r.csv"),
    file.path(DATA, "corr_rect_epi_immune_overall_padj.csv"),
    "panel_b_epi_immune_overall", 180, 110,
    "b  Rare epithelial x immune co-abundance",
    support_line("overall_epi_immune"),
    tl_cex = 0.38, segment = NULL, annotate_support = TRUE
  )

  # Segment panels: powered only (n epi+immune >= 40)
  segs <- powered_segments()
  # remove stale underpowered panel outputs
  stale <- list.files(OUT, pattern = "^panel_c_epi_immune_(jejunum|duodenum)",
                      full.names = TRUE)
  if (length(stale)) unlink(stale)
  stale2 <- list.files(OUT, pattern = "^panel_c_epi_immune_by_segment_2x2",
                       full.names = TRUE)
  if (length(stale2)) unlink(stale2)

  for (seg in segs) {
    sub <- support_line("segment_epi_immune", seg)
    save_square_corr(
      file.path(DATA, paste0("corr_rect_epi_immune_segment_", seg, "_r.csv")),
      file.path(DATA, paste0("corr_rect_epi_immune_segment_", seg, "_padj.csv")),
      paste0("panel_c_epi_immune_", seg), 160, 100,
      paste0("Rare epithelial x immune - ", seg),
      sub,
      tl_cex = 0.36, segment = seg, annotate_support = TRUE
    )
  }

  # Side-by-side powered segments
  if (length(segs) >= 1) {
    draw_seg <- function() {
      nc <- length(segs)
      par(mfrow = c(1, nc), family = "Helvetica")
      for (seg in segs) {
        r <- read_mat(file.path(DATA, paste0("corr_rect_epi_immune_segment_", seg, "_r.csv")))
        p <- read_mat(file.path(DATA, paste0("corr_rect_epi_immune_segment_", seg, "_padj.csv")))
        keep_c <- colSums(is.finite(r)) > 0
        if (!any(keep_c)) keep_c <- rep(TRUE, ncol(r))
        r <- r[, keep_c, drop = FALSE]
        p <- p[, keep_c, drop = FALSE]
        if (nrow(r) >= 2 && any(is.finite(r))) {
          rc <- r; rc[!is.finite(rc)] <- 0
          o <- hclust(dist(rc), method = "ward.D2")$order
          r <- r[o, , drop = FALSE]; p <- p[o, , drop = FALSE]
        }
        if (ncol(r) >= 2 && any(is.finite(r))) {
          rc <- r; rc[!is.finite(rc)] <- 0
          o <- hclust(dist(t(rc)), method = "ward.D2")$order
          r <- r[, o, drop = FALSE]; p <- p[, o, drop = FALSE]
        }
        rn <- rownames(r); cn <- colnames(r)
        rownames(r) <- label_with_support(rn, seg)
        colnames(r) <- label_with_support(cn, seg)
        rownames(p) <- rownames(r); colnames(p) <- colnames(r)
        draw_circle_corr(
          r, p,
          title = seg,
          subtitle = support_line("segment_epi_immune", seg),
          tl_cex = 0.28
        )
      }
    }
    save_devices("panel_c_epi_immune_by_segment", 180, 105, draw_seg)
  }

  for (lin in c("epithelial", "lymphoid", "myeloid", "stroma")) {
    rp <- file.path(DATA, paste0("corr_matrix_lineage_", lin, ".csv"))
    pp <- file.path(DATA, paste0("corr_pmatrix_lineage_", lin, ".csv"))
    if (!file.exists(rp)) next
    r <- read_mat(rp); p <- read_mat(pp)
    rownames(r) <- shorten(rownames(r)); colnames(r) <- shorten(colnames(r))
    rownames(p) <- shorten(rownames(p)); colnames(p) <- shorten(colnames(p))
    save_devices(paste0("panel_lineage_", lin), 100, 100, function() {
      draw_circle_corr(
        r, p,
        title = paste0("Compositional correlations - ", lin),
        subtitle = support_line("overall_epi_immune"),
        tl_cex = 0.36, do_hclust = TRUE
      )
    })
  }

  # helper: draw a set of chemfrac/LP rect panels side by side
  draw_rect_set <- function(tags, prefix, title_fun, scope, seg_ann_fun = NULL) {
    function() {
      # drop tags with no tested pairs
      ok <- vapply(tags, function(tag) {
        r <- read_mat(file.path(DATA, paste0("corr_rect_epi_immune_", prefix, tag, "_r.csv")))
        any(is.finite(r))
      }, logical(1))
      tags <- tags[ok]
      if (length(tags) == 0) {
        plot.new(); title(main = "No tested pairs")
        return(invisible(NULL))
      }
      nc <- length(tags)
      par(mfrow = c(1, nc), family = "Helvetica")
      for (tag in tags) {
        r <- read_mat(file.path(DATA, paste0("corr_rect_epi_immune_", prefix, tag, "_r.csv")))
        p <- read_mat(file.path(DATA, paste0("corr_rect_epi_immune_", prefix, tag, "_padj.csv")))
        keep_c <- colSums(is.finite(r)) > 0
        if (!any(keep_c)) keep_c <- rep(TRUE, ncol(r))
        r <- r[, keep_c, drop = FALSE]
        p <- p[, keep_c, drop = FALSE]
        if (nrow(r) >= 2 && any(is.finite(r))) {
          rc <- r; rc[!is.finite(rc)] <- 0
          o <- hclust(dist(rc), method = "ward.D2")$order
          r <- r[o, , drop = FALSE]; p <- p[o, , drop = FALSE]
        }
        if (ncol(r) >= 2 && any(is.finite(r))) {
          rc <- r; rc[!is.finite(rc)] <- 0
          o <- hclust(dist(t(rc)), method = "ward.D2")$order
          r <- r[, o, drop = FALSE]; p <- p[, o, drop = FALSE]
        }
        seg_ann <- if (!is.null(seg_ann_fun)) seg_ann_fun(tag) else NULL
        if (!is.null(seg_ann)) {
          rn <- rownames(r); cn <- colnames(r)
          rownames(r) <- label_with_support(rn, seg_ann)
          colnames(r) <- label_with_support(cn, seg_ann)
        } else {
          rownames(r) <- shorten(rownames(r)); colnames(r) <- shorten(colnames(r))
        }
        rownames(p) <- rownames(r); colnames(p) <- colnames(r)
        draw_circle_corr(
          r, p,
          title = title_fun(tag),
          subtitle = support_line(scope, tag),
          tl_cex = 0.26
        )
      }
    }
  }

  # Chemical fractionation × collection (+ UF biopsy ileum/colon)
  tags <- shown_chemfrac()
  for (tag in tags) {
    lab <- gsub("_", " ", tag)
    seg_ann <- if (grepl("ileum$", tag)) "ileum" else if (grepl("colon$", tag)) "colon" else NULL
    save_square_corr(
      file.path(DATA, paste0("corr_rect_epi_immune_chemfrac_", tag, "_r.csv")),
      file.path(DATA, paste0("corr_rect_epi_immune_chemfrac_", tag, "_padj.csv")),
      paste0("panel_d_epi_immune_", tag), 160, 100,
      paste0("Rare epithelial x immune - ", lab),
      support_line("chemfrac_epi_immune", tag),
      tl_cex = 0.34, segment = seg_ann, annotate_support = TRUE
    )
  }
  tags_main <- shown_chemfrac("^(biopsy_unfractionated|biopsy_fractionated|resection_unfractionated)$")
  if (length(tags_main) >= 1) {
    h_mm <- if (length(tags_main) > 2) 190 else 105
    save_devices(
      "panel_d_epi_immune_chemfrac_by_collection", 180, h_mm,
      draw_rect_set(tags_main, "chemfrac_", function(t) gsub("_", " ", t),
                    "chemfrac_epi_immune")
    )
  }
  tags_uf_seg <- shown_chemfrac("^biopsy_unfractionated_(ileum|colon)$")
  if (length(tags_uf_seg) >= 1) {
    save_devices(
      "panel_d_epi_immune_biopsy_UF_by_segment", 180, 105,
      draw_rect_set(
        tags_uf_seg, "chemfrac_",
        function(t) gsub("biopsy_unfractionated_", "UF biopsy ", t),
        "chemfrac_epi_immune",
        seg_ann_fun = function(t) if (grepl("ileum", t)) "ileum" else "colon"
      )
    )
  }

  # LP-only libraries (ileum vs colon)
  lp_tags <- shown_lp()
  for (tag in lp_tags) {
    lab <- paste("LP", gsub("_", " ", tag))
    seg_ann <- if (grepl("ileum", tag)) "ileum" else if (grepl("colon", tag)) "colon" else NULL
    save_square_corr(
      file.path(DATA, paste0("corr_rect_epi_immune_lp_", tag, "_r.csv")),
      file.path(DATA, paste0("corr_rect_epi_immune_lp_", tag, "_padj.csv")),
      paste0("panel_e_epi_immune_lp_", tag), 160, 100,
      paste0("Rare epithelial x immune - ", lab),
      support_line("lp_epi_immune", tag),
      tl_cex = 0.34, segment = seg_ann, annotate_support = TRUE
    )
  }
  lp_seg <- shown_lp("^biopsy_(ileum|colon)$")
  if (length(lp_seg) >= 1) {
    save_devices(
      "panel_e_epi_immune_lp_biopsy_by_segment", 180, 105,
      draw_rect_set(
        lp_seg, "lp_",
        function(t) paste("LP biopsy", sub("biopsy_", "", t)),
        "lp_epi_immune",
        seg_ann_fun = function(t) if (grepl("ileum", t)) "ileum" else "colon"
      )
    )
  }
  if (length(lp_tags) >= 1) {
    save_devices(
      "panel_e_epi_immune_lp_by_stratum", 180, 105,
      draw_rect_set(lp_tags, "lp_", function(t) paste("LP", gsub("_", " ", t)),
                    "lp_epi_immune")
    )
  }

  save_het_bars()
  save_top_pairs()
  save_prevalence_heatmap()
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) render_all()
