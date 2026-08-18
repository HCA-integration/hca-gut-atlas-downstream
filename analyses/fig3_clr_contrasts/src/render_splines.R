#!/usr/bin/env Rscript
# Splines of CLR composition split by sample collection method (biopsy vs
# surgical resection), for EVERY cell type that passes FDR < 0.05 in the
# biopsy-vs-resection CLR Wilcoxon (clr_wilcoxon_collection.csv). One faceted
# figure per lineage so panels stay legible at Nature size.
#
# Two x-axes, both requested:
#   AGE   (panels j-m): x = donor age (mid-point of age_range)
#   TISSUE(panels n-q): x = gut segment proximal -> distal
#                       (duodenum, jejunum, ileum, colon only)
#
# Each facet: raw per-sample points + one natural-spline curve per collection
# method with 95% band. Curve separation = a sampling-method effect on that
# cell type not explained by the x-axis (age, or gut segment).
#
# Curves are fit manually per (cell type x method) with a tryCatch fallback to
# a linear fit, so sparse groups (e.g. only 4 gut segments) never break a panel.
#
# plot_specs.md: Helvetica 5-7 pt, no gridlines, open axes, Wong palette,
# vector cairo PDF + SVG + 300 dpi PNG at exact final size.

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(svglite); library(ragg); library(splines)
})

HERE <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 error = function(e) ".")
if (length(HERE) == 0 || HERE == "") HERE <- "."
DATA <- file.path(HERE, "..", "data")
OUT  <- file.path(HERE, "..", "out")
MM <- 25.4

METHOD_COL <- c("biopsy" = "#0072B2", "surgical resection" = "#D55E00")

# proximal -> distal gut segments only
TISSUE_ORDER <- c("duodenum" = 1, "jejunum" = 2, "ileum" = 3, "colon" = 4)
TISSUE_ABBR  <- c("1" = "Duo", "2" = "Jej", "3" = "Ile", "4" = "Col")

coll <- suppressMessages(read_csv(file.path(DATA, "clr_wilcoxon_collection.csv"),
                                  show_col_types = FALSE))
sig <- coll %>% filter(p_adj < 0.05)

long <- suppressMessages(read_csv(file.path(DATA, "clr_long.csv"),
                                  show_col_types = FALSE)) %>%
  filter(sample_collection_method %in% names(METHOD_COL)) %>%
  mutate(
    sample_collection_method = factor(sample_collection_method, levels = names(METHOD_COL)),
    tissue_order = unname(TISSUE_ORDER[tissue_level_1])
  )

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text            = element_text(colour = "black", family = "Helvetica"),
      plot.title      = element_text(size = 7, hjust = 0, colour = "black"),
      axis.line       = element_line(colour = "black", linewidth = 0.25),
      axis.ticks      = element_line(colour = "black", linewidth = 0.25),
      axis.text       = element_text(colour = "black", size = 5),
      axis.title      = element_text(colour = "black", size = base),
      panel.grid      = element_blank(),
      legend.position = "bottom",
      legend.title    = element_blank(),
      legend.text     = element_text(size = base, colour = "black"),
      legend.key.size = unit(3, "mm"),
      strip.background = element_blank(),
      strip.text      = element_text(size = 5, colour = "black"),
      panel.spacing   = unit(1.5, "mm"),
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

# per-group spline fit with graceful fallback -> tidy prediction frame
fit_curve <- function(d, xk, df_spline = 3, n_grid = 80) {
  d <- d[is.finite(d[[xk]]) & is.finite(d$clr), c(xk, "clr")]
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
  data.frame(x = grid$x, fit = pr$fit,
             lo = pr$fit - 1.96 * pr$se.fit, hi = pr$fit + 1.96 * pr$se.fit)
}

build_curves <- function(d, xk) {
  out <- list()
  for (ct in unique(d$celltype)) {
    for (m in levels(d$sample_collection_method)) {
      sub <- d[d$celltype == ct & d$sample_collection_method == m, ]
      f <- fit_curve(sub, xk)
      if (!is.null(f)) {
        f$celltype <- ct; f$sample_collection_method <- m
        out[[paste(ct, m)]] <- f
      }
    }
  }
  if (!length(out)) return(NULL)
  bind_rows(out)
}

LIN_ORDER <- c("epithelial", "lymphoid", "myeloid", "stroma")

render_axis <- function(xk, x_title, letters_by_lin, stem_axis, x_breaks = NULL,
                        x_labels = NULL) {
  for (ln in LIN_ORDER) {
    cts <- sig %>% filter(lineage == ln) %>% arrange(p_value) %>% pull(celltype)
    if (!length(cts)) next
    d <- long %>%
      filter(celltype %in% cts, is.finite(.data[[xk]])) %>%
      mutate(celltype = factor(celltype, levels = cts))
    if (!nrow(d)) next
    curves <- build_curves(d, xk)
    if (is.null(curves)) next
    curves$celltype <- factor(curves$celltype, levels = cts)

    ncol <- 5
    nrow <- ceiling(length(cts) / ncol)
    height_mm <- min(170, 20 + nrow * 27)

    p <- ggplot() +
      geom_point(data = d, aes(.data[[xk]], clr, colour = sample_collection_method),
                 size = 0.35, alpha = 0.35, stroke = 0,
                 position = if (is.null(x_breaks)) "identity"
                            else position_jitter(width = 0.08, height = 0)) +
      geom_ribbon(data = curves,
                  aes(x = x, ymin = lo, ymax = hi, fill = sample_collection_method),
                  alpha = 0.12) +
      geom_line(data = curves,
                aes(x = x, y = fit, colour = sample_collection_method),
                linewidth = 0.5) +
      facet_wrap(~ celltype, ncol = ncol, scales = "free_y",
                 labeller = label_wrap_gen(width = 22)) +
      scale_colour_manual(values = METHOD_COL, drop = FALSE) +
      scale_fill_manual(values = METHOD_COL, drop = FALSE, guide = "none") +
      labs(x = x_title, y = "CLR composition",
           title = paste0(letters_by_lin[[ln]], "  ", str_to_title(ln),
                          " cell types affected by sampling method (FDR < 0.05): ",
                          "CLR by ", stem_axis, ", split by collection method")) +
      theme_gca()

    if (!is.null(x_breaks)) {
      p <- p + scale_x_continuous(breaks = x_breaks, labels = x_labels)
    }
    save_panel(p, paste0("panel_", letters_by_lin[[ln]], "_splines_by_collection_",
                         stem_axis, "_", ln), 180, height_mm)
  }
}

# ---- AGE axis (panels j-m) ----
render_axis(
  xk = "age_order", x_title = "Age (years)",
  letters_by_lin = c(epithelial = "j", lymphoid = "k", myeloid = "l", stroma = "m"),
  stem_axis = "age"
)

# ---- TISSUE contrast (panels n-q): ileum vs colon box plots ----
# Only ileum and colon are well powered for both collection methods; the other
# segments (duodenum, jejunum) lack samples. Per FDR-significant cell type we
# show within-lineage % as a Nature-style grouped box plot, ileum vs colon,
# split biopsy (blue) vs surgical resection (orange), with Wilcoxon brackets.
TISSUE_KEEP <- c("ileum" = "Ileum", "colon" = "Colon")
MIN_PER_BOX <- 3
DODGE_W <- 0.72

p_to_stars <- function(p) {
  ifelse(is.na(p), "n.s.",
    ifelse(p < 1e-4, "****",
    ifelse(p < 1e-3, "***",
    ifelse(p < 1e-2, "**",
    ifelse(p < 0.05, "*", "n.s.")))))
}

wilcox_brackets <- function(d) {
  # Within each celltype × tissue: Wilcoxon biopsy vs resection on %
  rows <- list()
  for (ct in levels(d$celltype)) {
    for (ti in levels(d$tissue)) {
      sub <- d[d$celltype == ct & d$tissue == ti, ]
      a <- sub$within_lineage_percentage[sub$sample_collection_method == "biopsy"]
      b <- sub$within_lineage_percentage[sub$sample_collection_method == "surgical resection"]
      a <- a[is.finite(a)]; b <- b[is.finite(b)]
      if (length(a) < MIN_PER_BOX || length(b) < MIN_PER_BOX) next
      p <- tryCatch(
        stats::wilcox.test(a, b, exact = FALSE)$p.value,
        error = function(e) NA_real_
      )
      ymax <- max(c(a, b), na.rm = TRUE)
      x_mid <- as.numeric(factor(ti, levels = levels(d$tissue)))
      rows[[length(rows) + 1L]] <- data.frame(
        celltype = ct,
        tissue = ti,
        x1 = x_mid - DODGE_W / 4,
        x2 = x_mid + DODGE_W / 4,
        y = ymax,
        p_value = p,
        stringsAsFactors = FALSE
      )
    }
  }
  if (!length(rows)) return(NULL)
  out <- bind_rows(rows) %>%
    mutate(
      p_adj = p.adjust(p_value, method = "BH"),
      label = p_to_stars(p_adj),
      celltype = factor(celltype, levels = levels(d$celltype)),
      tissue = factor(tissue, levels = levels(d$tissue)),
      # room above the highest point in that facet for the bracket
      y_bar = y + pmax(0.04 * abs(y), 0.8),
      y_lab = y + pmax(0.09 * abs(y), 1.8)
    )
  out
}

render_tissue_boxplots <- function(letters_by_lin) {
  base <- long %>%
    filter(tissue_level_1 %in% names(TISSUE_KEEP)) %>%
    mutate(tissue = factor(unname(TISSUE_KEEP[tissue_level_1]),
                           levels = unname(TISSUE_KEEP)))
  for (ln in LIN_ORDER) {
    cts <- sig %>% filter(lineage == ln) %>% arrange(p_value) %>% pull(celltype)
    if (!length(cts)) next
    d <- base %>% filter(celltype %in% cts, is.finite(within_lineage_percentage))
    # keep only cell types with >=MIN_PER_BOX samples in each tissue x method box
    grp_n <- d %>% count(celltype, tissue, sample_collection_method)
    ok_ct <- grp_n %>% group_by(celltype) %>%
      summarise(ok = sum(n >= MIN_PER_BOX) >= 4, .groups = "drop") %>%
      filter(ok) %>% pull(celltype)
    cts <- intersect(cts, ok_ct)
    if (!length(cts)) next
    d <- d %>% filter(celltype %in% cts) %>%
      mutate(celltype = factor(celltype, levels = cts))

    br <- wilcox_brackets(d)

    ncol <- 5
    nrow <- ceiling(length(cts) / ncol)
    height_mm <- min(210, 26 + nrow * 34)

    p <- ggplot(d, aes(tissue, within_lineage_percentage,
                       fill = sample_collection_method)) +
      geom_boxplot(outlier.shape = NA, linewidth = 0.25, width = 0.62,
                   position = position_dodge(width = DODGE_W),
                   colour = "grey25", alpha = 0.9) +
      geom_point(aes(colour = sample_collection_method),
                 position = position_jitterdodge(jitter.width = 0.12,
                                                 dodge.width = DODGE_W),
                 size = 0.28, alpha = 0.35, stroke = 0, show.legend = FALSE) +
      facet_wrap(~ celltype, ncol = ncol, scales = "free_y",
                 labeller = label_wrap_gen(width = 22)) +
      scale_fill_manual(
        values = METHOD_COL, drop = FALSE,
        labels = c("biopsy" = "Biopsy", "surgical resection" = "Resection")
      ) +
      scale_colour_manual(values = METHOD_COL, drop = FALSE) +
      scale_y_continuous(expand = expansion(mult = c(0.02, 0.18))) +
      labs(x = NULL, y = "Within lineage (%)",
           title = paste0(letters_by_lin[[ln]], "  ", str_to_title(ln),
                          " cell types affected by sampling method (FDR<0.05): ",
                          "ileum vs colon, biopsy vs resection (Wilcoxon, BH within panel)")) +
      theme_gca() +
      theme(axis.text.x = element_text(size = 5.5))

    if (!is.null(br)) {
      p <- p +
        geom_segment(
          data = br,
          aes(x = x1, xend = x2, y = y_bar, yend = y_bar),
          inherit.aes = FALSE, linewidth = 0.25, colour = "black"
        ) +
        geom_segment(
          data = br,
          aes(x = x1, xend = x1, y = y_bar - 0.35 * (y_lab - y_bar), yend = y_bar),
          inherit.aes = FALSE, linewidth = 0.25, colour = "black"
        ) +
        geom_segment(
          data = br,
          aes(x = x2, xend = x2, y = y_bar - 0.35 * (y_lab - y_bar), yend = y_bar),
          inherit.aes = FALSE, linewidth = 0.25, colour = "black"
        ) +
        geom_text(
          data = br,
          aes(x = (x1 + x2) / 2, y = y_lab, label = label),
          inherit.aes = FALSE, size = 1.7, family = "Helvetica",
          colour = "black", vjust = 0
        )
    }

    save_panel(p, paste0("panel_", letters_by_lin[[ln]],
                         "_tissue_boxplot_ileum_colon_", ln), 180, height_mm)
  }
}

render_tissue_boxplots(
  letters_by_lin = c(epithelial = "n", lymphoid = "o", myeloid = "p", stroma = "q"))

message("done splines + tissue boxplots.")
