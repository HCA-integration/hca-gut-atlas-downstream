#!/usr/bin/env Rscript
# Supplemental figure: follicle (GC B) capture diagnostics.
#
# a  Threshold scan — capture rate vs GC B cell-count cutoff (+ best-k marker)
# b  GSVA violins by follicle+/− at best cutoff (follicle-associated programs)
# c  Variance explained by covariates + pairwise confounding (Cramér's V)
# d  Logistic mixed model (estimable terms)
#
# Inputs written by compute_follicle_supplement.py

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(tidyr); library(patchwork); library(svglite); library(scales)
  library(forcats); library(lme4)
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
PT <- 2.845

COL_SEG <- c(ileum = "#0072B2", colon = "#E69F00", pooled = "#000000")
COL_DET <- c("Follicle−" = "#D0D0D0", "Follicle+" = "#0072B2")
COL_R2 <- c(
  "Univariate" = "#0072B2",
  "Unique after study" = "#D55E00"
)
PROG_ORDER <- c("GC_module", "GC_DZ", "GC_LZ", "Tfh", "Tfr", "FARM", "fDC")
PROG_LAB <- c(
  GC_module = "GC module", GC_DZ = "GC B DZ", GC_LZ = "GC B LZ",
  Tfh = "Tfh", Tfr = "Tfr", FARM = "FARM", fDC = "fDC"
)

theme_gca <- function(base = 7) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 7.5, hjust = 0, face = "plain",
                               margin = margin(b = 2, t = 0)),
      plot.subtitle = element_text(size = 5.5, hjust = 0, colour = "grey25",
                                  margin = margin(b = 2)),
      axis.line = element_line(colour = "black", linewidth = 0.3),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length = unit(0.8, "mm"),
      axis.text = element_text(colour = "black", size = 7),
      axis.title = element_text(colour = "black", size = 7.5),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 6.5),
      legend.key.size = unit(3.0, "mm"),
      legend.margin = margin(0, 0, 0, 0),
      strip.background = element_blank(),
      strip.text = element_text(size = 7, colour = "black",
                               margin = margin(b = 1)),
      panel.spacing = unit(1.5, "mm"),
      plot.margin = margin(1.5, 2, 1, 1.5, "pt")
    )
}

save_panel <- function(p, stem, w_mm, h_mm) {
  wi <- w_mm / MM; hi <- h_mm / MM
  ggsave(file.path(OUT, paste0(stem, ".pdf")), p, width = wi, height = hi,
         device = cairo_pdf, bg = "transparent")
  ggsave(file.path(OUT, paste0(stem, ".svg")), p, width = wi, height = hi,
         device = svglite, bg = "transparent")
  ggsave(file.path(OUT, paste0(stem, ".png")), p, width = wi, height = hi,
         dpi = 300, bg = "white")
  message("wrote ", stem, " (", w_mm, "×", h_mm, " mm)")
}

best_meta <- read_csv(file.path(DATA, "follicle_threshold_best.csv"),
                      show_col_types = FALSE)
BEST_K <- best_meta$best_cutoff[1]
PRIMARY_K <- best_meta$primary_cutoff[1]

# ── a: threshold scan ──────────────────────────────────────────────────────
make_a_threshold_scan <- function() {
  rates <- read_csv(file.path(DATA, "follicle_threshold_scan_rates.csv"),
                    show_col_types = FALSE) %>%
    filter(strata %in% c("ileum", "colon", "pooled")) %>%
    mutate(strata = factor(strata, levels = c("ileum", "colon", "pooled")))
  sep <- read_csv(file.path(DATA, "follicle_threshold_scan_separation.csv"),
                  show_col_types = FALSE)

  p_rate <- ggplot(rates, aes(cutoff, rate, colour = strata, fill = strata)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 0.5) +
    geom_vline(xintercept = BEST_K, linetype = "dashed",
               colour = "grey30", linewidth = 0.35) +
    geom_vline(xintercept = PRIMARY_K, linetype = "dotted",
               colour = "grey50", linewidth = 0.35) +
    annotate("text", x = BEST_K, y = max(rates$ci_hi, na.rm = TRUE) * 0.98,
             label = sprintf("best k=%d", BEST_K), hjust = -0.05,
             size = 6 / PT, family = "Helvetica") +
    scale_colour_manual(values = COL_SEG) +
    scale_fill_manual(values = COL_SEG) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0.02, 0.08))) +
    scale_x_continuous(breaks = c(1, 5, 10, 20, 30, 40)) +
    labs(
      title = "a",
      x = "Minimum GC B cells (LZ or DZ)", y = "Detection rate"
    ) +
    theme_gca() +
    theme(legend.position = c(0.82, 0.82), legend.background = element_blank())

  # Plain-English separation metrics (no Cohen's d / Youden labels)
  sep_long <- sep %>%
    select(cutoff, cohens_d, youden_vs_gsva_pos) %>%
    pivot_longer(-cutoff, names_to = "metric", values_to = "value") %>%
    mutate(
      metric = recode(
        metric,
        cohens_d = "GSVA score gap (follicle+ vs −)",
        youden_vs_gsva_pos = "Agreement with high GSVA"
      )
    )
  p_sep <- ggplot(sep_long, aes(cutoff, value, colour = metric)) +
    geom_line(linewidth = 0.5) +
    geom_vline(xintercept = BEST_K, linetype = "dashed",
               colour = "grey30", linewidth = 0.35) +
    scale_colour_manual(values = c(
      "GSVA score gap (follicle+ vs −)" = "#D55E00",
      "Agreement with high GSVA" = "#009E73"
    )) +
    scale_x_continuous(breaks = c(1, 5, 10, 20, 30, 40)) +
    labs(title = NULL, x = "Minimum GC B cells", y = NULL) +
    theme_gca() +
    theme(
      legend.position = c(0.55, 0.22),
      legend.background = element_blank(),
      legend.text = element_text(size = 6)
    )

  p_rate | p_sep
}

# ── b: GSVA violins at best k ──────────────────────────────────────────────
make_b_gsva_violins <- function() {
  d <- read_csv(file.path(DATA, "follicle_gsva_by_call_bestk.csv"),
                show_col_types = FALSE) %>%
    filter(program %in% PROG_ORDER) %>%
    mutate(
      program = factor(program, levels = PROG_ORDER, labels = PROG_LAB[PROG_ORDER]),
      follicle_label = factor(follicle_label, levels = c("Follicle−", "Follicle+"))
    )

  ggplot(d, aes(follicle_label, gsva, fill = follicle_label)) +
    geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.25) +
    geom_violin(scale = "width", colour = "black", linewidth = 0.15, alpha = 0.9) +
    geom_boxplot(width = 0.18, outlier.shape = NA, fill = "white", linewidth = 0.2) +
    facet_wrap(~ program, nrow = 1) +
    scale_fill_manual(values = COL_DET, guide = "none") +
    labs(title = "b", x = NULL, y = "GSVA score") +
    theme_gca() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5))
}

# ── c: variance explained + covariate confounding heatmap ─────────────────
make_c_confounds <- function() {
  ve <- read_csv(file.path(DATA, "follicle_var_explained.csv"),
                 show_col_types = FALSE) %>%
    filter(is.finite(r2_univariate)) %>%
    mutate(label = factor(label, levels = rev(label[order(r2_univariate)])))

  ve_long <- ve %>%
    select(label, `Univariate` = r2_univariate,
           `Unique after study` = r2_unique_after_study) %>%
    pivot_longer(-label, names_to = "metric", values_to = "r2") %>%
    filter(is.finite(r2)) %>%
    mutate(metric = factor(metric, levels = c("Univariate", "Unique after study")))

  p_r2 <- ggplot(ve_long, aes(r2, label, fill = metric)) +
    geom_col(
      position = position_dodge(width = 0.72), width = 0.68,
      colour = "black", linewidth = 0.15
    ) +
    scale_fill_manual(values = COL_R2) +
    scale_x_continuous(
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.08))
    ) +
    labs(
      title = "c",
      x = "Share of follicle-call variation explained",
      y = NULL
    ) +
    theme_gca() +
    theme(
      legend.position = "top",
      legend.justification = "left",
      legend.margin = margin(0, 0, -2, 0),
      axis.text.y = element_text(size = 6.5)
    )

  # Focus heatmap on the design factors that matter for confounding
  keep_labs <- c(
    "Study", "Fractionation", "Collection", "Radial layer",
    "Site condition", "Gut segment", "Age", "Assay"
  )
  cv <- read_csv(file.path(DATA, "follicle_cov_cramers_v.csv"),
                 show_col_types = FALSE) %>%
    filter(label_a %in% keep_labs, label_b %in% keep_labs) %>%
    mutate(
      label_a = factor(label_a, levels = keep_labs),
      label_b = factor(label_b, levels = keep_labs)
    )

  p_hm <- ggplot(cv, aes(label_a, label_b, fill = cramers_v)) +
    geom_tile(colour = "white", linewidth = 0.35) +
    geom_text(
      aes(label = ifelse(is.na(cramers_v), "", sprintf("%.2f", cramers_v))),
      size = 5 / PT, family = "Helvetica", colour = "grey10"
    ) +
    scale_fill_gradient(
      low = "#F7F7F7", high = "#D55E00",
      limits = c(0, 1), na.value = "grey90",
      name = "How linked\n(0–1)"
    ) +
    coord_fixed() +
    labs(title = NULL, x = NULL, y = NULL) +
    theme_gca() +
    theme(
      axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1, size = 6),
      axis.text.y = element_text(size = 6),
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "right",
      legend.key.height = unit(4, "mm"),
      legend.key.width = unit(2.2, "mm"),
      legend.title = element_text(size = 5.5),
      legend.text = element_text(size = 5.5)
    )

  (p_r2 | p_hm) + plot_layout(widths = c(1.15, 1))
}

# ── d: logistic mixed model ────────────────────────────────────────────────
fit_and_plot_model <- function() {
  d <- read_csv(file.path(DATA, "follicle_mixed_model_input.csv"),
                show_col_types = FALSE) %>%
    mutate(
      site = factor(site, levels = c("Healthy", "Disease-adjacent")),
      collection = factor(collection, levels = c("Biopsy", "Resection")),
      segment = factor(segment, levels = c("ileum", "colon")),
      frac = factor(frac, levels = c("unfractionated", "fractionated")),
      dataset_id = factor(dataset_id),
      donor_id = factor(donor_id)
    )

  form <- gc ~ site + collection + segment + scale(log_total_cells) +
    (1 | dataset_id) + (1 | donor_id)

  fit <- tryCatch(
    glmer(
      form, data = d, family = binomial,
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    ),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    fit <- glmer(
      gc ~ site + collection + segment + scale(log_total_cells) + (1 | dataset_id),
      data = d, family = binomial,
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
  }

  sm <- summary(fit)$coefficients
  ct <- data.frame(term = rownames(sm), sm, check.names = FALSE) %>%
    filter(term != "(Intercept)") %>%
    mutate(
      OR = exp(Estimate),
      OR_lo = exp(Estimate - 1.96 * `Std. Error`),
      OR_hi = exp(Estimate + 1.96 * `Std. Error`),
      p = `Pr(>|z|)`,
      label = dplyr::recode(
        term,
        "siteDisease-adjacent" = "Disease-adjacent vs healthy",
        "collectionResection" = "Resection vs biopsy",
        "segmentcolon" = "Colon vs ileum",
        "scale(log_total_cells)" = "log(n_cells/sample), per SD",
        .default = term
      )
    )
  write_csv(ct, file.path(DATA, "follicle_mixed_model_coefs.csv"))
  sink(file.path(DATA, "follicle_mixed_model_summary.txt"))
  cat("Formula:", deparse(formula(fit)), "\n")
  cat("n =", nobs(fit), "\n\n")
  print(summary(fit))
  cat("\nNote: chemical_fractionation omitted as fixed effect — fully nested\n")
  cat("within dataset_id (0 studies span both UF and F).\n")
  sink()

  p_forest <- ggplot(ct, aes(OR, fct_rev(label))) +
    geom_vline(xintercept = 1, linetype = "dashed",
               colour = "grey50", linewidth = 0.3) +
    geom_errorbar(aes(xmin = OR_lo, xmax = OR_hi),
                  orientation = "y", width = 0.18, linewidth = 0.35) +
    geom_point(shape = 21, fill = "#0072B2", colour = "black",
               size = 2.0, stroke = 0.3) +
    geom_text(
      aes(label = sprintf("OR=%.2f  P=%.3g", OR, p)),
      hjust = -0.08, size = 5.5 / PT, family = "Helvetica"
    ) +
    scale_x_log10(breaks = c(0.1, 0.25, 0.5, 1, 2)) +
    coord_cartesian(xlim = c(min(ct$OR_lo, na.rm = TRUE) * 0.7,
                             max(ct$OR_hi, na.rm = TRUE) * 2.8)) +
    labs(
      title = "d",
      subtitle = "glmer: gc ~ site + collection + segment + log(n_cells/sample) + (1|dataset) + (1|donor)",
      x = "Odds ratio (log scale)", y = NULL
    ) +
    theme_gca() +
    theme(
      legend.position = "none",
      axis.text.y = element_text(size = 7),
      plot.subtitle = element_text(size = 5.5, family = "Helvetica")
    )

  list(plot = p_forest, fit = fit, coefs = ct)
}

render <- function() {
  p_a <- make_a_threshold_scan()
  p_b <- make_b_gsva_violins()
  p_c <- make_c_confounds()
  mod <- fit_and_plot_model()
  p_d <- mod$plot

  save_panel(p_a, "sfig_follicle_a_threshold_scan", 180, 50)
  save_panel(p_b, "sfig_follicle_b_gsva_violins", 180, 46)
  save_panel(p_c, "sfig_follicle_c_varpart", 180, 70)
  save_panel(p_d, "sfig_follicle_d_mixed_model", 120, 48)

  fig <- p_a / p_b / p_c / p_d +
    plot_layout(heights = c(1.0, 0.9, 1.4, 0.85)) +
    plot_annotation(
      title = "Supplemental: Follicle (GC B) capture threshold, GSVA concordance, and confounds",
      theme = theme(
        plot.title = element_text(
          family = "Helvetica", size = 7.5, hjust = 0, face = "plain",
          margin = margin(b = 3)
        )
      )
    )
  save_panel(fig, "sfig_follicle_capture_supplement", 180, 180)
  save_panel(fig, "sfig_follicle_capture_supplement_tall", 180, 220)
}

.is_main <- length(grep("^--file=", commandArgs(FALSE))) > 0
if (.is_main && !interactive()) render()
