#!/usr/bin/env Rscript
# Gut-axis CLR splines (duodenum → jejunum → ileum → colon) for the top
# non-absorptive segment-ω² cell types, stratified by full thickness vs not.
# Styled like panel_g (pDC / Th17 age splines): thick smooth curves, jittered
# points, dashed zero line, Wong palette, sentence-case labels.
#
# Inputs:  ../data/clr_long.csv
# Outputs: ../out/panel_gut_axis_depth_splines_top16.*
#          ../data/gut_axis_depth_splines_top16_celltypes.csv

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(stringr)
  library(tidyr); library(svglite); library(splines)
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

SEGS <- c("duodenum", "jejunum", "ileum", "colon")
SEG_LAB <- c(duodenum = "Duo", jejunum = "Jej", ileum = "Ile", colon = "Col")
SEG_NUM <- c(duodenum = 1, jejunum = 2, ileum = 3, colon = 4)

# Match panel_g collection colours: mucosal/biopsy-like = orange,
# deep/resection-like = blue
DEPTH_COL <- c(
  "Not full thickness" = "#E69F00",
  "Full thickness" = "#0072B2"
)

ABSORPTIVE_RE <- paste(
  "Enterocyte|Colonocyte|BEST4 Enterocyte|BEST4 Colonocyte|",
  "Villus Tip|Villus tip|Villus-tip",
  sep = ""
)

SHORT_LAB <- c(
  "CD8 TRM" = "CD8 TRM",
  "Transiently Amplifying Cells (TA)" = "TA",
  "Medullary Sinus Endothelial" = "Med. sinus endo.",
  "Mast Cells" = "Mast",
  "NK Cells" = "NK",
  "CD8 IEL" = "CD8 IEL",
  "CD8 Effector Memory" = "CD8 TEM",
  "Mature Goblet Cells" = "Mature goblet",
  "CD4 Memory" = "CD4 memory",
  "NKT Cells" = "NKT",
  "CD4 Th17" = "CD4 Th17",
  "CD4 Tfr" = "CD4 Tfr",
  "Tuft Progenitors" = "Tuft prog.",
  "CD4 Tfh" = "CD4 Tfh",
  "cDC1" = "cDC1",
  "GC B Light Zone (GC B LZ)" = "GC B LZ"
)

theme_gca <- function(base = 6) {
  theme_classic(base_size = base, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "black", family = "Helvetica"),
      plot.title = element_text(size = 7, hjust = 0, colour = "black",
                               margin = margin(b = 1)),
      plot.subtitle = element_text(size = 5.5, hjust = 0, colour = "grey25",
                                  margin = margin(b = 3)),
      axis.line = element_line(colour = "black", linewidth = 0.3),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      axis.text = element_text(colour = "black", size = 5.5),
      axis.title = element_text(colour = "black", size = 6.5),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 6, colour = "black"),
      legend.key.size = unit(3.2, "mm"),
      legend.margin = margin(t = 1),
      strip.background = element_blank(),
      strip.text = element_text(size = 6, colour = "black", hjust = 0,
                               margin = margin(b = 2, t = 1)),
      panel.spacing = unit(2.2, "mm"),
      plot.margin = margin(2, 4, 2, 2, "pt")
    )
}

omega_sq <- function(y, g) {
  ok <- is.finite(y) & !is.na(g)
  y <- y[ok]; g <- as.character(g[ok])
  if (length(y) < 10) return(NA_real_)
  lev <- unique(g)
  if (length(lev) < 2) return(NA_real_)
  n <- length(y)
  grand <- mean(y)
  ss_tot <- sum((y - grand)^2)
  if (ss_tot <= 0) return(0)
  ss_b <- 0
  for (lv in lev) {
    m <- g == lv
    ss_b <- ss_b + sum(m) * (mean(y[m]) - grand)^2
  }
  k <- length(lev)
  ms_w <- (ss_tot - ss_b) / (n - k)
  as.numeric(max(0, min(1, (ss_b - (k - 1) * ms_w) / (ss_tot + ms_w))))
}

long0 <- read_csv(file.path(DATA, "clr_long.csv"), show_col_types = FALSE) %>%
  filter(tolower(tissue_level_1) %in% SEGS) %>%
  mutate(
    tissue = factor(tolower(tissue_level_1), levels = SEGS),
    x = as.numeric(SEG_NUM[as.character(tissue)]),
    depth = ifelse(
      tolower(as.character(radial_tissue_term)) == "epi_lp_musc",
      "Full thickness", "Not full thickness"
    ),
    depth = factor(depth, levels = c("Not full thickness", "Full thickness"))
  )

rank_df <- long0 %>%
  group_by(celltype, lineage) %>%
  summarise(omega2 = omega_sq(clr, tissue), n = n(), .groups = "drop") %>%
  filter(
    is.finite(omega2),
    !str_detect(celltype, regex(ABSORPTIVE_RE, ignore_case = TRUE))
  ) %>%
  arrange(desc(omega2)) %>%
  slice_head(n = 16) %>%
  mutate(
    rank = row_number(),
    short = ifelse(
      celltype %in% names(SHORT_LAB),
      unname(SHORT_LAB[celltype]),
      celltype
    ),
    # panel_g style facet titles: "Type · by sampling depth"
    facet_lab = paste0(short, " · by sampling depth")
  )

write_csv(rank_df, file.path(DATA, "gut_axis_depth_splines_top16_celltypes.csv"))
message("Top 16:")
print(rank_df %>% select(rank, lineage, celltype, omega2), n = 16)

d <- long0 %>%
  filter(celltype %in% rank_df$celltype) %>%
  left_join(rank_df %>% select(celltype, facet_lab, rank), by = "celltype") %>%
  mutate(facet_lab = factor(facet_lab, levels = rank_df$facet_lab))

# Keep arms with enough points for a smooth; drop ultra-sparse depth×type
arm_n <- d %>% count(facet_lab, depth)
keep_arms <- arm_n %>% filter(n >= 6)
d <- d %>% semi_join(keep_arms, by = c("facet_lab", "depth"))

# panel_g-like thick curves: natural spline / linear fallback per arm
# (4 ordered segments; GAM overfits / fails when an arm skips a segment)
fit_curve <- function(sub) {
  sub <- sub[is.finite(sub$x) & is.finite(sub$clr), c("x", "clr", "depth", "facet_lab")]
  if (nrow(sub) < 6) return(NULL)
  nu <- length(unique(sub$x))
  if (nu < 2) return(NULL)
  grid <- data.frame(x = seq(min(sub$x), max(sub$x), length.out = 80))
  fit <- tryCatch({
    # poly avoids ns() boundary-knot warnings on 4 discrete segments
    deg <- min(3L, nu - 1L)
    if (deg >= 2) lm(clr ~ poly(x, degree = deg, raw = TRUE), data = sub)
    else lm(clr ~ x, data = sub)
  }, error = function(e) lm(clr ~ x, data = sub))
  pr <- predict(fit, newdata = grid, se.fit = TRUE)
  data.frame(
    x = grid$x,
    fit = as.numeric(pr$fit),
    lo = as.numeric(pr$fit - 1.96 * pr$se.fit),
    hi = as.numeric(pr$fit + 1.96 * pr$se.fit),
    depth = sub$depth[1],
    facet_lab = sub$facet_lab[1]
  )
}

curves <- d %>%
  group_split(facet_lab, depth) %>%
  lapply(fit_curve) %>%
  bind_rows() %>%
  mutate(
    facet_lab = factor(facet_lab, levels = levels(d$facet_lab)),
    depth = factor(depth, levels = levels(d$depth))
  )

p <- ggplot() +
  geom_hline(yintercept = 0, linetype = "dashed",
             colour = "grey55", linewidth = 0.3) +
  geom_point(
    data = d,
    aes(x, clr, colour = depth),
    size = 0.9, alpha = 0.45, stroke = 0,
    position = position_jitter(width = 0.08, height = 0)
  ) +
  geom_ribbon(
    data = curves,
    aes(x = x, ymin = lo, ymax = hi, fill = depth),
    alpha = 0.14, colour = NA
  ) +
  geom_line(
    data = curves,
    aes(x = x, y = fit, colour = depth),
    linewidth = 1.05
  ) +
  facet_wrap(~ facet_lab, ncol = 4, scales = "free_y") +
  scale_colour_manual(values = DEPTH_COL, drop = FALSE) +
  scale_fill_manual(values = DEPTH_COL, drop = FALSE, guide = "none") +
  scale_x_continuous(
    breaks = 1:4,
    labels = unname(SEG_LAB[SEGS])
  ) +
  labs(
    title = "Gut-axis composition for top segment-variable cell types",
    subtitle = "CLR vs proximal → distal segment, stratified by full thickness (EPI_LP_MUSC) vs not",
    x = "Gut segment",
    y = "CLR (composition)"
  ) +
  theme_gca() +
  theme(
    axis.text.x = element_text(size = 5.5, angle = 35, hjust = 1, vjust = 1)
  )

# Roomier facets like panel_g (still within ~Nature full-page height)
w_mm <- 180
h_mm <- 168
wi <- w_mm / MM; hi <- h_mm / MM
stem <- "panel_gut_axis_depth_splines_top16"
ggsave(file.path(OUT, paste0(stem, ".pdf")), p, width = wi, height = hi,
       device = cairo_pdf, bg = "transparent")
ggsave(file.path(OUT, paste0(stem, ".svg")), p, width = wi, height = hi,
       device = svglite, bg = "transparent")
ggsave(file.path(OUT, paste0(stem, ".png")), p, width = wi, height = hi,
       dpi = 300, bg = "white")
message("wrote ", stem, " (", w_mm, "×", h_mm, " mm)")
