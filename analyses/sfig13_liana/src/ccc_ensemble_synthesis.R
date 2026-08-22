#!/usr/bin/env Rscript
## Genome-wide ensemble-robust CCC synthesis across the 4 major gut segments.
##
## Input must be LIANA `rank_aggregate` output (RRA over CellPhoneDB, Connectome,
## log2FC, NATMI, SingleCellSignalR). Robustness uses the true consensus ranks:
##   * magnitude_rank    — LIANA magnitude consensus (lower = stronger)
##   * specificity_rank  — LIANA specificity consensus (lower = more specific)
##   * ensemble_rank     — sqrt(magnitude_rank * specificity_rank)
## An edge is "robust" if its magnitude_rank is in the top RANK_Q within a
## segment (smallest ranks) in >= MIN_SEG segments.
##
## Outputs (under <OUT>/ensemble_synthesis/):
##   pillars_edges.csv          robust source->target LR edges (cell-resolved)
##   pillars_lr_axes.csv        robust ligand->receptor axes (cell-agnostic)
##   lineage_channels.csv       source_lineage -> target_lineage flux
##   specific_novel_edges.csv   high-specificity robust neuro/GPCR edges

set.seed(1L)
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr, warn.conflicts = FALSE)
  library(readr)
  library(stringr, warn.conflicts = FALSE)
})

INPUT_CSV <- Sys.getenv("CCC_EDGE_CSV", "")
if (!nzchar(INPUT_CSV))
  stop("Set CCC_EDGE_CSV to combined_lr_per_tissue_level_1.csv")
BASE_OUT  <- Sys.getenv("CCC_OUTPUT_DIR", "")
if (!nzchar(BASE_OUT))
  stop("Set CCC_OUTPUT_DIR")
OUT_DIR   <- file.path(BASE_OUT, "ensemble_synthesis")
LINEAGE_CSV <- Sys.getenv("CCC_LINEAGE_LOOKUP_CSV",
                          "data/hgca_celltype_v1_lineage.csv")
DROP_STATES <- trimws(strsplit(Sys.getenv("CCC_DROP_STATES", "Epithelial"),
                               "[;,]")[[1]])
SEG_ORDER  <- c("duodenum", "jejunum", "ileum", "colon")
RANK_Q     <- as.numeric(Sys.getenv("CCC_COMBINED_Q", "0.99"))  # top 1%
MIN_SEG    <- as.integer(Sys.getenv("CCC_MIN_SEG", "4"))
MAX_PVAL   <- as.numeric(Sys.getenv("CCC_MAX_PVAL", "1"))  # 1 = no CPDB p filter
EPS <- 1e-300
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

message("Reading ", INPUT_CSV)
dt <- fread(INPUT_CSV)
need <- c("magnitude_rank", "specificity_rank")
miss <- setdiff(need, names(dt))
if (length(miss))
  stop("Input is not LIANA rank_aggregate output; missing columns: ",
       paste(miss, collapse = ", "))

setnames(dt, "tissue_level_1", "segment")
dt[, segment := tolower(trimws(segment))]
dt <- dt[segment %in% SEG_ORDER &
           !is.na(magnitude_rank) & !is.na(specificity_rank)]
dt <- dt[!(source %in% DROP_STATES) & !(target %in% DROP_STATES)]
if ("cellphone_pvals" %in% names(dt) && is.finite(MAX_PVAL) && MAX_PVAL < 1)
  dt <- dt[cellphone_pvals <= MAX_PVAL]
dt[, lr_pair := paste(ligand_complex, receptor_complex, sep = "->")]

## LIANA RRA ranks: lower = better. Score for plotting/ordering: -log10(rank).
dt[, ensemble_rank := sqrt(pmax(magnitude_rank, EPS) *
                             pmax(specificity_rank, EPS))]
dt[, mag_score := -log10(pmax(magnitude_rank, EPS))]
dt[, spec_score := -log10(pmax(specificity_rank, EPS))]
dt[, ensemble_score := -log10(pmax(ensemble_rank, EPS))]

## top RANK_Q by magnitude consensus within each segment
dt[, is_top := magnitude_rank <= quantile(magnitude_rank, 1 - RANK_Q),
   by = segment]

## ---- lineage annotation ----
lk <- fread(LINEAGE_CSV)[, .(cell_state, plot_lineage)]
setkey(lk, cell_state)
dt[, src_lin := lk[.(source),  plot_lineage]]
dt[, tgt_lin := lk[.(target),  plot_lineage]]
dt[is.na(src_lin), src_lin := "Other"]
dt[is.na(tgt_lin), tgt_lin := "Other"]

## ---------- 1. cell-resolved robust edges (pillars) ----------
edge_key <- c("source", "target", "lr_pair", "src_lin", "tgt_lin")
edges <- dt[, .(
  n_seg_top        = sum(is_top),
  n_seg_seen       = .N,
  ensemble_score_mean = mean(ensemble_score),
  ensemble_rank_mean  = mean(ensemble_rank),
  mag_rank_mean    = mean(magnitude_rank),
  mag_rank_min     = min(magnitude_rank),
  spec_rank_mean   = mean(specificity_rank),
  lr_means_mean    = if ("lr_means" %in% names(dt)) mean(lr_means) else NA_real_
), by = edge_key]
pillars <- edges[n_seg_top >= MIN_SEG][order(-ensemble_score_mean)]
fwrite(pillars, file.path(OUT_DIR, "pillars_edges.csv"))

## ---------- 2. cell-agnostic robust LR axes ----------
ax <- dt[, .(ensemble_score = max(ensemble_score),
             ensemble_rank  = min(ensemble_rank),
             magnitude_rank = min(magnitude_rank),
             specificity_rank = min(specificity_rank),
             lr_means = if ("lr_means" %in% names(dt)) max(lr_means) else NA_real_,
             is_top   = as.integer(any(is_top))),
         by = .(segment, lr_pair)]
axes <- ax[, .(n_seg_top = sum(is_top),
               ensemble_score_mean = mean(ensemble_score),
               ensemble_rank_mean  = mean(ensemble_rank),
               mag_rank_mean  = mean(magnitude_rank),
               mag_rank_min   = min(magnitude_rank),
               spec_rank_mean = mean(specificity_rank),
               lr_means_mean  = mean(lr_means)),
           by = lr_pair]
axes <- axes[n_seg_top >= MIN_SEG][order(-ensemble_score_mean)]
fwrite(axes, file.path(OUT_DIR, "pillars_lr_axes.csv"))

## ---------- 3. lineage -> lineage channels ----------
chan <- dt[, .(
  flux_ensemble = sum(ensemble_score),
  mean_ensemble = mean(ensemble_score),
  n_top_edges   = sum(is_top),
  n_edges       = .N
), by = .(segment, src_lin, tgt_lin)]
chan_all <- dt[, .(
  mean_ensemble = mean(ensemble_score),
  n_top_edges   = sum(is_top),
  n_edges       = .N,
  frac_top      = mean(is_top)
), by = .(src_lin, tgt_lin)][order(-mean_ensemble)]
fwrite(chan[order(segment, -mean_ensemble)],
       file.path(OUT_DIR, "lineage_channels_by_segment.csv"))
fwrite(chan_all, file.path(OUT_DIR, "lineage_channels.csv"))

## ---------- 4. specific & novel (neuro / GPCR / secreted) ----------
novel_rx <- paste(c("ADRA", "ADRB", "HTR", "CHRM", "CHRN", "NPY", "NPY1R",
                    "VIPR", "GRM", "GPR", "SORT1", "RET", "GFRA", "PLXN",
                    "EDNR", "F2RL", "DRD", "SSTR", "GALR", "CALCR", "TRP"),
                  collapse = "|")
nov <- edges[grepl(novel_rx, lr_pair) & n_seg_top >= max(2L, MIN_SEG - 1L)]
nov <- nov[order(spec_rank_mean)]
fwrite(nov, file.path(OUT_DIR, "specific_novel_edges.csv"))

## ---------- console summary ----------
cat("\n==== PILLAR LR AXES (LIANA magnitude_rank top ",
    sprintf("%.0f%%", (1 - RANK_Q) * 100),
    " in >=", MIN_SEG, " segments) ====\n", sep = "")
print(head(axes, 30))
cat("\n==== TOP LINEAGE->LINEAGE CHANNELS (by mean ensemble score) ====\n")
print(head(chan_all, 20))
cat("\n==== SPECIFIC / NEURO-GPCR ROBUST EDGES (by mean specificity_rank) ====\n")
print(head(nov[, .(source, target, lr_pair, src_lin, tgt_lin,
                   n_seg_top, spec_rank_mean, ensemble_score_mean)], 35))
cat("\n==== PILLAR CELL-RESOLVED EDGES (top ensemble score) ====\n")
print(head(pillars[, .(source, target, lr_pair, src_lin, tgt_lin,
                       n_seg_top, ensemble_score_mean, mag_rank_mean)], 30))
message("\nWrote synthesis under: ", OUT_DIR)

## ============================================================================
## 5. Where do the highlighted concept axes rank among ALL interactions?
## ============================================================================
suppressPackageStartupMessages({
  library(ggplot2); library(ggrepel); library(scales)
})

axes_all <- ax[, .(ensemble_score_mean = mean(ensemble_score),
                   mag_rank_mean = mean(magnitude_rank),
                   spec_rank_mean = mean(specificity_rank),
                   n_seg_top     = sum(is_top)),
               by = lr_pair][order(-ensemble_score_mean)]
axes_all[, rank := .I]
N_axes  <- nrow(axes_all)
N_edges <- nrow(unique(dt[, .(source, target, lr_pair)]))
N_rows  <- nrow(dt)
top_cut <- axes_all[, quantile(ensemble_score_mean, RANK_Q)]

hl <- data.table(
  lr_pair = c("PCSK1N->GPR171",
              "GUCA2B->GUCY2C", "GUCA2A->GUCY2C",
              "TPSAB1->F2RL1", "TPSB2->F2RL1",
              "NRXN1->NLGN1", "NRXN1->NLGN2",
              "MADCAM1->ITGA4_ITGB7",
              "CCL2->ACKR1", "CCL14->ACKR1",
              "PYY->NPY1R", "NTS->SORT1", "GCG->GLP1R",
              "PDGFA->PDGFRA", "MFGE8->PDGFRB", "JAG1->NOTCH3", "VEGFC->LYVE1",
              "CXCL8->CXCR2", "CXCL1->CXCR2"),
  concept = c("C2 EEC->CD8 checkpoint",
              "C1 guanylin->GUCY2C", "C1 guanylin->GUCY2C",
              "C3 mast->PAR2", "C3 mast->PAR2",
              "C4 glia->stroma", "C4 glia->stroma",
              "Pillar MADCAM1 addressin",
              "Pillar ACKR1 sink", "Pillar ACKR1 sink",
              "C1 EEC peptides", "C1 EEC peptides", "C1 EEC peptides",
              "C6 perivascular wiring", "C6 perivascular wiring",
              "C6 perivascular wiring", "C6 perivascular wiring",
              "C7 chemokine recruit", "C7 chemokine recruit"),
  label = c("PCSK1N->GPR171",
            "GUCA2B->GUCY2C", "GUCA2A->GUCY2C",
            "TPSAB1->F2RL1", "TPSB2->F2RL1",
            "NRXN1->NLGN1", "NRXN1->NLGN2",
            "MADCAM1->ITGA4/B7",
            "CCL2->ACKR1", "CCL14->ACKR1",
            "PYY->NPY1R", "NTS->SORT1", "GCG->GLP1R",
            "PDGFA->PDGFRA", "MFGE8->PDGFRB", "JAG1->NOTCH3", "VEGFC->LYVE1",
            "CXCL8->CXCR2", "CXCL1->CXCR2"))
hl <- merge(hl, axes_all, by = "lr_pair", all.x = TRUE)
hl[, pct_top := 100 * rank / N_axes]
hl[, lab_full := sprintf("%s  (#%d)", label, rank)]
fwrite(hl[order(rank)], file.path(OUT_DIR, "highlight_axis_ranks.csv"))
cat("\n==== HIGHLIGHTED CONCEPT AXES: rank among", N_axes, "LR axes ====\n")
print(hl[order(rank), .(concept, label, rank, pct_top = round(pct_top, 2),
                        ensemble_score_mean = round(ensemble_score_mean, 3),
                        n_seg_top)])

concept_cols <- c(
  "C1 EEC peptides"          = "#E69F00",
  "C1 guanylin->GUCY2C"      = "#F0B860",
  "C2 EEC->CD8 checkpoint"   = "#0072B2",
  "C3 mast->PAR2"            = "#009E73",
  "C4 glia->stroma"          = "#CC79A7",
  "Pillar ACKR1 sink"        = "#56B4E9",
  "Pillar MADCAM1 addressin" = "#D55E00",
  "C6 perivascular wiring"   = "#8C564B",
  "C7 chemokine recruit"     = "#111111")

theme_gca <- function() {
  theme_classic(base_size = 6, base_family = "Helvetica") +
    theme(text = element_text(colour = "black"),
          plot.title = element_text(face = "bold", size = 7, hjust = 0),
          plot.subtitle = element_text(size = 6),
          axis.line = element_line(linewidth = 0.25),
          axis.ticks = element_line(linewidth = 0.25),
          axis.text = element_text(colour = "black", size = 6),
          axis.title = element_text(size = 6),
          panel.grid = element_blank(),
          legend.key.size = unit(3, "mm"),
          legend.text = element_text(size = 5.5),
          legend.title = element_text(size = 6, face = "bold"))
}

p_rank <- ggplot(axes_all, aes(rank, ensemble_score_mean)) +
  geom_line(colour = "#BBBBBB", linewidth = 0.4) +
  geom_hline(yintercept = top_cut, linetype = "dashed",
             colour = "#999999", linewidth = 0.25) +
  annotate("text", x = N_axes, y = top_cut,
           label = sprintf("top %.0f%% axis-score cut", (1 - RANK_Q) * 100),
           hjust = 1, vjust = -0.5, size = 1.9, colour = "#666666") +
  geom_point(data = hl[!is.na(rank)], aes(rank, ensemble_score_mean, fill = concept),
             shape = 21, size = 1.8, stroke = 0.3, colour = "black") +
  geom_text_repel(data = hl[!is.na(rank)],
                  aes(rank, ensemble_score_mean, label = lab_full,
                      colour = concept),
                  size = 1.85, max.overlaps = Inf, box.padding = 0.6,
                  force = 4, force_pull = 0.2, nudge_y = -0.12,
                  segment.size = 0.2, min.segment.length = 0,
                  seed = 1, show.legend = FALSE) +
  scale_x_log10(labels = comma) +
  scale_fill_manual(values = concept_cols, name = "Concept") +
  scale_colour_manual(values = concept_cols, guide = "none") +
  labs(
    title = "Where the concept axes rank among all gut CCC interactions",
    subtitle = sprintf(
      "LIANA rank_aggregate: -log10(sqrt(magnitude_rank x specificity_rank)), mean over 4 segments.  N = %s L-R axes;  %s cell-resolved edges;  %s edge x segment tests.",
      comma(N_axes), comma(N_edges), comma(N_rows)),
    x = "Rank among all L-R axes (log scale; 1 = strongest)",
    y = "Ensemble score (mean over segments)") +
  theme_gca()

ggsave(file.path(OUT_DIR, "fig_rank_highlight.pdf"), p_rank,
       width = 180 / 25.4, height = 96 / 25.4, device = grDevices::cairo_pdf)
ggsave(file.path(OUT_DIR, "fig_rank_highlight.png"), p_rank,
       width = 180 / 25.4, height = 96 / 25.4, dpi = 300)
message("Wrote rank-highlight figure -> ", file.path(OUT_DIR, "fig_rank_highlight.png"))
