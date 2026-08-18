#!/usr/bin/env Rscript
# Task 3: composition mixed-model variance fractions + study-level bootstrap
# on lineage aggregates. Expression variancePartition is run from Python once
# embeddings exist (04_expression_varpart.py).

suppressPackageStartupMessages({
  library(lme4)
  library(data.table)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(file_arg)) normalizePath(file.path(dirname(file_arg), "..")) else getwd()
fig <- dirname(root)
tables <- file.path(root, "tables")
logs <- file.path(root, "logs")
dir.create(tables, showWarnings = FALSE, recursive = TRUE)
dir.create(logs, showWarnings = FALSE, recursive = TRUE)

SEED <- 20260804
set.seed(SEED)
N_BOOT <- 100
MIN_N <- 40
MAIN <- c("duodenum", "jejunum", "ileum", "colon")
COVS <- c(
  "radial_tissue_term", "tissue_level_1", "sample_collection_method",
  "age_range", "assay", "sample_preservation_method",
  "sampled_site_condition", "sex_ontology_term",
  "sequenced_fragment", "gene_annotation_version", "dataset_id"
)

is_unknown <- function(x) {
  x <- trimws(tolower(as.character(x)))
  is.na(x) | x %in% c("", "unknown", "nan", "none", "n/a", "na", "not applicable")
}

var_frac_mixed <- function(y, g, dataset, donor, include_g = TRUE) {
  d <- data.frame(
    y = as.numeric(y),
    g = factor(as.character(g)),
    dataset = factor(as.character(dataset)),
    donor = factor(as.character(donor))
  )
  d <- d[is.finite(d$y) & !is_unknown(d$g), ]
  if (nrow(d) < MIN_N) return(c(fixed = NA, dataset = NA, donor = NA, resid = NA))
  if (include_g && nlevels(droplevels(d$g)) < 2) {
    return(c(fixed = NA, dataset = NA, donor = NA, resid = NA))
  }
  form <- if (include_g) {
    y ~ g + (1 | dataset) + (1 | donor)
  } else {
    y ~ (1 | dataset) + (1 | donor)
  }
  fit <- tryCatch(
    lmer(form, data = d, REML = TRUE,
         control = lmerControl(check.conv.singular = "ignore",
                               optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    error = function(e) tryCatch(
      lmer(if (include_g) y ~ g + (1 | dataset) else y ~ (1 | dataset),
           data = d, REML = TRUE,
           control = lmerControl(check.conv.singular = "ignore")),
      error = function(e2) NULL
    )
  )
  if (is.null(fit)) return(c(fixed = NA, dataset = NA, donor = NA, resid = NA))
  vc <- as.data.frame(VarCorr(fit))
  ve <- sigma(fit)^2
  fe <- 0
  if (include_g) {
    fe <- tryCatch({
      mm <- model.matrix(~ g, data = d)
      b <- fixef(fit)
      common <- intersect(colnames(mm), names(b))
      xb <- as.numeric(mm[, common, drop = FALSE] %*% b[common])
      stats::var(xb)
    }, error = function(e) NA_real_)
  }
  v_dataset <- vc$vcov[vc$grp == "dataset"][1]
  v_donor <- vc$vcov[vc$grp == "donor"][1]
  if (length(v_donor) == 0 || is.na(v_donor)) v_donor <- 0
  if (length(v_dataset) == 0 || is.na(v_dataset)) v_dataset <- 0
  if (!is.finite(fe)) fe <- 0
  tot <- fe + v_dataset + v_donor + ve
  if (!is.finite(tot) || tot <= 0) return(c(fixed = NA, dataset = NA, donor = NA, resid = NA))
  c(fixed = fe / tot, dataset = v_dataset / tot, donor = v_donor / tot, resid = ve / tot)
}

clr <- fread(file.path(fig, "data", "clr_long.csv"))
meta <- clr[, lapply(.SD, function(x) x[which(!is.na(x) & !is_unknown(x))[1]]),
            .SDcols = c(
              "donor_id", "dataset_id", "tissue_level_1", "sampled_site_condition",
              "radial_tissue_term", "sample_preservation_method", "sex_ontology_term",
              "age_range", "assay", "sample_collection_method", "sequenced_fragment",
              "gene_annotation_version"
            ),
            by = sample_id]

cts <- unique(clr[, .(celltype, lineage)])
comp_rows <- list()
i <- 0L
for (r in seq_len(nrow(cts))) {
  ct <- cts$celltype[r]
  lin <- cts$lineage[r]
  d <- unique(clr[celltype == ct, .(sample_id, clr)])
  d <- merge(d, meta, by = "sample_id", all.x = TRUE)
  d <- d[as.character(tissue_level_1) %in% MAIN]
  if (nrow(d) < MIN_N) next
  tc <- d[, .N, by = tissue_level_1]
  if (nrow(tc) < 4 || min(tc$N) < 3) next
  for (cov in COVS) {
    if (cov == "dataset_id") {
      # random-effect only partition
      vf <- var_frac_mixed(d$clr, d$dataset_id, d$dataset_id, d$donor_id, include_g = FALSE)
      fixed <- NA_real_
      dataset_frac <- vf["dataset"]
      donor_frac <- vf["donor"]
    } else {
      vf <- var_frac_mixed(d$clr, d[[cov]], d$dataset_id, d$donor_id, include_g = TRUE)
      fixed <- vf["fixed"]
      dataset_frac <- vf["dataset"]
      donor_frac <- vf["donor"]
    }
    i <- i + 1L
    comp_rows[[i]] <- data.frame(
      celltype = ct, lineage = lin, covariate = cov, modality = "composition",
      estimator = "lmer_varfrac",
      n = sum(!is_unknown(d[[cov]])),
      fixed_frac = as.numeric(fixed),
      dataset_frac = as.numeric(dataset_frac),
      donor_frac = as.numeric(donor_frac),
      stringsAsFactors = FALSE
    )
  }
  if (r %% 5 == 0) message("composition mixed: ", r, "/", nrow(cts), " ", lin, "/", ct)
}
comp <- rbindlist(comp_rows, fill = TRUE)
fwrite(comp, file.path(tables, "composition_mixed_varfrac.csv"))
message("Wrote composition_mixed_varfrac.csv: ", nrow(comp), " rows")

# Study-level bootstrap of lineage-mean fixed fractions (not per cell type)
boot_rows <- list()
bi <- 0L
for (lin in unique(comp$lineage)) {
  dlin <- clr[lineage == lin]
  # pick cell types that passed support
  keep_ct <- unique(comp[lineage == lin, celltype])
  studies <- unique(as.character(meta$dataset_id))
  for (cov in setdiff(COVS, "dataset_id")) {
    boots <- rep(NA_real_, N_BOOT)
    for (b in seq_len(N_BOOT)) {
      draw <- sample(studies, length(studies), replace = TRUE)
      fracs <- c()
      for (ct in keep_ct) {
        d <- unique(dlin[celltype == ct, .(sample_id, clr)])
        d <- merge(d, meta, by = "sample_id")
        d <- d[as.character(tissue_level_1) %in% MAIN]
        parts <- lapply(draw, function(s) d[as.character(dataset_id) == s])
        db <- rbindlist(parts)
        if (nrow(db) < MIN_N) next
        fracs <- c(fracs, var_frac_mixed(db$clr, db[[cov]], db$dataset_id, db$donor_id)["fixed"])
      }
      boots[b] <- if (length(fracs)) mean(fracs, na.rm = TRUE) else NA_real_
    }
    boots <- boots[is.finite(boots)]
    point <- mean(comp[lineage == lin & covariate == cov, fixed_frac], na.rm = TRUE)
    bi <- bi + 1L
    boot_rows[[bi]] <- data.frame(
      lineage = lin, covariate = cov, modality = "composition",
      fixed_frac = point,
      boot_lo = if (length(boots)) as.numeric(quantile(boots, 0.025)) else NA_real_,
      boot_hi = if (length(boots)) as.numeric(quantile(boots, 0.975)) else NA_real_,
      boot_mean = if (length(boots)) mean(boots) else NA_real_,
      n_boot = length(boots),
      stringsAsFactors = FALSE
    )
    message("bootstrap ", lin, " ", cov, " n_boot=", length(boots))
  }
}
boot <- rbindlist(boot_rows, fill = TRUE)
fwrite(boot, file.path(tables, "composition_mixed_study_bootstrap.csv"))

# OLS vs mixed inflation
ols_path <- file.path(tables, "composition_celltype_estimates.csv")
if (file.exists(ols_path)) {
  ols <- fread(ols_path)
  m <- merge(
    ols[, .(celltype, covariate, ols = omega2_trunc)],
    comp[!is.na(fixed_frac), .(celltype, covariate, mixed = fixed_frac)],
    by = c("celltype", "covariate")
  )
  m[, inflation := ols / mixed]
  fwrite(m, file.path(tables, "task3_ols_vs_mixed_composition.csv"))
  summ <- m[, .(
    median_ols = median(ols, na.rm = TRUE),
    median_mixed = median(mixed, na.rm = TRUE),
    median_inflation = median(inflation[is.finite(inflation) & mixed > 0.01], na.rm = TRUE),
    n = .N
  ), by = covariate][order(-median_ols)]
  fwrite(summ, file.path(tables, "task3_inflation_by_covariate.csv"))
  print(summ)
}

writeLines(
  c(paste("SEED", SEED), paste("N_BOOT", N_BOOT), capture.output(sessionInfo())),
  file.path(logs, "mixed_models_sessionInfo.txt")
)
message("Done")
