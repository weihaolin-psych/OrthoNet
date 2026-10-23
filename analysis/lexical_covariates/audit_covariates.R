#!/usr/bin/env Rscript
# Describe lexical overlap, AoA coding, and collinearity in retrieval models.
# Run from the repository root: Rscript --vanilla analysis/lexical_covariates/audit_covariates.R
# Reads canonical processed trials and saved main models; never fits a model.
# Writes item correlations, AoA counts, and model/design VIFs to output/.
# First run analysis/mixed_effects_entropy/01_fit_models.R to create fitted models.

# ---- Configuration ---------------------------------------------------------
DATA_DIR <- file.path("data", "processed")
MODEL_DIR <- file.path("analysis", "mixed_effects_entropy", "output")
OUTPUT_DIR <- file.path("analysis", "lexical_covariates", "output")
PROJECT_LIBRARY <- file.path("renv", "library", "windows", "R-4.4", "x86_64-w64-mingw32")
DATASETS <- c("huang", "wang_small", "wang_full")
PREDICTORS <- c("log_freq", "aoa", "context_familiarity", "phonetic_radical_order", "regularity", "n_stroke", "entropy")
MODEL_TERMS <- c("log_freq_c", "aoa_c", "context_familiarity_c", "radical_order_c", "regularity_c", "n_stroke_c", "entropy_c")
AOA_SENTINEL <- 12.5
RT_MAX_MS <- 10000
TOLERANCE <- 1e-7
REQUIRED_PACKAGES <- c("lme4", "lmerTest")

# ---- Dependencies and preservation -----------------------------------------
if (dir.exists(PROJECT_LIBRARY)) .libPaths(c(PROJECT_LIBRARY, .libPaths()))
if (.Platform$OS.type == "windows") invisible(Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"))
missing_packages <- REQUIRED_PACKAGES[!vapply(REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("Missing packages: ", paste(missing_packages, collapse = ", "), ". Nothing was installed.")
library(lme4)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
source_files <- c(file.path(DATA_DIR, paste0(DATASETS, "_trials_with_entropy.csv")),
                  list.files(MODEL_DIR, pattern = "\\.rds$", full.names = TRUE))
source_hashes <- tools::md5sum(source_files)
assert_close <- function(a, b) stopifnot(length(a) == length(b),
  isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = TOLERANCE, scale = 1)))

# ---- Item summaries: equal weight per character ----------------------------
item_data <- function(df) {
  items <- unique(df[, c("item", "Character", "Word", PREDICTORS)])
  if (anyDuplicated(items$item) || anyDuplicated(items$Character))
    stop("Item predictors vary within character, or character is not a unique item.")
  items[order(items$item), ]
}
aoa_summary <- function(items, dataset, sample) {
  known <- !is.na(items$aoa)
  data.frame(dataset, sample, n_items = nrow(items), n_aoa_known = sum(known),
    n_aoa_missing = sum(!known), n_aoa_12_5 = sum(items$aoa == AOA_SENTINEL, na.rm = TRUE),
    pct_of_known = 100 * mean(items$aoa[known] == AOA_SENTINEL),
    mean_aoa = mean(items$aoa[known]), sd_aoa = sd(items$aoa[known]))
}
save_correlations <- function(items, dataset, sample) {
  complete <- items[complete.cases(items[, PREDICTORS]), ]
  for (method in c("pearson", "spearman")) {
    correlation <- cor(complete[, PREDICTORS], method = method)
    write.csv(correlation, file.path(OUTPUT_DIR, paste(dataset, sample, method, "correlation.csv", sep = "_")))
  }
  data.frame(dataset, sample, n_items = nrow(items), n_complete_items = nrow(complete),
    frequency_aoa_r = cor(complete$log_freq, complete$aoa),
    frequency_aoa_rho = cor(complete$log_freq, complete$aoa, method = "spearman"))
}

# ---- VIF: distinguish fitted-model covariance from raw design overlap ------
# Every predictor has one degree of freedom. diag(solve(R)) is the ordinary
# VIF (not its square root). For mixed models R is the fixed-slope coefficient
# correlation matrix, excluding the intercept (the car::vif convention).
vif_from_correlation <- function(r) {
  if (min(eigen(r, symmetric = TRUE, only.values = TRUE)$values) <= 0)
    stop("Singular correlation matrix; finite VIFs cannot be reported.")
  diag(solve(r))
}

# ---- Audit canonical samples against the saved model frames ----------------
aoa_rows <- correlation_rows <- vif_rows <- check_rows <- list()
all_items <- list()
for (dataset in DATASETS) {
  raw <- read.csv(file.path(DATA_DIR, paste0(dataset, "_trials_with_entropy.csv")), fileEncoding = "UTF-8")
  items <- item_data(raw)
  all_items[[dataset]] <- items
  write.csv(items, file.path(OUTPUT_DIR, paste0(dataset, "_items.csv")), row.names = FALSE, fileEncoding = "UTF-8")
  aoa_rows[[paste(dataset, "all")]] <- aoa_summary(items, dataset, "all")
  correlation_rows[[paste(dataset, "all")]] <- save_correlations(items, dataset, "all")
  write.csv(as.data.frame(table(items$aoa, useNA = "ifany")),
            file.path(OUTPUT_DIR, paste0(dataset, "_aoa_distribution.csv")), row.names = FALSE)
  for (outcome in c("top", "rt")) {
    eligible <- if (outcome == "top") as.logical(raw$top_eligible) else as.logical(raw$rt_eligible) & raw$rt <= RT_MAX_MS
    df <- raw[which(eligible), ]
    for (i in seq_along(PREDICTORS)) {
      value <- df[[PREDICTORS[i]]]
      df[[MODEL_TERMS[i]]] <- if (PREDICTORS[i] == "phonetic_radical_order")
        ifelse(value == 1, 0.5, -0.5) else value - mean(value, na.rm = TRUE)
    }
    response <- if (outcome == "top") "top_binary" else "log_rt"
    df <- df[complete.cases(df[, c(MODEL_TERMS, response, "subject", "item")]), ]
    used_items <- item_data(df)
    key <- paste(dataset, outcome)
    aoa_rows[[key]] <- aoa_summary(used_items, dataset, outcome)
    correlation_rows[[key]] <- save_correlations(used_items, dataset, outcome)
    previous_frame <- NULL
    for (model_name in c("base", "entropy")) {
      model <- readRDS(file.path(MODEL_DIR, paste0(dataset, "_", outcome, "_", model_name, ".rds")))
      frame <- model.frame(model)
      terms <- if (model_name == "base") MODEL_TERMS[-length(MODEL_TERMS)] else MODEL_TERMS
      stopifnot(nrow(df) == nrow(frame), identical(as.character(df$item), as.character(frame$item)),
        identical(as.character(df$subject), as.character(frame$subject)))
      for (column in c(response, terms)) assert_close(df[[column]], frame[[column]])
      if (!is.null(previous_frame)) for (column in names(previous_frame))
        stopifnot(identical(previous_frame[[column]], frame[[column]]))
      previous_frame <- frame
      covariance <- as.matrix(vcov(model))[terms, terms, drop = FALSE]
      model_vif <- vif_from_correlation(cov2cor(covariance))
      # Independent determinant identity for one-df GVIFs.
      coefficient_correlation <- cov2cor(covariance)
      determinant_vif <- vapply(seq_along(terms), function(i)
        det(coefficient_correlation[-i, -i, drop = FALSE]) / det(coefficient_correlation), numeric(1))
      assert_close(model_vif, determinant_vif)
      design_vif <- vif_from_correlation(cor(df[, terms]))
      item_vif <- vif_from_correlation(cor(df[!duplicated(df$item), terms]))
      vif_rows[[paste(key, model_name)]] <- data.frame(dataset, outcome, model = model_name,
        term = PREDICTORS[match(terms, MODEL_TERMS)], model_vif, trial_design_vif = design_vif,
        item_design_vif = item_vif, n_trials = nrow(frame), n_items = nrow(used_items))
      check_rows[[paste(key, model_name)]] <- data.frame(dataset, outcome, model = model_name,
        canonical_frame_verified = TRUE, n_trials = nrow(frame), n_items = nrow(used_items))
    }
  }
}

# ---- Check overlapping material values and save complete audit outputs -----
overlap <- merge(all_items$huang, all_items$wang_small, by = "Character", suffixes = c("_huang", "_wang"))
overlap_checks <- do.call(rbind, lapply(PREDICTORS, function(column) {
  a <- overlap[[paste0(column, "_huang")]]
  b <- overlap[[paste0(column, "_wang")]]
  data.frame(variable = column, n_overlap = nrow(overlap),
    n_different = sum(xor(is.na(a), is.na(b)) | (!is.na(a) & !is.na(b) & abs(a-b) > TOLERANCE)))
}))
outputs <- list(aoa_summary = do.call(rbind, aoa_rows), correlation_summary = do.call(rbind, correlation_rows),
  vif = do.call(rbind, vif_rows), sample_checks = do.call(rbind, check_rows), overlap_checks = overlap_checks)
for (name in names(outputs)) write.csv(outputs[[name]], file.path(OUTPUT_DIR, paste0(name, ".csv")), row.names = FALSE)

# ---- Human-readable numerical report ---------------------------------------
markdown_table <- function(df) {
  display <- df
  for (column in names(display)) if (is.numeric(display[[column]]))
    display[[column]] <- format(round(display[[column]], 3), trim = TRUE, scientific = FALSE)
  c(paste0("| ", paste(names(display), collapse = " | "), " |"),
    paste0("| ", paste(rep("---", ncol(display)), collapse = " | "), " |"),
    apply(display, 1, function(row) paste0("| ", paste(row, collapse = " | "), " |")), "")
}
report <- c("# Lexical covariate audit", "", "## Frequency–AoA correlations", "",
  "Character-level complete cases across all seven predictors; Pearson r and Spearman rho.", "",
  markdown_table(outputs$correlation_summary), "## AoA = 12.5", "",
  markdown_table(outputs$aoa_summary))
for (dataset in DATASETS) {
  matrix <- read.csv(file.path(OUTPUT_DIR, paste(dataset, "all", "pearson", "correlation.csv", sep = "_")), row.names = 1)
  report <- c(report, paste0("## ", dataset, ": seven-predictor Pearson matrix"), "",
    markdown_table(data.frame(predictor = rownames(matrix), matrix, row.names = NULL)))
}
for (dataset in DATASETS) for (outcome in c("top", "rt")) {
  values <- outputs$vif[outputs$vif$dataset == dataset & outputs$vif$outcome == outcome, ]
  wide <- merge(values[values$model == "base", c("term", "model_vif")],
                values[values$model == "entropy", c("term", "model_vif")], by = "term", all = TRUE,
                suffixes = c("_base", "_entropy"))
  wide <- wide[match(PREDICTORS, wide$term), ]
  report <- c(report, paste0("## ", dataset, " / ", outcome, ": fitted-model VIF"), "", markdown_table(wide))
}
report <- c(report, "NA for entropy in Base means the term is absent.", "",
  "VIF uses the fixed-slope coefficient correlation matrix, excluding the intercept; all terms have one degree of freedom.",
  "vif.csv also includes trial-weighted and equal-item design VIFs, which need not equal fitted-model VIFs.",
  "AoA counts refer to unique characters, not trials. Wang Small and Huang share the same 200 characters and predictors.",
  "No outcome models were refitted. Every fitted sample and predictor was checked against canonical processed data.")
writeLines(report, file.path(OUTPUT_DIR, "report.md"), useBytes = TRUE)
stopifnot(identical(source_hashes, tools::md5sum(source_files)))
write.csv(data.frame(path = names(source_hashes), md5 = unname(source_hashes)), file.path(OUTPUT_DIR, "input_manifest.csv"), row.names = FALSE)
writeLines(c("Canonical samples verified; original data/models unchanged; no model refitting.", capture.output(sessionInfo())), file.path(OUTPUT_DIR, "verification.txt"))
print(outputs$aoa_summary, row.names = FALSE)
print(outputs$correlation_summary, row.names = FALSE)
print(aggregate(model_vif ~ dataset + outcome + model, outputs$vif, max), row.names = FALSE)
print(overlap_checks, row.names = FALSE)
