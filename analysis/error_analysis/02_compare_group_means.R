#!/usr/bin/env Rscript
# Compare selected model-human difficulty groups with the overall item mean.
# Run: Rscript --vanilla analysis/error_analysis/02_compare_group_means.R
# First run 01_classify_and_rank.R. Reads classified_*.csv and writes numerical
# group means, t statistics and p values to output/group_mean_tests.csv.

# ---- Configuration ----------------------------------------------------------
if (!requireNamespace("here", quietly = TRUE)) {
  stop("Missing R package: here. Install it manually.")
}
DATASETS <- c("huang", "wang_small", "wang_full")
INPUT_DIR <- here::here("analysis", "error_analysis")
OUTPUT_FILE <- here::here("analysis", "error_analysis", "output", "group_mean_tests.csv")

VARS <- c(
  "log_freq"               = "Frequency",
  "aoa"                    = "Age of acquisition",
  "context_familiarity"    = "Context familiarity",
  "phonetic_radical_order" = "Radical order",
  "regularity"             = "Regularity",
  "n_stroke"               = "Stroke count"
)

DATASET_BASE <- c(
  huang      = "Huang",
  wang_small = "Wang Small",
  wang_full  = "Wang Full"
)

TOP_T <- 30   # fallback group size if a `selected` column is missing


# ---- Test selected groups against the overall item mean ---------------------
compute_ttest_rows <- function(df) {
  if ("selected" %in% names(df)) {
    is_selected <- as.logical(df$selected)
  } else {
    is_selected <- df$rank_in_type <= TOP_T
  }

  results <- list()
  for (vname in names(VARS)) {
    mu_all <- mean(df[[vname]], na.rm = TRUE)   # reference and test mu
    row <- list(variable = vname, m_all = mu_all)

    for (qname in c("Q1", "Q2", "Q3", "Q4")) {
      in_group <- is_selected & df$type == qname
      grp <- df[[vname]][in_group]
      grp <- grp[!is.na(grp)]
      m_type <- if (length(grp) > 0) mean(grp) else NA

      if (length(unique(grp)) >= 2) {
        tt <- t.test(grp, mu = mu_all)         # one-sample against the grand mean
        row[[paste0(qname, "_m")]] <- m_type
        row[[paste0(qname, "_t")]] <- unname(tt$statistic)
        row[[paste0(qname, "_p")]] <- tt$p.value
      } else {
        row[[paste0(qname, "_m")]] <- m_type
        row[[paste0(qname, "_t")]] <- NA
        row[[paste0(qname, "_p")]] <- NA
      }
    }
    results[[vname]] <- row
  }
  results
}


# ---- Save the original unrounded calculations without typesetting ----------
# The reference is the overall mean including the selected items. It is treated
# as the fixed mu in a two-sided one-sample t-test, preserving the reported test.
results <- lapply(DATASETS, function(dataset) {
  path <- file.path(INPUT_DIR, paste0("classified_", dataset, ".csv"))
  if (!file.exists(path)) stop("Missing input: ", path, "; run 01_classify_and_rank.R first.")
  data <- read.csv(path, check.names = FALSE)
  required <- c(names(VARS), "type")
  if (!all(required %in% names(data)) ||
      !any(c("selected", "rank_in_type") %in% names(data))) {
    stop("Missing classification or lexical columns: ", dataset)
  }
  rows <- do.call(rbind, lapply(compute_ttest_rows(data), as.data.frame))
  cbind(dataset = dataset, n_all = nrow(data), rows)
})
dir.create(dirname(OUTPUT_FILE), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, results), OUTPUT_FILE, row.names = FALSE, fileEncoding = "UTF-8")
message("Saved ", OUTPUT_FILE)
