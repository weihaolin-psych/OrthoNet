#!/usr/bin/env Rscript
# Compare entropy and two relative-logit measures within experimental contexts.
# Run from the repository root: Rscript analysis/logit_measures/correlate_uncertainty_measures.R
# Reads canonical trial files; writes a nine-row numerical CSV.

# ---- Configuration ----------------------------------------------------------
REQUIRED_PACKAGES <- c("here", "dplyr")
missing_packages <- REQUIRED_PACKAGES[!vapply(REQUIRED_PACKAGES,
  requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("Missing R packages: ",
  paste(missing_packages, collapse = ", "), ". Install before running; nothing was installed.")
suppressPackageStartupMessages(library(dplyr))

DATA_FILES <- c(
  huang = here::here("data", "processed", "huang_trials_with_logit_measures.csv"),
  wang_small = here::here("data", "processed", "wang_small_trials_with_logit_measures.csv"),
  wang_full = here::here("data", "processed", "wang_full_trials_with_logit_measures.csv"))
DATASET_LABELS <- c(huang = "Huang", wang_small = "Wang Small", wang_full = "Wang Full")
EXPECTED_TOTAL <- c(huang = 200L, wang_small = 200L, wang_full = 1600L)
EXPECTED_MATCHED <- c(huang = 200L, wang_small = 200L, wang_full = 1592L)
MEASURES <- c("entropy", "deficit_measure", "blocking_measure")
PAIRS <- data.frame(
  measure_x = c("entropy", "entropy", "deficit_measure"),
  measure_y = c("deficit_measure", "blocking_measure", "blocking_measure"))
MEASURE_LABELS <- c(entropy = "Entropy",
  deficit_measure = "Target - mean(non-target logits)",
  blocking_measure = "Target - max(non-target logits)")
PAIR_TITLES <- c("Entropy vs. mean contrast", "Entropy vs. max contrast", "Mean vs. max contrast")
OUTPUT_DIR <- here::here("analysis", "logit_measures", "output")
OUTPUT_CSV <- file.path(OUTPUT_DIR, "uncertainty_measure_correlations.csv")
VALIDATION_TOLERANCE <- 1e-14

# ---- Deduplicate trials without averaging different word contexts ------------
context_scores <- function(trials) {
  required <- c("Character", "Word", MEASURES)
  absent <- setdiff(required, names(trials))
  if (length(absent)) stop("Missing columns: ", paste(absent, collapse = ", "))
  if (anyNA(trials[c("Character", "Word")]) ||
      any(!nzchar(trials$Character) | !nzchar(trials$Word))) stop("Missing Character-Word keys")
  for (measure in MEASURES) {
    if (!is.numeric(trials[[measure]]) || any(is.infinite(trials[[measure]]))) {
      stop("Non-numeric or infinite values: ", measure)
    }
  }
  consistency <- trials %>% group_by(Character, Word) %>%
    summarise(across(all_of(MEASURES), ~ n_distinct(.x, na.rm = FALSE)), .groups = "drop")
  if (any(as.matrix(consistency[MEASURES]) != 1L)) {
    stop("Model scores vary between trials of the same Character-Word")
  }
  contexts <- trials %>% select(all_of(required)) %>% distinct()
  if (anyDuplicated(contexts[c("Character", "Word")])) stop("Duplicated context keys")
  contexts
}

read_contexts <- function(path, dataset_name) {
  if (!file.exists(path)) stop("Missing canonical trials: ", path)
  contexts <- context_scores(read.csv(path, stringsAsFactors = FALSE,
    fileEncoding = "UTF-8", check.names = FALSE))
  if (nrow(contexts) != EXPECTED_TOTAL[[dataset_name]]) stop("Unexpected total contexts: ", dataset_name)
  # One common complete sample for all three pairs; no behavioral eligibility filter.
  complete <- contexts[complete.cases(contexts[MEASURES]), ]
  if (nrow(complete) != EXPECTED_MATCHED[[dataset_name]]) stop("Unexpected matched contexts: ", dataset_name)
  if (!all(vapply(complete[MEASURES], function(x) all(is.finite(x)), logical(1)))) {
    stop("Non-finite complete-case values")
  }
  message(dataset_name, ": ", nrow(complete), "/", nrow(contexts), " complete Character-Word contexts")
  complete$dataset <- dataset_name
  complete
}

# ---- Pearson as primary association; Spearman as a rank-based check ----------
correlate_pair <- function(x, y) {
  if (length(x) != length(y) || length(x) < 3L ||
      any(!is.finite(x)) || any(!is.finite(y)) || sd(x) == 0 || sd(y) == 0) {
    stop("Correlation requires at least three finite pairs with nonzero variance")
  }
  pearson <- cor.test(x, y, method = "pearson", alternative = "two.sided")
  # R assigns average ranks to ties; exact=FALSE uses an approximate p-value.
  spearman <- cor.test(x, y, method = "spearman", alternative = "two.sided", exact = FALSE)
  data.frame(n = length(x), pearson_r = unname(pearson$estimate), pearson_p = pearson$p.value,
    spearman_rho = unname(spearman$estimate), spearman_p = spearman$p.value)
}

calculate_correlations <- function(contexts) {
  rows <- list()
  for (dataset_name in names(DATA_FILES)) {
    sample <- contexts[contexts$dataset == dataset_name, ]
    for (i in seq_len(nrow(PAIRS))) {
      pair <- PAIRS[i, ]
      rows[[length(rows) + 1L]] <- cbind(data.frame(dataset = dataset_name,
        measure_x = pair$measure_x, measure_y = pair$measure_y),
        correlate_pair(sample[[pair$measure_x]], sample[[pair$measure_y]]))
    }
  }
  results <- bind_rows(rows)
  if (nrow(results) != length(DATA_FILES) * nrow(PAIRS) ||
      anyDuplicated(results[c("dataset", "measure_x", "measure_y")])) stop("Incomplete correlation grid")
  results
}

# ---- Round-trip CSV at sufficient precision to preserve double values ---------
write_results <- function(results, path) {
  output <- results
  columns <- c("pearson_r", "pearson_p", "spearman_rho", "spearman_p")
  for (column in columns) output[[column]] <- sprintf("%.17g", results[[column]])
  write.csv(output, path, row.names = FALSE, fileEncoding = "UTF-8", na = "NA")
  restored <- read.csv(path, stringsAsFactors = FALSE)
  for (column in names(results)) {
    if (!isTRUE(all.equal(restored[[column]], results[[column]],
                         tolerance = VALIDATION_TOLERANCE))) stop("CSV readback mismatch: ", column)
  }
  restored
}

# ---- Save numerical correlations -------------------------------------------
main <- function() {
  contexts <- bind_rows(lapply(names(DATA_FILES), function(dataset_name)
    read_contexts(DATA_FILES[[dataset_name]], dataset_name)))
  results <- calculate_correlations(contexts)
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  restored <- write_results(results, OUTPUT_CSV)
  print(restored, digits = 8, row.names = FALSE)
  message("Saved ", OUTPUT_CSV)
  invisible(list(contexts = contexts, results = restored))
}

if (sys.nframe() == 0L) main()
